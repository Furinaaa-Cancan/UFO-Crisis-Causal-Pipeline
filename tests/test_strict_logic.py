import unittest
import json
import tempfile
from datetime import timedelta
from pathlib import Path

import causal_analyzer
import control_panel_builder
import model_did
import model_causal_ml
import model_synth_control
import panel_pipeline
import historical_backfill
import scraper
import strict_reviewer


class TestStrictLogic(unittest.TestCase):
    def test_build_scraped_snapshot_uses_run_day_only_counts(self):
        payload = {
            "scraped_at": "2026-02-21T12:00:00+00:00",
            "policy": "strict-balanced",
            "ufo_news": [
                {"date": "2026-02-21", "title": "ufo a", "authenticity": {"final_score": 80}},
                {"date": "2026-02-20", "title": "ufo b", "authenticity": {"final_score": 75}},
            ],
            "crisis_news": [
                {"date": "2026-02-21", "title": "crisis a", "authenticity": {"final_score": 82}},
                {"date": "2026-02-20", "title": "crisis b", "authenticity": {"final_score": 70}},
            ],
            "rejected_news": [
                {"date": "2026-02-21", "title": "rej a"},
                {"date": "2026-02-18", "title": "rej b"},
            ],
            "stats": {"accepted_events": 99, "raw_items": 120},
        }

        snap = causal_analyzer.build_scraped_snapshot(payload)
        self.assertEqual(snap["date"], "2026-02-21")
        self.assertEqual(snap["date_scope"], "run_day_only")
        self.assertEqual(snap["ufo_count"], 1)
        self.assertEqual(snap["crisis_count"], 1)
        self.assertEqual(snap["rejected_count"], 1)
        self.assertEqual(snap["accepted_events"], 2)
        self.assertEqual(snap["window_ufo_count"], 2)
        self.assertEqual(snap["window_crisis_count"], 2)
        self.assertEqual(snap["window_rejected_count"], 2)

    def test_dual_review_excludes_legacy_non_run_day_rows(self):
        payload = {
            "rows": [
                {"date": "2026-02-20", "policy": "strict", "ufo_count": 1, "crisis_count": 0, "date_scope": "run_day_only"},
                {"date": "2026-02-20", "policy": "strict-balanced", "ufo_count": 1, "crisis_count": 1, "date_scope": "run_day_only"},
                {"date": "2026-02-19", "policy": "strict", "ufo_count": 5, "crisis_count": 3},  # legacy row
            ]
        }
        review = panel_pipeline.compute_dual_policy_review(payload, min_overlap_days=1)
        self.assertIn("current", review)
        self.assertEqual(review["current"]["overlap_days"], 1)
        self.assertEqual(review["current"]["excluded_non_run_day_rows"], 1)

    def test_classify_level_l4_requires_reproducibility(self):
        summary = {
            "gates": {
                "core_passed": True,
                "falsification_passed": True,
                "reproducibility_passed": True,
            },
            "signals": {
                "has_temporal_signal": True,
                "verdict_has_correlation_phrase": True,
            },
        }
        self.assertEqual(strict_reviewer.classify_level(summary), "L4")

        summary["gates"]["reproducibility_passed"] = False
        self.assertEqual(strict_reviewer.classify_level(summary), "L3")

    def test_collapse_claim_clusters_merges_same_day_crisis_event_signature(self):
        items = [
            {
                "category": "crisis",
                "title": "Supreme Court strikes down Trump's tariffs",
                "description": "Court ruling and White House response",
                "date": "2026-02-21",
                "source": "A",
                "source_type": "rss",
                "domain": "a.com",
                "url": "https://a.com/1",
                "weight": 3,
                "authenticity": {"final_score": 90, "corroboration_count": 2, "trusted_corroboration": 2},
            },
            {
                "category": "crisis",
                "title": "Trump attacks Supreme Court after tariff ruling",
                "description": "White House says it will appeal",
                "date": "2026-02-21",
                "source": "B",
                "source_type": "rss",
                "domain": "b.com",
                "url": "https://b.com/2",
                "weight": 2,
                "authenticity": {"final_score": 85, "corroboration_count": 1, "trusted_corroboration": 1},
            },
            {
                "category": "crisis",
                "title": "DOJ files immigration lawsuit at border",
                "description": "Federal action in court",
                "date": "2026-02-21",
                "source": "C",
                "source_type": "rss",
                "domain": "c.com",
                "url": "https://c.com/3",
                "weight": 2,
                "authenticity": {"final_score": 84, "corroboration_count": 1, "trusted_corroboration": 1},
            },
        ]

        merged = scraper.collapse_claim_clusters(items)
        crisis = [x for x in merged if x.get("category") == "crisis"]
        # The first two are the same same-day tariff/supreme-court shock and should merge into one.
        self.assertEqual(len(crisis), 2)
        biggest = max(crisis, key=lambda x: x.get("merged_claims", 1))
        self.assertGreaterEqual(biggest.get("merged_claims", 1), 2)

    def test_analyze_scraped_reason_only_includes_failed_constraints(self):
        payload = {
            "policy": "strict-balanced",
            "ufo_news": [{"date": "2026-02-21", "title": "ufo"} for _ in range(4)],
            "crisis_news": [{"date": "2026-02-21", "title": "crisis"} for _ in range(12)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            fp = Path(tmp) / "scraped.json"
            fp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            stats = causal_analyzer.analyze_scraped(fp)

        self.assertFalse(stats.sufficient)
        # crisis=12 should not appear as a failed constraint; ufo and coverage should.
        self.assertNotIn("crisis=12 (<10)", stats.reason)
        self.assertIn("ufo=4 (<30)", stats.reason)

    def test_shock_threshold_uses_absolute_floor(self):
        self.assertEqual(causal_analyzer.compute_shock_threshold([1.0, 1.0, 1.0]), 2.0)
        self.assertEqual(panel_pipeline.compute_shock_threshold([]), 2.0)

    def test_did_min_distance_handles_empty_shocks(self):
        d = causal_analyzer.parse_date("2026-02-21")
        self.assertGreater(model_did.min_distance_to_shocks(d, []), 1000)

    def test_analyze_scraped_uses_reachable_coverage_target_from_lookback(self):
        # With lookback=60, coverage target should be 60 (not hard-coded 180).
        payload = {
            "policy": "strict-balanced",
            "lookback_days": 60,
            "ufo_news": [{"date": "2026-01-01", "title": "ufo"} for _ in range(29)] + [{"date": "2026-03-01", "title": "ufo"}],
            "crisis_news": [{"date": "2026-01-15", "title": "crisis"} for _ in range(9)] + [{"date": "2026-02-20", "title": "crisis"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            fp = Path(tmp) / "scraped.json"
            fp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            stats = causal_analyzer.analyze_scraped(fp)

        # crisis/ufo counts meet thresholds and date span is 60+ days, so this gate should be reachable.
        self.assertTrue(stats.sufficient)

    def test_panel_readers_prefer_run_day_rows(self):
        panel_payload = {
            "rows": [
                {"date": "2026-02-20", "policy": "strict-balanced", "ufo_count": 9, "crisis_count": 9},  # legacy
                {
                    "date": "2026-02-20",
                    "policy": "strict-balanced",
                    "ufo_count": 2,
                    "crisis_count": 3,
                    "date_scope": "run_day_only",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            panel_path = Path(tmp) / "panel.json"
            panel_path.write_text(json.dumps(panel_payload, ensure_ascii=False), encoding="utf-8")

            # model_did reader
            old_panel_file = model_did.PANEL_FILE
            model_did.PANEL_FILE = panel_path
            try:
                rows = model_did.read_panel_rows("strict-balanced")
            finally:
                model_did.PANEL_FILE = old_panel_file
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["ufo_count"]), 2.0)

            # panel_pipeline reader
            old_pipeline_panel = panel_pipeline.PANEL_FILE
            panel_pipeline.PANEL_FILE = panel_path
            try:
                rows2 = panel_pipeline.load_panel_rows("strict-balanced")
            finally:
                panel_pipeline.PANEL_FILE = old_pipeline_panel
            self.assertEqual(len(rows2), 1)
            self.assertEqual(float(rows2[0]["crisis_count"]), 3.0)

    def test_keyword_matching_avoids_substring_false_positive(self):
        self.assertFalse(control_panel_builder.keyword_match("he said this today", {"ai"}))
        self.assertTrue(control_panel_builder.keyword_match("AI policy update", {"ai"}))

        self.assertEqual(scraper.keyword_hits("forward guidance update", ["war"]), [])
        self.assertEqual(scraper.keyword_hits("war update", ["war"]), ["war"])

    def test_synth_pre_dates_exclude_shock_day(self):
        start = causal_analyzer.parse_date("2026-01-01")
        us_dates = [start + timedelta(days=i) for i in range(10)]
        shocks = [start + timedelta(days=4)]  # 2026-01-05
        pre_dates, post_dates = model_synth_control.split_pre_post_dates(us_dates, shocks, post_horizon_days=7)

        self.assertNotIn(start + timedelta(days=4), pre_dates)
        self.assertIn(start + timedelta(days=5), post_dates)

    def test_progress_earliest_ready_date_considers_shock_gap(self):
        start = causal_analyzer.parse_date("2026-01-01")
        rows = []
        for i in range(10):
            d = (start + timedelta(days=i)).isoformat()
            rows.append({
                "date": d,
                "crisis_count": 3 if i == 0 else 0,
                "ufo_count": 0,
            })

        report = panel_pipeline.compute_progress(
            rows=rows,
            min_days=12,
            min_shocks=5,
            min_observed_ratio=0.5,
            policy="strict-balanced",
        )
        self.assertEqual(report["remaining"]["days"], 2)
        self.assertEqual(report["remaining"]["shocks"], 4)
        self.assertEqual(report["remaining"]["earliest_ready_date_if_daily"], "2026-01-14")

    def test_reproducibility_same_day_rerun_does_not_regress_after_cross_day(self):
        signature = {"approval_status": "REJECTED"}

        # Cross-day repeat establishes reproducibility.
        prev_cross_day = {
            "generated_at": "2026-02-20T10:00:00+00:00",
            "meta": {"signature": signature},
            "gates": {"reproducibility_passed": True},
        }
        curr_ts = strict_reviewer._parse_iso_ts("2026-02-21T10:00:00+00:00")
        passed, cross_day, same_day = strict_reviewer.evaluate_reproducibility(
            prev_cross_day,
            signature,
            curr_ts,
        )
        self.assertTrue(passed)
        self.assertTrue(cross_day)
        self.assertFalse(same_day)

        # Same-day rerun keeps reproducibility if previous snapshot already passed.
        prev_same_day_passed = {
            "generated_at": "2026-02-21T10:05:00+00:00",
            "meta": {"signature": signature},
            "gates": {"reproducibility_passed": True},
        }
        curr_ts2 = strict_reviewer._parse_iso_ts("2026-02-21T10:06:00+00:00")
        passed2, cross_day2, same_day2 = strict_reviewer.evaluate_reproducibility(
            prev_same_day_passed,
            signature,
            curr_ts2,
        )
        self.assertTrue(passed2)
        self.assertFalse(cross_day2)
        self.assertTrue(same_day2)

        # If previous same-day snapshot had not passed, keep it failed.
        prev_same_day_failed = {
            "generated_at": "2026-02-21T10:07:00+00:00",
            "meta": {"signature": signature},
            "gates": {"reproducibility_passed": False},
        }
        curr_ts3 = strict_reviewer._parse_iso_ts("2026-02-21T10:08:00+00:00")
        passed3, _, same_day3 = strict_reviewer.evaluate_reproducibility(
            prev_same_day_failed,
            signature,
            curr_ts3,
        )
        self.assertFalse(passed3)
        self.assertTrue(same_day3)

    def test_coverage_audit_includes_source_recency_summary(self):
        raw_items = [
            {"source": "A", "date": "2026-02-21", "title": "x"},
            {"source": "A", "date": "2026-02-20", "title": "y"},
            {"source": "B", "date": "2026-01-01", "title": "z"},
        ]
        source_stats = [
            {"source": "A", "status": "ok", "item_count": 2, "used_fallback": False},
            {"source": "B", "status": "ok", "item_count": 1, "used_fallback": False},
        ]
        coverage = scraper.build_coverage_audit(
            raw_items=raw_items,
            unique_items=raw_items,
            candidates=raw_items,
            accepted_events=[raw_items[0]],
            rejected_items=[{"reason": "stale_item_outside_lookback"}],
            source_stats=source_stats,
            lookback_days=120,
        )

        self.assertEqual(coverage["rejections"]["stale_item_outside_lookback"], 1)
        self.assertEqual(coverage["date_spans"]["raw_items"]["min_date"], "2026-01-01")
        self.assertEqual(coverage["source_window_summary"]["sources_total"], 2)

    def test_causal_ml_build_dataset_has_lag_features_and_treatment(self):
        rows = []
        start = causal_analyzer.parse_date("2026-01-01")
        for i in range(20):
            d = (start + timedelta(days=i)).isoformat()
            rows.append({
                "date": d,
                "ufo_count": 1 if i % 3 == 0 else 0,
                "crisis_count": 3 if i in (10, 12, 14, 16) else 0,
                "control_total": 0,
                "control_density_accepted": 0.0,
            })

        dataset, feature_names = model_causal_ml.build_dataset(rows, crisis_threshold=2.0)
        treated = sum(int(r["t"]) for r in dataset)
        self.assertGreater(len(feature_names), 8)
        self.assertGreater(len(dataset), 5)
        self.assertGreaterEqual(treated, 1)

    def test_causal_ml_strict_pass_requires_non_fallback_models(self):
        passed = model_causal_ml.compute_causal_ml_pass(
            ate_positive=True,
            ate_significant=True,
            nuisance_model_ready=False,
            cate_model_ready=False,
            heterogeneity_estimated=False,
        )
        self.assertFalse(passed)

        passed2 = model_causal_ml.compute_causal_ml_pass(
            ate_positive=True,
            ate_significant=True,
            nuisance_model_ready=True,
            cate_model_ready=True,
            heterogeneity_estimated=True,
        )
        self.assertTrue(passed2)

    def test_control_panel_date_grid_is_inclusive(self):
        start = causal_analyzer.parse_date("2026-01-01")
        end = causal_analyzer.parse_date("2026-01-03")
        grid = control_panel_builder.build_date_grid(start, end)
        self.assertEqual(grid, ["2026-01-01", "2026-01-02", "2026-01-03"])

    def test_official_source_detection(self):
        self.assertTrue(strict_reviewer.is_official_source("Pentagon / DoD 新闻稿"))
        self.assertTrue(strict_reviewer.is_official_source("White House 新闻稿"))
        self.assertFalse(strict_reviewer.is_official_source("BBC 美国&加拿大新闻"))

    def test_mechanism_summary_detects_official_lead_proxy(self):
        scraped = {
            "ufo_news": [
                {
                    "source": "Pentagon / DoD 新闻稿",
                    "primary_source": "Pentagon / DoD 新闻稿",
                    "corroborated_sources": ["BBC 美国&加拿大新闻", "NYT 美国新闻"],
                },
                {
                    "source": "BBC 美国&加拿大新闻",
                    "primary_source": "BBC 美国&加拿大新闻",
                    "corroborated_sources": ["Pentagon / DoD 新闻稿"],
                },
                {
                    "source": "NYT 美国新闻",
                    "primary_source": "NYT 美国新闻",
                    "corroborated_sources": [],
                },
            ]
        }
        m = strict_reviewer.summarize_mechanism_signals(
            scraped,
            min_official_share=0.3,
            min_official_lead_events=1,
        )
        self.assertEqual(m["metrics"]["ufo_events_total"], 3)
        self.assertGreaterEqual(m["metrics"]["official_source_share"], 0.3)
        self.assertGreaterEqual(m["metrics"]["official_primary_with_media_followup_events"], 1)
        self.assertEqual(m["lead_basis"], "source_order_proxy")
        self.assertTrue(m["mechanism_passed"])

    def test_mechanism_summary_prefers_lag_when_available(self):
        scraped = {
            "ufo_news": [
                {"source": "NYT 美国新闻", "official_to_media_lag_days": 2, "corroborated_sources": ["Pentagon / DoD 新闻稿"]},
                {"source": "BBC 美国&加拿大新闻", "official_to_media_lag_days": 0, "corroborated_sources": ["White House 新闻稿"]},
                {"source": "Reuters", "official_to_media_lag_days": -1, "corroborated_sources": ["White House 新闻稿"]},
            ]
        }
        m = strict_reviewer.summarize_mechanism_signals(
            scraped,
            min_official_share=0.3,
            min_official_lead_events=2,
        )
        self.assertEqual(m["lead_basis"], "lag")
        self.assertEqual(m["metrics"]["lag_observed_events"], 3)
        self.assertEqual(m["metrics"]["official_lead_by_lag_events"], 2)
        self.assertTrue(m["gates"]["official_lead_events>=2"])

    def test_inference_matrix_transitions(self):
        mechanism_fail = {"mechanism_passed": False}
        mechanism_pass = {"mechanism_passed": True}

        s1 = {
            "signals": {"has_temporal_signal": True, "verdict_has_correlation_phrase": True},
            "gates": {"core_passed": False, "falsification_passed": False},
        }
        i1 = strict_reviewer.build_inference_matrix(s1, mechanism_fail)
        self.assertEqual(i1["level"], "TEMPORAL_ASSOCIATION_ONLY")

        s2 = {
            "signals": {"has_temporal_signal": True, "verdict_has_correlation_phrase": True},
            "gates": {"core_passed": True, "falsification_passed": True},
        }
        i2 = strict_reviewer.build_inference_matrix(s2, mechanism_fail)
        self.assertEqual(i2["level"], "CAUSAL_SIGNAL_WITHOUT_STRATEGIC_MECHANISM")

        i3 = strict_reviewer.build_inference_matrix(s2, mechanism_pass)
        self.assertEqual(i3["level"], "STRATEGIC_COMMUNICATION_INDICATION")

    def test_collapse_claim_clusters_outputs_official_media_lag(self):
        items = [
            {
                "category": "ufo",
                "claim_fingerprint": "fp-ufo-lag",
                "title": "DoD releases UAP memo",
                "description": "Official briefing",
                "date": "2026-02-20",
                "source": "Pentagon / DoD 新闻稿",
                "source_type": "rss",
                "domain": "defense.gov",
                "url": "https://defense.gov/a",
                "weight": 3,
                "authenticity": {"final_score": 90, "corroboration_count": 1, "trusted_corroboration": 1},
            },
            {
                "category": "ufo",
                "claim_fingerprint": "fp-ufo-lag",
                "title": "Media report on DoD UAP memo",
                "description": "Coverage follow-up",
                "date": "2026-02-21",
                "source": "NYT 美国新闻",
                "source_type": "rss",
                "domain": "nytimes.com",
                "url": "https://nytimes.com/b",
                "weight": 3,
                "authenticity": {"final_score": 87, "corroboration_count": 1, "trusted_corroboration": 1},
            },
        ]

        collapsed = scraper.collapse_claim_clusters(items)
        self.assertEqual(len(collapsed), 1)
        e = collapsed[0]
        self.assertEqual(e.get("first_official_date"), "2026-02-20")
        self.assertEqual(e.get("first_media_date"), "2026-02-21")
        self.assertEqual(e.get("official_to_media_lag_days"), 1)
        self.assertTrue(e.get("official_leads_media"))
        self.assertGreaterEqual(len(e.get("corroboration_timeline", [])), 2)

    def test_historical_backfill_build_row_shapes_panel_fields(self):
        row = historical_backfill.build_row(
            day_iso="2020-01-02",
            policy="strict-balanced",
            ufo_count=5,
            crisis_count=7,
            ctrl_economy=3,
            ctrl_security=4,
            ctrl_immigration=1,
        )
        self.assertEqual(row["date"], "2020-01-02")
        self.assertEqual(row["policy"], "strict-balanced")
        self.assertEqual(row["ufo_count"], 5)
        self.assertEqual(row["crisis_count"], 7)
        self.assertEqual(row["control_total"], 8)
        self.assertEqual(row["data_source"], historical_backfill.BACKFILL_TAG)
        self.assertEqual(row["date_scope"], "run_day_only")

    def test_historical_backfill_history_is_append_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "runs.json"
            old_history = historical_backfill.REPORT_HISTORY_FILE
            historical_backfill.REPORT_HISTORY_FILE = history_path
            try:
                historical_backfill.append_report_history({"run_id": "r1"})
                historical_backfill.append_report_history({"run_id": "r2"})
            finally:
                historical_backfill.REPORT_HISTORY_FILE = old_history

            payload = json.loads(history_path.read_text(encoding="utf-8"))
            self.assertEqual([x["run_id"] for x in payload["runs"]], ["r1", "r2"])

    def test_historical_backfill_query_subset_parser(self):
        q = historical_backfill.parse_selected_queries("ufo, crisis")
        self.assertEqual(q, ["ufo", "crisis"])
        with self.assertRaises(ValueError):
            historical_backfill.parse_selected_queries("ufo,unknown_query")

    def test_historical_backfill_split_date_range(self):
        s = causal_analyzer.parse_date("1990-01-01")
        e = causal_analyzer.parse_date("1990-01-10")
        (l1, l2), (r1, r2) = historical_backfill.split_date_range(s, e)
        self.assertEqual(l1.isoformat(), "1990-01-01")
        self.assertEqual(l2.isoformat(), "1990-01-05")
        self.assertEqual(r1.isoformat(), "1990-01-06")
        self.assertEqual(r2.isoformat(), "1990-01-10")


if __name__ == "__main__":
    unittest.main()
