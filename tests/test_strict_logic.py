import unittest
import json
import tempfile
from pathlib import Path

import causal_analyzer
import model_did
import panel_pipeline
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


if __name__ == "__main__":
    unittest.main()
