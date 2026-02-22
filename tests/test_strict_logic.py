import unittest
from unittest import mock
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import causal_analyzer
import control_panel_builder
import model_did
import model_event_study
import model_causal_ml
import model_synth_control
import panel_pipeline
import historical_backfill
import replay_backfill_failures
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

    def test_collect_source_urls_for_retry_merges_and_dedupes(self):
        src = {
            "url": "https://a.example/rss",
            "fallback_url": "https://b.example/rss",
            "fallback_urls": [
                "https://b.example/rss",
                "https://c.example/rss",
                "",
                "https://a.example/rss",
            ],
        }
        urls = scraper.collect_source_urls_for_retry(src)
        self.assertEqual(
            urls,
            [
                "https://a.example/rss",
                "https://b.example/rss",
                "https://c.example/rss",
            ],
        )

    def test_collect_paged_source_urls_builds_expected_range(self):
        src = {
            "paged_url_template": "https://x.example/feed/?paged={page}",
            "paged_start": 2,
            "paged_end": 4,
        }
        self.assertEqual(
            scraper.collect_paged_source_urls(src),
            [
                "https://x.example/feed/?paged=2",
                "https://x.example/feed/?paged=3",
                "https://x.example/feed/?paged=4",
            ],
        )

    def test_collect_source_urls_for_retry_can_include_paged_urls(self):
        src = {
            "url": "https://a.example/rss",
            "fallback_url": "https://b.example/rss",
            "paged_url_template": "https://a.example/rss?paged={page}",
            "paged_start": 1,
            "paged_end": 2,
        }
        urls = scraper.collect_source_urls_for_retry(src, include_paged=True)
        self.assertEqual(
            urls,
            [
                "https://a.example/rss",
                "https://b.example/rss",
                "https://a.example/rss?paged=1",
                "https://a.example/rss?paged=2",
            ],
        )

    def test_fetch_rss_merges_paged_urls_and_stops_after_repeated_no_new_pages(self):
        source = {
            "name": "Test Feed",
            "category": "ufo",
            "type": "rss",
            "url": "https://x.example/feed/?paged=1",
            "paged_url_template": "https://x.example/feed/?paged={page}",
            "paged_start": 1,
            "paged_end": 5,
        }

        def item(url, title, date):
            return {
                "source": source["name"],
                "source_type": "rss",
                "category": "ufo",
                "weight": 2,
                "title": title,
                "date": date,
                "published_at": f"{date}T00:00:00+00:00",
                "url": url,
                "domain": "x.example",
                "description": "",
            }

        page_payload = {
            "https://x.example/feed/?paged=1": ([item("https://x.example/a", "A", "2026-01-05")], None),
            "https://x.example/feed/?paged=2": ([item("https://x.example/b", "B", "2026-01-04")], None),
            "https://x.example/feed/?paged=3": ([item("https://x.example/b", "B", "2026-01-04")], None),
            "https://x.example/feed/?paged=4": ([item("https://x.example/b", "B", "2026-01-04")], None),
            "https://x.example/feed/?paged=5": ([item("https://x.example/c", "C", "2026-01-03")], None),
        }

        def fake_fetch_rss_url(url, src, max_items):
            self.assertEqual(src["name"], source["name"])
            return page_payload[url]

        with mock.patch.object(scraper, "fetch_rss_url", side_effect=fake_fetch_rss_url):
            out = scraper.fetch_rss(source, max_items_per_source=10)

        self.assertIsNone(out["error"])
        self.assertEqual(len(out["items"]), 2)
        self.assertEqual(out["attempted_urls"], [
            "https://x.example/feed/?paged=1",
            "https://x.example/feed/?paged=2",
            "https://x.example/feed/?paged=3",
            "https://x.example/feed/?paged=4",
        ])

    def test_extract_date_from_url_fallback(self):
        self.assertEqual(
            scraper.extract_date_from_url("https://whitehouse.gov/briefings-statements/2026/02/21/example"),
            "2026-02-21",
        )
        self.assertEqual(
            scraper.extract_date_from_url("https://example.com/news/2026-01-09-uap-hearing"),
            "2026-01-09",
        )
        self.assertIsNone(scraper.extract_date_from_url("https://example.com/news/latest"))

    def test_deduplicate_within_source_keeps_distinct_urls_when_date_missing(self):
        items = [
            {"source": "Senate Feed", "title": "Closed Briefing: Intelligence Matters", "date": "", "url": "https://x/2026/01/30/a"},
            {"source": "Senate Feed", "title": "Closed Briefing: Intelligence Matters", "date": "", "url": "https://x/2026/01/08/b"},
        ]
        unique, rejected = scraper.deduplicate_within_source(items)
        self.assertEqual(len(unique), 2)
        self.assertEqual(len(rejected), 0)

    def test_base_authenticity_ambiguous_ufo_term_needs_core_signal(self):
        today = datetime.now(timezone.utc).date().isoformat()
        items = [
            {
                "source": "Pentagon / DoD 新闻稿",
                "source_type": "rss",
                "category": "ufo",
                "weight": 3,
                "title": "Department expands reverse engineering training program",
                "description": "Procurement and workforce update",
                "date": today,
                "published_at": "2026-02-22T09:30:00+00:00",
                "date_parsed_ok": True,
                "url": "https://www.defense.gov/news/2026/02/22/program",
                "domain": "www.defense.gov",
            },
        ]
        accepted, rejected = scraper.evaluate_base_authenticity(
            items=items,
            lookback_days=30,
            policy=scraper.POLICY_CONFIGS[scraper.POLICY_LENIENT],
        )
        self.assertEqual(len(accepted), 0)
        self.assertEqual(len(rejected), 1)
        self.assertIn("no_relevance_keywords", rejected[0]["reason"])

    def test_infer_ufo_topic_tag_ignores_immigration_alien_context(self):
        text = "The administration said it will deport a violent illegal alien at the border."
        self.assertEqual(scraper.infer_ufo_topic_tag(text), "na")

        nhi_text = "Trump said UFO files and alien life disclosures will be released."
        self.assertEqual(scraper.infer_ufo_topic_tag(nhi_text), "ufo_uap")

    def test_base_authenticity_rejects_astrobiology_alien_noise_in_strict_balanced(self):
        today = datetime.now(timezone.utc).date().isoformat()
        items = [
            {
                "source": "Space.com",
                "source_type": "rss",
                "category": "ufo",
                "weight": 2,
                "title": "Thousands of alien plant species found in deep ocean simulation",
                "description": "A biology study on alien-like ecosystems in extreme environments.",
                "date": today,
                "published_at": f"{today}T10:00:00+00:00",
                "date_parsed_ok": True,
                "url": "https://www.space.com/alien-plant-species-study",
                "domain": "space.com",
            },
        ]
        accepted, rejected = scraper.evaluate_base_authenticity(
            items=items,
            lookback_days=30,
            policy=scraper.POLICY_CONFIGS[scraper.POLICY_STRICT_BALANCED],
        )
        self.assertEqual(len(accepted), 0)
        self.assertEqual(len(rejected), 1)
        self.assertIn("ufo_without_strong_signal", rejected[0]["reason"])

    def test_base_authenticity_rejects_ufo_entertainment_without_government_context(self):
        today = datetime.now(timezone.utc).date().isoformat()
        items = [
            {
                "source": "NBC News Science（补充）",
                "source_type": "rss",
                "category": "ufo",
                "weight": 2,
                "title": "Watch the trailer for new UFO documentary series",
                "description": "The streaming series follows a filmmaker interviewing witnesses.",
                "date": today,
                "published_at": f"{today}T09:00:00+00:00",
                "date_parsed_ok": True,
                "url": "https://www.nbcnews.com/science/ufo-documentary-series-trailer",
                "domain": "nbcnews.com",
            },
        ]
        accepted, rejected = scraper.evaluate_base_authenticity(
            items=items,
            lookback_days=30,
            policy=scraper.POLICY_CONFIGS[scraper.POLICY_STRICT_BALANCED],
        )
        self.assertEqual(len(accepted), 0)
        self.assertEqual(len(rejected), 1)
        self.assertIn("ufo_without_government_context", rejected[0]["reason"])

    def test_ufo_signature_merge_combines_same_event_across_sources(self):
        items = [
            {
                "category": "ufo",
                "claim_fingerprint": "fp-a",
                "title": "Trump tells Pentagon to release files on UFOs",
                "description": "White House disclosure move",
                "date": "2026-02-21",
                "published_at": "2026-02-21T00:30:00+00:00",
                "source": "CBS News 政治（补充）",
                "source_type": "rss",
                "domain": "cbsnews.com",
                "url": "https://cbsnews.com/a",
                "weight": 2,
                "authenticity": {"final_score": 88, "corroboration_count": 1, "trusted_corroboration": 1},
            },
            {
                "category": "ufo",
                "claim_fingerprint": "fp-b",
                "title": "Trump says US government will declassify its UFO files",
                "description": "Disclosure discussion",
                "date": "2026-02-20",
                "published_at": "2026-02-20T19:00:00+00:00",
                "source": "Space.com",
                "source_type": "rss",
                "domain": "space.com",
                "url": "https://space.com/b",
                "weight": 2,
                "authenticity": {"final_score": 85, "corroboration_count": 1, "trusted_corroboration": 1},
            },
        ]
        collapsed = scraper.collapse_claim_clusters(items)
        self.assertEqual(len(collapsed), 1)
        e = collapsed[0]
        self.assertGreaterEqual(e.get("cluster_size", 0), 2)
        self.assertGreaterEqual(len(e.get("corroborated_sources", [])), 2)
        self.assertTrue(str(e.get("ufo_event_signature", "")).startswith("2026-02-16|act:disclosure"))

    def test_build_official_media_pairs_detects_strict_positive_lag(self):
        items = [
            {
                "category": "ufo",
                "source": "Pentagon / DoD 新闻稿",
                "source_type": "rss",
                "weight": 3,
                "title": "Department of Defense Releases Annual UAP Report",
                "description": "Official UAP annual assessment release",
                "date": "2024-11-14",
                "published_at": "2024-11-14T12:00:00+00:00",
                "url": "https://www.defense.gov/a",
                "domain": "defense.gov",
                "authenticity": {"final_score": 88},
            },
            {
                "category": "ufo",
                "source": "BBC 美国&加拿大新闻",
                "source_type": "rss",
                "weight": 2,
                "title": "Pentagon UAP annual report draws new scrutiny",
                "description": "Media follow-up on the UAP report",
                "date": "2024-11-15",
                "published_at": "2024-11-15T06:00:00+00:00",
                "url": "https://www.bbc.com/b",
                "domain": "bbc.com",
                "authenticity": {"final_score": 84},
            },
        ]
        pairs = scraper.build_official_media_pairs(items, max_lag_days=10, min_semantic_score=2, min_base_score=55)
        summary = pairs.get("summary", {})
        self.assertGreaterEqual(summary.get("official_items_considered", 0), 1)
        self.assertGreaterEqual(summary.get("media_items_considered", 0), 1)
        self.assertGreaterEqual(summary.get("strict_pairs", 0), 1)
        self.assertGreaterEqual(summary.get("strict_positive_lag_pairs", 0), 1)

    def test_build_official_media_pairs_keeps_google_proxy_separate_from_strict(self):
        items = [
            {
                "category": "ufo",
                "source": "Pentagon / DoD 新闻稿",
                "source_type": "rss",
                "weight": 3,
                "title": "Department of Defense Releases Annual UAP Report",
                "description": "Official UAP annual assessment release",
                "date": "2024-11-14",
                "published_at": "2024-11-14T12:00:00+00:00",
                "url": "https://www.defense.gov/a",
                "domain": "defense.gov",
                "authenticity": {"final_score": 88},
            },
            {
                "category": "ufo",
                "source": "Google News - UAP政府披露（补充）",
                "source_type": "rss",
                "weight": 1,
                "title": "DoD annual UAP report sparks congressional scrutiny - DefenseScoop",
                "description": "Defense reporting on official UAP report",
                "date": "2024-11-15",
                "published_at": "2024-11-15T06:00:00+00:00",
                "url": "https://news.google.com/rss/articles/abc",
                "domain": "news.google.com",
                "publisher_name": "DefenseScoop",
                "publisher_url": "https://defensescoop.com",
                "publisher_domain": "defensescoop.com",
                "authenticity": {"final_score": 62, "is_aggregator": True},
            },
        ]
        pairs = scraper.build_official_media_pairs(items, max_lag_days=10, min_semantic_score=2, min_base_score=55)
        summary = pairs.get("summary", {})
        self.assertEqual(summary.get("strict_pairs", 0), 0)
        self.assertGreaterEqual(summary.get("proxy_strict_pairs", 0), 1)
        self.assertGreaterEqual(summary.get("pairs_with_resolved_publisher", 0), 1)
        self.assertEqual(pairs["pairs"][0]["evidence_tier"], "proxy_strict")
        self.assertTrue(pairs["pairs"][0]["media_is_aggregator_proxy"])

    def test_build_official_media_pairs_ignores_immigration_alien_false_positive(self):
        items = [
            {
                "category": "crisis",
                "source": "White House 新闻稿",
                "source_type": "rss",
                "weight": 3,
                "title": "Minnesota Democrats Protected This Violent Illegal Alien",
                "description": "The administration is deporting this criminal illegal alien.",
                "date": "2026-02-03",
                "published_at": "2026-02-03T20:12:20+00:00",
                "url": "https://whitehouse.gov/articles/a",
                "domain": "whitehouse.gov",
                "authenticity": {"final_score": 90},
            },
            {
                "category": "ufo",
                "source": "CBS News 政治（补充）",
                "source_type": "rss",
                "weight": 2,
                "title": "Trump tells Pentagon to release UFO files",
                "description": "UFO disclosure in the United States.",
                "date": "2026-02-21",
                "published_at": "2026-02-21T00:38:39+00:00",
                "url": "https://cbsnews.com/news/ufo-files",
                "domain": "cbsnews.com",
                "authenticity": {"final_score": 83},
            },
        ]
        pairs = scraper.build_official_media_pairs(items, max_lag_days=30, min_semantic_score=2, min_base_score=55)
        self.assertEqual(pairs.get("summary", {}).get("official_items_considered"), 0)
        self.assertEqual(pairs.get("summary", {}).get("total_pairs"), 0)

    def test_base_authenticity_prefers_official_items_with_parseable_timestamp(self):
        today = datetime.now(timezone.utc).date().isoformat()
        items = [
            {
                "source": "White House 新闻稿",
                "source_type": "rss",
                "category": "crisis",
                "weight": 3,
                "title": "White House and DOJ announce investigation update",
                "description": "Federal investigation and White House response",
                "date": today,
                "published_at": "2026-02-22T10:30:00+00:00",
                "date_parsed_ok": True,
                "url": "https://www.whitehouse.gov/a",
                "domain": "www.whitehouse.gov",
            },
            {
                "source": "White House 新闻稿",
                "source_type": "rss",
                "category": "crisis",
                "weight": 3,
                "title": "White House and DOJ announce investigation update 2",
                "description": "Federal investigation and White House response",
                "date": today,
                "published_at": None,
                "date_parsed_ok": True,
                "url": "https://www.whitehouse.gov/b",
                "domain": "www.whitehouse.gov",
            },
        ]
        accepted, rejected = scraper.evaluate_base_authenticity(
            items=items,
            lookback_days=30,
            policy=scraper.POLICY_CONFIGS[scraper.POLICY_LENIENT],
        )
        self.assertEqual(len(rejected), 0)
        self.assertEqual(len(accepted), 2)
        by_url = {x["url"]: x for x in accepted}
        score_ts = by_url["https://www.whitehouse.gov/a"]["authenticity"]["base_score"]
        score_no_ts = by_url["https://www.whitehouse.gov/b"]["authenticity"]["base_score"]
        self.assertGreater(score_ts, score_no_ts)
        self.assertIn(
            "official_missing_published_at",
            by_url["https://www.whitehouse.gov/b"]["authenticity"]["flags"],
        )

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

    def test_causal_ml_nuisance_uses_linear_ridge_without_sklearn(self):
        old_ready = model_causal_ml.SKLEARN_READY
        model_causal_ml.SKLEARN_READY = False
        try:
            data = []
            for i in range(60):
                x1 = float(i % 10)
                x2 = float((i // 10) % 6)
                t = 1 if (i % 5) in (0, 1) else 0
                y = 0.4 * x1 + 1.5 * t + 0.1 * x2
                data.append({"x": [x1, x2], "y": y, "t": t})

            m_hat, e_hat, method = model_causal_ml.estimate_nuisance(data, folds=5)
            self.assertEqual(method, "cross_fitted_linear_ridge")
            self.assertEqual(len(m_hat), 60)
            self.assertEqual(len(e_hat), 60)
            self.assertTrue(all(0.03 <= x <= 0.97 for x in e_hat))
        finally:
            model_causal_ml.SKLEARN_READY = old_ready

    def test_causal_ml_cate_uses_linear_proxy_without_sklearn(self):
        old_ready = model_causal_ml.SKLEARN_READY
        model_causal_ml.SKLEARN_READY = False
        try:
            data = []
            for i in range(80):
                x1 = float(i % 10)
                x2 = float((i // 10) % 8)
                t = 1 if (i % 4) in (0, 1) else 0
                y = 0.3 * x1 + 2.0 * t + 0.2 * x2
                data.append({"x": [x1, x2], "y": y, "t": t})

            ys = [float(r["y"]) for r in data]
            ts = [int(r["t"]) for r in data]
            m_hat, e_hat, _ = model_causal_ml.estimate_nuisance(data, folds=4)
            _, y_res, t_res, _ = model_causal_ml.orthogonal_ate(ys, ts, m_hat, e_hat)
            tau, method = model_causal_ml.estimate_cate(data, y_res, t_res)

            self.assertEqual(method, "orthogonal_linear_ridge_proxy")
            self.assertEqual(len(tau), 80)
            self.assertGreater(max(tau) - min(tau), 0.0)
        finally:
            model_causal_ml.SKLEARN_READY = old_ready

    def test_control_panel_date_grid_is_inclusive(self):
        start = causal_analyzer.parse_date("2026-01-01")
        end = causal_analyzer.parse_date("2026-01-03")
        grid = control_panel_builder.build_date_grid(start, end)
        self.assertEqual(grid, ["2026-01-01", "2026-01-02", "2026-01-03"])

    def test_event_study_downsample_dates_evenly_keeps_bounds(self):
        start = causal_analyzer.parse_date("2026-01-01")
        dates = [start + timedelta(days=i) for i in range(10)]
        sampled = model_event_study.downsample_dates_evenly(dates, 4)
        self.assertEqual(len(sampled), 4)
        self.assertEqual(sampled[0], dates[0])
        self.assertEqual(sampled[-1], dates[-1])
        self.assertTrue(all(sampled[i] < sampled[i + 1] for i in range(len(sampled) - 1)))

    def test_event_study_placebo_respects_permutations_arg(self):
        start = causal_analyzer.parse_date("2025-01-01")
        dates = [start + timedelta(days=i) for i in range(140)]
        series = {d: float((i % 7) + 1) for i, d in enumerate(dates)}
        shocks = [start + timedelta(days=i) for i in range(45, 80, 3)]
        candidates = [start + timedelta(days=i) for i in range(90, 130, 2)]
        obs_peak, p_val, draws = model_event_study.placebo_peak_test(
            series=series,
            shocks=shocks,
            candidates=candidates,
            permutations=15,
            baseline_cache={},
        )
        self.assertIsInstance(obs_peak, float)
        self.assertGreaterEqual(p_val, 0.0)
        self.assertLessEqual(p_val, 1.0)
        self.assertGreater(draws, 0)
        self.assertLessEqual(draws, 15)

    def test_official_source_detection(self):
        self.assertTrue(strict_reviewer.is_official_source("Pentagon / DoD 新闻稿"))
        self.assertTrue(strict_reviewer.is_official_source("White House 新闻稿"))
        self.assertFalse(strict_reviewer.is_official_source("BBC 美国&加拿大新闻"))
        self.assertTrue(scraper.source_is_official("Pentagon / DoD 新闻稿"))
        self.assertFalse(scraper.source_is_official("Google News - AARO非人类智慧（补充）"))

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

    def test_mechanism_summary_uses_pair_strict_lag_when_event_lag_missing(self):
        scraped = {
            "ufo_news": [
                {"source": "BBC 美国&加拿大新闻", "corroborated_sources": []},
            ]
        }
        pair_payload = {
            "summary": {
                "total_pairs": 2,
                "strict_pairs": 2,
                "balanced_pairs": 0,
                "strict_nonnegative_lag_pairs": 2,
                "strict_positive_lag_pairs": 1,
                "strict_with_timestamp_pairs": 1,
                "official_events_with_strict_followup": 1,
            }
        }
        m = strict_reviewer.summarize_mechanism_signals(
            scraped=scraped,
            min_official_share=0.0,
            min_official_lead_events=1,
            min_ufo_events=1,
            official_media_pairs=pair_payload,
        )
        self.assertEqual(m["lead_basis"], "pair_strict_lag")
        self.assertEqual(m["metrics"]["official_lead_events"], 1)
        self.assertEqual(m["metrics"]["pair_strict"], 2)
        self.assertTrue(m["gates"]["official_lead_events>=1"])

    def test_historical_mechanism_summary_uses_us_cases_only(self):
        events_v2 = {
            "correlations": [
                {"region": "USA", "confidence": "HIGH", "gap_days": 10, "ufo_event": {"government_action": True}},
                {"region": "US", "confidence": "MEDIUM", "gap_days": 5, "ufo_event": {"government_action": False}},
                {"region": "UK", "confidence": "HIGH", "gap_days": 20, "ufo_event": {"government_action": True}},
            ]
        }
        h = strict_reviewer.summarize_historical_mechanism(events_v2)
        self.assertEqual(h["status"], "ok")
        self.assertEqual(h["metrics"]["ufo_events_total"], 2)
        self.assertEqual(h["metrics"]["government_action_events"], 1)
        self.assertEqual(h["metrics"]["positive_gap_events"], 2)
        self.assertEqual(h["metrics"]["confidence_counts"]["HIGH"], 1)
        self.assertEqual(h["metrics"]["confidence_counts"]["MEDIUM"], 1)

    def test_mechanism_summary_blends_live_and_historical_for_sample_and_share(self):
        scraped = {
            "ufo_news": [
                {"source": "NYT 美国新闻", "primary_source": "NYT 美国新闻", "corroborated_sources": []},
                {"source": "BBC 美国&加拿大新闻", "primary_source": "BBC 美国&加拿大新闻", "corroborated_sources": []},
            ]
        }
        historical = {
            "status": "ok",
            "metrics": {
                "ufo_events_total": 8,
                "government_action_events": 5,
            },
        }
        m = strict_reviewer.summarize_mechanism_signals(
            scraped=scraped,
            min_official_share=0.30,
            min_official_lead_events=1,
            min_ufo_events=8,
            historical_mechanism=historical,
        )
        self.assertEqual(m["metrics"]["ufo_events_total"], 2)
        self.assertEqual(m["metrics"]["historical_ufo_events_total"], 8)
        self.assertEqual(m["metrics"]["effective_ufo_events_total"], 10)
        self.assertGreaterEqual(m["metrics"]["effective_official_share"], 0.3)
        self.assertTrue(m["gates"]["ufo_events>=8"])
        self.assertTrue(m["gates"]["official_share>=0.30"])
        # official_lead_events still from live lag/source-order evidence.
        self.assertFalse(m["gates"]["official_lead_events>=1"])

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

    def test_parse_feed_datetime_returns_utc_timestamp(self):
        d, ts, ok = scraper.parse_feed_datetime("Wed, 02 Oct 2002 13:00:00 GMT")
        self.assertTrue(ok)
        self.assertEqual(d, "2002-10-02")
        self.assertTrue(str(ts).startswith("2002-10-02T13:00:00"))

    def test_collapse_claim_clusters_uses_timestamp_lag_when_available(self):
        items = [
            {
                "category": "ufo",
                "claim_fingerprint": "fp-ufo-lag-ts",
                "title": "DoD memo",
                "description": "official",
                "date": "2026-02-20",
                "published_at": "2026-02-20T10:00:00+00:00",
                "source": "Pentagon / DoD 新闻稿",
                "source_type": "rss",
                "domain": "defense.gov",
                "url": "https://defense.gov/a",
                "weight": 3,
                "authenticity": {"final_score": 90},
            },
            {
                "category": "ufo",
                "claim_fingerprint": "fp-ufo-lag-ts",
                "title": "NYT follow-up",
                "description": "media",
                "date": "2026-02-20",
                "published_at": "2026-02-20T18:30:00+00:00",
                "source": "NYT 美国新闻",
                "source_type": "rss",
                "domain": "nytimes.com",
                "url": "https://nytimes.com/b",
                "weight": 3,
                "authenticity": {"final_score": 88},
            },
        ]
        collapsed = scraper.collapse_claim_clusters(items)
        self.assertEqual(len(collapsed), 1)
        e = collapsed[0]
        self.assertEqual(e.get("official_media_lag_basis"), "timestamp")
        self.assertEqual(e.get("official_to_media_lag_days"), 0)
        self.assertAlmostEqual(float(e.get("official_to_media_lag_hours", 0.0)), 8.5, places=3)
        self.assertTrue(e.get("official_leads_media_by_timestamp"))

    def test_build_official_lead_diagnostics_counts_blockers_and_candidates(self):
        ufo_news = [
            {
                "title": "lead",
                "date": "2026-02-20",
                "official_timeline_observations": 1,
                "media_timeline_observations": 1,
                "official_to_media_lag_days": 2,
                "official_to_media_lag_hours": 36.0,
                "cluster_size": 2,
            },
            {
                "title": "no official",
                "date": "2026-02-20",
                "official_timeline_observations": 0,
                "media_timeline_observations": 1,
                "official_to_media_lag_days": None,
                "official_to_media_lag_hours": None,
                "cluster_size": 1,
            },
        ]
        diag = scraper.build_official_lead_diagnostics(ufo_news)
        summary = diag.get("summary", {})
        self.assertEqual(summary.get("total_ufo_events"), 2)
        self.assertEqual(summary.get("official_lead_strict_candidates"), 1)
        self.assertEqual(summary.get("with_official_source"), 1)
        self.assertEqual(summary.get("with_lag_days"), 1)
        reasons = {x.get("reason") for x in summary.get("top_blockers", [])}
        self.assertIn("no_official_source_in_timeline", reasons)
        self.assertIn("official_leads_cross_day", reasons)

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

    def test_historical_backfill_expand_failed_days(self):
        failed = [
            {"start": "1990-01-01", "end": "1990-01-03", "error": "x"},
            {"start": "1990-01-06", "end": "1990-01-06", "error": "y"},
        ]
        out = historical_backfill._expand_failed_days(
            failed,
            causal_analyzer.parse_date("1990-01-02"),
            causal_analyzer.parse_date("1990-01-06"),
        )
        self.assertEqual(out, {"1990-01-02", "1990-01-03", "1990-01-06"})

    def test_historical_backfill_retry_sleep_seconds(self):
        s1 = historical_backfill._compute_retry_sleep_seconds(
            attempt=1,
            is_429=False,
            rate_limit_cooldown=60.0,
            retry_backoff_max=180.0,
        )
        self.assertAlmostEqual(s1, 1.2, places=5)
        s2 = historical_backfill._compute_retry_sleep_seconds(
            attempt=1,
            is_429=True,
            rate_limit_cooldown=60.0,
            retry_backoff_max=180.0,
        )
        self.assertAlmostEqual(s2, 60.0, places=5)

    def test_historical_backfill_build_day_source_map_inclusive(self):
        s = causal_analyzer.parse_date("1990-01-01")
        e = causal_analyzer.parse_date("1990-01-03")
        m = historical_backfill._build_day_source_map(s, e, "gdelt_timeline_volraw")
        self.assertEqual(len(m), 3)
        self.assertEqual(m["1990-01-01"], "gdelt_timeline_volraw")
        self.assertEqual(m["1990-01-03"], "gdelt_timeline_volraw")

    def test_historical_backfill_parse_google_pub_date(self):
        d = historical_backfill._parse_google_pub_date("Wed, 02 Oct 2002 13:00:00 GMT")
        self.assertIsNotNone(d)
        self.assertEqual(d.isoformat(), "2002-10-02")
        self.assertIsNone(historical_backfill._parse_google_pub_date("not a date"))

    def test_replay_select_runs_for_replay(self):
        runs = [
            {"run_id": "r0", "stats": {"failed_chunks_total": 0}},
            {"run_id": "r1", "stats": {"failed_chunks_total": 3}},
            {"run_id": "r2", "stats": {"failed_chunks_total": 2}},
        ]
        picked = replay_backfill_failures.select_runs_for_replay(runs, run_id="", last_n_runs=1)
        self.assertEqual([x["run_id"] for x in picked], ["r2"])
        picked2 = replay_backfill_failures.select_runs_for_replay(runs, run_id="", last_n_runs=2)
        self.assertEqual([x["run_id"] for x in picked2], ["r1", "r2"])
        picked3 = replay_backfill_failures.select_runs_for_replay(runs, run_id="r1", last_n_runs=1)
        self.assertEqual([x["run_id"] for x in picked3], ["r1"])

    def test_replay_collect_failed_jobs_dedup_and_limit(self):
        selected_runs = [
            {
                "run_id": "r1",
                "policy": "strict-balanced",
                "failed_chunks": {
                    "ufo": [
                        {"start": "1990-01-01", "end": "1990-01-03", "error": "x"},
                        {"start": "1990-01-01", "end": "1990-01-03", "error": "dup"},
                    ],
                    "crisis": [
                        {"start": "1990-01-02", "end": "1990-01-02", "error": "y"},
                    ],
                },
            },
            {
                "run_id": "r2",
                "policy": "strict-balanced",
                "failed_chunks": {
                    "ufo": [
                        {"start": "1990-01-04", "end": "1990-01-05", "error": "z"},
                    ],
                },
            },
        ]
        jobs = replay_backfill_failures.collect_failed_jobs(
            selected_runs=selected_runs,
            selected_queries=set(),
            max_chunks=0,
            slice_days=0,
            schedule_order="none",
        )
        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0]["query"], "crisis")
        self.assertEqual(jobs[1]["query"], "ufo")
        self.assertEqual(jobs[2]["query"], "ufo")

        jobs2 = replay_backfill_failures.collect_failed_jobs(
            selected_runs=selected_runs,
            selected_queries={"ufo"},
            max_chunks=1,
            slice_days=0,
            schedule_order="none",
        )
        self.assertEqual(len(jobs2), 1)
        self.assertEqual(jobs2[0]["query"], "ufo")

    def test_replay_collect_failed_jobs_with_slicing_and_shortest_order(self):
        selected_runs = [
            {
                "run_id": "r1",
                "policy": "strict-balanced",
                "failed_chunks": {
                    "ufo": [
                        {"start": "1990-01-01", "end": "1990-01-05", "error": "x"},
                    ],
                    "crisis": [
                        {"start": "1990-01-01", "end": "1990-01-02", "error": "y"},
                    ],
                },
            }
        ]
        jobs = replay_backfill_failures.collect_failed_jobs(
            selected_runs=selected_runs,
            selected_queries=set(),
            max_chunks=0,
            slice_days=2,
            schedule_order="shortest",
        )
        self.assertEqual(len(jobs), 4)
        self.assertEqual(jobs[0]["span_days"], 1)
        self.assertEqual(jobs[0]["query"], "ufo")
        self.assertEqual(jobs[-1]["span_days"], 2)

    def test_replay_derive_job_status(self):
        self.assertEqual(
            replay_backfill_failures.derive_job_status(0, {"failed_chunks_total": 0}),
            "full_success",
        )
        self.assertEqual(
            replay_backfill_failures.derive_job_status(0, {"failed_chunks_total": 2}),
            "partial_success",
        )
        self.assertEqual(
            replay_backfill_failures.derive_job_status(1, {"failed_chunks_total": 0}),
            "failed",
        )

    def test_replay_build_backfill_command_includes_google_flags(self):
        args = type(
            "Args",
            (),
            {
                "python_bin": ".venv/bin/python",
                "chunk_days": 7,
                "request_timeout": 12,
                "request_retries": 1,
                "rate_limit_cooldown": 5.0,
                "retry_backoff_max": 10.0,
                "google_max_span_days": 3,
                "pause_between_chunks": 0.2,
                "min_split_days": 3,
                "allow_partial": True,
                "overwrite_backfill": True,
                "google_fallback": True,
                "use_env_proxy": False,
                "verbose_chunks": True,
            },
        )()
        cmd = replay_backfill_failures.build_backfill_command(
            args=args,
            job={"query": "ufo", "start": "2002-01-01", "end": "2002-01-03"},
            policy="strict-balanced",
        )
        self.assertIn("--google-fallback", cmd)
        self.assertIn("--google-max-span-days", cmd)
        self.assertIn("3", cmd)

    def test_replay_filter_jobs_by_replay_history(self):
        now_iso = datetime.now(timezone.utc).isoformat()
        old_iso = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        jobs = [
            {"query": "ufo", "start": "1990-01-01", "end": "1990-01-01", "policy": "strict-balanced"},
            {"query": "ufo", "start": "1990-01-02", "end": "1990-01-02", "policy": "strict-balanced"},
            {"query": "ufo", "start": "1990-01-03", "end": "1990-01-03", "policy": "strict-balanced"},
        ]
        replay_runs = [
            {
                "generated_at": now_iso,
                "jobs": [
                    {"query": "ufo", "start": "1990-01-01", "end": "1990-01-01", "policy": "strict-balanced", "status": "partial_success"},
                    {"query": "ufo", "start": "1990-01-02", "end": "1990-01-02", "policy": "strict-balanced", "status": "full_success"},
                ],
            },
            {
                "generated_at": old_iso,
                "jobs": [
                    {"query": "ufo", "start": "1990-01-03", "end": "1990-01-03", "policy": "strict-balanced", "status": "partial_success"},
                ],
            },
        ]
        kept, skipped = replay_backfill_failures.filter_jobs_by_replay_history(
            jobs=jobs,
            replay_runs=replay_runs,
            failure_cooldown_hours=6.0,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["start"], "1990-01-03")
        self.assertEqual(len(skipped), 2)
        reasons = {x["reason"] for x in skipped}
        self.assertIn("cooldown_active", reasons)
        self.assertIn("already_full_success", reasons)


if __name__ == "__main__":
    unittest.main()
