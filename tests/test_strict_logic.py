import unittest

import causal_analyzer
import panel_pipeline
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


if __name__ == "__main__":
    unittest.main()
