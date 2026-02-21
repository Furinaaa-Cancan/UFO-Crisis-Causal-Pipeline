"""
统一严格评审器

读取核心产物并生成统一审查快照：
- data/scraped_news.json
- data/causal_report.json
- data/panel_progress.json
- data/strict_dual_review.json
- data/model_*.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SCRAPED_FILE = DATA_DIR / "scraped_news.json"
CAUSAL_REPORT_FILE = DATA_DIR / "causal_report.json"
PANEL_PROGRESS_FILE = DATA_DIR / "panel_progress.json"
DUAL_REVIEW_FILE = DATA_DIR / "strict_dual_review.json"
MODEL_DID_FILE = DATA_DIR / "model_did_report.json"
MODEL_EVENT_FILE = DATA_DIR / "model_event_study_report.json"
MODEL_SYNTH_FILE = DATA_DIR / "model_synth_control_report.json"
OUT_FILE = DATA_DIR / "strict_review_snapshot.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="统一严格评审")
    p.add_argument("--expected-policy", choices=["strict", "strict-balanced"], default="strict-balanced")
    p.add_argument("--min-source-availability", type=float, default=0.9)
    p.add_argument("--max-failed-source-ratio", type=float, default=0.10)
    p.add_argument("--min-observed-ratio", type=float, default=0.85)
    p.add_argument("--max-missing-streak", type=int, default=7)
    return p.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _gate_pass(gate_map: Dict[str, bool], prefix: str) -> bool:
    for name, passed in gate_map.items():
        if str(name).startswith(prefix):
            return bool(passed)
    return False


def build_signature(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "approval_status": summary["decision"]["approval_status"],
        "approval_level": summary["decision"]["approval_level"],
        "core_passed": summary["gates"]["core_passed"],
        "falsification_passed": summary["gates"]["falsification_passed"],
        "policy_consistency_passed": summary["gates"]["policy_consistency_passed"],
        "did_passed": summary["quality"]["models"]["did_passed"],
        "event_passed": summary["quality"]["models"]["event_passed"],
        "synth_passed": summary["quality"]["models"]["synth_passed"],
        "panel_observed_days": summary["quality"]["panel_observed_days"],
        "panel_shock_days": summary["quality"]["panel_shock_days"],
        "dual_status": summary["signals"]["dual_review_status"],
    }


def _parse_iso_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def classify_level(summary: Dict[str, Any]) -> str:
    if (
        summary["gates"]["core_passed"]
        and summary["gates"]["falsification_passed"]
        and summary["gates"]["reproducibility_passed"]
    ):
        return "L4"
    if summary["gates"]["core_passed"] and summary["gates"]["falsification_passed"]:
        return "L3"
    if summary["gates"]["core_passed"]:
        return "L2"
    if summary["signals"]["has_temporal_signal"] or summary["signals"]["verdict_has_correlation_phrase"]:
        return "L1"
    return "L0"


def build_review(args: argparse.Namespace) -> Dict[str, Any]:
    scraped = read_json(SCRAPED_FILE)
    causal = read_json(CAUSAL_REPORT_FILE)
    progress = read_json(PANEL_PROGRESS_FILE)
    dual = read_json(DUAL_REVIEW_FILE)
    did = read_json(MODEL_DID_FILE)
    event = read_json(MODEL_EVENT_FILE)
    synth = read_json(MODEL_SYNTH_FILE)
    prev_snapshot = read_json(OUT_FILE)

    approval = causal.get("approval", {})
    panel = causal.get("panel", {})
    gates = {g.get("name"): bool(g.get("passed")) for g in approval.get("gates", [])}

    n_ufo = int(scraped.get("events_by_type_count", {}).get("ufo", 0) or 0)
    n_crisis = int(scraped.get("events_by_type_count", {}).get("crisis", 0) or 0)
    verdict = str(causal.get("verdict", ""))
    verdict_has_corr = ("相关" in verdict) or ("因果信号" in verdict)

    source_health = scraped.get("source_health", {})
    avail = float(source_health.get("availability_rate", 0.0) or 0.0)
    failed_sources = int(source_health.get("failed_sources", 0) or 0)
    total_active_sources = int(source_health.get("total_active_sources", 0) or 0)
    allowed_failed_sources = (
        max(1, int(total_active_sources * args.max_failed_source_ratio)) if total_active_sources > 0 else 0
    )
    source_gate_passed = (
        avail >= args.min_source_availability and failed_sources <= allowed_failed_sources
    )

    panel_observed_days = int(progress.get("current", {}).get("observed_days", panel.get("observed_days", 0)) or 0)
    panel_shock_days = int(progress.get("current", {}).get("shock_days", panel.get("n_shocks", 0)) or 0)
    observed_ratio = float(progress.get("current", {}).get("observed_ratio", panel.get("observed_ratio", 0.0)) or 0.0)
    max_missing_streak = int(
        progress.get("current", {}).get("max_missing_streak", panel.get("max_missing_streak", 0)) or 0
    )
    continuity_gate_passed = (
        observed_ratio >= args.min_observed_ratio and max_missing_streak <= args.max_missing_streak
    )

    approval_gate_passed = approval.get("status") == "APPROVED"
    sample_gates_passed = (
        _gate_pass(gates, "panel_observed_days>=")
        and _gate_pass(gates, "panel_shocks>=")
        and _gate_pass(gates, "panel_observed_ratio>=")
    )
    directional_gates_passed = (
        _gate_pass(gates, ">=2_significant_windows")
        and _gate_pass(gates, "lead_lag_positive")
        and _gate_pass(gates, "reverse_not_dominant")
    )

    dual_gates = dual.get("gates", [])
    dual_status = dual.get("status", "unknown")
    dual_stable_passed = (
        dual_status == "stable"
        and len(dual_gates) > 0
        and all(bool(g.get("passed")) for g in dual_gates)
    )

    scraped_policy = scraped.get("policy")
    causal_policy = panel.get("policy")
    progress_policy = progress.get("policy")
    policies = [p for p in [scraped_policy, causal_policy, progress_policy] if p]
    policy_consistency_passed = bool(policies) and all(p == args.expected_policy for p in policies)

    core_passed = (
        approval_gate_passed
        and sample_gates_passed
        and directional_gates_passed
        and continuity_gate_passed
        and source_gate_passed
        and dual_stable_passed
        and policy_consistency_passed
    )

    did_status = did.get("status", did.get("result", {}).get("status", "missing"))
    did_passed = bool(did.get("gates", {}).get("did_passed", False))
    did_neg_ctrl_available = bool(did.get("gates", {}).get("negative_controls_available", False))
    event_status = event.get("status", "missing")
    event_passed = bool(event.get("gates", {}).get("event_study_passed", False))
    synth_status = synth.get("status", "missing")
    synth_passed = bool(synth.get("gates", {}).get("synth_passed", False))

    models_estimated = did_status == "ok" and event_status == "ok" and synth_status == "ok"
    model_falsification_ok = did_passed and event_passed and synth_passed and did_neg_ctrl_available
    falsification_passed = models_estimated and model_falsification_ok

    has_temporal_signal = (panel_observed_days > 0 and panel_shock_days > 0) or (n_ufo > 0 and n_crisis > 0)

    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": {
            "verdict": verdict or "no_verdict",
            "approval_status": approval.get("status", "UNKNOWN"),
            "approval_level": approval.get("level", "UNKNOWN"),
            "expected_policy": args.expected_policy,
            "observed_policies": {
                "scraped_policy": scraped_policy,
                "causal_policy": causal_policy,
                "progress_policy": progress_policy,
            },
        },
        "signals": {
            "ufo_events_latest_window": n_ufo,
            "crisis_events_latest_window": n_crisis,
            "has_temporal_signal": has_temporal_signal,
            "verdict_has_correlation_phrase": verdict_has_corr,
            "dual_review_status": dual_status,
            "dual_overlap_days": int(dual.get("current", {}).get("overlap_days", 0) or 0),
        },
        "quality": {
            "source_availability_rate": avail,
            "source_failed_count": failed_sources,
            "source_total_active": total_active_sources,
            "source_allowed_failed_threshold": allowed_failed_sources,
            "panel_status": progress.get("status", "unknown"),
            "panel_observed_days": panel_observed_days,
            "panel_shock_days": panel_shock_days,
            "panel_observed_ratio": round(observed_ratio, 6),
            "panel_max_missing_streak": max_missing_streak,
            "models": {
                "did_status": did_status,
                "did_passed": did_passed,
                "did_negative_controls_available": did_neg_ctrl_available,
                "event_status": event_status,
                "event_passed": event_passed,
                "synth_status": synth_status,
                "synth_passed": synth_passed,
            },
        },
        "gates": {
            "core_passed": core_passed,
            "falsification_passed": falsification_passed,
            "approval_gate_passed": approval_gate_passed,
            "sample_gates_passed": sample_gates_passed,
            "directional_gates_passed": directional_gates_passed,
            "continuity_gate_passed": continuity_gate_passed,
            "source_gate_passed": source_gate_passed,
            "dual_stable_passed": dual_stable_passed,
            "policy_consistency_passed": policy_consistency_passed,
            "reproducibility_passed": False,
            "gate_map": gates,
        },
        "meta": {},
        "research_level": "",
        "next_actions": [
            "继续累计 observed_days 与 shock_days 至门槛",
            "保持 strict 与 strict-balanced 双档同步累计并提升 overlap",
            "仅在 core+falsification 全部通过后才宣称因果",
        ],
    }

    signature = build_signature(summary)
    prev_sig = prev_snapshot.get("meta", {}).get("signature") if prev_snapshot else None
    prev_ts = _parse_iso_ts(prev_snapshot.get("generated_at")) if prev_snapshot else None
    curr_ts = _parse_iso_ts(summary.get("generated_at"))
    cross_day_repeat = bool(prev_ts and curr_ts and prev_ts.date() < curr_ts.date())
    summary["gates"]["reproducibility_passed"] = bool(prev_sig) and prev_sig == signature and cross_day_repeat
    summary["meta"] = {
        "signature": signature,
        "previous_signature_exists": bool(prev_sig),
        "previous_generated_at": prev_snapshot.get("generated_at") if prev_snapshot else None,
        "cross_day_repeat": cross_day_repeat,
    }
    summary["research_level"] = classify_level(summary)
    return summary


def main() -> None:
    args = parse_args()
    report = build_review(args)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== Strict Unified Review ===")
    print(f"approval: {report['decision']['approval_status']} ({report['decision']['approval_level']})")
    print(f"research_level: {report['research_level']}")
    print(f"core_passed: {report['gates']['core_passed']}")
    print(f"falsification_passed: {report['gates']['falsification_passed']}")
    print(f"reproducibility_passed: {report['gates']['reproducibility_passed']}")
    print(f"source_availability_rate: {report['quality']['source_availability_rate']:.4f}")
    print(f"panel_observed_days: {report['quality']['panel_observed_days']}")
    print(f"panel_shock_days: {report['quality']['panel_shock_days']}")
    print(f"[输出] {OUT_FILE}")


if __name__ == "__main__":
    main()
