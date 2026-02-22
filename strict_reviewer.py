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

from utils import percentile  # type: ignore[import]


BASE_DIR = Path(__file__).resolve().parent  # type: ignore
DATA_DIR = BASE_DIR / "data"
SCRAPED_FILE = DATA_DIR / "scraped_news.json"
CAUSAL_REPORT_FILE = DATA_DIR / "causal_report.json"
PANEL_PROGRESS_FILE = DATA_DIR / "panel_progress.json"
DUAL_REVIEW_FILE = DATA_DIR / "strict_dual_review.json"
MODEL_DID_FILE = DATA_DIR / "model_did_report.json"
MODEL_EVENT_FILE = DATA_DIR / "model_event_study_report.json"
MODEL_SYNTH_FILE = DATA_DIR / "model_synth_control_report.json"
MODEL_CAUSAL_ML_FILE = DATA_DIR / "model_causal_ml_report.json"
EVENTS_V2_FILE = DATA_DIR / "events_v2.json"
OFFICIAL_LEAD_DIAG_FILE = DATA_DIR / "official_lead_event_candidates.json"
OFFICIAL_MEDIA_PAIRS_FILE = DATA_DIR / "official_media_pairs.json"
OUT_FILE = DATA_DIR / "strict_review_snapshot.json"

OFFICIAL_SOURCE_HINTS = (
    "white house",
    "pentagon",
    "department of defense",
    "dod",
    "state department",
    "congress",
    "senate",
    "house",
    "doj",
    "fbi",
    "cia",
    "nasa",
    "aaro",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="统一严格评审")
    p.add_argument("--expected-policy", choices=["strict", "strict-balanced"], default="strict-balanced")
    p.add_argument("--min-source-availability", type=float, default=0.9)
    p.add_argument("--max-failed-source-ratio", type=float, default=0.10)
    p.add_argument("--min-observed-ratio", type=float, default=0.85)
    p.add_argument("--max-missing-streak", type=int, default=7)
    p.add_argument("--min-official-share", type=float, default=0.30)
    p.add_argument("--min-official-lead-events", type=int, default=1)
    p.add_argument("--min-mechanism-ufo-events", type=int, default=8)
    return p.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():  # type: ignore
        return {}
    with path.open("r", encoding="utf-8") as f:  # type: ignore
        return json.load(f)  # type: ignore


def _gate_pass(gate_map: Dict[str, bool], prefix: str) -> bool:
    for name, passed in gate_map.items():  # type: ignore
        if str(name).startswith(prefix):
            return bool(passed)
    return False


def is_official_source(source_name: str | None) -> bool:
    if not source_name:
        return False
    lowered = source_name.lower()
    return any(h in lowered for h in OFFICIAL_SOURCE_HINTS)


def summarize_mechanism_signals(
    scraped: Dict[str, Any],
    min_official_share: float,
    min_official_lead_events: int,
    min_ufo_events: int = 3,
    historical_mechanism: Dict[str, Any] | None = None,
    official_media_pairs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    ufo_events = scraped.get("ufo_news", [])  # type: ignore
    total = len(ufo_events)
    official_involved = 0
    official_primary = 0
    official_primary_with_media_followup = 0
    media_primary_with_official_secondary = 0
    lag_observed_events = 0
    official_lead_by_lag_events = 0
    lag_values = []

    for row in ufo_events:  # type: ignore
        source_pool = set()
        for k in ("source", "primary_source"):
            v = row.get(k)  # type: ignore
            if v:
                source_pool.add(str(v))
        for s in row.get("corroborated_sources", []) or []:  # type: ignore
            if s:
                source_pool.add(str(s))

        official_sources = [s for s in source_pool if is_official_source(s)]
        non_official_sources = [s for s in source_pool if not is_official_source(s)]
        primary_source = row.get("primary_source") or row.get("source")  # type: ignore
        primary_is_official = is_official_source(str(primary_source) if primary_source else None)

        if official_sources:
            official_involved += 1  # type: ignore
        if primary_is_official:
            official_primary += 1  # type: ignore
            if non_official_sources:
                official_primary_with_media_followup += 1  # type: ignore
        elif official_sources:
            media_primary_with_official_secondary += 1  # type: ignore

        lag = row.get("official_to_media_lag_days")  # type: ignore
        if isinstance(lag, (int, float)):
            lag_observed_events += 1  # type: ignore
            lag_values.append(float(lag))  # type: ignore
            if lag >= 0:
                official_lead_by_lag_events += 1  # type: ignore

    official_share = (official_involved / float(total)) if total > 0 else 0.0
    official_primary_share = (official_primary / float(total)) if total > 0 else 0.0
    hist_metrics = (historical_mechanism or {}).get("metrics", {}) if isinstance(historical_mechanism, dict) else {}
    hist_total = int(hist_metrics.get("ufo_events_total", 0) or 0)
    hist_official = int(hist_metrics.get("government_action_events", 0) or 0)
    effective_total = total + hist_total
    effective_official = official_involved + hist_official
    effective_share = (effective_official / float(effective_total)) if effective_total > 0 else 0.0
    enough_ufo_events = effective_total >= min_ufo_events

    pair_summary = (official_media_pairs or {}).get("summary", {}) if isinstance(official_media_pairs, dict) else {}
    pair_total = int(pair_summary.get("total_pairs", 0) or 0)
    pair_strict = int(pair_summary.get("strict_pairs", 0) or 0)
    pair_balanced = int(pair_summary.get("balanced_pairs", 0) or 0)
    pair_nonnegative = int(pair_summary.get("strict_nonnegative_lag_pairs", 0) or 0)
    pair_positive = int(pair_summary.get("strict_positive_lag_pairs", 0) or 0)
    pair_lag_observed = int(pair_summary.get("strict_with_timestamp_pairs", 0) or 0)
    pair_official_with_followup = int(pair_summary.get("official_events_with_strict_followup", 0) or 0)
    pair_lead_events = pair_positive if pair_positive > 0 else pair_nonnegative

    # Prefer timestamp/day lag evidence first, then pair-based strict lag evidence,
    # and fallback to source-order proxy only when no lag evidence exists.
    if lag_observed_events > 0:
        official_lead_events = official_lead_by_lag_events
        lead_basis = "lag"
    elif pair_lead_events > 0:
        official_lead_events = pair_lead_events
        lead_basis = "pair_strict_lag"
    else:
        official_lead_events = official_primary_with_media_followup
        lead_basis = "source_order_proxy"
    lag_mean = (sum(lag_values) / float(len(lag_values))) if lag_values else None

    gates = {
        f"ufo_events>={min_ufo_events}": enough_ufo_events,
        f"official_share>={min_official_share:.2f}": effective_share >= min_official_share,
        f"official_lead_events>={min_official_lead_events}": official_lead_events >= min_official_lead_events,
    }
    mechanism_passed = all(gates.values())

    return {
        "metrics": {
            "ufo_events_total": total,
            "min_ufo_events_required": min_ufo_events,
            "official_involved_events": official_involved,
            "official_primary_events": official_primary,
            "official_primary_with_media_followup_events": official_primary_with_media_followup,
            "media_primary_with_official_secondary_events": media_primary_with_official_secondary,
            "official_lead_events": official_lead_events,
            "lag_observed_events": lag_observed_events,
            "official_lead_by_lag_events": official_lead_by_lag_events,
            "official_to_media_lag_days_mean": (round(lag_mean, 6) if lag_mean is not None else None),
            "official_to_media_lag_days_q50": (round(percentile(lag_values, 50), 6) if lag_values else None),
            "official_source_share": round(official_share, 6),  # type: ignore
            "official_primary_share": round(official_primary_share, 6),  # type: ignore
            "pair_total": pair_total,
            "pair_strict": pair_strict,
            "pair_balanced": pair_balanced,
            "pair_strict_nonnegative_lag_events": pair_nonnegative,
            "pair_strict_positive_lag_events": pair_positive,
            "pair_lag_observed_events": pair_lag_observed,
            "pair_official_events_with_strict_followup": pair_official_with_followup,
            "historical_ufo_events_total": hist_total,
            "historical_government_action_events": hist_official,
            "effective_ufo_events_total": effective_total,
            "effective_official_events_total": effective_official,
            "effective_official_share": round(effective_share, 6),  # type: ignore
        },
        "gates": gates,
        "lead_basis": lead_basis,
        "mechanism_passed": mechanism_passed,
    }


def summarize_historical_mechanism(events_v2: Dict[str, Any]) -> Dict[str, Any]:
    correlations = events_v2.get("correlations", []) if isinstance(events_v2, dict) else []  # type: ignore
    if not isinstance(correlations, list):
        correlations = []

    total = 0
    gov_action = 0
    positive_gap = 0
    gov_and_positive_gap = 0
    confidence_counts: Dict[str, int] = {}

    for row in correlations:
        if not isinstance(row, dict):
            continue
        region_raw = str(row.get("region", "USA") or "USA").strip().upper()
        if region_raw not in ("USA", "US"):
            continue

        ufo_event = row.get("ufo_event", {}) if isinstance(row.get("ufo_event", {}), dict) else {}
        confidence = str(row.get("confidence", "UNKNOWN") or "UNKNOWN").upper()
        confidence_counts[confidence] = int(confidence_counts.get(confidence, 0) + 1)

        total += 1
        is_gov = bool(ufo_event.get("government_action", False))
        if is_gov:
            gov_action += 1

        gap = row.get("gap_days")
        try:
            gap_v = float(gap)
        except Exception:
            gap_v = None
        if gap_v is not None and gap_v >= 0:
            positive_gap += 1
            if is_gov:
                gov_and_positive_gap += 1

    gov_share = (gov_action / float(total)) if total > 0 else 0.0
    positive_gap_share = (positive_gap / float(total)) if total > 0 else 0.0

    return {
        "status": "ok",
        "metrics": {
            "ufo_events_total": total,
            "government_action_events": gov_action,
            "government_action_share": round(gov_share, 6),  # type: ignore
            "positive_gap_events": positive_gap,
            "positive_gap_share": round(positive_gap_share, 6),  # type: ignore
            "government_action_and_positive_gap_events": gov_and_positive_gap,
            "confidence_counts": confidence_counts,
        },
        "notes": [
            "historical_mechanism 来自 events_v2（仅 USA/US 案例）",
            "government_action 表示事件本身含官方动作，不等同于官方先发媒体滞后证据",
        ],
    }


def build_inference_matrix(summary: Dict[str, Any], mechanism: Dict[str, Any]) -> Dict[str, Any]:
    stage_a_temporal = bool(
        summary.get("signals", {}).get("has_temporal_signal")
        or summary.get("signals", {}).get("verdict_has_correlation_phrase")
    )
    stage_b_causal = bool(
        summary.get("gates", {}).get("core_passed")
        and summary.get("gates", {}).get("falsification_passed")
    )
    stage_c_mechanism = bool(stage_b_causal and mechanism.get("mechanism_passed"))

    if not stage_a_temporal:
        level = "NO_TEMPORAL_SIGNAL"
        conclusion = "未观察到稳定时序信号，当前不支持相关/因果解释。"
    elif not stage_b_causal:
        level = "TEMPORAL_ASSOCIATION_ONLY"
        conclusion = "仅观察到时序相关，因果识别闸门尚未通过。"
    elif not stage_c_mechanism:
        level = "CAUSAL_SIGNAL_WITHOUT_STRATEGIC_MECHANISM"
        conclusion = "存在因果信号，但缺少“官方先发→媒体跟进”机制证据。"
    else:
        level = "STRATEGIC_COMMUNICATION_INDICATION"
        conclusion = "因果与机制闸门均通过，存在策略性沟通迹象（非动机终局证据）。"

    return {
        "stage_a_temporal_association": stage_a_temporal,
        "stage_b_causal_identification": stage_b_causal,
        "stage_c_strategic_mechanism": stage_c_mechanism,
        "level": level,
        "conclusion": conclusion,
    }


def build_next_actions(inference: Dict[str, Any]) -> list[str]:
    level = inference.get("level", "")
    if level == "NO_TEMPORAL_SIGNAL":
        return [
            "核查关键词与事件定义，避免把非政治议题混入冲击日。",
            "扩展历史窗口并补齐来源，先确认时序相关是否稳定存在。",
            "在出现稳定相关前，不进入因果或动机解释。",
        ]
    if level == "TEMPORAL_ASSOCIATION_ONLY":
        return [
            "优先补齐历史面板与冲击日样本，先过因果识别闸门。",
            "持续跑 DID/事件研究/合成控制/Causal ML，并执行安慰剂与反向因果检验。",
            "因果闸门未通过前，禁止使用“故意引导”表述。",
        ]
    if level == "CAUSAL_SIGNAL_WITHOUT_STRATEGIC_MECHANISM":
        return [
            "补充机制变量：official_source_share、official_primary_with_media_followup、official_to_media_lag。",
            "区分“官方主动发布”与“媒体自发放大”两条路径，增加事件级证据链。",
            "机制闸门通过前，仅可表述为“因果效应”，不可上升到“策略意图”。",
        ]
    return [
        "继续跨日复现并做跨时期外部验证，防止单阶段偶然结果。",
        "补充定性证据（官方文件/听证材料）强化动机解释的外部效度。",
        "报告中保持边界：策略迹象≠直接证明主观动机。",
    ]


def build_signature(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "approval_status": summary["decision"]["approval_status"],  # type: ignore
        "approval_level": summary["decision"]["approval_level"],  # type: ignore
        "core_passed": summary["gates"]["core_passed"],  # type: ignore
        "falsification_passed": summary["gates"]["falsification_passed"],  # type: ignore
        "policy_consistency_passed": summary["gates"]["policy_consistency_passed"],  # type: ignore
        "did_passed": summary["quality"]["models"]["did_passed"],  # type: ignore
        "event_passed": summary["quality"]["models"]["event_passed"],  # type: ignore
        "synth_passed": summary["quality"]["models"]["synth_passed"],  # type: ignore
        "causal_ml_passed": summary["quality"]["models"]["causal_ml_passed"],  # type: ignore
        "panel_observed_days": summary["quality"]["panel_observed_days"],  # type: ignore
        "panel_shock_days": summary["quality"]["panel_shock_days"],  # type: ignore
        "dual_status": summary["signals"]["dual_review_status"],  # type: ignore
        "inference_level": summary["inference"]["level"],  # type: ignore
    }


def _parse_iso_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def evaluate_reproducibility(
    prev_snapshot: Dict[str, Any],
    signature: Dict[str, Any],
    curr_ts: datetime | None,
) -> tuple[bool, bool, bool]:
    prev_sig = prev_snapshot.get("meta", {}).get("signature") if prev_snapshot else None
    if not prev_sig or prev_sig != signature:
        return False, False, False

    prev_ts = _parse_iso_ts(prev_snapshot.get("generated_at")) if prev_snapshot else None
    same_day_repeat = bool(prev_ts and curr_ts and prev_ts.date() == curr_ts.date())
    cross_day_repeat = bool(prev_ts and curr_ts and prev_ts.date() < curr_ts.date())
    prev_repro = bool(prev_snapshot.get("gates", {}).get("reproducibility_passed", False))

    # Same-day rerun should not regress a previously achieved cross-day reproducibility.
    reproducibility_passed = cross_day_repeat or (same_day_repeat and prev_repro)
    return reproducibility_passed, cross_day_repeat, same_day_repeat


def classify_level(summary: Dict[str, Any]) -> str:
    if (
        summary["gates"]["core_passed"]  # type: ignore
        and summary["gates"]["falsification_passed"]  # type: ignore
        and summary["gates"]["reproducibility_passed"]  # type: ignore
    ):
        return "L4"
    if summary["gates"]["core_passed"] and summary["gates"]["falsification_passed"]:  # type: ignore
        return "L3"
    if summary["gates"]["core_passed"]:  # type: ignore
        return "L2"
    if summary["signals"]["has_temporal_signal"] or summary["signals"]["verdict_has_correlation_phrase"]:  # type: ignore
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
    causal_ml = read_json(MODEL_CAUSAL_ML_FILE)
    events_v2 = read_json(EVENTS_V2_FILE)
    official_lead_diag = read_json(OFFICIAL_LEAD_DIAG_FILE)
    official_media_pairs = read_json(OFFICIAL_MEDIA_PAIRS_FILE)
    prev_snapshot = read_json(OUT_FILE)

    approval = causal.get("approval", {})  # type: ignore
    panel = causal.get("panel", {})  # type: ignore
    gates = {g.get("name"): bool(g.get("passed")) for g in approval.get("gates", [])}  # type: ignore

    n_ufo = int(scraped.get("events_by_type_count", {}).get("ufo", 0) or 0)  # type: ignore
    n_crisis = int(scraped.get("events_by_type_count", {}).get("crisis", 0) or 0)  # type: ignore
    verdict = str(causal.get("verdict", ""))  # type: ignore
    verdict_has_corr = ("相关" in verdict) or ("因果信号" in verdict)

    source_health = scraped.get("source_health", {})  # type: ignore
    avail = float(source_health.get("availability_rate", 0.0) or 0.0)  # type: ignore
    failed_sources = int(source_health.get("failed_sources", 0) or 0)  # type: ignore
    total_active_sources = int(source_health.get("total_active_sources", 0) or 0)  # type: ignore
    allowed_failed_sources = (
        max(1, int(total_active_sources * args.max_failed_source_ratio)) if total_active_sources > 0 else 0
    )
    source_gate_passed = (
        avail >= args.min_source_availability and failed_sources <= allowed_failed_sources
    )

    panel_observed_days = int(progress.get("current", {}).get("observed_days", panel.get("observed_days", 0)) or 0)  # type: ignore
    panel_shock_days = int(progress.get("current", {}).get("shock_days", panel.get("n_shocks", 0)) or 0)  # type: ignore
    observed_ratio = float(progress.get("current", {}).get("observed_ratio", panel.get("observed_ratio", 0.0)) or 0.0)  # type: ignore
    max_missing_streak = int(
        progress.get("current", {}).get("max_missing_streak", panel.get("max_missing_streak", 0)) or 0  # type: ignore
    )
    continuity_gate_passed = (
        observed_ratio >= args.min_observed_ratio and max_missing_streak <= args.max_missing_streak
    )

    approval_gate_passed = approval.get("status") == "APPROVED"  # type: ignore
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

    dual_gates = dual.get("gates", [])  # type: ignore
    dual_status = dual.get("status", "unknown")  # type: ignore
    dual_stable_passed = (
        dual_status == "stable"
        and len(dual_gates) > 0
        and all(bool(g.get("passed")) for g in dual_gates)  # type: ignore
    )

    scraped_policy = scraped.get("policy")  # type: ignore
    causal_policy = panel.get("policy")  # type: ignore
    progress_policy = progress.get("policy")  # type: ignore
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

    did_status = did.get("status", did.get("result", {}).get("status", "missing"))  # type: ignore
    did_passed = bool(did.get("gates", {}).get("did_passed", False))  # type: ignore
    did_neg_ctrl_available = bool(did.get("gates", {}).get("negative_controls_available", False))  # type: ignore
    event_status = event.get("status", "missing")  # type: ignore
    event_passed = bool(event.get("gates", {}).get("event_study_passed", False))  # type: ignore
    synth_status = synth.get("status", "missing")  # type: ignore
    synth_passed = bool(synth.get("gates", {}).get("synth_passed", False))  # type: ignore
    causal_ml_status = causal_ml.get("status", "missing")  # type: ignore
    causal_ml_passed = bool(causal_ml.get("gates", {}).get("causal_ml_passed", False))  # type: ignore

    models_estimated = did_status == "ok" and event_status == "ok" and synth_status == "ok"
    model_falsification_ok = did_passed and event_passed and synth_passed and did_neg_ctrl_available
    # Causal ML is an additional check. If estimated and failed, treat as inconsistency.
    causal_ml_consistent = causal_ml_status != "ok" or causal_ml_passed
    falsification_passed = models_estimated and model_falsification_ok and causal_ml_consistent

    has_temporal_signal = (panel_observed_days > 0 and panel_shock_days > 0) or (n_ufo > 0 and n_crisis > 0)
    historical_mechanism = summarize_historical_mechanism(events_v2)
    mechanism = summarize_mechanism_signals(
        scraped,
        min_official_share=args.min_official_share,
        min_official_lead_events=args.min_official_lead_events,
        min_ufo_events=args.min_mechanism_ufo_events,
        historical_mechanism=historical_mechanism,
        official_media_pairs=official_media_pairs,
    )

    summary: Dict[str, Any] = {  # type: ignore
        "generated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "decision": {
            "verdict": verdict or "no_verdict",
            "approval_status": approval.get("status", "UNKNOWN"),  # type: ignore
            "approval_level": approval.get("level", "UNKNOWN"),  # type: ignore
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
            "dual_overlap_days": int(dual.get("current", {}).get("overlap_days", 0) or 0),  # type: ignore
        },
        "mechanism": mechanism,
        "mechanism_historical": historical_mechanism,
        "quality": {
            "source_availability_rate": avail,
            "source_failed_count": failed_sources,
            "source_total_active": total_active_sources,
            "source_allowed_failed_threshold": allowed_failed_sources,
            "panel_status": progress.get("status", "unknown"),  # type: ignore
            "panel_observed_days": panel_observed_days,
            "panel_shock_days": panel_shock_days,
            "panel_observed_ratio": round(observed_ratio, 6),  # type: ignore
            "panel_max_missing_streak": max_missing_streak,
            "models": {
                "did_status": did_status,
                "did_passed": did_passed,
                "did_negative_controls_available": did_neg_ctrl_available,
                "event_status": event_status,
                "event_passed": event_passed,
                "synth_status": synth_status,
                "synth_passed": synth_passed,
                "causal_ml_status": causal_ml_status,
                "causal_ml_passed": causal_ml_passed,
            },
            "official_lead_diagnostics": {
                "available": bool(official_lead_diag),
                "summary": official_lead_diag.get("summary", {}) if isinstance(official_lead_diag, dict) else {},
            },
            "official_media_pairs": {
                "available": bool(official_media_pairs),
                "summary": official_media_pairs.get("summary", {}) if isinstance(official_media_pairs, dict) else {},
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
            "mechanism_gate_passed": bool(mechanism.get("mechanism_passed")),
            "reproducibility_passed": False,
            "gate_map": gates,
        },
        "inference": {},
        "meta": {},
        "research_level": "",
        "next_actions": [],
    }

    summary["inference"] = build_inference_matrix(summary, mechanism)  # type: ignore
    summary["next_actions"] = build_next_actions(summary["inference"])  # type: ignore

    signature = build_signature(summary)
    curr_ts = _parse_iso_ts(summary.get("generated_at"))  # type: ignore
    reproducibility_passed, cross_day_repeat, same_day_repeat = evaluate_reproducibility(
        prev_snapshot,
        signature,
        curr_ts,
    )
    summary["gates"]["reproducibility_passed"] = reproducibility_passed  # type: ignore
    summary["meta"] = {  # type: ignore
        "signature": signature,
        "previous_signature_exists": bool(prev_snapshot.get("meta", {}).get("signature")) if prev_snapshot else False,
        "previous_generated_at": prev_snapshot.get("generated_at") if prev_snapshot else None,  # type: ignore
        "cross_day_repeat": cross_day_repeat,
        "same_day_repeat": same_day_repeat,
    }
    summary["research_level"] = classify_level(summary)  # type: ignore
    return summary


def main() -> None:
    args = parse_args()
    report = build_review(args)
    with OUT_FILE.open("w", encoding="utf-8") as f:  # type: ignore
        json.dump(report, f, ensure_ascii=False, indent=2)  # type: ignore

    print("=== Strict Unified Review ===")
    print(f"approval: {report['decision']['approval_status']} ({report['decision']['approval_level']})")  # type: ignore
    print(f"research_level: {report['research_level']}")  # type: ignore
    print(f"core_passed: {report['gates']['core_passed']}")  # type: ignore
    print(f"falsification_passed: {report['gates']['falsification_passed']}")  # type: ignore
    print(f"reproducibility_passed: {report['gates']['reproducibility_passed']}")  # type: ignore
    print(f"source_availability_rate: {report['quality']['source_availability_rate']:.4f}")  # type: ignore
    print(f"panel_observed_days: {report['quality']['panel_observed_days']}")  # type: ignore
    print(f"panel_shock_days: {report['quality']['panel_shock_days']}")  # type: ignore
    print(f"inference_level: {report['inference']['level']}")  # type: ignore
    print(f"inference_conclusion: {report['inference']['conclusion']}")  # type: ignore
    print(
        "causal_ml: "
        f"{report['quality']['models']['causal_ml_status']} "
        f"(passed={report['quality']['models']['causal_ml_passed']})"
    )  # type: ignore
    print(
        "mechanism: "
        f"official_share_live={report['mechanism']['metrics']['official_source_share']}, "
        f"official_share_effective={report['mechanism']['metrics']['effective_official_share']}, "
        f"official_lead_events={report['mechanism']['metrics']['official_lead_events']}, "
        f"lag_observed={report['mechanism']['metrics']['lag_observed_events']}, "
        f"pair_strict={report['mechanism']['metrics']['pair_strict']}, "
        f"pair_strict_positive={report['mechanism']['metrics']['pair_strict_positive_lag_events']}, "
        f"lag_q50={report['mechanism']['metrics']['official_to_media_lag_days_q50']}, "
        f"passed={report['mechanism']['mechanism_passed']}"
    )  # type: ignore
    print(
        "mechanism_historical: "
        f"ufo_events={report['mechanism_historical']['metrics']['ufo_events_total']}, "
        f"government_action_share={report['mechanism_historical']['metrics']['government_action_share']}"
    )  # type: ignore
    lead_diag_summary = report.get("quality", {}).get("official_lead_diagnostics", {}).get("summary", {})  # type: ignore
    if lead_diag_summary:
        print(
            "official_lead_diagnostics: "
            f"strict_candidates={lead_diag_summary.get('official_lead_strict_candidates')}, "
            f"with_lag_days={lead_diag_summary.get('with_lag_days')}, "
            f"total_ufo={lead_diag_summary.get('total_ufo_events')}"
        )
    pair_summary = report.get("quality", {}).get("official_media_pairs", {}).get("summary", {})  # type: ignore
    if pair_summary:
        print(
            "official_media_pairs: "
            f"total={pair_summary.get('total_pairs')}, "
            f"strict={pair_summary.get('strict_pairs')}, "
            f"strict_positive={pair_summary.get('strict_positive_lag_pairs')}, "
            f"official_items={pair_summary.get('official_items_considered')}, "
            f"media_items={pair_summary.get('media_items_considered')}"
        )
    print(f"[输出] {OUT_FILE}")  # type: ignore


if __name__ == "__main__":
    main()
