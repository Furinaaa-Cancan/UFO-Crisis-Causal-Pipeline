"""
面板累计管道（唯一有效路径执行器）

作用：
1) 运行 scraper（可选）
2) 运行 causal_analyzer（可选）
3) 计算面板累计进度并写入 data/panel_progress.json

推荐每日执行一次（同一 policy）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PANEL_FILE = DATA_DIR / "causal_panel.json"
PROGRESS_FILE = DATA_DIR / "panel_progress.json"
DUAL_REVIEW_FILE = DATA_DIR / "strict_dual_review.json"
SHOCK_COUNT_FLOOR = 2.0


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    idx = (len(xs) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    w = idx - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def pearson_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = 0.0
    dx2 = 0.0
    dy2 = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        num += dx * dy
        dx2 += dx * dx
        dy2 += dy * dy
    den = (dx2 * dy2) ** 0.5
    if den == 0:
        return 0.0
    return num / den


def compute_shock_threshold(nonzero_values: List[float], q: float = 75.0, floor: float = SHOCK_COUNT_FLOOR) -> float:
    if not nonzero_values:
        return floor
    return max(floor, percentile(nonzero_values, q))


def max_missing_streak(dates: List[date], all_dates: List[date]) -> int:
    observed = set(dates)
    longest = 0
    current = 0
    for d in all_dates:
        if d in observed:
            current = 0
            continue
        current += 1
        if current > longest:
            longest = current
    return longest


def run_cmd(cmd: List[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def load_panel_rows(policy: str) -> List[dict]:
    if not PANEL_FILE.exists():
        return []
    with PANEL_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("rows", [])
    run_day_rows = [r for r in rows if r.get("date_scope") == "run_day_only"]
    effective_rows = run_day_rows if run_day_rows else rows
    filtered = [r for r in effective_rows if r.get("policy") == policy]
    by_date: Dict[str, dict] = {}
    for row in filtered:
        by_date[row.get("date", "")] = row
    return [by_date[k] for k in sorted(by_date.keys()) if k]


def load_panel_payload() -> dict:
    if not PANEL_FILE.exists():
        return {"meta": {"version": 1}, "rows": []}
    with PANEL_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload.setdefault("meta", {"version": 1})
    payload.setdefault("rows", [])
    return payload


def compute_progress(
    rows: List[dict],
    min_days: int,
    min_shocks: int,
    min_observed_ratio: float,
    policy: str,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    if not rows:
        return {
            "generated_at": now,
            "policy": policy,
            "status": "no_data",
            "message": "面板为空，请先运行 scraper + causal_analyzer",
            "targets": {
                "min_days": min_days,
                "min_shocks": min_shocks,
                "min_observed_ratio": min_observed_ratio,
            },
            "current": {
                "span_days": 0,
                "observed_days": 0,
                "missing_days": 0,
                "observed_ratio": 0.0,
                "max_missing_streak": 0,
                "shock_days": 0,
            },
            "remaining": {"days": min_days, "shocks": min_shocks},
            "progress": {"days": 0.0, "shocks": 0.0, "overall": 0.0},
        }

    dates = [parse_date(r["date"]) for r in rows]
    start = min(dates)
    end = max(dates)
    span_days = (end - start).days + 1
    observed_days = len(rows)
    missing_days = max(0, span_days - observed_days)
    all_dates = [start + timedelta(days=i) for i in range(span_days)]
    observed_ratio = (observed_days / float(span_days)) if span_days > 0 else 0.0
    longest_gap = max_missing_streak(dates, all_dates)

    crisis_nonzero = [float(r.get("crisis_count", 0)) for r in rows if float(r.get("crisis_count", 0)) > 0]
    shock_threshold = compute_shock_threshold(crisis_nonzero)
    shock_days = sum(1 for r in rows if float(r.get("crisis_count", 0)) >= shock_threshold)

    remaining_days = max(0, min_days - observed_days)
    remaining_shocks = max(0, min_shocks - shock_days)

    days_progress = min(1.0, observed_days / float(min_days)) if min_days > 0 else 1.0
    shocks_progress = min(1.0, shock_days / float(min_shocks)) if min_shocks > 0 else 1.0
    ratio_progress = min(1.0, observed_ratio / float(min_observed_ratio)) if min_observed_ratio > 0 else 1.0
    overall = min(days_progress, shocks_progress, ratio_progress)

    status = (
        "ready_for_strict_approval"
        if (remaining_days == 0 and remaining_shocks == 0 and observed_ratio >= min_observed_ratio)
        else "accumulating"
    )
    earliest_ready_date = (end + timedelta(days=remaining_days)).isoformat() if remaining_days > 0 else end.isoformat()

    return {
        "generated_at": now,
        "policy": policy,
        "status": status,
        "message": (
            "已达到严格审批样本门槛，可开启强因果闸门"
            if status == "ready_for_strict_approval"
            else "继续按日累计面板数据"
        ),
        "targets": {
            "min_days": min_days,
            "min_shocks": min_shocks,
            "min_observed_ratio": min_observed_ratio,
        },
        "current": {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "span_days": span_days,
            "observed_days": observed_days,
            "missing_days": missing_days,
            "observed_ratio": round(observed_ratio, 4),
            "max_missing_streak": longest_gap,
            "shock_days": shock_days,
            "shock_threshold": shock_threshold,
            "rows": len(rows),
        },
        "remaining": {
            "days": remaining_days,
            "shocks": remaining_shocks,
            "earliest_ready_date_if_daily": earliest_ready_date,
        },
        "progress": {
            "days": round(days_progress, 4),
            "shocks": round(shocks_progress, 4),
            "coverage_ratio": round(ratio_progress, 4),
            "overall": round(overall, 4),
        },
        "next_actions": [
            f"每天固定跑：python scraper.py --policy {policy}",
            (
                "每天固定跑：python causal_analyzer.py --panel-policy "
                f"{policy} --min-panel-observed-ratio {min_observed_ratio}"
            ),
            "达到门槛后跑：python causal_analyzer.py --fail-on-reject",
        ],
    }


def compute_dual_policy_review(panel_payload: dict, min_overlap_days: int = 30) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    rows = panel_payload.get("rows", [])
    run_day_rows = [r for r in rows if r.get("date_scope") == "run_day_only"]
    excluded_legacy_rows = len(rows) - len(run_day_rows)

    strict_rows = {r.get("date"): r for r in run_day_rows if r.get("policy") == "strict" and r.get("date")}
    balanced_rows = {
        r.get("date"): r for r in run_day_rows if r.get("policy") == "strict-balanced" and r.get("date")
    }

    overlap_dates = sorted(set(strict_rows.keys()) & set(balanced_rows.keys()))
    if not strict_rows or not balanced_rows:
        return {
            "generated_at": now,
            "status": "insufficient_policies",
            "message": "需要 strict 和 strict-balanced 两档同时累计后才能评审稳定性",
            "current": {
                "strict_days": len(strict_rows),
                "strict_balanced_days": len(balanced_rows),
                "overlap_days": len(overlap_dates),
                "excluded_non_run_day_rows": excluded_legacy_rows,
            },
            "targets": {"min_overlap_days": min_overlap_days},
        }

    if not overlap_dates:
        return {
            "generated_at": now,
            "status": "no_overlap",
            "message": "双档暂无同日重叠样本",
            "current": {
                "strict_days": len(strict_rows),
                "strict_balanced_days": len(balanced_rows),
                "overlap_days": 0,
                "excluded_non_run_day_rows": excluded_legacy_rows,
            },
            "targets": {"min_overlap_days": min_overlap_days},
        }

    strict_crisis = [float(strict_rows[d].get("crisis_count", 0)) for d in overlap_dates]
    balanced_crisis = [float(balanced_rows[d].get("crisis_count", 0)) for d in overlap_dates]
    strict_ufo = [float(strict_rows[d].get("ufo_count", 0)) for d in overlap_dates]
    balanced_ufo = [float(balanced_rows[d].get("ufo_count", 0)) for d in overlap_dates]

    mean_abs_delta_crisis = sum(abs(a - b) for a, b in zip(strict_crisis, balanced_crisis)) / len(overlap_dates)
    mean_abs_delta_ufo = sum(abs(a - b) for a, b in zip(strict_ufo, balanced_ufo)) / len(overlap_dates)

    denom_crisis = sum(max(1.0, (a + b) / 2.0) for a, b in zip(strict_crisis, balanced_crisis))
    denom_ufo = sum(max(1.0, (a + b) / 2.0) for a, b in zip(strict_ufo, balanced_ufo))
    rel_delta_crisis = sum(abs(a - b) for a, b in zip(strict_crisis, balanced_crisis)) / denom_crisis
    rel_delta_ufo = sum(abs(a - b) for a, b in zip(strict_ufo, balanced_ufo)) / denom_ufo

    strict_thr = compute_shock_threshold([x for x in strict_crisis if x > 0])
    balanced_thr = compute_shock_threshold([x for x in balanced_crisis if x > 0])
    strict_shock = [x >= strict_thr for x in strict_crisis]
    balanced_shock = [x >= balanced_thr for x in balanced_crisis]
    shock_agreement = (
        sum(1 for a, b in zip(strict_shock, balanced_shock) if a == b) / float(len(overlap_dates))
    )

    gates = [
        {
            "name": "overlap_days>=target",
            "passed": len(overlap_dates) >= min_overlap_days,
            "detail": f"overlap_days={len(overlap_dates)}, target={min_overlap_days}",
        },
        {
            "name": "shock_agreement>=0.70",
            "passed": shock_agreement >= 0.7,
            "detail": f"shock_agreement={shock_agreement:.4f}",
        },
        {
            "name": "crisis_rel_delta<=0.60",
            "passed": rel_delta_crisis <= 0.6,
            "detail": f"rel_delta_crisis={rel_delta_crisis:.4f}",
        },
        {
            "name": "ufo_rel_delta<=0.60",
            "passed": rel_delta_ufo <= 0.6,
            "detail": f"rel_delta_ufo={rel_delta_ufo:.4f}",
        },
    ]
    stable = all(g["passed"] for g in gates)
    status = "stable" if stable else "needs_more_data_or_tuning"

    return {
        "generated_at": now,
        "status": status,
        "message": "双档稳定" if stable else "双档稳定性未达严格门槛",
        "current": {
            "strict_days": len(strict_rows),
            "strict_balanced_days": len(balanced_rows),
            "overlap_days": len(overlap_dates),
            "window_start": overlap_dates[0],
            "window_end": overlap_dates[-1],
            "excluded_non_run_day_rows": excluded_legacy_rows,
        },
        "targets": {"min_overlap_days": min_overlap_days},
        "metrics": {
            "mean_abs_delta_crisis": round(mean_abs_delta_crisis, 4),
            "mean_abs_delta_ufo": round(mean_abs_delta_ufo, 4),
            "rel_delta_crisis": round(rel_delta_crisis, 4),
            "rel_delta_ufo": round(rel_delta_ufo, 4),
            "crisis_corr": round(pearson_corr(strict_crisis, balanced_crisis), 4),
            "ufo_corr": round(pearson_corr(strict_ufo, balanced_ufo), 4),
            "strict_shock_threshold": round(strict_thr, 4),
            "balanced_shock_threshold": round(balanced_thr, 4),
            "shock_agreement": round(shock_agreement, 4),
        },
        "gates": gates,
        "next_actions": [
            "继续按日累计 strict 与 strict-balanced 两档数据",
            "若长期分歧过大，回看来源健康审计与拒绝原因分布",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="面板累计管道")
    parser.add_argument(
        "--policy",
        default="strict-balanced",
        choices=["strict", "strict-balanced", "lenient"],
        help="抓取与面板分析策略",
    )
    parser.add_argument("--skip-scrape", action="store_true", help="跳过 scraper")
    parser.add_argument("--skip-causal", action="store_true", help="跳过 causal_analyzer")
    parser.add_argument("--min-days", type=int, default=180, help="严格审批最少覆盖天数")
    parser.add_argument("--min-shocks", type=int, default=12, help="严格审批最少冲击日数")
    parser.add_argument(
        "--min-observed-ratio",
        type=float,
        default=0.85,
        help="严格审批最少有效观测覆盖率（默认 0.85）",
    )
    parser.add_argument(
        "--enforce-gate",
        action="store_true",
        help="将 --fail-on-reject 传给 causal_analyzer（未通过即非零退出）",
    )
    parser.add_argument(
        "--skip-dual-review",
        action="store_true",
        help="跳过 strict 双档稳定性评审输出",
    )
    parser.add_argument(
        "--dual-min-overlap-days",
        type=int,
        default=30,
        help="双档稳定性评审最少重叠天数（默认 30）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    py = sys.executable
    if args.skip_causal and not args.skip_scrape:
        print("[warn] --skip-causal 已开启：本次抓取结果不会写入 causal_panel（仅更新 scraped_news）。")

    if not args.skip_scrape:
        run_cmd([py, "scraper.py", "--policy", args.policy])

    if not args.skip_causal:
        cmd = [
            py, "causal_analyzer.py",
            "--panel-policy", args.policy,
            "--min-panel-days", str(args.min_days),
            "--min-panel-shocks", str(args.min_shocks),
            "--min-panel-observed-ratio", str(args.min_observed_ratio),
        ]
        if args.enforce_gate:
            cmd.append("--fail-on-reject")
        run_cmd(cmd)

    rows = load_panel_rows(args.policy)
    report = compute_progress(
        rows,
        args.min_days,
        args.min_shocks,
        args.min_observed_ratio,
        args.policy,
    )
    with PROGRESS_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    dual_report = None
    if not args.skip_dual_review:
        panel_payload = load_panel_payload()
        dual_report = compute_dual_policy_review(panel_payload, min_overlap_days=args.dual_min_overlap_days)
        with DUAL_REVIEW_FILE.open("w", encoding="utf-8") as f:
            json.dump(dual_report, f, ensure_ascii=False, indent=2)

    print("\n=== 面板累计进度 ===")
    print(f"policy: {report['policy']}")
    print(f"status: {report['status']}")
    print(f"message: {report['message']}")
    print(
        f"observed_days: {report['current']['observed_days']} / {report['targets']['min_days']}, "
        f"shock_days: {report['current']['shock_days']} / {report['targets']['min_shocks']}"
    )
    print(
        f"observed_ratio: {report['current']['observed_ratio']} / "
        f"{report['targets']['min_observed_ratio']}"
    )
    print(
        f"remaining_days: {report['remaining']['days']}, "
        f"remaining_shocks: {report['remaining']['shocks']}"
    )
    print(f"progress_overall: {report['progress']['overall']}")
    print(f"[输出] {PROGRESS_FILE}")
    if dual_report is not None:
        print("\n=== 严格双档稳定性评审 ===")
        print(f"status: {dual_report['status']}")
        print(f"message: {dual_report['message']}")
        current = dual_report.get("current", {})
        print(
            f"strict_days: {current.get('strict_days', 0)}, "
            f"strict_balanced_days: {current.get('strict_balanced_days', 0)}, "
            f"overlap_days: {current.get('overlap_days', 0)}"
        )
        print(f"[输出] {DUAL_REVIEW_FILE}")


if __name__ == "__main__":
    main()
