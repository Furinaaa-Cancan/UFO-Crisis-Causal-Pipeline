"""
面板累计管道（唯一有效路径执行器）

作用：
1) 运行 scraper（可选）
2) 运行 causal_analyzer（可选）
3) 计算面板累计进度并写入 data/panel_progress.json

推荐每日执行一次（同一 policy）。
"""
# pyre-ignore-all-errors
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

from utils import (  # type: ignore[import]
    compute_shock_threshold,
    max_missing_streak,
    parse_date,
    pearson_corr,
    percentile,
)


BASE_DIR = Path(__file__).resolve().parent  # type: ignore
DATA_DIR = BASE_DIR / "data"
SCRAPED_FILE = DATA_DIR / "scraped_news.json"
PANEL_FILE = DATA_DIR / "causal_panel.json"  # type: ignore
PROGRESS_FILE = DATA_DIR / "panel_progress.json"
DUAL_REVIEW_FILE = DATA_DIR / "strict_dual_review.json"
SHOCK_LOCK_FILE = DATA_DIR / "crisis_shock_catalog_lock.json"
SHOCK_COUNT_FLOOR = 2.0


# parse_date, percentile, pearson_corr, compute_shock_threshold, max_missing_streak 已移至 utils.py


def run_cmd(cmd: List[str]) -> None:
    print(f"[run] {' '.join(cmd)}")  # type: ignore
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def validate_shock_catalog_lock(
    catalog_path: Path,
    lock_path: Path,
    shock_catalog_key: str,
) -> tuple[bool, str]:
    if not catalog_path.exists():  # type: ignore
        return False, f"catalog_missing:{catalog_path}"
    if not lock_path.exists():  # type: ignore
        return False, f"lock_missing:{lock_path}"

    try:
        with catalog_path.open("r", encoding="utf-8") as f:  # type: ignore
            catalog = json.load(f)  # type: ignore
    except Exception as e:
        return False, f"catalog_invalid_json:{e}"
    try:
        with lock_path.open("r", encoding="utf-8") as f:  # type: ignore
            lock = json.load(f)  # type: ignore
    except Exception as e:
        return False, f"lock_invalid_json:{e}"

    if not isinstance(catalog, dict):
        return False, "catalog_not_object"
    if not isinstance(lock, dict):
        return False, "lock_not_object"

    catalog_sig = str(catalog.get("catalog_signature_sha256", "") or "")
    lock_sig = str(lock.get("catalog_signature_sha256", "") or "")
    if not catalog_sig:
        return False, "catalog_signature_missing"
    if not lock_sig:
        return False, "lock_signature_missing"
    if catalog_sig != lock_sig:
        return False, f"signature_mismatch:catalog={catalog_sig},lock={lock_sig}"

    key = str(shock_catalog_key or "shock_dates").strip() or "shock_dates"
    if key not in catalog:
        return False, f"shock_catalog_key_missing:{key}"
    return True, "ok"


def load_panel_rows(policy: str) -> List[dict]:
    if not PANEL_FILE.exists():  # type: ignore
        return []
    with PANEL_FILE.open("r", encoding="utf-8") as f:  # type: ignore
        payload = json.load(f)  # type: ignore
    rows = payload.get("rows", [])  # type: ignore
    run_day_rows = [r for r in rows if r.get("date_scope") == "run_day_only"]  # type: ignore
    effective_rows = run_day_rows if run_day_rows else rows
    filtered = [r for r in effective_rows if r.get("policy") == policy]  # type: ignore
    by_date: Dict[str, dict] = {}  # type: ignore
    for row in filtered:
        by_date[row.get("date", "")] = row  # type: ignore
    return [by_date[k] for k in sorted(by_date.keys()) if k]  # type: ignore


def load_panel_payload() -> dict:
    if not PANEL_FILE.exists():  # type: ignore
        return {"meta": {"version": 1}, "rows": []}
    with PANEL_FILE.open("r", encoding="utf-8") as f:  # type: ignore
        payload = json.load(f)  # type: ignore
    payload.setdefault("meta", {"version": 1})  # type: ignore
    payload.setdefault("rows", [])  # type: ignore
    return payload


def compute_progress(
    rows: List[dict],  # type: ignore
    min_days: int,
    min_shocks: int,
    min_observed_ratio: float,
    policy: str,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()  # type: ignore
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

    dates = [parse_date(r["date"]) for r in rows]  # type: ignore
    start = min(dates)
    end = max(dates)
    span_days = (end - start).days + 1
    observed_days = len(rows)
    missing_days = max(0, span_days - observed_days)
    all_dates = [start + timedelta(days=i) for i in range(span_days)]  # type: ignore
    observed_ratio = (observed_days / float(span_days)) if span_days > 0 else 0.0
    longest_gap = max_missing_streak(all_dates, set(dates))

    crisis_nonzero = [float(r.get("crisis_count", 0)) for r in rows if float(r.get("crisis_count", 0)) > 0]  # type: ignore
    shock_threshold = compute_shock_threshold(crisis_nonzero)
    shock_days = sum(1 for r in rows if float(r.get("crisis_count", 0)) >= shock_threshold)  # type: ignore

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
    # Earliest readiness must satisfy both day and shock thresholds.
    required_extra_days = max(remaining_days, remaining_shocks)
    earliest_ready_date = (
        (end + timedelta(days=required_extra_days)).isoformat()
        if required_extra_days > 0
        else end.isoformat()
    )  # type: ignore

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
            "start_date": start.isoformat(),  # type: ignore
            "end_date": end.isoformat(),  # type: ignore
            "span_days": span_days,
            "observed_days": observed_days,
            "missing_days": missing_days,
            "observed_ratio": round(observed_ratio, 4),  # type: ignore
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
            "days": round(days_progress, 4),  # type: ignore
            "shocks": round(shocks_progress, 4),  # type: ignore
            "coverage_ratio": round(ratio_progress, 4),  # type: ignore
            "overall": round(overall, 4),  # type: ignore
        },
        "next_actions": [
            f"每天固定跑：python scraper.py --policy {policy}",
            (
                "每天固定跑：python causal_analyzer.py --panel-policy "  # type: ignore
                f"{policy} --min-panel-observed-ratio {min_observed_ratio}"  # type: ignore
            ),
            "达到门槛后跑：python causal_analyzer.py --fail-on-reject",
        ],
    }


def compute_dual_policy_review(panel_payload: dict, min_overlap_days: int = 30) -> dict:
    now = datetime.now(timezone.utc).isoformat()  # type: ignore
    rows = panel_payload.get("rows", [])  # type: ignore
    run_day_rows = [r for r in rows if r.get("date_scope") == "run_day_only"]  # type: ignore
    excluded_legacy_rows = len(rows) - len(run_day_rows)

    strict_rows = {r.get("date"): r for r in run_day_rows if r.get("policy") == "strict" and r.get("date")}  # type: ignore
    balanced_rows = {
        r.get("date"): r for r in run_day_rows if r.get("policy") == "strict-balanced" and r.get("date")  # type: ignore
    }

    overlap_dates = sorted(set(strict_rows.keys()) & set(balanced_rows.keys()))  # type: ignore
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

    strict_crisis = [float(strict_rows[d].get("crisis_count", 0)) for d in overlap_dates]  # type: ignore
    balanced_crisis = [float(balanced_rows[d].get("crisis_count", 0)) for d in overlap_dates]  # type: ignore
    strict_ufo = [float(strict_rows[d].get("ufo_count", 0)) for d in overlap_dates]  # type: ignore
    balanced_ufo = [float(balanced_rows[d].get("ufo_count", 0)) for d in overlap_dates]  # type: ignore

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
    stable = all(g["passed"] for g in gates)  # type: ignore
    status = "stable" if stable else "needs_more_data_or_tuning"

    return {
        "generated_at": now,
        "status": status,
        "message": "双档稳定" if stable else "双档稳定性未达严格门槛",
        "current": {
            "strict_days": len(strict_rows),
            "strict_balanced_days": len(balanced_rows),
            "overlap_days": len(overlap_dates),
            "window_start": overlap_dates[0],  # type: ignore
            "window_end": overlap_dates[-1],  # type: ignore
            "excluded_non_run_day_rows": excluded_legacy_rows,
        },
        "targets": {"min_overlap_days": min_overlap_days},
        "metrics": {
            "mean_abs_delta_crisis": round(mean_abs_delta_crisis, 4),  # type: ignore
            "mean_abs_delta_ufo": round(mean_abs_delta_ufo, 4),  # type: ignore
            "rel_delta_crisis": round(rel_delta_crisis, 4),  # type: ignore
            "rel_delta_ufo": round(rel_delta_ufo, 4),  # type: ignore
            "crisis_corr": round(pearson_corr(strict_crisis, balanced_crisis), 4),  # type: ignore
            "ufo_corr": round(pearson_corr(strict_ufo, balanced_ufo), 4),  # type: ignore
            "strict_shock_threshold": round(strict_thr, 4),  # type: ignore
            "balanced_shock_threshold": round(balanced_thr, 4),  # type: ignore
            "shock_agreement": round(shock_agreement, 4),  # type: ignore
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
        choices=["strict", "strict-balanced"],
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
    parser.add_argument(
        "--shock-catalog-file",
        default="",
        help="可选冲击日目录（传递给 causal_analyzer）；为空则仅使用 events_v2 主轨",
    )
    parser.add_argument(
        "--shock-catalog-key",
        default="shock_dates",
        help="冲击目录字段名（默认 shock_dates；可切到 shock_dates_nonoverlap_30d 等）",
    )
    parser.add_argument(
        "--shock-lock-file",
        default=str(SHOCK_LOCK_FILE),
        help="冲击目录锁文件路径（默认 data/crisis_shock_catalog_lock.json）",
    )
    parser.add_argument(
        "--skip-shock-lock-check",
        action="store_true",
        help="跳过冲击目录锁一致性检查（仅探索场景）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    py = sys.executable
    shock_catalog_raw = str(args.shock_catalog_file or "").strip()  # type: ignore
    if shock_catalog_raw and not args.skip_shock_lock_check:  # type: ignore
        lock_raw = str(args.shock_lock_file or SHOCK_LOCK_FILE)
        lock_path = Path(lock_raw)
        if not lock_path.is_absolute():  # type: ignore
            lock_path = BASE_DIR / lock_path
        ok, reason = validate_shock_catalog_lock(
            catalog_path=Path(shock_catalog_raw),  # type: ignore
            lock_path=lock_path,
            shock_catalog_key=str(args.shock_catalog_key or "shock_dates"),
        )
        if not ok:
            print(f"[error] shock catalog lock check failed: {reason}")  # type: ignore
            raise SystemExit(3)
        print(f"[info] shock catalog lock check passed: {lock_path}")  # type: ignore

    if args.skip_causal and not args.skip_scrape:
        print("[warn] --skip-causal 已开启：本次抓取结果不会写入 causal_panel（仅更新 scraped_news）。")  # type: ignore

    if not args.skip_scrape:
        run_cmd([py, "scraper.py", "--policy", args.policy])

    if not args.skip_causal:
        cmd = [
            py, "causal_analyzer.py",
            "--panel-policy", args.policy,  # type: ignore
            "--min-panel-days", str(args.min_days),  # type: ignore
            "--min-panel-shocks", str(args.min_shocks),  # type: ignore
            "--min-panel-observed-ratio", str(args.min_observed_ratio),  # type: ignore
        ]
        if args.enforce_gate:
            cmd.append("--fail-on-reject")
        if str(args.shock_catalog_file or "").strip():  # type: ignore
            cmd.extend(["--shock-catalog-file", str(args.shock_catalog_file)])  # type: ignore
            cmd.extend(["--shock-catalog-key", str(args.shock_catalog_key or "shock_dates")])  # type: ignore
        run_cmd(cmd)
    else:
        # skip-causal 场景下，刷新一次 causal_report（不写面板），避免后续严格评审读到旧口径。
        if not SCRAPED_FILE.exists():  # type: ignore
            print("[warn] --skip-causal：未找到 scraped_news.json，跳过 causal_report 刷新。")  # type: ignore
        else:
            cmd = [
                py,
                "causal_analyzer.py",
                "--no-update-panel",
                "--panel-policy",
                args.policy,  # type: ignore
                "--min-panel-days",
                str(args.min_days),  # type: ignore
                "--min-panel-shocks",
                str(args.min_shocks),  # type: ignore
                "--min-panel-observed-ratio",
                str(args.min_observed_ratio),  # type: ignore
            ]
            if str(args.shock_catalog_file or "").strip():  # type: ignore
                cmd.extend(["--shock-catalog-file", str(args.shock_catalog_file)])  # type: ignore
                cmd.extend(["--shock-catalog-key", str(args.shock_catalog_key or "shock_dates")])  # type: ignore
            print("[info] --skip-causal 已开启：将刷新 causal_report（--no-update-panel）以保持口径一致。")  # type: ignore
            run_cmd(cmd)

    rows = load_panel_rows(args.policy)
    report = compute_progress(
        rows,
        args.min_days,
        args.min_shocks,
        args.min_observed_ratio,
        args.policy,
    )
    with PROGRESS_FILE.open("w", encoding="utf-8") as f:  # type: ignore
        json.dump(report, f, ensure_ascii=False, indent=2)  # type: ignore

    dual_report = None
    if not args.skip_dual_review:
        panel_payload = load_panel_payload()
        dual_report = compute_dual_policy_review(panel_payload, min_overlap_days=args.dual_min_overlap_days)
        with DUAL_REVIEW_FILE.open("w", encoding="utf-8") as f:  # type: ignore
            json.dump(dual_report, f, ensure_ascii=False, indent=2)  # type: ignore

    print("\n=== 面板累计进度 ===")
    print(f"policy: {report['policy']}")  # type: ignore
    print(f"status: {report['status']}")  # type: ignore
    print(f"message: {report['message']}")  # type: ignore
    print(
        f"observed_days: {report['current']['observed_days']} / {report['targets']['min_days']}, "  # type: ignore
        f"shock_days: {report['current']['shock_days']} / {report['targets']['min_shocks']}"  # type: ignore
    )
    print(
        f"observed_ratio: {report['current']['observed_ratio']} / "  # type: ignore
        f"{report['targets']['min_observed_ratio']}"  # type: ignore
    )
    print(
        f"remaining_days: {report['remaining']['days']}, "  # type: ignore
        f"remaining_shocks: {report['remaining']['shocks']}"  # type: ignore
    )
    print(f"progress_overall: {report['progress']['overall']}")  # type: ignore
    print(f"[输出] {PROGRESS_FILE}")  # type: ignore
    if dual_report is not None:
        print("\n=== 严格双档稳定性评审 ===")
        print(f"status: {dual_report['status']}")  # type: ignore
        print(f"message: {dual_report['message']}")  # type: ignore
        current = dual_report.get("current", {})  # type: ignore
        print(
            f"strict_days: {current.get('strict_days', 0)}, "  # type: ignore
            f"strict_balanced_days: {current.get('strict_balanced_days', 0)}, "  # type: ignore
            f"overlap_days: {current.get('overlap_days', 0)}"  # type: ignore
        )
        print(f"[输出] {DUAL_REVIEW_FILE}")  # type: ignore


if __name__ == "__main__":
    main()
