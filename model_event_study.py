"""
事件研究（动态效应）脚本

输出：
- event-time 动态效应曲线（-14..+14）
- 预趋势强度检验
- 后效应峰值安慰剂检验
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PANEL_FILE = DATA_DIR / "causal_panel.json"
OUT_FILE = DATA_DIR / "model_event_study_report.json"

SEED = 20260221
PERMUTATIONS = 2000
MIN_OBS_DAYS = 90
MIN_SHOCK_DAYS = 10
PLACEBO_BUFFER_DAYS = 14
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


def compute_shock_threshold(nonzero_values: List[float], q: float = 75.0, floor: float = SHOCK_COUNT_FLOOR) -> float:
    if not nonzero_values:
        return floor
    return max(floor, percentile(nonzero_values, q))


def read_rows(policy: str = "strict-balanced") -> List[dict]:
    if not PANEL_FILE.exists():
        return []
    with PANEL_FILE.open("r", encoding="utf-8") as f:
        rows = json.load(f).get("rows", [])
    run_day_rows = [r for r in rows if r.get("date_scope") == "run_day_only"]
    effective_rows = run_day_rows if run_day_rows else rows
    by_date = {}
    for r in effective_rows:
        if r.get("policy") == policy and r.get("date"):
            by_date[r["date"]] = r
    return [by_date[d] for d in sorted(by_date)]


def baseline(series: Dict[date, float], d: date) -> float | None:
    vals = []
    for k in range(-21, -7):
        dd = d + timedelta(days=k)
        if dd in series:
            vals.append(series[dd])
    if len(vals) < 5:
        return None
    return mean(vals)


def dynamic_effect(series: Dict[date, float], shocks: List[date]) -> Dict[int, float | None]:
    out: Dict[int, float | None] = {}
    for k in range(-14, 15):
        vals = []
        for d in shocks:
            b = baseline(series, d)
            if b is None:
                continue
            dd = d + timedelta(days=k)
            if dd in series:
                vals.append(series[dd] - b)
        out[k] = round(mean(vals), 6) if vals else None
    return out


def placebo_peak_test(series: Dict[date, float], shocks: List[date], candidates: List[date]) -> tuple[float, float, int]:
    dyn = dynamic_effect(series, shocks)
    post_vals = [v for k, v in dyn.items() if 1 <= k <= 14 and v is not None]
    obs_peak = max(post_vals) if post_vals else 0.0

    m = len(shocks)
    valid_candidates = [d for d in candidates if baseline(series, d) is not None]
    if len(valid_candidates) < m or m == 0:
        return obs_peak, 1.0, 0

    random.seed(SEED)
    null = []
    for _ in range(PERMUTATIONS):
        sample = random.sample(valid_candidates, m)
        d2 = dynamic_effect(series, sample)
        pvals = [v for k, v in d2.items() if 1 <= k <= 14 and v is not None]
        if not pvals:
            continue
        null.append(max(pvals))

    if not null:
        return obs_peak, 1.0, 0
    p = sum(x >= obs_peak for x in null) / float(len(null))
    return obs_peak, p, len(null)


def min_distance_to_shocks(d: date, shocks: List[date]) -> int:
    if not shocks:
        return 10**9
    return min(abs((d - s).days) for s in shocks)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="事件研究（动态效应）")
    p.add_argument(
        "--policy",
        choices=["strict", "strict-balanced"],
        default="strict-balanced",
        help="读取哪个面板 policy",
    )
    p.add_argument("--min-observed-days", type=int, default=MIN_OBS_DAYS)
    p.add_argument("--min-shock-days", type=int, default=MIN_SHOCK_DAYS)
    p.add_argument("--placebo-buffer-days", type=int, default=PLACEBO_BUFFER_DAYS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.policy)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "event_study_dynamic",
        "policy": args.policy,
        "observed_days": len(rows),
        "status": "pending",
        "reason": "",
        "shock_days": 0,
        "shock_threshold": None,
        "dynamic_effect": {},
        "metrics": {
            "pretrend_abs_mean": None,
            "post_peak": None,
            "post_peak_p_value": None,
            "placebo_draws": 0,
        },
        "gates": {
            "pretrend_ok": False,
            "post_peak_positive": False,
            "p_peak_significant": False,
            "event_study_passed": False,
        },
    }

    if len(rows) < args.min_observed_days:
        out["reason"] = f"observed_days < {args.min_observed_days}，事件研究不稳定"
    else:
        dates = [parse_date(r["date"]) for r in rows]
        ufo_series = {parse_date(r["date"]): float(r.get("ufo_count", 0)) for r in rows}
        crisis_series = {parse_date(r["date"]): float(r.get("crisis_count", 0)) for r in rows}

        crisis_nonzero = [v for v in crisis_series.values() if v > 0]
        thr = compute_shock_threshold(crisis_nonzero)
        shocks = sorted([d for d in dates if crisis_series.get(d, 0.0) >= thr])
        out["shock_days"] = len(shocks)
        out["shock_threshold"] = round(thr, 6)

        if len(shocks) < args.min_shock_days:
            out["reason"] = f"shock_days < {args.min_shock_days}，事件研究不稳定"
        else:
            dyn = dynamic_effect(ufo_series, shocks)
            out["dynamic_effect"] = {str(k): v for k, v in dyn.items()}

            pre_vals = [v for k, v in dyn.items() if -14 <= k <= -1 and v is not None]
            post_vals = [v for k, v in dyn.items() if 1 <= k <= 14 and v is not None]
            pre_abs_mean = mean(abs(v) for v in pre_vals) if pre_vals else None
            post_peak = max(post_vals) if post_vals else None

            placebo_candidates = [
                d
                for d in dates
                if d not in set(shocks) and min_distance_to_shocks(d, shocks) > args.placebo_buffer_days
            ]
            obs_peak, p_peak, draws = placebo_peak_test(ufo_series, shocks, placebo_candidates)

            out["metrics"]["pretrend_abs_mean"] = round(pre_abs_mean, 6) if pre_abs_mean is not None else None
            out["metrics"]["post_peak"] = round(post_peak, 6) if post_peak is not None else None
            out["metrics"]["post_peak_p_value"] = round(p_peak, 6)
            out["metrics"]["placebo_draws"] = draws

            if draws < 200:
                out["status"] = "pending"
                out["reason"] = "placebo_draws_insufficient"
            else:
                pretrend_ok = pre_abs_mean is not None and pre_abs_mean <= 0.25
                post_peak_positive = post_peak is not None and post_peak > 0
                p_peak_sig = p_peak < 0.05

                out["gates"]["pretrend_ok"] = pretrend_ok
                out["gates"]["post_peak_positive"] = post_peak_positive
                out["gates"]["p_peak_significant"] = p_peak_sig
                out["gates"]["event_study_passed"] = pretrend_ok and post_peak_positive and p_peak_sig

                out["status"] = "ok"
                out["reason"] = "event_study_estimated"

    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=== Event Study Dynamic ===")
    print(f"status: {out['status']}")
    print(f"reason: {out['reason']}")
    print(f"shock_days: {out['shock_days']}")
    print(f"event_study_passed: {out['gates']['event_study_passed']}")
    print(f"[输出] {OUT_FILE}")


if __name__ == "__main__":
    main()
