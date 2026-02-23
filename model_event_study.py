"""
事件研究（动态效应）脚本

输出：
- event-time 动态效应曲线（-14..+14）
- 预趋势强度检验
- 后效应峰值安慰剂检验
"""
# pyre-ignore-all-errors
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List

from utils import (  # type: ignore[import]
    compute_shock_threshold,
    make_rng,
    min_distance_to_shocks,
    parse_date,
    read_panel_rows as _read_panel_rows,
)


BASE_DIR = Path(__file__).resolve().parent  # type: ignore
DATA_DIR = BASE_DIR / "data"
PANEL_FILE = DATA_DIR / "causal_panel.json"  # type: ignore
OUT_FILE = DATA_DIR / "model_event_study_report.json"
EVENTS_FILE = DATA_DIR / "events_v2.json"

SEED = 20260221
PERMUTATIONS = 2000
MIN_OBS_DAYS = 90
MIN_SHOCK_DAYS = 10
PLACEBO_BUFFER_DAYS = 14
SHOCK_COUNT_FLOOR = 2.0
MAX_SHOCKS_FOR_PLACEBO = 180
EV2_MIN_DATES = 5


# parse_date, percentile, compute_shock_threshold 已移至 utils.py


def read_rows(policy: str = "strict-balanced") -> List[dict]:
    return _read_panel_rows(PANEL_FILE, policy)


def downsample_dates_evenly(dates: List[date], max_n: int) -> List[date]:
    if max_n <= 0 or len(dates) <= max_n:
        return list(dates)
    if max_n == 1:
        return [dates[len(dates) // 2]]
    last = len(dates) - 1
    return [dates[(i * last) // (max_n - 1)] for i in range(max_n)]  # type: ignore


def _load_events_v2_crisis_dates(events_path: Path) -> List[date]:
    if not events_path.exists():  # type: ignore
        return []
    try:
        with events_path.open("r", encoding="utf-8") as f:  # type: ignore
            payload = json.load(f)  # type: ignore
        out = []
        for row in payload.get("correlations", []):  # type: ignore
            d_str = row.get("crisis", {}).get("date")  # type: ignore
            if not d_str:
                continue
            try:
                out.append(parse_date(d_str))  # type: ignore
            except Exception:
                continue
        return sorted(set(out))  # type: ignore
    except Exception:
        return []


def _load_shock_catalog_dates(catalog_path: Path | None) -> List[date]:
    if catalog_path is None or not catalog_path.exists():  # type: ignore
        return []
    try:
        with catalog_path.open("r", encoding="utf-8") as f:  # type: ignore
            payload = json.load(f)  # type: ignore
        ds = payload.get("shock_dates", []) if isinstance(payload, dict) else []
        out = []
        for d in ds:  # type: ignore
            if not isinstance(d, str):
                continue
            try:
                out.append(parse_date(d))  # type: ignore
            except Exception:
                continue
        return sorted(set(out))  # type: ignore
    except Exception:
        return []


def select_shock_days(
    crisis_series: Dict[date, float],  # type: ignore
    start: date,
    end: date,
    events_path: Path = EVENTS_FILE,
    shock_catalog_path: Path | None = None,
) -> tuple[List[date], float | None, str]:
    catalog_dates = _load_shock_catalog_dates(shock_catalog_path)
    if catalog_dates:
        base_dates = catalog_dates
        base_source_name = "shock_catalog_dates"
    else:
        base_dates = _load_events_v2_crisis_dates(events_path)
        base_source_name = "events_v2_crisis_dates"

    ev2_in_panel = [d for d in base_dates if start <= d <= end]
    if len(ev2_in_panel) >= EV2_MIN_DATES:
        return sorted(ev2_in_panel), None, base_source_name

    crisis_nonzero = [v for v in crisis_series.values() if v > 0]  # type: ignore
    thr75 = compute_shock_threshold(crisis_nonzero)
    news75 = sorted([d for d, v in crisis_series.items() if v >= thr75])  # type: ignore
    thr90 = compute_shock_threshold(crisis_nonzero, q=90.0)
    news90 = sorted([d for d, v in crisis_series.items() if v >= thr90])  # type: ignore
    if news90:
        return news90, thr90, "news_volume_90pct_fallback"
    return news75, thr75, "news_volume_75pct_fallback"


def baseline(
    series: Dict[date, float],
    d: date,
    cache: Dict[date, float | None] | None = None,
) -> float | None:
    if cache is not None and d in cache:
        return cache[d]
    vals = []
    for k in range(-21, -7):
        dd = d + timedelta(days=k)  # type: ignore
        if dd in series:
            vals.append(series[dd])  # type: ignore
    if len(vals) < 5:
        out = None
    else:
        out = mean(vals)
    if cache is not None:
        cache[d] = out
    return out


def dynamic_effect(
    series: Dict[date, float],
    shocks: List[date],
    baseline_cache: Dict[date, float | None] | None = None,
) -> Dict[int, float | None]:
    out: Dict[int, float | None] = {}  # type: ignore
    for k in range(-14, 15):
        vals = []
        for d in shocks:
            b = baseline(series, d, cache=baseline_cache)
            if b is None:
                continue
            dd = d + timedelta(days=k)  # type: ignore
            if dd in series:
                vals.append(series[dd] - b)  # type: ignore
        out[k] = round(mean(vals), 6) if vals else None  # type: ignore
    return out


def placebo_peak_test(
    series: Dict[date, float],
    shocks: List[date],
    candidates: List[date],
    permutations: int,
    baseline_cache: Dict[date, float | None] | None = None,
) -> tuple[float, float, int]:
    dyn = dynamic_effect(series, shocks, baseline_cache=baseline_cache)
    post_vals = [v for k, v in dyn.items() if 1 <= k <= 14 and v is not None]  # type: ignore
    obs_peak = max(post_vals) if post_vals else 0.0

    m = len(shocks)
    valid_candidates = [d for d in candidates if baseline(series, d, cache=baseline_cache) is not None]
    rng = make_rng(SEED)
    if len(valid_candidates) < m or m == 0:
        return obs_peak, 1.0, 0

    null = []
    for _ in range(max(1, int(permutations))):
        sample = rng.sample(valid_candidates, m)
        d2 = dynamic_effect(series, sample, baseline_cache=baseline_cache)
        pvals = [v for k, v in d2.items() if 1 <= k <= 14 and v is not None]  # type: ignore
        if not pvals:
            continue
        null.append(max(pvals))

    if not null:
        return obs_peak, 1.0, 0
    p = sum(x >= obs_peak for x in null) / float(len(null))
    return obs_peak, p, len(null)


# min_distance_to_shocks 已移至 utils.py


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
    p.add_argument("--permutations", type=int, default=PERMUTATIONS)
    p.add_argument("--max-shocks-for-placebo", type=int, default=MAX_SHOCKS_FOR_PLACEBO)
    p.add_argument(
        "--pretrend-ratio-threshold",
        type=float,
        default=0.6,
        help="预趋势绝对均值相对 post_peak 的比例阈值（默认 0.6）",
    )
    p.add_argument(
        "--pretrend-abs-floor",
        type=float,
        default=2.0,
        help="预趋势绝对阈值下限（默认 2.0）",
    )
    p.add_argument(
        "--shock-catalog-file",
        default="",
        help="可选冲击日目录（json，含 shock_dates[]）；为空则仅使用 events_v2",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shock_catalog_path: Path | None = None
    raw_catalog = str(args.shock_catalog_file or "").strip()  # type: ignore
    if raw_catalog:
        shock_catalog_path = Path(raw_catalog)  # type: ignore
    rows = read_rows(args.policy)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "model": "event_study_dynamic",
        "policy": args.policy,
        "observed_days": len(rows),
        "status": "pending",
        "reason": "",
        "shock_days": 0,
        "shock_threshold": None,
        "shock_source": "none",
        "dynamic_effect": {},
        "metrics": {
            "pretrend_abs_mean": None,
            "pretrend_ratio_to_post_peak": None,
            "post_peak": None,
            "post_peak_p_value": None,
            "placebo_draws": 0,
            "permutations": int(args.permutations),
            "shock_days_used": 0,
            "shock_sampling": "full",
        },
        "gates": {
            "pretrend_ok": False,
            "post_peak_positive": False,
            "p_peak_significant": False,
            "event_study_passed": False,
        },
    }

    if len(rows) < args.min_observed_days:
        out["reason"] = f"observed_days < {args.min_observed_days}，事件研究不稳定"  # type: ignore
    else:
        dates = [parse_date(r["date"]) for r in rows]  # type: ignore
        ufo_series = {parse_date(r["date"]): float(r.get("ufo_count", 0)) for r in rows}  # type: ignore
        crisis_series = {parse_date(r["date"]): float(r.get("crisis_count", 0)) for r in rows}  # type: ignore

        start = min(dates)
        end = max(dates)
        shocks, thr, shock_source = select_shock_days(
            crisis_series,
            start,
            end,
            events_path=EVENTS_FILE,
            shock_catalog_path=shock_catalog_path,
        )
        out["shock_days"] = len(shocks)  # type: ignore
        out["shock_threshold"] = round(thr, 6) if isinstance(thr, (int, float)) else None  # type: ignore
        out["shock_source"] = shock_source  # type: ignore

        effective_min_shocks = EV2_MIN_DATES if shock_source in ("events_v2_crisis_dates", "shock_catalog_dates") else args.min_shock_days
        if len(shocks) < effective_min_shocks:
            out["reason"] = f"shock_days < {effective_min_shocks}，事件研究不稳定"  # type: ignore
        else:
            max_shocks = max(effective_min_shocks, int(args.max_shocks_for_placebo))
            shocks_for_event = downsample_dates_evenly(shocks, max_shocks)
            baseline_cache: Dict[date, float | None] = {}
            dyn = dynamic_effect(ufo_series, shocks_for_event, baseline_cache=baseline_cache)
            out["dynamic_effect"] = {str(k): v for k, v in dyn.items()}  # type: ignore
            out["metrics"]["shock_days_used"] = len(shocks_for_event)  # type: ignore
            out["metrics"]["shock_sampling"] = (  # type: ignore
                "downsample_evenly" if len(shocks_for_event) < len(shocks) else "full"
            )

            pre_vals = [v for k, v in dyn.items() if -14 <= k <= -1 and v is not None]  # type: ignore
            post_vals = [v for k, v in dyn.items() if 1 <= k <= 14 and v is not None]  # type: ignore
            pre_abs_mean = mean(abs(v) for v in pre_vals) if pre_vals else None
            post_peak = max(post_vals) if post_vals else None

            placebo_candidates = [
                d
                for d in dates
                if d not in set(shocks) and min_distance_to_shocks(d, shocks) > args.placebo_buffer_days
            ]
            obs_peak, p_peak, draws = placebo_peak_test(
                ufo_series,
                shocks_for_event,
                placebo_candidates,
                permutations=args.permutations,
                baseline_cache=baseline_cache,
            )

            out["metrics"]["pretrend_abs_mean"] = round(pre_abs_mean, 6) if pre_abs_mean is not None else None  # type: ignore
            out["metrics"]["post_peak"] = round(post_peak, 6) if post_peak is not None else None  # type: ignore
            out["metrics"]["post_peak_p_value"] = round(p_peak, 6)  # type: ignore
            out["metrics"]["placebo_draws"] = draws  # type: ignore

            if draws < 200:
                out["status"] = "pending"  # type: ignore
                out["reason"] = "placebo_draws_insufficient"  # type: ignore
            else:
                pretrend_ratio = None
                pretrend_limit = None
                if pre_abs_mean is not None and post_peak is not None and abs(post_peak) > 1e-9:
                    pretrend_ratio = pre_abs_mean / abs(post_peak)
                    pretrend_limit = max(float(args.pretrend_abs_floor), float(args.pretrend_ratio_threshold) * abs(post_peak))
                if pretrend_ratio is not None:
                    out["metrics"]["pretrend_ratio_to_post_peak"] = round(pretrend_ratio, 6)  # type: ignore

                pretrend_ok = (
                    pre_abs_mean is not None
                    and pretrend_limit is not None
                    and pre_abs_mean <= pretrend_limit
                )
                post_peak_positive = post_peak is not None and post_peak > 0
                p_peak_sig = p_peak < 0.05

                out["gates"]["pretrend_ok"] = pretrend_ok  # type: ignore
                out["gates"]["post_peak_positive"] = post_peak_positive  # type: ignore
                out["gates"]["p_peak_significant"] = p_peak_sig  # type: ignore
                out["gates"]["event_study_passed"] = pretrend_ok and post_peak_positive and p_peak_sig  # type: ignore

                out["status"] = "ok"  # type: ignore
                out["reason"] = "event_study_estimated"  # type: ignore

    with OUT_FILE.open("w", encoding="utf-8") as f:  # type: ignore
        json.dump(out, f, ensure_ascii=False, indent=2)  # type: ignore

    print("=== Event Study Dynamic ===")
    print(f"status: {out['status']}")  # type: ignore
    print(f"reason: {out['reason']}")  # type: ignore
    print(f"shock_days: {out['shock_days']} (source={out['shock_source']})")  # type: ignore
    print(f"event_study_passed: {out['gates']['event_study_passed']}")  # type: ignore
    print(f"[输出] {OUT_FILE}")  # type: ignore


if __name__ == "__main__":
    main()
