"""
窗口均值差检验（Window Mean-Difference Test）

注意：本脚本实现的是"冲击日前后窗口均值差 + 置换检验"，
不是 Callaway-Sant'Anna 多期 DID（后者需要单位固定效应和平行趋势检验）。
结论中应表述为"窗口均值差检验"，不应声称"DID 因果识别"。

目标：
- 基于冲击日与非冲击日做窗口 ATT 估计（7/14/30）
- 用置换检验给出 p 值
- 用负对照主题（control_topics.csv）做证伪检查
"""
# pyre-ignore-all-errors
from __future__ import annotations

import argparse
import csv
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
    quantile,
    read_panel_rows as _read_panel_rows,
)


BASE_DIR = Path(__file__).resolve().parent  # type: ignore
DATA_DIR = BASE_DIR / "data"
PANEL_FILE = DATA_DIR / "causal_panel.json"  # type: ignore
TOPIC_FILE = DATA_DIR / "control_panels" / "control_topics.csv"
OUT_FILE = DATA_DIR / "model_did_report.json"
EVENTS_FILE = DATA_DIR / "events_v2.json"

WINDOWS = (7, 14, 30)
PERMUTATIONS = 2000
SEED = 20260221
MIN_OBS_DAYS = 60
MIN_SHOCK_DAYS = 8
PLACEBO_BUFFER_DAYS = 7
SHOCK_COUNT_FLOOR = 2.0
MAX_SHOCKS_FOR_ESTIMATION = 180
EV2_MIN_DATES = 5
NEG_CTRL_MIN_EFFECT = 0.05


# parse_date, percentile, quantile, compute_shock_threshold 已移至 utils.py


def read_panel_rows(policy: str = "strict-balanced") -> List[dict]:
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


def select_shock_days(
    crisis_series: Dict[date, float],  # type: ignore
    start: date,
    end: date,
    events_path: Path = EVENTS_FILE,
) -> tuple[List[date], float | None, str]:
    ev2_dates = _load_events_v2_crisis_dates(events_path)
    ev2_in_panel = [d for d in ev2_dates if start <= d <= end]
    if len(ev2_in_panel) >= EV2_MIN_DATES:
        return sorted(ev2_in_panel), None, "events_v2_crisis_dates"

    crisis_nonzero = [v for v in crisis_series.values() if v > 0]  # type: ignore
    thr75 = compute_shock_threshold(crisis_nonzero)
    news75 = sorted([d for d, v in crisis_series.items() if v >= thr75])  # type: ignore
    thr90 = compute_shock_threshold(crisis_nonzero, q=90.0)
    news90 = sorted([d for d, v in crisis_series.items() if v >= thr90])  # type: ignore
    if news90:
        return news90, thr90, "news_volume_90pct_fallback"
    return news75, thr75, "news_volume_75pct_fallback"


def load_topic_series() -> Dict[str, Dict[date, float]]:
    out: Dict[str, Dict[date, float]] = {}  # type: ignore
    if not TOPIC_FILE.exists():  # type: ignore
        return out

    with TOPIC_FILE.open("r", encoding="utf-8") as f:  # type: ignore
        reader = csv.DictReader(f)
        for row in reader:
            try:
                d = parse_date(row.get("date", ""))  # type: ignore
                topic = row.get("topic", "")  # type: ignore
                cnt = float(row.get("count", "0") or 0)  # type: ignore
            except Exception:
                continue
            if not topic:
                continue
            out.setdefault(topic, {})[d] = cnt  # type: ignore
    return out


def window_effect(series: Dict[date, float], d: date, w: int) -> float | None:
    pre = []
    post = []
    for k in range(1, w + 1):
        dp = d - timedelta(days=k)  # type: ignore
        dn = d + timedelta(days=k)  # type: ignore
        if dp in series:
            pre.append(series[dp])  # type: ignore
        if dn in series:
            post.append(series[dn])  # type: ignore
    min_points = max(3, w // 3)
    if len(pre) < min_points or len(post) < min_points:
        return None
    return mean(post) - mean(pre)


def evaluate_window(
    series: Dict[date, float],  # type: ignore
    shocks: List[date],  # type: ignore
    placebo_pool: List[date],  # type: ignore
    w: int,
) -> dict:
    obs_vals = [window_effect(series, d, w) for d in shocks]
    obs_vals = [v for v in obs_vals if v is not None]
    if len(obs_vals) < max(3, len(shocks) // 2):
        return {
            "status": "pending",
            "reason": "insufficient_window_observations",
            "att": None,
            "p_value": None,
            "ci95": None,
        }

    obs_att = mean(obs_vals)
    m = len(obs_vals)
    candidates = [d for d in placebo_pool if window_effect(series, d, w) is not None]
    if len(candidates) < m:
        return {
            "status": "pending",
            "reason": "insufficient_placebo_pool",
            "att": round(obs_att, 6),  # type: ignore
            "p_value": None,
            "ci95": None,
        }

    random_inst = make_rng(SEED + w)
    null = []
    for _ in range(PERMUTATIONS):
        sample = random_inst.sample(candidates, m)
        vals = [window_effect(series, d, w) for d in sample]
        vals = [v for v in vals if v is not None]
        if len(vals) < max(3, m // 2):
            continue
        null.append(mean(vals))

    if not null:
        return {
            "status": "pending",
            "reason": "null_distribution_empty",
            "att": round(obs_att, 6),  # type: ignore
            "p_value": None,
            "ci95": None,
        }

    p_val = sum(v >= obs_att for v in null) / float(len(null))
    ci = [round(quantile(null, 0.025), 6), round(quantile(null, 0.975), 6)]  # type: ignore

    return {
        "status": "ok",
        "reason": "estimated",
        "att": round(obs_att, 6),  # type: ignore
        "p_value": round(p_val, 6),  # type: ignore
        "ci95": ci,
        "n_obs_events": len(obs_vals),
        "n_null": len(null),
    }


def _p_below(value: float | None, threshold: float) -> bool:
    return isinstance(value, (int, float)) and float(value) < float(threshold)


# min_distance_to_shocks 已移至 utils.py


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DID（准实验）")
    p.add_argument(
        "--policy",
        choices=["strict", "strict-balanced"],
        default="strict-balanced",
        help="读取哪个面板 policy",
    )
    p.add_argument("--min-observed-days", type=int, default=MIN_OBS_DAYS)
    p.add_argument("--min-shock-days", type=int, default=MIN_SHOCK_DAYS)
    p.add_argument("--placebo-buffer-days", type=int, default=PLACEBO_BUFFER_DAYS)
    p.add_argument("--max-shocks-for-estimation", type=int, default=MAX_SHOCKS_FOR_ESTIMATION)
    p.add_argument(
        "--negative-control-min-effect",
        type=float,
        default=NEG_CTRL_MIN_EFFECT,
        help="负对照违规的最小效应阈值（避免极小幅度噪声触发违规）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_panel_rows(args.policy)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "model": "window_mean_diff_permutation_test",
        "policy": args.policy,
        "observed_days": len(rows),
        "status": "pending",
        "reason": "",
        "shock_threshold": None,
        "shock_source": "none",
        "shock_days": 0,
        "shock_days_used": 0,
        "shock_sampling": "full",
        "windows": {},
        "negative_controls": {},
        "gates": {
            "significant_positive_windows": 0,
            "negative_controls_available": False,
            "negative_controls_estimated": 0,
            "negative_control_violations": 0,
            "negative_control_borderline_positives": 0,
            "negative_control_min_effect": float(args.negative_control_min_effect),
            "did_passed": False,
        },
    }

    if len(rows) < args.min_observed_days:
        out["reason"] = f"observed_days < {args.min_observed_days}，先累计样本"  # type: ignore
    else:
        dates = [parse_date(r["date"]) for r in rows]  # type: ignore
        ufo_series = {parse_date(r["date"]): float(r.get("ufo_count", 0)) for r in rows}  # type: ignore
        crisis_series = {parse_date(r["date"]): float(r.get("crisis_count", 0)) for r in rows}  # type: ignore

        start = min(dates)
        end = max(dates)
        shocks, thr, shock_source = select_shock_days(crisis_series, start, end)
        shock_set = set(shocks)
        placebo_pool = [
            d for d in dates
            if d not in shock_set and min_distance_to_shocks(d, shocks) > args.placebo_buffer_days
        ]

        out["shock_threshold"] = round(thr, 6) if isinstance(thr, (int, float)) else None  # type: ignore
        out["shock_source"] = shock_source  # type: ignore
        out["shock_days"] = len(shocks)  # type: ignore

        effective_min_shocks = EV2_MIN_DATES if shock_source == "events_v2_crisis_dates" else args.min_shock_days
        if len(shocks) < effective_min_shocks:
            out["reason"] = f"shock_days < {effective_min_shocks}，DID 估计不稳定"  # type: ignore
        else:
            max_shocks = max(effective_min_shocks, int(args.max_shocks_for_estimation))
            shocks_for_estimation = downsample_dates_evenly(shocks, max_shocks)
            out["shock_days_used"] = len(shocks_for_estimation)  # type: ignore
            out["shock_sampling"] = "downsample_evenly" if len(shocks_for_estimation) < len(shocks) else "full"  # type: ignore

            sig_pos = 0
            for w in WINDOWS:
                wr = evaluate_window(ufo_series, shocks_for_estimation, placebo_pool, w)
                out["windows"][str(w)] = wr  # type: ignore
                if wr.get("status") == "ok" and wr.get("att", 0) > 0 and _p_below(wr.get("p_value"), 0.05):  # type: ignore
                    sig_pos += 1

            topic_series = load_topic_series()
            if not topic_series:
                out["status"] = "blocked"  # type: ignore
                out["reason"] = "negative_controls_missing（control_topics.csv 无可用负对照）"  # type: ignore
            else:
                neg_viol = 0
                neg_est = 0
                neg_borderline = 0
                for topic, series in topic_series.items():  # type: ignore
                    wr = evaluate_window(series, shocks_for_estimation, placebo_pool, 7)
                    out["negative_controls"][topic] = wr  # type: ignore
                    if wr.get("status") == "ok":  # type: ignore
                        neg_est += 1  # type: ignore
                    if wr.get("status") == "ok" and wr.get("att", 0) > 0 and _p_below(wr.get("p_value"), 0.1):  # type: ignore
                        if float(wr.get("att", 0) or 0.0) > float(args.negative_control_min_effect):  # type: ignore
                            neg_viol += 1
                        else:
                            neg_borderline += 1

                out["gates"]["significant_positive_windows"] = sig_pos  # type: ignore
                out["gates"]["negative_controls_available"] = neg_est > 0  # type: ignore
                out["gates"]["negative_controls_estimated"] = neg_est  # type: ignore
                out["gates"]["negative_control_violations"] = neg_viol  # type: ignore
                out["gates"]["negative_control_borderline_positives"] = neg_borderline  # type: ignore
                out["gates"]["did_passed"] = sig_pos >= 1 and neg_viol == 0 and neg_est > 0  # type: ignore

                out["status"] = "ok" if neg_est > 0 else "pending"  # type: ignore
                out["reason"] = "did_estimated" if neg_est > 0 else "negative_controls_insufficient"  # type: ignore

    with OUT_FILE.open("w", encoding="utf-8") as f:  # type: ignore
        json.dump(out, f, ensure_ascii=False, indent=2)  # type: ignore

    print("=== Window Mean-Difference Test ===")
    print(f"status: {out['status']}")  # type: ignore
    print(f"reason: {out['reason']}")  # type: ignore
    print(
        f"shock_days: {out['shock_days']} "
        f"(used={out['shock_days_used']}, sampling={out['shock_sampling']}, source={out['shock_source']})"
    )  # type: ignore
    print(f"did_passed: {out['gates']['did_passed']}")  # type: ignore
    print(f"[输出] {OUT_FILE}")  # type: ignore


if __name__ == "__main__":
    main()
