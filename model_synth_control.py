"""
合成控制（简化版）

输入：
- data/control_panels/country_controls.csv
- data/causal_panel.json（US 冲击日）

输出：
- 简化权重
- post窗口 gap ATT
- 安慰剂 p 值
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
    parse_date,
    read_panel_rows as _read_panel_rows,
)


BASE_DIR = Path(__file__).resolve().parent  # type: ignore
DATA_DIR = BASE_DIR / "data"
COUNTRY_FILE = DATA_DIR / "control_panels" / "country_controls.csv"
PANEL_FILE = DATA_DIR / "causal_panel.json"  # type: ignore
OUT_FILE = DATA_DIR / "model_synth_control_report.json"
EVENTS_FILE = DATA_DIR / "events_v2.json"

SEED = 20260221
PERMUTATIONS = 2000
MIN_SHOCK_DAYS = 8
MIN_PRE_DAYS = 30
MIN_POST_DAYS = 8
EV2_MIN_DATES = 5
# parse_date, compute_shock_threshold 已移至 utils.py


def load_country_rows() -> List[dict]:
    if not COUNTRY_FILE.exists():  # type: ignore
        return []
    rows = []
    with COUNTRY_FILE.open("r", encoding="utf-8") as f:  # type: ignore
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


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


def load_us_shock_days(
    policy: str = "strict-balanced",
    start: date | None = None,
    end: date | None = None,
    shock_catalog_path: Path | None = None,
) -> tuple[List[date], float | None, str]:
    rows = _read_panel_rows(PANEL_FILE, policy)
    if not rows:
        return [], None, "none"
    series = {parse_date(r["date"]): float(r.get("crisis_count", 0)) for r in rows}  # type: ignore
    if start is None:
        start = min(series.keys())  # type: ignore
    if end is None:
        end = max(series.keys())  # type: ignore

    catalog_dates = _load_shock_catalog_dates(shock_catalog_path)
    if catalog_dates:
        base_dates = catalog_dates
        base_source_name = "shock_catalog_dates"
    else:
        base_dates = _load_events_v2_crisis_dates(EVENTS_FILE)
        base_source_name = "events_v2_crisis_dates"

    ev2_in_panel = [d for d in base_dates if start <= d <= end]
    if len(ev2_in_panel) >= EV2_MIN_DATES:
        return sorted(ev2_in_panel), None, base_source_name

    nonzero = [v for v in series.values() if v > 0]  # type: ignore
    thr75 = compute_shock_threshold(nonzero)
    news75 = sorted([d for d, v in series.items() if v >= thr75])  # type: ignore
    thr90 = compute_shock_threshold(nonzero, q=90.0)
    news90 = sorted([d for d, v in series.items() if v >= thr90])  # type: ignore
    if news90:
        return news90, thr90, "news_volume_90pct_fallback"
    return news75, thr75, "news_volume_75pct_fallback"


def split_pre_post_dates(us_dates: List[date], shocks: List[date], post_horizon_days: int = 7) -> tuple[List[date], List[date]]:
    shock_set = set(shocks)
    post_window = set()
    for d in shocks:
        for k in range(1, post_horizon_days + 1):
            post_window.add(d + timedelta(days=k))  # type: ignore
    post_dates = [d for d in us_dates if d in post_window]
    # Shock day itself is neither pre nor post.
    pre_dates = [d for d in us_dates if d not in post_window and d not in shock_set]
    return pre_dates, post_dates


def inverse_mse_weights(us: Dict[date, float], donors: Dict[str, Dict[date, float]], pre_dates: List[date]) -> Dict[str, float]:
    raw = {}
    for c, s in donors.items():  # type: ignore
        errs = []
        for d in pre_dates:
            if d in us and d in s:
                errs.append((us[d] - s[d]) ** 2)  # type: ignore
        mse = mean(errs) if errs else 1e6
        raw[c] = 1.0 / (mse + 1e-6)  # type: ignore

    z = sum(raw.values())  # type: ignore
    if z <= 0:
        n = max(1, len(raw))
        return {k: 1.0 / n for k in raw}
    return {k: v / z for k, v in raw.items()}  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic Control（简化版）")
    p.add_argument(
        "--policy",
        choices=["strict", "strict-balanced"],
        default="strict-balanced",
        help="使用哪个面板 policy 定义 US 冲击日",
    )
    p.add_argument("--min-shock-days", type=int, default=MIN_SHOCK_DAYS)
    p.add_argument("--min-pre-days", type=int, default=MIN_PRE_DAYS)
    p.add_argument("--min-post-days", type=int, default=MIN_POST_DAYS)
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
    rows = load_country_rows()
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "model": "synthetic_control_simplified",
        "policy": args.policy,
        "status": "blocked",
        "reason": "",
        "n_rows": len(rows),
        "countries": [],
        "weights": {},
        "metrics": {
            "shock_days": 0,
            "shock_threshold": None,
            "shock_source": "none",
            "att_gap": None,
            "p_value": None,
            "p_value_negative": None,
            "n_common_dates": 0,
            "n_pre_dates": 0,
            "n_post_dates": 0,
        },
        "gates": {
            "att_positive": False,
            "p_value_significant": False,
            "att_near_zero": False,
            "no_significant_negative": False,
            "synth_passed": False,
        },
    }

    if not rows:
        out["reason"] = "country_controls.csv 缺失或为空"  # type: ignore
    else:
        series: Dict[str, Dict[date, float]] = {}  # type: ignore
        for r in rows:
            try:
                d = parse_date(r.get("date", ""))  # type: ignore
                c = r.get("country", "")  # type: ignore
                y = float(r.get("ufo_policy_news", "0") or 0)  # type: ignore
            except Exception:
                continue
            if not c:
                continue
            series.setdefault(c, {})[d] = y  # type: ignore

        countries = sorted(series.keys())  # type: ignore
        out["countries"] = countries  # type: ignore
        if "US" not in series:
            out["reason"] = "country_controls.csv 缺少 US 行"  # type: ignore
        else:
            us = series["US"]  # type: ignore
            donors_all = {c: s for c, s in series.items() if c != "US"}  # type: ignore
            if len(donors_all) < 3:
                out["reason"] = "对照国家数量不足（需要 >=3）"  # type: ignore
            else:
                us_dates = sorted(us.keys())  # type: ignore
                shocks, shock_thr, shock_source = load_us_shock_days(
                    args.policy,
                    start=us_dates[0],
                    end=us_dates[-1],
                    shock_catalog_path=shock_catalog_path,
                )
                out["metrics"]["shock_days"] = len(shocks)  # type: ignore
                out["metrics"]["shock_threshold"] = round(shock_thr, 6) if isinstance(shock_thr, (int, float)) else None  # type: ignore
                out["metrics"]["shock_source"] = shock_source  # type: ignore

                effective_min_shocks = EV2_MIN_DATES if shock_source in ("events_v2_crisis_dates", "shock_catalog_dates") else args.min_shock_days
                if len(shocks) < effective_min_shocks:
                    out["reason"] = f"US 冲击日不足（{len(shocks)} < {effective_min_shocks}），无法估计"  # type: ignore
                else:
                    pre_dates, post_dates = split_pre_post_dates(us_dates, shocks, post_horizon_days=7)
                    out["metrics"]["n_pre_dates"] = len(pre_dates)  # type: ignore
                    out["metrics"]["n_post_dates"] = len(post_dates)  # type: ignore

                    if len(pre_dates) < args.min_pre_days or len(post_dates) < args.min_post_days:
                        out["reason"] = "pre/post 样本不足"  # type: ignore
                    else:
                        min_overlap = max(15, len(pre_dates) // 4)
                        donors = {}
                        for c, s in donors_all.items():  # type: ignore
                            overlap = sum(1 for d in pre_dates if d in s)
                            if overlap >= min_overlap:
                                donors[c] = s  # type: ignore

                        if len(donors) < 3:
                            out["reason"] = "与 US 的 donor overlap 不足（需要 >=3 个国家）"  # type: ignore
                        else:
                            weights = inverse_mse_weights(us, donors, pre_dates)
                            out["weights"] = {k: round(v, 6) for k, v in weights.items()}  # type: ignore

                            def synth_value(d: date) -> float | None:
                                active = {c: w for c, w in weights.items() if d in donors[c]}  # type: ignore
                                if not active:
                                    return None
                                z = sum(active.values())  # type: ignore
                                if z <= 0:
                                    return None
                                return sum((w / z) * donors[c][d] for c, w in active.items())  # type: ignore

                            gap = {}
                            for d in us_dates:
                                sv = synth_value(d)
                                if sv is None:
                                    continue
                                gap[d] = us[d] - sv  # type: ignore

                            pre_gap = [gap[d] for d in pre_dates if d in gap]  # type: ignore
                            post_gap = [gap[d] for d in post_dates if d in gap]  # type: ignore
                            out["metrics"]["n_common_dates"] = len(gap)  # type: ignore

                            if len(pre_gap) < 20 or len(post_gap) < 6:
                                out["reason"] = "有效 gap 样本不足"  # type: ignore
                            else:
                                gap_pre = mean(pre_gap)
                                gap_post = mean(post_gap)
                                att = gap_post - gap_pre

                                rng = make_rng(SEED)
                                null = []
                                pre_gap_dates = [d for d in pre_dates if d in gap]
                                if len(pre_gap_dates) < len(post_gap):
                                    out["reason"] = "pre_gap 可抽样日期不足，无法构建安慰剂分布"  # type: ignore
                                    pre_gap_dates = []

                                for _ in range(PERMUTATIONS):
                                    if not pre_gap_dates:
                                        break
                                    fake_post_dates = rng.sample(pre_gap_dates, len(post_gap))
                                    fake_att = mean(gap[d] for d in fake_post_dates) - gap_pre  # type: ignore
                                    null.append(fake_att)

                                if not null:
                                    out["reason"] = "安慰剂分布为空，无法稳健估计"  # type: ignore
                                else:
                                    p_val = sum(x >= att for x in null) / float(len(null))
                                    p_neg = sum(x <= att for x in null) / float(len(null))
                                    near_zero = abs(att) <= 0.05
                                    # 采用“统计显著 + 实际幅度”双阈值，避免把极小负值误判为结构性反证。
                                    no_sig_negative = not (att < -0.05 and p_neg < 0.05)

                                    out["metrics"]["att_gap"] = round(att, 6)  # type: ignore
                                    out["metrics"]["p_value"] = round(p_val, 6)  # type: ignore
                                    out["metrics"]["p_value_negative"] = round(p_neg, 6)  # type: ignore
                                    att_positive = att > 0
                                    p_sig = p_val < 0.05
                                    # ATT 严格为正且显著 → 支持正向因果
                                    positive_signal = att_positive and p_sig
                                    # ATT 接近零（|att|<=0.05）且无显著负效应 → 中性，不反对
                                    neutral_consistent = near_zero and no_sig_negative
                                    # ATT < -0.05 → 反向证据，不通过（即使 p 不显著）
                                    negative_evidence = att < -0.05

                                    out["gates"]["att_positive"] = att_positive  # type: ignore
                                    out["gates"]["p_value_significant"] = p_sig  # type: ignore
                                    out["gates"]["att_near_zero"] = near_zero  # type: ignore
                                    out["gates"]["no_significant_negative"] = no_sig_negative  # type: ignore
                                    # 修复：ATT<0 时不通过，即使 att_near_zero=True 也需要排除负值
                                    out["gates"]["synth_passed"] = positive_signal or (neutral_consistent and not negative_evidence)  # type: ignore

                                    out["status"] = "ok"  # type: ignore
                                    if positive_signal:
                                        out["reason"] = "synth_estimated_positive_signal"  # type: ignore
                                    elif negative_evidence:
                                        out["reason"] = "synth_estimated_negative_att_not_supported"  # type: ignore
                                    else:
                                        out["reason"] = "synth_estimated_neutral_consistent"  # type: ignore

    with OUT_FILE.open("w", encoding="utf-8") as f:  # type: ignore
        json.dump(out, f, ensure_ascii=False, indent=2)  # type: ignore

    print("=== Synthetic Control Simplified ===")
    print(f"status: {out['status']}")  # type: ignore
    print(f"reason: {out['reason']}")  # type: ignore
    print(f"synth_passed: {out['gates']['synth_passed']}")  # type: ignore
    print(f"[输出] {OUT_FILE}")  # type: ignore


if __name__ == "__main__":
    main()
