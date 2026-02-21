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

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
COUNTRY_FILE = DATA_DIR / "control_panels" / "country_controls.csv"
PANEL_FILE = DATA_DIR / "causal_panel.json"
OUT_FILE = DATA_DIR / "model_synth_control_report.json"

SEED = 20260221
PERMUTATIONS = 2000
MIN_SHOCK_DAYS = 8
MIN_PRE_DAYS = 30
MIN_POST_DAYS = 8
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


def load_country_rows() -> List[dict]:
    if not COUNTRY_FILE.exists():
        return []
    rows = []
    with COUNTRY_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_us_shock_days(policy: str = "strict-balanced") -> List[date]:
    if not PANEL_FILE.exists():
        return []
    with PANEL_FILE.open("r", encoding="utf-8") as f:
        rows = json.load(f).get("rows", [])
    us = [r for r in rows if r.get("policy") == policy and r.get("date")]
    if not us:
        return []
    series = {parse_date(r["date"]): float(r.get("crisis_count", 0)) for r in us}
    nonzero = [v for v in series.values() if v > 0]
    thr = compute_shock_threshold(nonzero)
    return sorted([d for d, v in series.items() if v >= thr])


def inverse_mse_weights(us: Dict[date, float], donors: Dict[str, Dict[date, float]], pre_dates: List[date]) -> Dict[str, float]:
    raw = {}
    for c, s in donors.items():
        errs = []
        for d in pre_dates:
            if d in us and d in s:
                errs.append((us[d] - s[d]) ** 2)
        mse = mean(errs) if errs else 1e6
        raw[c] = 1.0 / (mse + 1e-6)

    z = sum(raw.values())
    if z <= 0:
        n = max(1, len(raw))
        return {k: 1.0 / n for k in raw}
    return {k: v / z for k, v in raw.items()}


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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_country_rows()
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "synthetic_control_simplified",
        "policy": args.policy,
        "status": "blocked",
        "reason": "",
        "n_rows": len(rows),
        "countries": [],
        "weights": {},
        "metrics": {
            "att_gap": None,
            "p_value": None,
            "n_common_dates": 0,
            "n_pre_dates": 0,
            "n_post_dates": 0,
        },
        "gates": {
            "att_positive": False,
            "p_value_significant": False,
            "synth_passed": False,
        },
    }

    if not rows:
        out["reason"] = "country_controls.csv 缺失或为空"
    else:
        series: Dict[str, Dict[date, float]] = {}
        for r in rows:
            try:
                d = parse_date(r.get("date", ""))
                c = r.get("country", "")
                y = float(r.get("ufo_policy_news", "0") or 0)
            except Exception:
                continue
            if not c:
                continue
            series.setdefault(c, {})[d] = y

        countries = sorted(series.keys())
        out["countries"] = countries
        if "US" not in series:
            out["reason"] = "country_controls.csv 缺少 US 行"
        else:
            us = series["US"]
            donors_all = {c: s for c, s in series.items() if c != "US"}
            if len(donors_all) < 3:
                out["reason"] = "对照国家数量不足（需要 >=3）"
            else:
                shocks = load_us_shock_days(args.policy)
                if len(shocks) < args.min_shock_days:
                    out["reason"] = f"US 冲击日不足（{len(shocks)} < {args.min_shock_days}），无法估计"
                else:
                    us_dates = sorted(us.keys())
                    shock_window = set()
                    for d in shocks:
                        for k in range(1, 8):
                            shock_window.add(d + timedelta(days=k))

                    post_dates = [d for d in us_dates if d in shock_window]
                    pre_dates = [d for d in us_dates if d not in shock_window]
                    out["metrics"]["n_pre_dates"] = len(pre_dates)
                    out["metrics"]["n_post_dates"] = len(post_dates)

                    if len(pre_dates) < args.min_pre_days or len(post_dates) < args.min_post_days:
                        out["reason"] = "pre/post 样本不足"
                    else:
                        min_overlap = max(15, len(pre_dates) // 4)
                        donors = {}
                        for c, s in donors_all.items():
                            overlap = sum(1 for d in pre_dates if d in s)
                            if overlap >= min_overlap:
                                donors[c] = s

                        if len(donors) < 3:
                            out["reason"] = "与 US 的 donor overlap 不足（需要 >=3 个国家）"
                        else:
                            weights = inverse_mse_weights(us, donors, pre_dates)
                            out["weights"] = {k: round(v, 6) for k, v in weights.items()}

                            def synth_value(d: date) -> float | None:
                                active = {c: w for c, w in weights.items() if d in donors[c]}
                                if not active:
                                    return None
                                z = sum(active.values())
                                if z <= 0:
                                    return None
                                return sum((w / z) * donors[c][d] for c, w in active.items())

                            gap = {}
                            for d in us_dates:
                                sv = synth_value(d)
                                if sv is None:
                                    continue
                                gap[d] = us[d] - sv

                            pre_gap = [gap[d] for d in pre_dates if d in gap]
                            post_gap = [gap[d] for d in post_dates if d in gap]
                            out["metrics"]["n_common_dates"] = len(gap)

                            if len(pre_gap) < 20 or len(post_gap) < 6:
                                out["reason"] = "有效 gap 样本不足"
                            else:
                                gap_pre = mean(pre_gap)
                                gap_post = mean(post_gap)
                                att = gap_post - gap_pre

                                random.seed(SEED)
                                null = []
                                pre_gap_dates = [d for d in pre_dates if d in gap]
                                if len(pre_gap_dates) < len(post_gap):
                                    out["reason"] = "pre_gap 可抽样日期不足，无法构建安慰剂分布"
                                    pre_gap_dates = []

                                for _ in range(PERMUTATIONS):
                                    if not pre_gap_dates:
                                        break
                                    fake_post_dates = random.sample(pre_gap_dates, len(post_gap))
                                    fake_att = mean(gap[d] for d in fake_post_dates) - gap_pre
                                    null.append(fake_att)

                                if not null:
                                    out["reason"] = "安慰剂分布为空，无法稳健估计"
                                else:
                                    p_val = sum(x >= att for x in null) / float(len(null))

                                    out["metrics"]["att_gap"] = round(att, 6)
                                    out["metrics"]["p_value"] = round(p_val, 6)
                                    out["gates"]["att_positive"] = att > 0
                                    out["gates"]["p_value_significant"] = p_val < 0.05
                                    out["gates"]["synth_passed"] = (
                                        out["gates"]["att_positive"] and out["gates"]["p_value_significant"]
                                    )

                                    out["status"] = "ok"
                                    out["reason"] = "synth_estimated"

    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=== Synthetic Control Simplified ===")
    print(f"status: {out['status']}")
    print(f"reason: {out['reason']}")
    print(f"synth_passed: {out['gates']['synth_passed']}")
    print(f"[输出] {OUT_FILE}")


if __name__ == "__main__":
    main()
