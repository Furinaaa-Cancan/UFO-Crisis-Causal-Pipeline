"""
危机冲击日目录构建器

用途：
1) 保留 events_v2 的人工核验危机日（baseline）
2) 从 causal_panel 中自动提名高强度危机局部峰值（candidate）
3) 输出可审计的冲击日目录（data/crisis_shock_catalog.json）

注意：
- 该目录用于“扩展功效分析”，不替代 events_v2 的人工核验样本。
- 主结论仍建议以 events_v2 主轨为准。
"""
# pyre-ignore-all-errors
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from utils import parse_date, percentile, read_panel_rows


BASE_DIR = Path(__file__).resolve().parent  # type: ignore
DATA_DIR = BASE_DIR / "data"
EVENTS_FILE = DATA_DIR / "events_v2.json"
PANEL_FILE = DATA_DIR / "causal_panel.json"
OUT_FILE = DATA_DIR / "crisis_shock_catalog.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建 crisis shock catalog")
    p.add_argument("--policy", choices=["strict", "strict-balanced"], default="strict-balanced")
    p.add_argument("--peak-percentile", type=float, default=99.0, help="候选冲击峰值阈值百分位")
    p.add_argument("--min-gap-days", type=int, default=21, help="候选冲击日最小间隔")
    p.add_argument("--max-candidates", type=int, default=80, help="最多保留自动候选数量")
    p.add_argument(
        "--exclude-near-manual-days",
        type=int,
        default=14,
        help="与人工危机日距离 <= N 天的自动候选将被剔除",
    )
    return p.parse_args()


def load_manual_crisis_dates(events_path: Path) -> List[str]:
    if not events_path.exists():  # type: ignore
        return []
    try:
        with events_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)  # type: ignore
    except Exception:
        return []

    out = []
    for row in payload.get("correlations", []):  # type: ignore
        d = row.get("crisis", {}).get("date")  # type: ignore
        if isinstance(d, str):
            out.append(d)
    return sorted(set(out))


def _is_local_peak(series: Dict[str, float], day_iso: str, radius: int = 3) -> bool:
    center = float(series.get(day_iso, 0.0))
    d = parse_date(day_iso)
    for k in range(1, radius + 1):
        left = (d - timedelta(days=k)).isoformat()  # type: ignore
        right = (d + timedelta(days=k)).isoformat()  # type: ignore
        if float(series.get(left, 0.0)) > center:
            return False
        if float(series.get(right, 0.0)) > center:
            return False
    return True


def _too_close(date_iso: str, selected: List[str], min_gap_days: int) -> bool:
    d = parse_date(date_iso)
    for s in selected:
        if abs((d - parse_date(s)).days) <= min_gap_days:
            return True
    return False


def _near_manual(date_iso: str, manual_dates: List[str], radius_days: int) -> bool:
    d = parse_date(date_iso)
    for m in manual_dates:
        if abs((d - parse_date(m)).days) <= radius_days:
            return True
    return False


def build_auto_candidates(
    rows: List[dict],  # type: ignore
    peak_percentile: float,
    min_gap_days: int,
    max_candidates: int,
    manual_dates: List[str],
    exclude_near_manual_days: int,
) -> Tuple[List[dict], float]:
    series: Dict[str, float] = {}  # type: ignore
    for r in rows:
        d = r.get("date")
        if not isinstance(d, str):
            continue
        series[d] = float(r.get("crisis_count", 0) or 0.0)  # type: ignore

    nonzero = [v for v in series.values() if v > 0]
    if not nonzero:
        return [], 0.0
    thr = percentile(nonzero, peak_percentile)

    sorted_days = sorted(series.keys(), key=lambda x: (series[x], x), reverse=True)  # type: ignore
    selected: List[str] = []
    out: List[dict] = []  # type: ignore
    for d in sorted_days:
        v = float(series.get(d, 0.0))
        if v < thr:
            continue
        if not _is_local_peak(series, d, radius=3):
            continue
        if _too_close(d, selected, min_gap_days):
            continue
        if _near_manual(d, manual_dates, exclude_near_manual_days):
            continue
        selected.append(d)
        out.append(
            {
                "date": d,
                "crisis_count": v,
                "candidate_type": "panel_local_peak",
                "peak_percentile_threshold": peak_percentile,
            }
        )
        if len(out) >= max(1, int(max_candidates)):
            break

    out.sort(key=lambda x: x["date"])  # type: ignore
    return out, float(thr)


def main() -> None:
    args = parse_args()
    manual_dates = load_manual_crisis_dates(EVENTS_FILE)
    rows = read_panel_rows(PANEL_FILE, args.policy)
    auto_candidates, thr = build_auto_candidates(
        rows,
        peak_percentile=args.peak_percentile,
        min_gap_days=args.min_gap_days,
        max_candidates=args.max_candidates,
        manual_dates=manual_dates,
        exclude_near_manual_days=args.exclude_near_manual_days,
    )

    all_dates = sorted(set(manual_dates + [c["date"] for c in auto_candidates]))  # type: ignore
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "policy": args.policy,
        "source_files": {
            "events_v2": str(EVENTS_FILE),
            "causal_panel": str(PANEL_FILE),
        },
        "config": {
            "peak_percentile": args.peak_percentile,
            "min_gap_days": args.min_gap_days,
            "max_candidates": args.max_candidates,
            "exclude_near_manual_days": args.exclude_near_manual_days,
        },
        "summary": {
            "manual_dates": len(manual_dates),
            "auto_candidates": len(auto_candidates),
            "total_shock_dates": len(all_dates),
            "auto_candidate_threshold_value": thr,
        },
        "manual_shock_dates": manual_dates,
        "auto_candidates": auto_candidates,
        "shock_dates": all_dates,
    }

    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)  # type: ignore

    print("=== Shock Catalog Builder ===")
    print(f"manual_dates: {len(manual_dates)}")
    print(f"auto_candidates: {len(auto_candidates)}")
    print(f"total_shock_dates: {len(all_dates)}")
    print(f"[输出] {OUT_FILE}")


if __name__ == "__main__":
    main()
