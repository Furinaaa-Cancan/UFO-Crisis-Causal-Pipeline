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
    p.add_argument(
        "--nonoverlap-gap-days",
        type=int,
        default=30,
        help="构造非重叠冲击集时的最小间隔（默认 30 天）",
    )
    p.add_argument(
        "--effect-lag-start",
        type=int,
        default=1,
        help="Causal ML 投影：处理后窗口起点（默认 t+1）",
    )
    p.add_argument(
        "--effect-window-days",
        type=int,
        default=7,
        help="Causal ML 投影：处理后窗口终点（默认 t+7）",
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


def _nearest_kept_distance(date_iso: str, selected: List[str]) -> Tuple[int | None, str | None]:
    if not selected:
        return None, None
    d = parse_date(date_iso)
    nearest = min(selected, key=lambda x: abs((d - parse_date(x)).days))  # type: ignore
    return abs((d - parse_date(nearest)).days), nearest


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


def build_nonoverlap_catalog(
    manual_dates: List[str],
    auto_candidates: List[dict],  # type: ignore
    min_gap_days: int,
) -> Tuple[List[str], List[dict]]:
    """
    按优先级构建“非重叠冲击集”：
    - 手工核验（manual）优先于自动候选
    - 同优先级下按 crisis_count 高分优先
    """
    entries = []
    for d in manual_dates:
        entries.append(
            {
                "date": d,
                "source": "manual",
                "priority": 2,
                "score": 1e18,
            }
        )
    for c in auto_candidates:
        entries.append(
            {
                "date": c.get("date"),
                "source": "auto",
                "priority": 1,
                "score": float(c.get("crisis_count", 0) or 0.0),  # type: ignore
            }
        )

    # 高优先 + 高分优先，最后按日期稳定排序
    entries = [e for e in entries if isinstance(e.get("date"), str)]
    entries.sort(key=lambda x: (int(x["priority"]), float(x["score"]), str(x["date"])), reverse=True)  # type: ignore

    kept: List[str] = []
    dropped: List[dict] = []
    for e in entries:
        d_iso = str(e["date"])
        if _too_close(d_iso, kept, min_gap_days):
            near_days, near_date = _nearest_kept_distance(d_iso, kept)
            dropped.append(
                {
                    "date": d_iso,
                    "source": e.get("source"),
                    "score": e.get("score"),
                    "reason": "too_close_to_selected",
                    "nearest_selected_date": near_date,
                    "nearest_selected_days": near_days,
                    "min_gap_days": int(min_gap_days),
                }
            )
            continue
        kept.append(d_iso)

    kept = sorted(set(kept))
    dropped.sort(key=lambda x: x.get("date", ""))  # type: ignore
    return kept, dropped


def project_treated_support(
    rows: List[dict],  # type: ignore
    shock_dates: List[str],
    effect_lag_start: int,
    effect_window_days: int,
) -> dict:
    """
    近似对齐 model_causal_ml.build_dataset 的可用样本口径，
    计算给定冲击集在 ML 设计中的 treated 支持度。
    """
    lag_start = max(0, int(effect_lag_start))
    lag_end = max(lag_start, int(effect_window_days))
    by_date: Dict[object, dict] = {}  # type: ignore
    for r in rows:
        d = r.get("date")
        if not isinstance(d, str):
            continue
        try:
            by_date[parse_date(d)] = r  # type: ignore
        except Exception:
            continue

    dates = sorted(by_date.keys())  # type: ignore
    shock_set = set()
    for s in shock_dates:
        try:
            shock_set.add(parse_date(s))  # type: ignore
        except Exception:
            continue

    n_samples = 0
    treated_samples = 0
    for d in dates:
        lag_ok = all((d - timedelta(days=k)) in by_date for k in (1, 2, 3, 7))  # type: ignore
        if not lag_ok:
            continue
        future_days = [d + timedelta(days=k) for k in range(lag_start, lag_end + 1)]  # type: ignore
        if any(fd not in by_date for fd in future_days):
            continue
        n_samples += 1
        if d in shock_set:
            treated_samples += 1

    treated_ratio = (treated_samples / float(n_samples)) if n_samples > 0 else 0.0
    return {
        "n_samples_for_ml_projection": int(n_samples),
        "treated_samples_projection": int(treated_samples),
        "treated_ratio_projection": round(treated_ratio, 6),  # type: ignore
        "effect_lag_start": int(lag_start),
        "effect_window_days": int(lag_end),
    }


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
    nonoverlap_key = f"shock_dates_nonoverlap_{int(args.nonoverlap_gap_days)}d"
    nonoverlap_dates, nonoverlap_drops = build_nonoverlap_catalog(
        manual_dates=manual_dates,
        auto_candidates=auto_candidates,
        min_gap_days=int(args.nonoverlap_gap_days),
    )
    support_projection_main = project_treated_support(
        rows=rows,
        shock_dates=all_dates,
        effect_lag_start=int(args.effect_lag_start),
        effect_window_days=int(args.effect_window_days),
    )
    support_projection_nonoverlap = project_treated_support(
        rows=rows,
        shock_dates=nonoverlap_dates,
        effect_lag_start=int(args.effect_lag_start),
        effect_window_days=int(args.effect_window_days),
    )

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
            "nonoverlap_gap_days": int(args.nonoverlap_gap_days),
            "effect_lag_start": int(args.effect_lag_start),
            "effect_window_days": int(args.effect_window_days),
        },
        "summary": {
            "manual_dates": len(manual_dates),
            "auto_candidates": len(auto_candidates),
            "total_shock_dates": len(all_dates),
            "total_shock_dates_nonoverlap": len(nonoverlap_dates),
            "nonoverlap_dropped": len(nonoverlap_drops),
            "auto_candidate_threshold_value": thr,
        },
        "manual_shock_dates": manual_dates,
        "auto_candidates": auto_candidates,
        "nonoverlap_drops": nonoverlap_drops,
        "shock_dates_nonoverlap": nonoverlap_dates,
        nonoverlap_key: nonoverlap_dates,
        "support_projection": {
            "shock_dates": support_projection_main,
            "shock_dates_nonoverlap": support_projection_nonoverlap,
            nonoverlap_key: support_projection_nonoverlap,
        },
        "shock_dates": all_dates,
    }

    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)  # type: ignore

    print("=== Shock Catalog Builder ===")
    print(f"manual_dates: {len(manual_dates)}")
    print(f"auto_candidates: {len(auto_candidates)}")
    print(f"total_shock_dates: {len(all_dates)}")
    print(f"{nonoverlap_key}: {len(nonoverlap_dates)}")
    print(
        "treated_projection(main/nonoverlap): "
        f"{support_projection_main['treated_samples_projection']}/"
        f"{support_projection_nonoverlap['treated_samples_projection']}"
    )
    print(f"[输出] {OUT_FILE}")


if __name__ == "__main__":
    main()
