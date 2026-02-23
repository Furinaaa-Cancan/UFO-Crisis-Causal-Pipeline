"""
危机冲击日目录构建器

用途：
1) 保留 events_v2 的人工核验危机日（baseline）
2) 从 causal_panel 中自动提名高强度危机局部峰值（candidate）
3) 可选合并外生冲击清单（exogenous catalog）
4) 输出可审计的冲击日目录（data/crisis_shock_catalog.json）

注意：
- 该目录用于“扩展功效分析”，不替代 events_v2 的人工核验样本。
- 主结论仍建议以 events_v2 主轨为准。
"""
# pyre-ignore-all-errors
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from utils import parse_date, percentile, read_panel_rows


BASE_DIR = Path(__file__).resolve().parent  # type: ignore
DATA_DIR = BASE_DIR / "data"
EVENTS_FILE = DATA_DIR / "events_v2.json"
PANEL_FILE = DATA_DIR / "causal_panel.json"
OUT_FILE = DATA_DIR / "crisis_shock_catalog.json"
EXOGENOUS_FILE = DATA_DIR / "exogenous_shocks_us.json"
LOCK_FILE = DATA_DIR / "crisis_shock_catalog_lock.json"
MIN_TREATED_SAMPLES_TARGET = 12
MIN_TREATED_RATIO_TARGET = 0.002


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
        "--nonoverlap-gaps",
        default="30,45,60",
        help="额外输出的非重叠间隔矩阵（逗号分隔，例如 30,45,60）",
    )
    p.add_argument(
        "--exogenous-file",
        default=str(EXOGENOUS_FILE),
        help="外生冲击清单 JSON 文件路径（默认 data/exogenous_shocks_us.json）",
    )
    p.add_argument(
        "--disable-exogenous",
        action="store_true",
        help="禁用外生冲击清单，仅使用 manual + auto",
    )
    p.add_argument(
        "--exogenous-categories",
        default="",
        help="仅保留指定 category（逗号分隔）；为空表示不过滤",
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
    p.add_argument(
        "--lock-file",
        default=str(LOCK_FILE),
        help="冲击目录锁文件路径（默认 data/crisis_shock_catalog_lock.json）",
    )
    p.add_argument(
        "--write-lock",
        action="store_true",
        help="写入锁文件（用于预注册冻结口径）",
    )
    p.add_argument(
        "--enforce-lock",
        action="store_true",
        help="校验当前目录签名必须与锁文件一致，不一致则退出",
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


def parse_gap_list(raw: str, fallback: int) -> List[int]:
    out = []
    for part in str(raw or "").split(","):
        s = part.strip()
        if not s:
            continue
        try:
            v = int(s)
        except Exception:
            continue
        if v > 0:
            out.append(v)
    if fallback > 0:
        out.append(int(fallback))
    return sorted(set(out))


def parse_category_filter(raw: str) -> set[str]:
    out = set()
    for part in str(raw or "").split(","):
        s = part.strip().lower()
        if s:
            out.add(s)
    return out


def load_exogenous_candidates(
    exogenous_path: Path,
    category_filter: set[str] | None = None,
) -> List[dict]:
    if not exogenous_path.exists():  # type: ignore
        return []
    try:
        with exogenous_path.open("r", encoding="utf-8") as f:  # type: ignore
            payload = json.load(f)  # type: ignore
    except Exception:
        return []

    rows = []
    if isinstance(payload, dict):
        maybe_rows = payload.get("events", [])  # type: ignore
        if isinstance(maybe_rows, list):
            rows = maybe_rows
    elif isinstance(payload, list):
        rows = payload

    out = []
    filters = category_filter or set()
    for row in rows:  # type: ignore
        if not isinstance(row, dict):
            continue
        d = row.get("date")
        if not isinstance(d, str):
            continue
        try:
            parse_date(d)
        except Exception:
            continue
        cat = str(row.get("category", "unspecified") or "unspecified").strip().lower()
        if filters and cat not in filters:
            continue
        out.append(
            {
                "date": d,
                "category": cat,
                "label": str(row.get("label", "") or ""),
                "source": str(row.get("source", "exogenous_catalog") or "exogenous_catalog"),
                "priority_score": float(row.get("priority_score", 1000.0) or 1000.0),
                "candidate_type": "exogenous",
            }
        )
    out.sort(key=lambda x: str(x.get("date", "")))  # type: ignore
    return out


def file_sha256(path: Path) -> str:
    if not path.exists():  # type: ignore
        return ""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:  # type: ignore
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def stable_payload_sha256(payload: object) -> str:
    normalized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_catalog_signature_payload(payload: dict) -> dict:
    nonoverlap_keys = sorted(
        [k for k in payload.keys() if isinstance(k, str) and k.startswith("shock_dates_nonoverlap_")]  # type: ignore
    )
    return {
        "policy": payload.get("policy", ""),
        "source_files": payload.get("source_files", {}),
        "source_hashes": payload.get("source_hashes", {}),
        "config": payload.get("config", {}),
        "manual_shock_dates": payload.get("manual_shock_dates", []),
        "exogenous_candidates": [
            {
                "date": r.get("date"),
                "category": r.get("category"),
                "source": r.get("source"),
            }
            for r in payload.get("exogenous_candidates", [])  # type: ignore
            if isinstance(r, dict)
        ],
        "auto_candidate_dates": [
            {
                "date": r.get("date"),
                "crisis_count": r.get("crisis_count"),
            }
            for r in payload.get("auto_candidates", [])  # type: ignore
            if isinstance(r, dict)
        ],
        "shock_dates": payload.get("shock_dates", []),
        "nonoverlap_keys": nonoverlap_keys,
        "nonoverlap_dates": {k: payload.get(k, []) for k in nonoverlap_keys},
    }


def compute_catalog_signature(payload: dict) -> str:
    return stable_payload_sha256(build_catalog_signature_payload(payload))


def load_lock_signature(lock_path: Path) -> str:
    if not lock_path.exists():  # type: ignore
        return ""
    try:
        with lock_path.open("r", encoding="utf-8") as f:  # type: ignore
            payload = json.load(f)  # type: ignore
        return str(payload.get("catalog_signature_sha256", "") or "")
    except Exception:
        return ""


def build_lock_payload(
    catalog_signature: str,
    policy: str,
    lock_path: Path,
    out_path: Path,
    source_hashes: dict,
    config: dict,
) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "schema": "shock_catalog_lock_v1",
        "policy": policy,
        "catalog_file": str(out_path),
        "catalog_signature_sha256": catalog_signature,
        "lock_file": str(lock_path),
        "source_hashes": source_hashes,
        "config": config,
    }


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
    exogenous_candidates: List[dict] | None = None,  # type: ignore
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
                "priority": 3,
                "score": 1e18,
            }
        )
    for exo in (exogenous_candidates or []):
        entries.append(
            {
                "date": exo.get("date"),
                "source": "exogenous",
                "source_label": exo.get("label", ""),
                "source_category": exo.get("category", ""),
                "priority": 2,
                "score": float(exo.get("priority_score", 1000.0) or 1000.0),  # type: ignore
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
                    "source_label": e.get("source_label", ""),
                    "source_category": e.get("source_category", ""),
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


def build_nonoverlap_matrix(
    manual_dates: List[str],
    auto_candidates: List[dict],  # type: ignore
    exogenous_candidates: List[dict],  # type: ignore
    gaps: Iterable[int],
) -> dict:
    normalized_gaps: List[int] = []
    for g in gaps:
        try:
            v = int(g)
        except Exception:
            continue
        if v > 0:
            normalized_gaps.append(v)
    out = {}
    for gap in sorted(set(normalized_gaps)):
        key = f"shock_dates_nonoverlap_{int(gap)}d"
        kept, dropped = build_nonoverlap_catalog(
            manual_dates=manual_dates,
            auto_candidates=auto_candidates,
            exogenous_candidates=exogenous_candidates,
            min_gap_days=int(gap),
        )
        out[key] = {
            "gap_days": int(gap),
            "shock_dates": kept,
            "dropped": dropped,
        }
    return out


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
    required_by_ratio = int((float(MIN_TREATED_RATIO_TARGET) * max(1, n_samples)) + 0.999999)
    required_treated = max(int(MIN_TREATED_SAMPLES_TARGET), required_by_ratio)
    treated_shortfall = max(0, required_treated - int(treated_samples))
    return {
        "n_samples_for_ml_projection": int(n_samples),
        "treated_samples_projection": int(treated_samples),
        "treated_ratio_projection": round(treated_ratio, 6),  # type: ignore
        "treated_samples_required_projection": int(required_treated),
        "treated_shortfall_projection": int(treated_shortfall),
        "treated_support_ok_projection": treated_shortfall == 0,
        "effect_lag_start": int(lag_start),
        "effect_window_days": int(lag_end),
    }


def main() -> None:
    args = parse_args()
    lock_path = Path(str(args.lock_file or LOCK_FILE))  # type: ignore
    manual_dates = load_manual_crisis_dates(EVENTS_FILE)
    rows = read_panel_rows(PANEL_FILE, args.policy)
    exogenous_filter = parse_category_filter(args.exogenous_categories)
    exogenous_candidates: List[dict] = []  # type: ignore
    if not args.disable_exogenous:
        exogenous_candidates = load_exogenous_candidates(
            Path(str(args.exogenous_file)),  # type: ignore
            category_filter=exogenous_filter,
        )
    auto_candidates, thr = build_auto_candidates(
        rows,
        peak_percentile=args.peak_percentile,
        min_gap_days=args.min_gap_days,
        max_candidates=args.max_candidates,
        manual_dates=manual_dates,
        exclude_near_manual_days=args.exclude_near_manual_days,
    )

    all_dates = sorted(
        set(
            manual_dates
            + [str(c.get("date")) for c in auto_candidates if isinstance(c.get("date"), str)]  # type: ignore
            + [str(c.get("date")) for c in exogenous_candidates if isinstance(c.get("date"), str)]  # type: ignore
        )
    )
    gaps = parse_gap_list(str(args.nonoverlap_gaps or ""), int(args.nonoverlap_gap_days))
    matrix = build_nonoverlap_matrix(
        manual_dates=manual_dates,
        auto_candidates=auto_candidates,
        exogenous_candidates=exogenous_candidates,
        gaps=gaps,
    )
    nonoverlap_key = f"shock_dates_nonoverlap_{int(args.nonoverlap_gap_days)}d"
    primary_profile = matrix.get(nonoverlap_key)
    if primary_profile is None and matrix:
        nonoverlap_key = sorted(matrix.keys())[0]
        primary_profile = matrix[nonoverlap_key]
    primary_dates = list(primary_profile.get("shock_dates", [])) if isinstance(primary_profile, dict) else []  # type: ignore
    primary_drops = list(primary_profile.get("dropped", [])) if isinstance(primary_profile, dict) else []  # type: ignore

    support_projection_main = project_treated_support(
        rows=rows,
        shock_dates=all_dates,
        effect_lag_start=int(args.effect_lag_start),
        effect_window_days=int(args.effect_window_days),
    )

    support_projection = {"shock_dates": support_projection_main}
    nonoverlap_profiles = []
    for key in sorted(matrix.keys()):
        profile = matrix[key]
        shock_dates = list(profile.get("shock_dates", []))  # type: ignore
        drops = list(profile.get("dropped", []))  # type: ignore
        proj = project_treated_support(
            rows=rows,
            shock_dates=shock_dates,
            effect_lag_start=int(args.effect_lag_start),
            effect_window_days=int(args.effect_window_days),
        )
        support_projection[key] = proj
        nonoverlap_profiles.append(
            {
                "key": key,
                "gap_days": int(profile.get("gap_days", 0)),  # type: ignore
                "n_dates": len(shock_dates),
                "dropped_dates": len(drops),
                "treated_samples_projection": int(proj["treated_samples_projection"]),
                "treated_ratio_projection": float(proj["treated_ratio_projection"]),
                "treated_shortfall_projection": int(proj["treated_shortfall_projection"]),
                "treated_support_ok_projection": bool(proj["treated_support_ok_projection"]),
            }
        )

    if nonoverlap_key in support_projection:
        support_projection_nonoverlap = support_projection[nonoverlap_key]
    else:
        support_projection_nonoverlap = project_treated_support(
            rows=rows,
            shock_dates=primary_dates,
            effect_lag_start=int(args.effect_lag_start),
            effect_window_days=int(args.effect_window_days),
        )

    exogenous_file_path = Path(str(args.exogenous_file or ""))
    source_hashes = {
        "events_v2_sha256": file_sha256(EVENTS_FILE),
        "causal_panel_sha256": file_sha256(PANEL_FILE),
        "exogenous_catalog_sha256": file_sha256(exogenous_file_path) if (not args.disable_exogenous) else "",
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "policy": args.policy,
        "source_files": {
            "events_v2": str(EVENTS_FILE),
            "causal_panel": str(PANEL_FILE),
            "exogenous_catalog": str(args.exogenous_file or ""),
        },
        "source_hashes": source_hashes,
        "config": {
            "peak_percentile": args.peak_percentile,
            "min_gap_days": args.min_gap_days,
            "max_candidates": args.max_candidates,
            "exclude_near_manual_days": args.exclude_near_manual_days,
            "nonoverlap_gap_days": int(args.nonoverlap_gap_days),
            "nonoverlap_gaps": gaps,
            "exogenous_enabled": not bool(args.disable_exogenous),
            "exogenous_category_filter": sorted(exogenous_filter),
            "effect_lag_start": int(args.effect_lag_start),
            "effect_window_days": int(args.effect_window_days),
        },
        "summary": {
            "manual_dates": len(manual_dates),
            "exogenous_candidates": len(exogenous_candidates),
            "auto_candidates": len(auto_candidates),
            "total_shock_dates": len(all_dates),
            "total_shock_dates_nonoverlap": len(primary_dates),
            "nonoverlap_dropped": len(primary_drops),
            "auto_candidate_threshold_value": thr,
            "nonoverlap_profiles": nonoverlap_profiles,
        },
        "manual_shock_dates": manual_dates,
        "exogenous_candidates": exogenous_candidates,
        "auto_candidates": auto_candidates,
        "nonoverlap_drops": primary_drops,
        "shock_dates_nonoverlap": primary_dates,
        nonoverlap_key: primary_dates,
        "nonoverlap_matrix": matrix,
        "support_projection": support_projection,
        "shock_dates": all_dates,
    }
    payload["support_projection"]["shock_dates_nonoverlap"] = support_projection_nonoverlap  # type: ignore
    payload["support_projection"][nonoverlap_key] = support_projection_nonoverlap  # type: ignore
    for key, profile in matrix.items():
        payload[key] = list(profile.get("shock_dates", []))  # type: ignore

    catalog_signature = compute_catalog_signature(payload)
    payload["catalog_signature_sha256"] = catalog_signature  # type: ignore

    if args.enforce_lock:
        locked_sig = load_lock_signature(lock_path)
        if not locked_sig:
            raise SystemExit(f"lock_missing_or_invalid: {lock_path}")
        if locked_sig != catalog_signature:
            raise SystemExit(
                "catalog_signature_mismatch: "
                f"expected={locked_sig}, got={catalog_signature}, lock_file={lock_path}"
            )

    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)  # type: ignore

    if args.write_lock:
        lock_payload = build_lock_payload(
            catalog_signature=catalog_signature,
            policy=str(args.policy),
            lock_path=lock_path,
            out_path=OUT_FILE,
            source_hashes=source_hashes,
            config=payload.get("config", {}) if isinstance(payload.get("config"), dict) else {},
        )
        with lock_path.open("w", encoding="utf-8") as f:
            json.dump(lock_payload, f, ensure_ascii=False, indent=2)  # type: ignore

    print("=== Shock Catalog Builder ===")
    print(f"manual_dates: {len(manual_dates)}")
    print(f"exogenous_candidates: {len(exogenous_candidates)}")
    print(f"auto_candidates: {len(auto_candidates)}")
    print(f"total_shock_dates: {len(all_dates)}")
    print(f"{nonoverlap_key}: {len(primary_dates)}")
    print(
        "treated_projection(main/nonoverlap): "
        f"{support_projection_main['treated_samples_projection']}/"
        f"{support_projection_nonoverlap['treated_samples_projection']}"
    )
    for p in nonoverlap_profiles:
        print(
            f"  - {p['key']}: n_dates={p['n_dates']}, "
            f"treated={p['treated_samples_projection']}, "
            f"ratio={p['treated_ratio_projection']:.6f}, "
            f"shortfall={p['treated_shortfall_projection']}"
        )
    print(f"catalog_signature_sha256: {catalog_signature}")
    if args.write_lock:
        print(f"[锁定] {lock_path}")
    print(f"[输出] {OUT_FILE}")


if __name__ == "__main__":
    main()
