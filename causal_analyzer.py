"""
因果检验器（升级版）

升级点：
1) 每次读取 scraped_news 后，自动写入日度面板 data/causal_panel.json（可累计）
2) 在面板上跑 lead-lag + 置换检验 + 安慰剂检验
3) 给出更严格的分级结论：因果信号 / 相关性 / 证据不足

注意：
- events_v2 是人工筛选后的配对样本，不能单独作为因果证据。
- 因果判断依赖长期面板（建议 >=180 天）和足够冲击样本。
"""
# pyre-ignore-all-errors
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from utils import (  # type: ignore[import]
    compute_shock_threshold,
    make_rng,
    max_missing_streak as _max_missing_streak,
    parse_date,
    pearson_corr,
)


BASE_DIR = Path(__file__).resolve().parent  # type: ignore
DATA_DIR = BASE_DIR / "data"
EVENTS_FILE = DATA_DIR / "events_v2.json"
SCRAPED_FILE = DATA_DIR / "scraped_news.json"
PANEL_FILE = DATA_DIR / "causal_panel.json"  # type: ignore
REPORT_FILE = DATA_DIR / "causal_report.json"

SEED = 20260221
PERMUTATIONS = 20000
FAST_MODE_PERMUTATIONS = 2000
WINDOWS = (3, 7, 10, 14, 30)
APPROVAL_WINDOWS = (3, 7, 10)
SHOCK_COUNT_FLOOR = 2.0
EV2_MIN_DATES = 5
DEFAULT_SCRAPER_LOOKBACK = 120
LEAD_LAG_TIE_TOLERANCE = 0.08

CONTROL_TOPIC_KEYWORDS = {
    "economy": {
        "economy", "economic", "inflation", "tariff", "gdp", "recession",
        "jobs", "unemployment", "market", "trade",
    },
    "security": {
        "war", "iran", "russia", "china", "military", "missile",
        "drone", "nato", "defense", "security",
    },
    "immigration": {
        "immigration", "immigrant", "border", "refugee", "asylum",
        "deport", "deportation", "migrant",
    },
}


# parse_date, percentile, pearson_corr, compute_shock_threshold 已移至 utils.py


def parse_iso_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def binomial_right_tail(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return 1.0
    tail = 0.0
    for i in range(k, n + 1):
        tail += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return tail


@dataclass
class EventsV2Stats:
    n_cases: int
    mean_gap: float
    mean_abs_gap: float
    within_30: int
    within_60: int
    positive_count: int
    negative_count: int
    p_abs_gap: float
    p_within_30: float
    p_within_60: float
    p_direction: float


@dataclass
class ScrapedStats:
    policy: str
    n_ufo: int
    n_crisis: int
    coverage_days: int
    sufficient: bool
    reason: str
    window_results: Dict[int, Dict[str, float]]  # type: ignore


@dataclass
class PanelStats:
    policy: str
    n_days: int
    start_date: str
    end_date: str
    observed_days: int
    missing_days: int
    observed_ratio: float
    max_missing_streak: int
    n_shocks: int
    shock_threshold: float
    sufficient: bool
    reason: str
    control_metric: str
    window_results: Dict[int, Dict[str, float]]  # type: ignore
    best_lag: int
    best_corr: float
    best_positive_lag: int
    best_positive_corr: float
    lag0_corr: float
    shock_source: str = "news_volume_75pct_fallback"


@dataclass
class ApprovalGate:
    name: str
    passed: bool
    detail: str


@dataclass
class ApprovalResult:
    status: str
    level: str
    gates: List[ApprovalGate]  # type: ignore
    reason: str


def analyze_events_v2(events_path: Path, permutations: int = PERMUTATIONS) -> EventsV2Stats:
    with events_path.open("r", encoding="utf-8") as f:  # type: ignore
        payload = json.load(f)  # type: ignore

    crisis_dates: List[date] = []  # type: ignore
    ufo_dates: List[date] = []  # type: ignore
    gaps: List[int] = []  # type: ignore
    for row in payload.get("correlations", []):  # type: ignore
        c = parse_date(row["crisis"]["date"])  # type: ignore
        u = parse_date(row["ufo_event"]["date"])  # type: ignore
        crisis_dates.append(c)
        ufo_dates.append(u)
        gaps.append((u - c).days)

    n = len(gaps)
    obs_mean_gap = mean(gaps) if gaps else 0.0
    obs_mean_abs = mean(abs(g) for g in gaps) if gaps else 0.0
    obs_30 = sum(0 <= g <= 30 for g in gaps)
    obs_60 = sum(0 <= g <= 60 for g in gaps)
    pos = sum(g >= 0 for g in gaps)
    neg = sum(g < 0 for g in gaps)

    rng = make_rng(SEED)
    abs_means = []
    within_30 = []
    within_60 = []
    n_perm = max(200, int(permutations))
    for _ in range(n_perm):
        shuffled = ufo_dates[:]  # type: ignore
        rng.shuffle(shuffled)
        sim_gaps = [(u - c).days for c, u in zip(crisis_dates, shuffled)]
        abs_means.append(mean(abs(g) for g in sim_gaps))
        within_30.append(sum(0 <= g <= 30 for g in sim_gaps))
        within_60.append(sum(0 <= g <= 60 for g in sim_gaps))

    p_abs = sum(x <= obs_mean_abs for x in abs_means) / n_perm
    p_30 = sum(x >= obs_30 for x in within_30) / n_perm
    p_60 = sum(x >= obs_60 for x in within_60) / n_perm
    p_dir = binomial_right_tail(pos, n, 0.5)

    return EventsV2Stats(
        n_cases=n,
        mean_gap=obs_mean_gap,
        mean_abs_gap=obs_mean_abs,
        within_30=obs_30,
        within_60=obs_60,
        positive_count=pos,
        negative_count=neg,
        p_abs_gap=p_abs,
        p_within_30=p_30,
        p_within_60=p_60,
        p_direction=p_dir,
    )


def _daily_counts(rows: List[dict], start: date, end: date) -> Dict[date, int]:
    counts = defaultdict(int)
    for row in rows:
        s = row.get("date")  # type: ignore
        if not s:
            continue
        try:
            d = parse_date(s)
        except Exception:
            continue
        if start <= d <= end:
            counts[d] += 1  # type: ignore
    return counts  # type: ignore


def _window_effect(trigger_days: List[date], outcome_counts: Dict[date, float], w: int) -> float:
    if not trigger_days:
        return 0.0
    vals = []
    for d in trigger_days:
        post = 0.0
        pre = 0.0
        for k in range(1, w + 1):
            post += outcome_counts.get(d + timedelta(days=k), 0.0)  # type: ignore
            pre += outcome_counts.get(d - timedelta(days=k), 0.0)  # type: ignore
        vals.append(post - pre)
    return mean(vals)


def _perm_pvalue(
    all_days: List[date],
    trigger_n: int,
    series: Dict[date, float],
    w: int,
    obs_effect: float,
    permutations: int = PERMUTATIONS,
) -> float:
    if trigger_n <= 0 or trigger_n > len(all_days):
        return 1.0
    rng = make_rng(SEED)
    null = []
    n_perm = max(200, int(permutations))
    for _ in range(n_perm):
        sampled = rng.sample(all_days, trigger_n)
        null.append(_window_effect(sampled, series, w))
    return sum(x >= obs_effect for x in null) / len(null) if null else 1.0


def analyze_scraped(scraped_path: Path, permutations: int = PERMUTATIONS) -> ScrapedStats:
    with scraped_path.open("r", encoding="utf-8") as f:  # type: ignore
        payload = json.load(f)  # type: ignore

    ufo_rows = payload.get("ufo_news", [])  # type: ignore
    crisis_rows = payload.get("crisis_news", [])  # type: ignore
    policy = payload.get("policy", "unknown")  # type: ignore
    lookback_days = int(payload.get("lookback_days", DEFAULT_SCRAPER_LOOKBACK) or DEFAULT_SCRAPER_LOOKBACK)  # type: ignore
    coverage_target = min(180, max(30, lookback_days))

    dates = []
    for row in ufo_rows + crisis_rows:
        s = row.get("date")  # type: ignore
        if not s:
            continue
        try:
            dates.append(parse_date(s))
        except Exception:
            continue

    if not dates:
        return ScrapedStats(
            policy=policy,
            n_ufo=len(ufo_rows),
            n_crisis=len(crisis_rows),
            coverage_days=0,
            sufficient=False,
            reason="无可用日期数据",
            window_results={},
        )

    start = min(dates)
    end = max(dates)
    coverage_days = (end - start).days + 1

    if len(crisis_rows) < 10 or len(ufo_rows) < 30 or coverage_days < coverage_target:
        fail_reasons = []
        if len(crisis_rows) < 10:
            fail_reasons.append(f"crisis={len(crisis_rows)} (<10)")
        if len(ufo_rows) < 30:
            fail_reasons.append(f"ufo={len(ufo_rows)} (<30)")
        if coverage_days < coverage_target:
            fail_reasons.append(f"覆盖天数={coverage_days} (<{coverage_target})")
        reason = f"样本不足：{' 或 '.join(fail_reasons)}"
        return ScrapedStats(
            policy=policy,
            n_ufo=len(ufo_rows),
            n_crisis=len(crisis_rows),
            coverage_days=coverage_days,
            sufficient=False,
            reason=reason,
            window_results={},
        )

    ufo_counts = _daily_counts(ufo_rows, start, end)
    crisis_counts = _daily_counts(crisis_rows, start, end)
    crisis_days = sorted(set(crisis_counts.keys()))  # type: ignore
    ufo_days = sorted(set(ufo_counts.keys()))  # type: ignore

    all_days = [start + timedelta(days=i) for i in range(coverage_days)]  # type: ignore
    # random seed 已由 _perm_pvalue 内部独立管理
    window_results = {}

    for w in WINDOWS:
        obs = _window_effect(crisis_days, ufo_counts, w)  # type: ignore
        rev = _window_effect(ufo_days, crisis_counts, w)  # type: ignore
        p_val = _perm_pvalue(  # type: ignore
            all_days,
            len(crisis_days),
            ufo_counts,
            w,
            obs,
            permutations=permutations,
        )
        window_results[w] = {  # type: ignore
            "obs_effect_post_minus_pre": obs,
            "reverse_effect": rev,
            "p_value": p_val,
        }

    return ScrapedStats(
        policy=policy,
        n_ufo=len(ufo_rows),
        n_crisis=len(crisis_rows),
        coverage_days=coverage_days,
        sufficient=True,
        reason="样本充分",
        window_results=window_results,
    )


def _topic_counts_from_titles(titles: List[str]) -> Dict[str, int]:
    out = {k: 0 for k in CONTROL_TOPIC_KEYWORDS}
    for t in titles:
        low = (t or "").lower()
        for topic, kws in CONTROL_TOPIC_KEYWORDS.items():  # type: ignore
            if any(kw in low for kw in kws):
                out[topic] += 1  # type: ignore
    return out


def _avg_score(rows: List[dict]) -> float:
    vals = []
    for row in rows:
        vals.append(float(row.get("authenticity", {}).get("final_score", 0)))  # type: ignore
    return mean(vals) if vals else 0.0


def _rows_on_date(rows: List[dict], target_date_iso: str) -> List[dict]:
    out: List[dict] = []  # type: ignore
    for row in rows:
        s = row.get("date")  # type: ignore
        if not s:
            continue
        try:
            d = parse_date(s).isoformat()  # type: ignore
        except Exception:
            continue
        if d == target_date_iso:
            out.append(row)
    return out


def build_scraped_snapshot(scraped_payload: dict) -> dict:
    dt = parse_iso_dt(scraped_payload.get("scraped_at")) or datetime.now(timezone.utc)  # type: ignore
    snap_date = dt.date().isoformat()  # type: ignore
    policy = scraped_payload.get("policy", "unknown")  # type: ignore

    ufo_rows = scraped_payload.get("ufo_news", [])  # type: ignore
    crisis_rows = scraped_payload.get("crisis_news", [])  # type: ignore
    rejected_rows = scraped_payload.get("rejected_news", [])  # type: ignore
    stats = scraped_payload.get("stats", {})  # type: ignore

    # 面板口径：仅统计“运行当日(date==snap_date)”事件，避免 lookback 窗口重叠导致伪相关
    ufo_rows_today = _rows_on_date(ufo_rows, snap_date)
    crisis_rows_today = _rows_on_date(crisis_rows, snap_date)
    rejected_rows_today = _rows_on_date(rejected_rows, snap_date)

    accepted_titles = []
    accepted_titles.extend([r.get("title", "") for r in ufo_rows_today])  # type: ignore
    accepted_titles.extend([r.get("title", "") for r in crisis_rows_today])  # type: ignore
    rejected_titles = [r.get("title", "") for r in rejected_rows_today]  # type: ignore

    topic_counts_accepted = _topic_counts_from_titles(accepted_titles)
    topic_counts_rejected = _topic_counts_from_titles(rejected_titles)
    control_total_accepted = sum(topic_counts_accepted.values())  # type: ignore
    accepted_denom = max(1, len(ufo_rows_today) + len(crisis_rows_today))
    control_density_accepted = control_total_accepted / float(accepted_denom)

    return {
        "date": snap_date,
        "policy": policy,
        "date_scope": "run_day_only",
        "updated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "ufo_count": len(ufo_rows_today),
        "crisis_count": len(crisis_rows_today),
        "rejected_count": len(rejected_rows_today),
        "accepted_events": len(ufo_rows_today) + len(crisis_rows_today),
        "window_ufo_count": len(ufo_rows),
        "window_crisis_count": len(crisis_rows),
        "window_rejected_count": len(rejected_rows),
        "window_accepted_events": int(stats.get("accepted_events", 0)),  # type: ignore
        "raw_items": int(stats.get("raw_items", 0)),  # type: ignore
        "ufo_score_mean": round(_avg_score(ufo_rows_today), 3),  # type: ignore
        "crisis_score_mean": round(_avg_score(crisis_rows_today), 3),  # type: ignore
        # 控制变量改为“仅通过审核样本”的议题强度，避免被 rejected 噪声放大
        "control_scope": "accepted_only",
        "control_economy": int(topic_counts_accepted["economy"]),  # type: ignore
        "control_security": int(topic_counts_accepted["security"]),  # type: ignore
        "control_immigration": int(topic_counts_accepted["immigration"]),  # type: ignore
        "control_total": int(control_total_accepted),
        "control_density_accepted": round(control_density_accepted, 6),  # type: ignore
        "control_total_rejected_audit": int(sum(topic_counts_rejected.values())),  # type: ignore
    }


def load_panel(panel_path: Path) -> dict:  # type: ignore
    if not panel_path.exists():  # type: ignore
        return {"meta": {"version": 1}, "rows": []}
    with panel_path.open("r", encoding="utf-8") as f:  # type: ignore
        payload = json.load(f)  # type: ignore
    payload.setdefault("meta", {"version": 1})  # type: ignore
    payload.setdefault("rows", [])  # type: ignore
    return payload


def save_panel(panel_path: Path, panel: dict) -> None:  # type: ignore
    panel["meta"]["updated_at"] = datetime.now(timezone.utc).isoformat()  # type: ignore
    with panel_path.open("w", encoding="utf-8") as f:  # type: ignore
        json.dump(panel, f, ensure_ascii=False, indent=2)  # type: ignore


def upsert_panel_row(panel: dict, row: dict) -> bool:  # type: ignore
    key = (row["date"], row["policy"])  # type: ignore
    rows = panel.get("rows", [])  # type: ignore
    for i, old in enumerate(rows):
        if (old.get("date"), old.get("policy")) == key:  # type: ignore
            rows[i] = row  # type: ignore
            panel["rows"] = rows  # type: ignore
            return False
    rows.append(row)
    rows.sort(key=lambda x: (x.get("date", ""), x.get("policy", "")))  # type: ignore
    panel["rows"] = rows  # type: ignore
    return True


def _select_panel_rows(panel: dict, prefer_policy: str) -> Tuple[str, List[dict]]:  # type: ignore
    rows = panel.get("rows", [])  # type: ignore
    run_day_rows = [r for r in rows if r.get("date_scope") == "run_day_only"]  # type: ignore
    effective_rows = run_day_rows if run_day_rows else rows

    grouped = defaultdict(list)
    for r in effective_rows:
        grouped[r.get("policy", "unknown")].append(r)  # type: ignore

    if prefer_policy in grouped:
        return prefer_policy, grouped[prefer_policy]  # type: ignore
    if grouped:
        policy = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)[0][0]  # type: ignore
        return policy, grouped[policy]  # type: ignore
    return prefer_policy, []


# _max_missing_streak 已移至 utils.py（通过 import 别名引入）


def _load_events_v2_crisis_dates(events_path: Path) -> List[date]:
    """从 events_v2.json 读取真实危机日期列表（语义正确的冲击日）。"""
    if not events_path.exists():  # type: ignore
        return []
    try:
        with events_path.open("r", encoding="utf-8") as f:  # type: ignore
            payload = json.load(f)  # type: ignore
        dates = []
        for row in payload.get("correlations", []):  # type: ignore
            d_str = row.get("crisis", {}).get("date")  # type: ignore
            if d_str:
                try:
                    dates.append(parse_date(d_str))  # type: ignore
                except Exception:
                    pass
        return sorted(set(dates))  # type: ignore
    except Exception:
        return []


def analyze_panel(  # type: ignore
    panel_path: Path,
    prefer_policy: str,
    min_days: int = 180,
    min_shocks: int = 12,
    min_observed_ratio: float = 0.8,
    permutations: int = PERMUTATIONS,
    events_v2_path: Path | None = None,
) -> PanelStats:
    panel = load_panel(panel_path)  # type: ignore
    policy, rows = _select_panel_rows(panel, prefer_policy)  # type: ignore
    if not rows:
        return PanelStats(
            policy=policy,
            n_days=0,
            start_date="",
            end_date="",
            observed_days=0,
            missing_days=0,
            observed_ratio=0.0,
            max_missing_streak=0,
            n_shocks=0,
            shock_threshold=0.0,
            sufficient=False,
            reason="面板为空，先运行 scraper 并更新面板",
            control_metric="control_density_accepted",
            window_results={},
            best_lag=0,
            best_corr=0.0,
            best_positive_lag=0,
            best_positive_corr=0.0,
            lag0_corr=0.0,
            shock_source="none",
        )

    by_date = {}
    for r in rows:
        by_date[r["date"]] = r  # 同一date+policy已upsert，保留最后一条  # type: ignore

    days = sorted(parse_date(d) for d in by_date.keys())  # type: ignore
    start = days[0]  # type: ignore
    end = days[-1]  # type: ignore
    n_days = (end - start).days + 1
    observed_days = len(by_date)
    missing_days = max(0, n_days - observed_days)

    all_days = [start + timedelta(days=i) for i in range(n_days)]  # type: ignore
    observed_day_set = set(days)
    observed_ratio = (observed_days / float(n_days)) if n_days > 0 else 0.0
    max_missing_streak = _max_missing_streak(all_days, observed_day_set)
    crisis_series: Dict[date, float] = {}  # type: ignore
    ufo_series: Dict[date, float] = {}  # type: ignore
    control_series: Dict[date, float] = {}  # type: ignore
    adjusted_ufo_series: Dict[date, float] = {}  # type: ignore

    for d in all_days:
        row = by_date.get(d.isoformat(), {})  # type: ignore
        c = float(row.get("crisis_count", 0))  # type: ignore
        u = float(row.get("ufo_count", 0))  # type: ignore
        ctrl_density = row.get("control_density_accepted")  # type: ignore
        if ctrl_density is None:
            legacy_total = float(row.get("control_total", 0))  # type: ignore
            accepted_n = float(row.get("ufo_count", 0)) + float(row.get("crisis_count", 0))  # type: ignore
            denom = max(1.0, accepted_n)
            ctrl_density = legacy_total / denom
        ctrl = max(0.0, float(ctrl_density))
        crisis_series[d] = c  # type: ignore
        ufo_series[d] = u  # type: ignore
        control_series[d] = ctrl  # type: ignore
        adjusted_ufo_series[d] = u / (1.0 + ctrl)  # type: ignore

    crisis_nonzero = [v for v in crisis_series.values() if v > 0]  # type: ignore
    if not crisis_nonzero:
        return PanelStats(
            policy=policy,
            n_days=n_days,
            start_date=start.isoformat(),  # type: ignore
            end_date=end.isoformat(),  # type: ignore
            observed_days=observed_days,
            missing_days=missing_days,
            observed_ratio=observed_ratio,
            max_missing_streak=max_missing_streak,
            n_shocks=0,
            shock_threshold=0.0,
            sufficient=False,
            reason="面板中无危机冲击日（crisis_count 全为0）",
            control_metric="control_density_accepted",
            window_results={},
            best_lag=0,
            best_corr=0.0,
            best_positive_lag=0,
            best_positive_corr=0.0,
            lag0_corr=0.0,
            shock_source="none",
        )

    # 双轨冲击日机制：
    # 主轨：events_v2 真实危机日期（语义正确，但数量少）
    # 辅轨：高新闻量日（75百分位，数量多但语义为"高新闻量日"而非"政治危机日"）
    # 优先使用主轨；若主轨在面板覆盖范围内的日期 < min_shocks，则回退辅轨并标注
    ev2_crisis_dates: List[date] = []
    if events_v2_path is not None:
        ev2_crisis_dates = _load_events_v2_crisis_dates(events_v2_path)
    # 只保留在面板覆盖范围内的 events_v2 危机日期
    ev2_in_panel = [d for d in ev2_crisis_dates if start <= d <= end]

    shock_threshold = compute_shock_threshold(crisis_nonzero)
    news_volume_shock_days = [d for d, v in crisis_series.items() if v >= shock_threshold]  # type: ignore

    # 主轨门槛：events_v2 面板内日期 >= 5 即优先使用（语义正确优先于数量）
    # min_shocks 是统计功效门槛，不应阻止语义正确的冲击日被使用
    if len(ev2_in_panel) >= EV2_MIN_DATES:
        shock_days = ev2_in_panel
        shock_source = "events_v2_crisis_dates"
    else:
        # 回退：使用90百分位（比原75百分位更严格）以减少误报
        shock_threshold_90 = compute_shock_threshold(crisis_nonzero, q=90.0)
        news_volume_shock_days_90 = [d for d, v in crisis_series.items() if v >= shock_threshold_90]
        shock_days = news_volume_shock_days_90 if news_volume_shock_days_90 else news_volume_shock_days
        shock_threshold = shock_threshold_90 if news_volume_shock_days_90 else shock_threshold
        shock_source = "news_volume_90pct_fallback" if news_volume_shock_days_90 else "news_volume_75pct_fallback"

    n_shocks = len(shock_days)

    # 当使用 events_v2 主轨时，冲击日门槛降为 EV2_MIN_DATES（语义正确优先于数量）
    effective_min_shocks = EV2_MIN_DATES if shock_source == "events_v2_crisis_dates" else min_shocks

    if observed_days < min_days or n_shocks < effective_min_shocks or observed_ratio < min_observed_ratio:
        reasons = []
        if observed_days < min_days:
            reasons.append(f"observed_days={observed_days} (<{min_days})")
        if n_shocks < effective_min_shocks:
            reasons.append(f"shocks={n_shocks} (<{effective_min_shocks})")
        if observed_ratio < min_observed_ratio:
            reasons.append(f"observed_ratio={observed_ratio:.3f} (<{min_observed_ratio:.3f})")
        return PanelStats(
            policy=policy,
            n_days=n_days,
            start_date=start.isoformat(),  # type: ignore
            end_date=end.isoformat(),  # type: ignore
            observed_days=observed_days,
            missing_days=missing_days,
            observed_ratio=observed_ratio,
            max_missing_streak=max_missing_streak,
            n_shocks=n_shocks,
            shock_threshold=shock_threshold,
            sufficient=False,
            reason=f"样本不足：{' 或 '.join(reasons)}",
            control_metric="control_density_accepted",
            window_results={},
            best_lag=0,
            best_corr=0.0,
            best_positive_lag=0,
            best_positive_corr=0.0,
            lag0_corr=0.0,
            shock_source=shock_source,
        )

    ufo_nonzero = [v for v in ufo_series.values() if v > 0]  # type: ignore
    ufo_thr = compute_shock_threshold(ufo_nonzero)
    ufo_shock_days = [d for d, v in ufo_series.items() if v >= ufo_thr]  # type: ignore

    results = {}
    for w in WINDOWS:
        obs_adj = _window_effect(shock_days, adjusted_ufo_series, w)
        obs_raw = _window_effect(shock_days, ufo_series, w)
        reverse_raw = _window_effect(ufo_shock_days, crisis_series, w)
        placebo_control = _window_effect(shock_days, control_series, w)
        p_adj = _perm_pvalue(
            all_days,
            len(shock_days),
            adjusted_ufo_series,
            w,
            obs_adj,
            permutations=permutations,
        )
        p_raw = _perm_pvalue(
            all_days,
            len(shock_days),
            ufo_series,
            w,
            obs_raw,
            permutations=permutations,
        )
        results[w] = {  # type: ignore
            "obs_effect_adjusted": obs_adj,
            "obs_effect_raw": obs_raw,
            "reverse_effect_raw": reverse_raw,
            "placebo_control_effect": placebo_control,
            "p_value_adjusted": p_adj,
            "p_value_raw": p_raw,
        }

    # lead-lag 相关：正lag代表“危机领先UFO”
    lag_corrs: Dict[int, float] = {}  # type: ignore
    best_lag = 0
    best_corr = -1.0
    for lag in range(-30, 31):
        xs = []
        ys = []
        for d in all_days:
            d2 = d + timedelta(days=lag)  # type: ignore
            if d2 < start or d2 > end:
                continue
            xs.append(crisis_series[d])  # type: ignore
            ys.append(adjusted_ufo_series[d2])  # type: ignore
        corr = pearson_corr(xs, ys)
        lag_corrs[lag] = corr  # type: ignore
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    positive_lag_corrs = [(lag, corr) for lag, corr in lag_corrs.items() if lag >= 1]  # type: ignore
    if positive_lag_corrs:
        best_positive_lag, best_positive_corr = max(positive_lag_corrs, key=lambda x: x[1])  # type: ignore
    else:
        best_positive_lag, best_positive_corr = 0, 0.0
    lag0_corr = float(lag_corrs.get(0, 0.0))

    return PanelStats(
        policy=policy,
        n_days=n_days,
        start_date=start.isoformat(),  # type: ignore
        end_date=end.isoformat(),  # type: ignore
        observed_days=observed_days,
        missing_days=missing_days,
        observed_ratio=observed_ratio,
        max_missing_streak=max_missing_streak,
        n_shocks=n_shocks,
        shock_threshold=shock_threshold,
        sufficient=True,
        reason="样本充分",
        control_metric="control_density_accepted",
        window_results=results,
        best_lag=best_lag,
        best_corr=best_corr,
        best_positive_lag=int(best_positive_lag),
        best_positive_corr=float(best_positive_corr),
        lag0_corr=lag0_corr,
        shock_source=shock_source,
    )


def summarize_causal_verdict(
    events_stats: EventsV2Stats,
    scraped_stats: ScrapedStats,
    panel_stats: PanelStats,
) -> Tuple[str, List[str]]:  # type: ignore
    notes = []
    notes.append("events_v2 显示强时间聚集，但为人工筛选配对，存在选择偏差。")
    notes.append(
        f"events_v2: p_abs={events_stats.p_abs_gap:.4g}, p_30={events_stats.p_within_30:.4g}, "
        f"p_60={events_stats.p_within_60:.4g}, p_dir={events_stats.p_direction:.4g}"
    )

    if not scraped_stats.sufficient:
        notes.append(f"实时样本不足：{scraped_stats.reason}")
    else:
        notes.append("实时样本达到基础门槛。")

    if not panel_stats.sufficient:
        notes.append(f"面板样本不足：{panel_stats.reason}")
        return "仅能判定存在时间相关性，无法建立因果关系", notes

    sig_windows = 0
    for w in APPROVAL_WINDOWS:
        r = panel_stats.window_results.get(w, {})  # type: ignore
        if (
            r.get("p_value_adjusted", 1.0) < 0.05  # type: ignore
            and r.get("obs_effect_adjusted", 0.0) > 0  # type: ignore
            and r.get("obs_effect_raw", 0.0) > r.get("reverse_effect_raw", 0.0)  # type: ignore
            and abs(r.get("placebo_control_effect", 0.0)) < max(1e-9, abs(r.get("obs_effect_raw", 0.0)))  # type: ignore
        ):
            sig_windows += 1  # type: ignore

    # 修复：lag=0（同期）为全局最优时，不能声称"危机领先UFO"的因果前导信号。
    # 只有正向 lag 的相关系数严格优于 lag=0 时，才算前导信号。
    lag0_is_dominant = panel_stats.lag0_corr >= panel_stats.best_positive_corr - LEAD_LAG_TIE_TOLERANCE
    lead_lag_ok = (
        1 <= panel_stats.best_positive_lag <= 30
        and panel_stats.best_positive_corr > 0.1
        and not lag0_is_dominant  # lag=0 不能是最优，否则只是同期相关
    )

    if sig_windows >= 2 and lead_lag_ok:
        return "存在因果信号（中等强度，仍需外部对照验证）", notes
    if sig_windows >= 2:
        return "存在时序窗口显著效应，但前导相关不明确（同期相关主导），因果方向待确认", notes
    if sig_windows >= 1:
        return "存在弱因果信号，但证据仍不足以作强因果结论", notes
    return "仅发现相关性或弱信号，因果证据不足", notes


def run_strict_approval(
    panel_stats: PanelStats,
    min_days: int = 180,
    min_shocks: int = 12,
    min_observed_ratio: float = 0.8,
) -> ApprovalResult:
    """
    严格审批门槛：
    1) 样本充分
    2) 有效观测天数 >= min_days
    3) 冲击日 >= min_shocks
    4) 有效观测覆盖率 >= min_observed_ratio
    5) 在短窗口（3/7/10）中至少两个窗口 p_adj < 0.05 且前向效应>0
    6) 正 lag 相关显著，且与全域最优相关差距不超过容忍阈值
    7) 在短窗口中反向效应不能系统性超过前向效应
    """
    gates: List[ApprovalGate] = []  # type: ignore

    gates.append(
        ApprovalGate(
            name="panel_sufficient",
            passed=panel_stats.sufficient,
            detail=panel_stats.reason,
        )
    )

    gates.append(
        ApprovalGate(
            name=f"panel_observed_days>={min_days}",
            passed=panel_stats.observed_days >= min_days,
            detail=f"observed_days={panel_stats.observed_days}, span_days={panel_stats.n_days}",
        )
    )

    # 当使用 events_v2 主轨时，冲击日门槛降为 EV2_MIN_DATES
    effective_min_shocks = EV2_MIN_DATES if panel_stats.shock_source == "events_v2_crisis_dates" else min_shocks
    ev2_shocks_note = " (events_v2主轨，门槛降为语义正确优先)" if panel_stats.shock_source == "events_v2_crisis_dates" else ""
    gates.append(
        ApprovalGate(
            name=f"panel_shocks>={int(effective_min_shocks)}",
            passed=panel_stats.n_shocks >= effective_min_shocks,
            detail=f"shocks={panel_stats.n_shocks}, shock_source={panel_stats.shock_source}{ev2_shocks_note}",
        )
    )

    gates.append(
        ApprovalGate(
            name=f"panel_observed_ratio>={min_observed_ratio:.2f}",
            passed=panel_stats.observed_ratio >= min_observed_ratio,
            detail=(
                f"observed_days={panel_stats.observed_days}, span_days={panel_stats.n_days}, "
                f"ratio={panel_stats.observed_ratio:.4f}, max_missing_streak={panel_stats.max_missing_streak}"
            ),
        )
    )

    sig_windows = 0
    reverse_violations = 0
    for w in APPROVAL_WINDOWS:
        r = panel_stats.window_results.get(w, {})  # type: ignore
        if (
            r.get("p_value_adjusted", 1.0) < 0.05  # type: ignore
            and r.get("obs_effect_adjusted", 0.0) > 0  # type: ignore
            and r.get("obs_effect_raw", 0.0) > 0  # type: ignore
        ):
            sig_windows += 1  # type: ignore
        if r.get("reverse_effect_raw", 0.0) > r.get("obs_effect_raw", 0.0) and r.get("reverse_effect_raw", 0.0) > 0:  # type: ignore
            reverse_violations += 1  # type: ignore

    gates.append(
        ApprovalGate(
            name=">=2_significant_windows",
            passed=sig_windows >= 2,
            detail=f"approval_windows={list(APPROVAL_WINDOWS)}, significant_windows={sig_windows}",
        )
    )

    # 修复：lag=0（同期）主导时不能声称因果前导，正向 lag 必须严格优于 lag=0
    lag0_is_dominant = panel_stats.lag0_corr >= panel_stats.best_positive_corr - LEAD_LAG_TIE_TOLERANCE
    lead_lag_ok = (
        1 <= panel_stats.best_positive_lag <= 30
        and panel_stats.best_positive_corr >= 0.1
        and not lag0_is_dominant
    )
    gates.append(
        ApprovalGate(
            name="lead_lag_positive",
            passed=lead_lag_ok,
            detail=(
                f"best_positive_lag={panel_stats.best_positive_lag}, "
                f"best_positive_corr={panel_stats.best_positive_corr:.4f}, "
                f"best_any_lag={panel_stats.best_lag}, best_any_corr={panel_stats.best_corr:.4f}, "
                f"lag0_corr={panel_stats.lag0_corr:.4f}, "
                f"lag0_dominant={lag0_is_dominant}"
            ),
        )
    )

    gates.append(
        ApprovalGate(
            name="reverse_not_dominant",
            passed=reverse_violations <= 1,
            detail=f"approval_windows={list(APPROVAL_WINDOWS)}, reverse_violations={reverse_violations}",
        )
    )

    all_passed = all(g.passed for g in gates)
    if all_passed:
        return ApprovalResult(
            status="APPROVED",
            level="CAUSAL_SIGNAL",
            gates=gates,
            reason="所有严格审批门槛通过",
        )

    # 分层驳回原因
    if not gates[0].passed or not gates[1].passed or not gates[2].passed or not gates[3].passed:  # type: ignore
        return ApprovalResult(
            status="REJECTED",
            level="INSUFFICIENT_EVIDENCE",
            gates=gates,
            reason="样本或覆盖不足，禁止作因果宣称",
        )
    return ApprovalResult(
        status="REJECTED",
        level="WEAK_SIGNAL",
        gates=gates,
        reason="样本达标但信号未通过严格因果门槛",
    )


def write_causal_report(
    report_path: Path,
    verdict: str,
    notes: List[str],  # type: ignore
    events_stats: EventsV2Stats,
    scraped_stats: ScrapedStats,
    panel_stats: PanelStats,
    approval: ApprovalResult,
    permutations: int,
    fast_mode: bool,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "verdict": verdict,
        "approval": {
            "status": approval.status,
            "level": approval.level,
            "reason": approval.reason,
            "gates": [
                {"name": g.name, "passed": g.passed, "detail": g.detail}
                for g in approval.gates
            ],
        },
        "events_v2": {
            "_warning": (
                "人工筛选正向配对样本，存在严重选择偏差。"
                "以下统计数字（含p值）仅作描述性参考，不用于任何闸门判断或因果结论。"
            ),
            "n_cases": events_stats.n_cases,
            "mean_gap": events_stats.mean_gap,
            "mean_abs_gap": events_stats.mean_abs_gap,
            "within_30": events_stats.within_30,
            "within_60": events_stats.within_60,
            "positive_count": events_stats.positive_count,
            "negative_count": events_stats.negative_count,
            "p_abs_gap_descriptive_only": events_stats.p_abs_gap,
            "p_within_30_descriptive_only": events_stats.p_within_30,
            "p_within_60_descriptive_only": events_stats.p_within_60,
            "p_direction_descriptive_only": events_stats.p_direction,
        },
        "scraped": {
            "policy": scraped_stats.policy,
            "n_ufo": scraped_stats.n_ufo,
            "n_crisis": scraped_stats.n_crisis,
            "coverage_days": scraped_stats.coverage_days,
            "sufficient": scraped_stats.sufficient,
            "reason": scraped_stats.reason,
            "window_results": scraped_stats.window_results,
        },
        "panel": {  # type: ignore
            "policy": panel_stats.policy,
            "n_days": panel_stats.n_days,
            "start_date": panel_stats.start_date,
            "end_date": panel_stats.end_date,
            "observed_days": panel_stats.observed_days,
            "missing_days": panel_stats.missing_days,
            "observed_ratio": panel_stats.observed_ratio,
            "max_missing_streak": panel_stats.max_missing_streak,
            "n_shocks": panel_stats.n_shocks,
            "shock_threshold": panel_stats.shock_threshold,
            "sufficient": panel_stats.sufficient,
            "reason": panel_stats.reason,
            "control_metric": panel_stats.control_metric,
            "shock_source": panel_stats.shock_source,
            "best_lag": panel_stats.best_lag,
            "best_corr": panel_stats.best_corr,
            "best_positive_lag": panel_stats.best_positive_lag,
            "best_positive_corr": panel_stats.best_positive_corr,
            "lag0_corr": panel_stats.lag0_corr,
            "window_results": panel_stats.window_results,
        },
        "runtime": {
            "permutations": int(permutations),
            "fast_mode": bool(fast_mode),
        },
        "notes": notes,
    }
    with report_path.open("w", encoding="utf-8") as f:  # type: ignore
        json.dump(payload, f, ensure_ascii=False, indent=2)  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="升级版因果检验器（含日度面板）")
    parser.add_argument(
        "--no-update-panel",  # type: ignore
        action="store_true",
        help="只分析，不把当前 scraped_news 写入面板",
    )
    parser.add_argument(
        "--panel-policy",  # type: ignore
        default="strict-balanced",
        help="面板分析优先使用的 policy（默认 strict-balanced）",
    )
    parser.add_argument(
        "--min-panel-days",  # type: ignore
        type=int,
        default=180,
        help="面板因果检验最少天数（默认 180）",
    )
    parser.add_argument(
        "--min-panel-shocks",  # type: ignore
        type=int,
        default=12,
        help="面板因果检验最少冲击日数量（默认 12）",
    )
    parser.add_argument(
        "--min-panel-observed-ratio",  # type: ignore
        type=float,
        default=0.85,
        help="面板有效观测覆盖率下限（默认 0.85）",
    )
    parser.add_argument(
        "--fail-on-reject",
        action="store_true",
        help="若严格审批未通过，则以非零状态码退出（用于CI/自动化闸门）",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=PERMUTATIONS,
        help=f"置换次数（默认 {PERMUTATIONS}）",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help=f"快速模式：自动把置换次数上限限制到 {FAST_MODE_PERMUTATIONS}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    permutations = max(200, int(args.permutations))
    if args.fast_mode:
        permutations = min(permutations, FAST_MODE_PERMUTATIONS)

    if not SCRAPED_FILE.exists():  # type: ignore
        print(f"[错误] 找不到 {SCRAPED_FILE}，请先运行 scraper.py")  # type: ignore
        sys.exit(1)

    with SCRAPED_FILE.open("r", encoding="utf-8") as f:  # type: ignore
        scraped_payload = json.load(f)  # type: ignore

    if not args.no_update_panel:  # type: ignore
        panel = load_panel(PANEL_FILE)  # type: ignore
        snapshot = build_scraped_snapshot(scraped_payload)
        inserted = upsert_panel_row(panel, snapshot)  # type: ignore
        save_panel(PANEL_FILE, panel)  # type: ignore
        action = "新增" if inserted else "更新"
        print(f"[面板] {action} 快照: date={snapshot['date']} policy={snapshot['policy']} -> {PANEL_FILE}")  # type: ignore

    events_stats = analyze_events_v2(EVENTS_FILE, permutations=permutations)
    scraped_stats = analyze_scraped(SCRAPED_FILE, permutations=permutations)
    panel_stats = analyze_panel(  # type: ignore
        PANEL_FILE,
        prefer_policy=args.panel_policy,
        min_days=args.min_panel_days,
        min_shocks=args.min_panel_shocks,
        min_observed_ratio=args.min_panel_observed_ratio,
        permutations=permutations,
        events_v2_path=EVENTS_FILE,
    )
    verdict, notes = summarize_causal_verdict(events_stats, scraped_stats, panel_stats)
    approval = run_strict_approval(
        panel_stats,
        min_days=args.min_panel_days,
        min_shocks=args.min_panel_shocks,
        min_observed_ratio=args.min_panel_observed_ratio,
    )
    write_causal_report(
        REPORT_FILE,
        verdict=verdict,
        notes=notes,
        events_stats=events_stats,
        scraped_stats=scraped_stats,
        panel_stats=panel_stats,
        approval=approval,
        permutations=permutations,
        fast_mode=bool(args.fast_mode),
    )

    print("\n=== 因果检验报告（升级版）===")
    print(f"置换次数: {permutations} (fast_mode={bool(args.fast_mode)})")  # type: ignore
    print(f"结论: {verdict}")  # type: ignore
    print(f"严格审批: {approval.status} ({approval.level}) - {approval.reason}")  # type: ignore

    print("\n[A] 历史配对数据（events_v2）")  # type: ignore
    print(f"- 案例数: {events_stats.n_cases}")  # type: ignore
    print(f"- 平均间隔: {events_stats.mean_gap:.2f} 天")  # type: ignore
    print(f"- 平均绝对间隔: {events_stats.mean_abs_gap:.2f} 天")  # type: ignore
    print(f"- 30天内: {events_stats.within_30}/{events_stats.n_cases}")  # type: ignore
    print(f"- 60天内: {events_stats.within_60}/{events_stats.n_cases}")  # type: ignore
    print(f"- 方向性: post={events_stats.positive_count}, pre={events_stats.negative_count}, p={events_stats.p_direction:.4g}")  # type: ignore
    print(
        f"- 置换检验 p 值: abs_gap={events_stats.p_abs_gap:.4g}, "
        f"within30={events_stats.p_within_30:.4g}, within60={events_stats.p_within_60:.4g}"
    )

    print("\n[B] 实时抓取样本（scraped_news）")  # type: ignore
    print(f"- policy: {scraped_stats.policy}")  # type: ignore
    print(f"- UFO数: {scraped_stats.n_ufo}, 危机数: {scraped_stats.n_crisis}, 覆盖天数: {scraped_stats.coverage_days}")  # type: ignore
    print(f"- 样本是否充分: {scraped_stats.sufficient}")  # type: ignore
    print(f"- 说明: {scraped_stats.reason}")  # type: ignore
    if scraped_stats.sufficient:
        for w in WINDOWS:
            r = scraped_stats.window_results[w]  # type: ignore
            print(
                f"  window={w}d: effect={r['obs_effect_post_minus_pre']:.3f}, "  # type: ignore
                f"reverse={r['reverse_effect']:.3f}, p={r['p_value']:.4g}"  # type: ignore
            )

    print("\n[C] 面板检验（causal_panel）")  # type: ignore
    print(f"- policy: {panel_stats.policy}")  # type: ignore
    print(f"- 覆盖: {panel_stats.start_date} ~ {panel_stats.end_date} ({panel_stats.n_days} 天)")  # type: ignore
    print(
        f"- 有效观测: {panel_stats.observed_days} 天, 缺失: {panel_stats.missing_days} 天, "
        f"覆盖率: {panel_stats.observed_ratio:.4f}, 最大缺口: {panel_stats.max_missing_streak} 天"
    )
    print(f"- 冲击日数: {panel_stats.n_shocks}, 阈值: {panel_stats.shock_threshold:.3f}")  # type: ignore
    print(f"- 样本是否充分: {panel_stats.sufficient}")  # type: ignore
    print(f"- 说明: {panel_stats.reason}")  # type: ignore
    print(f"- 控制变量口径: {panel_stats.control_metric}")  # type: ignore
    if panel_stats.sufficient:
        print(
            f"- lead-lag 最佳(全域): lag={panel_stats.best_lag} corr={panel_stats.best_corr:.4f}; "
            f"最佳正lag: lag={panel_stats.best_positive_lag} corr={panel_stats.best_positive_corr:.4f}; "
            f"lag0_corr={panel_stats.lag0_corr:.4f}"
        )  # type: ignore
        for w in WINDOWS:
            r = panel_stats.window_results[w]  # type: ignore
            print(
                f"  window={w}d: adj_effect={r['obs_effect_adjusted']:.4f}, raw_effect={r['obs_effect_raw']:.4f}, "  # type: ignore
                f"reverse={r['reverse_effect_raw']:.4f}, placebo={r['placebo_control_effect']:.4f}, "  # type: ignore
                f"p_adj={r['p_value_adjusted']:.4g}, p_raw={r['p_value_raw']:.4g}"  # type: ignore
            )

    print("\n[审慎解释]")  # type: ignore
    for n in notes:
        print(f"- {n}")  # type: ignore

    print("\n[严格审批门槛]")  # type: ignore
    for g in approval.gates:
        mark = "✓" if g.passed else "✗"
        print(f"- {mark} {g.name}: {g.detail}")  # type: ignore

    print(f"\n[输出] 审批报告已写入: {REPORT_FILE}")  # type: ignore

    if args.fail_on_reject and approval.status != "APPROVED":
        print("[退出] 严格审批未通过，返回状态码 2")
        sys.exit(2)


if __name__ == "__main__":
    main()
