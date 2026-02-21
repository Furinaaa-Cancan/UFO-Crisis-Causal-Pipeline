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

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVENTS_FILE = DATA_DIR / "events_v2.json"
SCRAPED_FILE = DATA_DIR / "scraped_news.json"
PANEL_FILE = DATA_DIR / "causal_panel.json"
REPORT_FILE = DATA_DIR / "causal_report.json"

SEED = 20260221
PERMUTATIONS = 20000
WINDOWS = (7, 14, 30)
SHOCK_COUNT_FLOOR = 2.0

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


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_iso_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    idx = (len(xs) - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return xs[lo]
    w = idx - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def pearson_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = 0.0
    dx2 = 0.0
    dy2 = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        num += dx * dy
        dx2 += dx * dx
        dy2 += dy * dy
    den = math.sqrt(dx2 * dy2)
    if den == 0:
        return 0.0
    return num / den


def compute_shock_threshold(nonzero_values: List[float], q: float = 75.0, floor: float = SHOCK_COUNT_FLOOR) -> float:
    if not nonzero_values:
        return floor
    return max(floor, percentile(nonzero_values, q))


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
    window_results: Dict[int, Dict[str, float]]


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
    window_results: Dict[int, Dict[str, float]]
    best_lag: int
    best_corr: float


@dataclass
class ApprovalGate:
    name: str
    passed: bool
    detail: str


@dataclass
class ApprovalResult:
    status: str
    level: str
    gates: List[ApprovalGate]
    reason: str


def analyze_events_v2(events_path: Path) -> EventsV2Stats:
    with events_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    crisis_dates: List[date] = []
    ufo_dates: List[date] = []
    gaps: List[int] = []
    for row in payload.get("correlations", []):
        c = parse_date(row["crisis"]["date"])
        u = parse_date(row["ufo_event"]["date"])
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

    random.seed(SEED)
    abs_means = []
    within_30 = []
    within_60 = []
    for _ in range(PERMUTATIONS):
        shuffled = ufo_dates[:]
        random.shuffle(shuffled)
        sim_gaps = [(u - c).days for c, u in zip(crisis_dates, shuffled)]
        abs_means.append(mean(abs(g) for g in sim_gaps))
        within_30.append(sum(0 <= g <= 30 for g in sim_gaps))
        within_60.append(sum(0 <= g <= 60 for g in sim_gaps))

    p_abs = sum(x <= obs_mean_abs for x in abs_means) / PERMUTATIONS
    p_30 = sum(x >= obs_30 for x in within_30) / PERMUTATIONS
    p_60 = sum(x >= obs_60 for x in within_60) / PERMUTATIONS
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
        s = row.get("date")
        if not s:
            continue
        try:
            d = parse_date(s)
        except Exception:
            continue
        if start <= d <= end:
            counts[d] += 1
    return counts


def _window_effect(trigger_days: List[date], outcome_counts: Dict[date, float], w: int) -> float:
    if not trigger_days:
        return 0.0
    vals = []
    for d in trigger_days:
        post = 0.0
        pre = 0.0
        for k in range(1, w + 1):
            post += outcome_counts.get(d + timedelta(days=k), 0.0)
            pre += outcome_counts.get(d - timedelta(days=k), 0.0)
        vals.append(post - pre)
    return mean(vals)


def _perm_pvalue(all_days: List[date], trigger_n: int, series: Dict[date, float], w: int, obs_effect: float) -> float:
    if trigger_n <= 0 or trigger_n > len(all_days):
        return 1.0
    random.seed(SEED)
    null = []
    for _ in range(PERMUTATIONS):
        sampled = random.sample(all_days, trigger_n)
        null.append(_window_effect(sampled, series, w))
    return sum(x >= obs_effect for x in null) / len(null) if null else 1.0


def analyze_scraped(scraped_path: Path) -> ScrapedStats:
    with scraped_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    ufo_rows = payload.get("ufo_news", [])
    crisis_rows = payload.get("crisis_news", [])
    policy = payload.get("policy", "unknown")

    dates = []
    for row in ufo_rows + crisis_rows:
        s = row.get("date")
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

    if len(crisis_rows) < 10 or len(ufo_rows) < 30 or coverage_days < 180:
        fail_reasons = []
        if len(crisis_rows) < 10:
            fail_reasons.append(f"crisis={len(crisis_rows)} (<10)")
        if len(ufo_rows) < 30:
            fail_reasons.append(f"ufo={len(ufo_rows)} (<30)")
        if coverage_days < 180:
            fail_reasons.append(f"覆盖天数={coverage_days} (<180)")
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
    crisis_days = sorted(set(crisis_counts.keys()))
    ufo_days = sorted(set(ufo_counts.keys()))

    all_days = [start + timedelta(days=i) for i in range(coverage_days)]
    random.seed(SEED)
    window_results = {}

    for w in WINDOWS:
        obs = _window_effect(crisis_days, ufo_counts, w)
        rev = _window_effect(ufo_days, crisis_counts, w)
        p_val = _perm_pvalue(all_days, len(crisis_days), ufo_counts, w, obs)
        window_results[w] = {
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
        for topic, kws in CONTROL_TOPIC_KEYWORDS.items():
            if any(kw in low for kw in kws):
                out[topic] += 1
    return out


def _avg_score(rows: List[dict]) -> float:
    vals = []
    for row in rows:
        vals.append(float(row.get("authenticity", {}).get("final_score", 0)))
    return mean(vals) if vals else 0.0


def _rows_on_date(rows: List[dict], target_date_iso: str) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        s = row.get("date")
        if not s:
            continue
        try:
            d = parse_date(s).isoformat()
        except Exception:
            continue
        if d == target_date_iso:
            out.append(row)
    return out


def build_scraped_snapshot(scraped_payload: dict) -> dict:
    dt = parse_iso_dt(scraped_payload.get("scraped_at")) or datetime.now(timezone.utc)
    snap_date = dt.date().isoformat()
    policy = scraped_payload.get("policy", "unknown")

    ufo_rows = scraped_payload.get("ufo_news", [])
    crisis_rows = scraped_payload.get("crisis_news", [])
    rejected_rows = scraped_payload.get("rejected_news", [])
    stats = scraped_payload.get("stats", {})

    # 面板口径：仅统计“运行当日(date==snap_date)”事件，避免 lookback 窗口重叠导致伪相关
    ufo_rows_today = _rows_on_date(ufo_rows, snap_date)
    crisis_rows_today = _rows_on_date(crisis_rows, snap_date)
    rejected_rows_today = _rows_on_date(rejected_rows, snap_date)

    accepted_titles = []
    accepted_titles.extend([r.get("title", "") for r in ufo_rows_today])
    accepted_titles.extend([r.get("title", "") for r in crisis_rows_today])
    rejected_titles = [r.get("title", "") for r in rejected_rows_today]

    topic_counts_accepted = _topic_counts_from_titles(accepted_titles)
    topic_counts_rejected = _topic_counts_from_titles(rejected_titles)
    control_total_accepted = sum(topic_counts_accepted.values())
    accepted_denom = max(1, len(ufo_rows_today) + len(crisis_rows_today))
    control_density_accepted = control_total_accepted / float(accepted_denom)

    return {
        "date": snap_date,
        "policy": policy,
        "date_scope": "run_day_only",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "ufo_count": len(ufo_rows_today),
        "crisis_count": len(crisis_rows_today),
        "rejected_count": len(rejected_rows_today),
        "accepted_events": len(ufo_rows_today) + len(crisis_rows_today),
        "window_ufo_count": len(ufo_rows),
        "window_crisis_count": len(crisis_rows),
        "window_rejected_count": len(rejected_rows),
        "window_accepted_events": int(stats.get("accepted_events", 0)),
        "raw_items": int(stats.get("raw_items", 0)),
        "ufo_score_mean": round(_avg_score(ufo_rows_today), 3),
        "crisis_score_mean": round(_avg_score(crisis_rows_today), 3),
        # 控制变量改为“仅通过审核样本”的议题强度，避免被 rejected 噪声放大
        "control_scope": "accepted_only",
        "control_economy": int(topic_counts_accepted["economy"]),
        "control_security": int(topic_counts_accepted["security"]),
        "control_immigration": int(topic_counts_accepted["immigration"]),
        "control_total": int(control_total_accepted),
        "control_density_accepted": round(control_density_accepted, 6),
        "control_total_rejected_audit": int(sum(topic_counts_rejected.values())),
    }


def load_panel(panel_path: Path) -> dict:
    if not panel_path.exists():
        return {"meta": {"version": 1}, "rows": []}
    with panel_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload.setdefault("meta", {"version": 1})
    payload.setdefault("rows", [])
    return payload


def save_panel(panel_path: Path, panel: dict) -> None:
    panel["meta"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    with panel_path.open("w", encoding="utf-8") as f:
        json.dump(panel, f, ensure_ascii=False, indent=2)


def upsert_panel_row(panel: dict, row: dict) -> bool:
    key = (row["date"], row["policy"])
    rows = panel.get("rows", [])
    for i, old in enumerate(rows):
        if (old.get("date"), old.get("policy")) == key:
            rows[i] = row
            panel["rows"] = rows
            return False
    rows.append(row)
    rows.sort(key=lambda x: (x.get("date", ""), x.get("policy", "")))
    panel["rows"] = rows
    return True


def _select_panel_rows(panel: dict, prefer_policy: str) -> Tuple[str, List[dict]]:
    rows = panel.get("rows", [])
    grouped = defaultdict(list)
    for r in rows:
        grouped[r.get("policy", "unknown")].append(r)

    if prefer_policy in grouped:
        return prefer_policy, grouped[prefer_policy]
    if grouped:
        policy = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)[0][0]
        return policy, grouped[policy]
    return prefer_policy, []


def _max_missing_streak(all_days: List[date], observed_days: set[date]) -> int:
    longest = 0
    current = 0
    for d in all_days:
        if d in observed_days:
            current = 0
            continue
        current += 1
        if current > longest:
            longest = current
    return longest


def analyze_panel(
    panel_path: Path,
    prefer_policy: str,
    min_days: int = 180,
    min_shocks: int = 12,
    min_observed_ratio: float = 0.8,
) -> PanelStats:
    panel = load_panel(panel_path)
    policy, rows = _select_panel_rows(panel, prefer_policy)
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
        )

    by_date = {}
    for r in rows:
        by_date[r["date"]] = r  # 同一date+policy已upsert，保留最后一条

    days = sorted(parse_date(d) for d in by_date.keys())
    start = days[0]
    end = days[-1]
    n_days = (end - start).days + 1
    observed_days = len(by_date)
    missing_days = max(0, n_days - observed_days)

    all_days = [start + timedelta(days=i) for i in range(n_days)]
    observed_day_set = set(days)
    observed_ratio = (observed_days / float(n_days)) if n_days > 0 else 0.0
    max_missing_streak = _max_missing_streak(all_days, observed_day_set)
    crisis_series: Dict[date, float] = {}
    ufo_series: Dict[date, float] = {}
    control_series: Dict[date, float] = {}
    adjusted_ufo_series: Dict[date, float] = {}

    for d in all_days:
        row = by_date.get(d.isoformat(), {})
        c = float(row.get("crisis_count", 0))
        u = float(row.get("ufo_count", 0))
        ctrl_density = row.get("control_density_accepted")
        if ctrl_density is None:
            legacy_total = float(row.get("control_total", 0))
            accepted_n = float(row.get("ufo_count", 0)) + float(row.get("crisis_count", 0))
            denom = max(1.0, accepted_n)
            ctrl_density = legacy_total / denom
        ctrl = max(0.0, float(ctrl_density))
        crisis_series[d] = c
        ufo_series[d] = u
        control_series[d] = ctrl
        adjusted_ufo_series[d] = u / (1.0 + ctrl)

    crisis_nonzero = [v for v in crisis_series.values() if v > 0]
    if not crisis_nonzero:
        return PanelStats(
            policy=policy,
            n_days=n_days,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
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
        )

    shock_threshold = compute_shock_threshold(crisis_nonzero)
    shock_days = [d for d, v in crisis_series.items() if v >= shock_threshold]
    n_shocks = len(shock_days)

    if observed_days < min_days or n_shocks < min_shocks or observed_ratio < min_observed_ratio:
        reasons = []
        if observed_days < min_days:
            reasons.append(f"observed_days={observed_days} (<{min_days})")
        if n_shocks < min_shocks:
            reasons.append(f"shocks={n_shocks} (<{min_shocks})")
        if observed_ratio < min_observed_ratio:
            reasons.append(f"observed_ratio={observed_ratio:.3f} (<{min_observed_ratio:.3f})")
        return PanelStats(
            policy=policy,
            n_days=n_days,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
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
        )

    ufo_nonzero = [v for v in ufo_series.values() if v > 0]
    ufo_thr = compute_shock_threshold(ufo_nonzero)
    ufo_shock_days = [d for d, v in ufo_series.items() if v >= ufo_thr]

    results = {}
    for w in WINDOWS:
        obs_adj = _window_effect(shock_days, adjusted_ufo_series, w)
        obs_raw = _window_effect(shock_days, ufo_series, w)
        reverse_raw = _window_effect(ufo_shock_days, crisis_series, w)
        placebo_control = _window_effect(shock_days, control_series, w)
        p_adj = _perm_pvalue(all_days, len(shock_days), adjusted_ufo_series, w, obs_adj)
        p_raw = _perm_pvalue(all_days, len(shock_days), ufo_series, w, obs_raw)
        results[w] = {
            "obs_effect_adjusted": obs_adj,
            "obs_effect_raw": obs_raw,
            "reverse_effect_raw": reverse_raw,
            "placebo_control_effect": placebo_control,
            "p_value_adjusted": p_adj,
            "p_value_raw": p_raw,
        }

    # lead-lag 相关：正lag代表“危机领先UFO”
    best_lag = 0
    best_corr = -1.0
    for lag in range(-30, 31):
        xs = []
        ys = []
        for d in all_days:
            d2 = d + timedelta(days=lag)
            if d2 < start or d2 > end:
                continue
            xs.append(crisis_series[d])
            ys.append(adjusted_ufo_series[d2])
        corr = pearson_corr(xs, ys)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return PanelStats(
        policy=policy,
        n_days=n_days,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
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
    )


def summarize_causal_verdict(
    events_stats: EventsV2Stats,
    scraped_stats: ScrapedStats,
    panel_stats: PanelStats,
) -> Tuple[str, List[str]]:
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
    for _, r in panel_stats.window_results.items():
        if (
            r["p_value_adjusted"] < 0.05
            and r["obs_effect_adjusted"] > 0
            and r["obs_effect_raw"] > r["reverse_effect_raw"]
            and abs(r["placebo_control_effect"]) < max(1e-9, abs(r["obs_effect_raw"]))
        ):
            sig_windows += 1

    lead_lag_ok = 1 <= panel_stats.best_lag <= 30 and panel_stats.best_corr > 0.1

    if sig_windows >= 2 and lead_lag_ok:
        return "存在因果信号（中等强度，仍需外部对照验证）", notes
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
    5) 至少两个窗口 p_adj < 0.05 且前向效应>0
    6) lead-lag 为正且在 1..30 日
    7) 反向效应不能系统性超过前向效应
    """
    gates: List[ApprovalGate] = []

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

    gates.append(
        ApprovalGate(
            name=f"panel_shocks>={min_shocks}",
            passed=panel_stats.n_shocks >= min_shocks,
            detail=f"shocks={panel_stats.n_shocks}",
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
    for w, r in panel_stats.window_results.items():
        if r["p_value_adjusted"] < 0.05 and r["obs_effect_adjusted"] > 0:
            sig_windows += 1
        if r["reverse_effect_raw"] > r["obs_effect_raw"]:
            reverse_violations += 1

    gates.append(
        ApprovalGate(
            name=">=2_significant_windows",
            passed=sig_windows >= 2,
            detail=f"significant_windows={sig_windows}",
        )
    )

    lead_lag_ok = 1 <= panel_stats.best_lag <= 30 and panel_stats.best_corr >= 0.1
    gates.append(
        ApprovalGate(
            name="lead_lag_positive",
            passed=lead_lag_ok,
            detail=f"best_lag={panel_stats.best_lag}, best_corr={panel_stats.best_corr:.4f}",
        )
    )

    gates.append(
        ApprovalGate(
            name="reverse_not_dominant",
            passed=reverse_violations <= 1,
            detail=f"reverse_violations={reverse_violations}",
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
    if not gates[0].passed or not gates[1].passed or not gates[2].passed or not gates[3].passed:
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
    notes: List[str],
    events_stats: EventsV2Stats,
    scraped_stats: ScrapedStats,
    panel_stats: PanelStats,
    approval: ApprovalResult,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
            "n_cases": events_stats.n_cases,
            "mean_gap": events_stats.mean_gap,
            "mean_abs_gap": events_stats.mean_abs_gap,
            "within_30": events_stats.within_30,
            "within_60": events_stats.within_60,
            "positive_count": events_stats.positive_count,
            "negative_count": events_stats.negative_count,
            "p_abs_gap": events_stats.p_abs_gap,
            "p_within_30": events_stats.p_within_30,
            "p_within_60": events_stats.p_within_60,
            "p_direction": events_stats.p_direction,
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
        "panel": {
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
            "best_lag": panel_stats.best_lag,
            "best_corr": panel_stats.best_corr,
            "window_results": panel_stats.window_results,
        },
        "notes": notes,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="升级版因果检验器（含日度面板）")
    parser.add_argument(
        "--no-update-panel",
        action="store_true",
        help="只分析，不把当前 scraped_news 写入面板",
    )
    parser.add_argument(
        "--panel-policy",
        default="strict-balanced",
        help="面板分析优先使用的 policy（默认 strict-balanced）",
    )
    parser.add_argument(
        "--min-panel-days",
        type=int,
        default=180,
        help="面板因果检验最少天数（默认 180）",
    )
    parser.add_argument(
        "--min-panel-shocks",
        type=int,
        default=12,
        help="面板因果检验最少冲击日数量（默认 12）",
    )
    parser.add_argument(
        "--min-panel-observed-ratio",
        type=float,
        default=0.85,
        help="面板有效观测覆盖率下限（默认 0.85）",
    )
    parser.add_argument(
        "--fail-on-reject",
        action="store_true",
        help="若严格审批未通过，则以非零状态码退出（用于CI/自动化闸门）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with SCRAPED_FILE.open("r", encoding="utf-8") as f:
        scraped_payload = json.load(f)

    if not args.no_update_panel:
        panel = load_panel(PANEL_FILE)
        snapshot = build_scraped_snapshot(scraped_payload)
        inserted = upsert_panel_row(panel, snapshot)
        save_panel(PANEL_FILE, panel)
        action = "新增" if inserted else "更新"
        print(f"[面板] {action} 快照: date={snapshot['date']} policy={snapshot['policy']} -> {PANEL_FILE}")

    events_stats = analyze_events_v2(EVENTS_FILE)
    scraped_stats = analyze_scraped(SCRAPED_FILE)
    panel_stats = analyze_panel(
        PANEL_FILE,
        prefer_policy=args.panel_policy,
        min_days=args.min_panel_days,
        min_shocks=args.min_panel_shocks,
        min_observed_ratio=args.min_panel_observed_ratio,
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
    )

    print("\n=== 因果检验报告（升级版）===")
    print(f"结论: {verdict}")
    print(f"严格审批: {approval.status} ({approval.level}) - {approval.reason}")

    print("\n[A] 历史配对数据（events_v2）")
    print(f"- 案例数: {events_stats.n_cases}")
    print(f"- 平均间隔: {events_stats.mean_gap:.2f} 天")
    print(f"- 平均绝对间隔: {events_stats.mean_abs_gap:.2f} 天")
    print(f"- 30天内: {events_stats.within_30}/{events_stats.n_cases}")
    print(f"- 60天内: {events_stats.within_60}/{events_stats.n_cases}")
    print(f"- 方向性: post={events_stats.positive_count}, pre={events_stats.negative_count}, p={events_stats.p_direction:.4g}")
    print(
        f"- 置换检验 p 值: abs_gap={events_stats.p_abs_gap:.4g}, "
        f"within30={events_stats.p_within_30:.4g}, within60={events_stats.p_within_60:.4g}"
    )

    print("\n[B] 实时抓取样本（scraped_news）")
    print(f"- policy: {scraped_stats.policy}")
    print(f"- UFO数: {scraped_stats.n_ufo}, 危机数: {scraped_stats.n_crisis}, 覆盖天数: {scraped_stats.coverage_days}")
    print(f"- 样本是否充分: {scraped_stats.sufficient}")
    print(f"- 说明: {scraped_stats.reason}")
    if scraped_stats.sufficient:
        for w in WINDOWS:
            r = scraped_stats.window_results[w]
            print(
                f"  window={w}d: effect={r['obs_effect_post_minus_pre']:.3f}, "
                f"reverse={r['reverse_effect']:.3f}, p={r['p_value']:.4g}"
            )

    print("\n[C] 面板检验（causal_panel）")
    print(f"- policy: {panel_stats.policy}")
    print(f"- 覆盖: {panel_stats.start_date} ~ {panel_stats.end_date} ({panel_stats.n_days} 天)")
    print(
        f"- 有效观测: {panel_stats.observed_days} 天, 缺失: {panel_stats.missing_days} 天, "
        f"覆盖率: {panel_stats.observed_ratio:.4f}, 最大缺口: {panel_stats.max_missing_streak} 天"
    )
    print(f"- 冲击日数: {panel_stats.n_shocks}, 阈值: {panel_stats.shock_threshold:.3f}")
    print(f"- 样本是否充分: {panel_stats.sufficient}")
    print(f"- 说明: {panel_stats.reason}")
    print(f"- 控制变量口径: {panel_stats.control_metric}")
    if panel_stats.sufficient:
        print(f"- lead-lag 最佳: lag={panel_stats.best_lag} corr={panel_stats.best_corr:.4f}")
        for w in WINDOWS:
            r = panel_stats.window_results[w]
            print(
                f"  window={w}d: adj_effect={r['obs_effect_adjusted']:.4f}, raw_effect={r['obs_effect_raw']:.4f}, "
                f"reverse={r['reverse_effect_raw']:.4f}, placebo={r['placebo_control_effect']:.4f}, "
                f"p_adj={r['p_value_adjusted']:.4g}, p_raw={r['p_value_raw']:.4g}"
            )

    print("\n[审慎解释]")
    for n in notes:
        print(f"- {n}")

    print("\n[严格审批门槛]")
    for g in approval.gates:
        mark = "✓" if g.passed else "✗"
        print(f"- {mark} {g.name}: {g.detail}")

    print(f"\n[输出] 审批报告已写入: {REPORT_FILE}")

    if args.fail_on_reject and approval.status != "APPROVED":
        print("[退出] 严格审批未通过，返回状态码 2")
        sys.exit(2)


if __name__ == "__main__":
    main()
