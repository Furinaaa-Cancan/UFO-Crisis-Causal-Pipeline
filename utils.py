"""
共用工具函数

集中存放多个模块共享的基础函数，避免代码重复：
- 日期解析
- 百分位数 / 分位数
- 冲击阈值计算
- 皮尔逊相关
- 最大连续缺失天数
- 面板行读取
"""

from __future__ import annotations

import json
import random as _random_module
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 日期解析
# ---------------------------------------------------------------------------

def parse_date(s: str) -> date:
    """将 YYYY-MM-DD 字符串解析为 date 对象。"""
    return datetime.strptime(s, "%Y-%m-%d").date()  # type: ignore


# ---------------------------------------------------------------------------
# 百分位数 / 分位数
# ---------------------------------------------------------------------------

def percentile(values: List[float], p: float) -> float:
    """线性插值百分位数（p 取 0-100）。"""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]  # type: ignore
    idx = (len(xs) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    w = idx - lo
    return xs[lo] * (1 - w) + xs[hi] * w  # type: ignore


def quantile(values: List[float], q: float) -> float:
    """线性插值分位数（q 取 0-1）。"""
    if not values:
        return 0.0
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    w = idx - lo
    return xs[lo] * (1 - w) + xs[hi] * w  # type: ignore


# ---------------------------------------------------------------------------
# 冲击阈值
# ---------------------------------------------------------------------------

SHOCK_COUNT_FLOOR = 2.0


def compute_shock_threshold(
    nonzero_values: List[float],  # type: ignore
    q: float = 75.0,
    floor: float = SHOCK_COUNT_FLOOR,
) -> float:
    """基于非零值的百分位数计算冲击阈值，保证不低于 floor。"""
    if not nonzero_values:
        return floor
    return max(floor, percentile(nonzero_values, q))


# ---------------------------------------------------------------------------
# 皮尔逊相关
# ---------------------------------------------------------------------------

def pearson_corr(xs: List[float], ys: List[float]) -> float:
    """计算两个序列的皮尔逊相关系数，样本量 < 3 返回 0。"""
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = 0.0
    dx2 = 0.0
    dy2 = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        num += dx * dy
        dx2 += dx * dx
        dy2 += dy * dy
    den = (dx2 * dy2) ** 0.5
    if den == 0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# 最大连续缺失天数
# ---------------------------------------------------------------------------

def max_missing_streak(all_days: List[date], observed_days: set[date]) -> int:
    """计算 all_days 中不在 observed_days 里的最长连续缺失天数。"""
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


# ---------------------------------------------------------------------------
# 面板行读取（共用逻辑）
# ---------------------------------------------------------------------------

def read_panel_rows(
    panel_path: Path,
    policy: str = "strict-balanced",
) -> List[dict]:  # type: ignore
    """
    从 causal_panel.json 读取指定 policy 的面板行。
    优先使用 date_scope='run_day_only' 的行；同一日期取最后一条。
    """
    if not panel_path.exists():  # type: ignore
        return []
    with panel_path.open("r", encoding="utf-8") as f:  # type: ignore
        rows = json.load(f).get("rows", [])  # type: ignore
    run_day_rows = [r for r in rows if r.get("date_scope") == "run_day_only"]  # type: ignore
    effective_rows = run_day_rows if run_day_rows else rows
    by_date: Dict[str, dict] = {}  # type: ignore
    for r in effective_rows:
        if r.get("policy") == policy and r.get("date"):  # type: ignore
            by_date[r["date"]] = r  # type: ignore
    return [by_date[d] for d in sorted(by_date)]  # type: ignore


# ---------------------------------------------------------------------------
# 距冲击日最小距离
# ---------------------------------------------------------------------------

def min_distance_to_shocks(d: date, shocks: List[date]) -> int:
    """返回日期 d 到最近冲击日的天数绝对值。"""
    if not shocks:
        return 10**9
    return min(abs((d - s).days) for s in shocks)


# ---------------------------------------------------------------------------
# 独立随机实例工厂
# ---------------------------------------------------------------------------

def make_rng(seed: int) -> _random_module.Random:
    """创建一个独立的 Random 实例，不影响全局随机状态。"""
    return _random_module.Random(seed)
