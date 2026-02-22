"""
因果机器学习（Causal ML）脚本

设计目标：
1) 在日度面板上用正交化方法估计 ATE（Double ML 思路）
2) 用随机森林学习 CATE（因果森林代理实现）
3) 输出严格闸门友好的结构化报告（data/model_causal_ml_report.json）
"""
# pyre-ignore-all-errors
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from utils import (  # type: ignore[import]
    compute_shock_threshold,
    make_rng,
    parse_date,
    percentile,
    quantile,
    read_panel_rows as _read_panel_rows,
)


try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # type: ignore
    from sklearn.model_selection import KFold  # type: ignore
    SKLEARN_READY = True
    SKLEARN_ERR = ""
except Exception as e:  # pragma: no cover - 环境相关分支
    SKLEARN_READY = False
    SKLEARN_ERR = str(e)


BASE_DIR = Path(__file__).resolve().parent  # type: ignore
DATA_DIR = BASE_DIR / "data"
PANEL_FILE = DATA_DIR / "causal_panel.json"  # type: ignore
OUT_FILE = DATA_DIR / "model_causal_ml_report.json"

MIN_OBS_DAYS = 120
MIN_SHOCK_DAYS = 10
SEED = 20260221
PERMUTATIONS = 2000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="因果机器学习（DML + 因果森林代理）")
    p.add_argument(
        "--policy",
        choices=["strict", "strict-balanced"],
        default="strict-balanced",
        help="读取哪个面板 policy",
    )
    p.add_argument("--min-observed-days", type=int, default=MIN_OBS_DAYS)
    p.add_argument("--min-shock-days", type=int, default=MIN_SHOCK_DAYS)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--permutations", type=int, default=PERMUTATIONS)
    return p.parse_args()


def load_rows(policy: str) -> List[dict]:
    return _read_panel_rows(PANEL_FILE, policy)


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def build_dataset(rows: List[dict], crisis_threshold: float) -> tuple[List[dict], List[str]]:
    """
    构建样本：
    - outcome: 当日 ufo_count
    - treatment: 当日 crisis_count 是否达到冲击阈值
    - covariates: 自回归滞后 + 控制变量 + 日历特征
    """
    by_date: Dict[date, dict] = {}  # type: ignore
    for r in rows:
        try:
            d = parse_date(r["date"])
        except Exception:
            continue
        by_date[d] = r  # type: ignore

    dates = sorted(by_date.keys())  # type: ignore
    if len(dates) < 15:
        return [], []

    lags = (1, 2, 3, 7)
    feature_names = [
        "ufo_lag1", "ufo_lag2", "ufo_lag3", "ufo_lag7",
        "crisis_lag1", "crisis_lag2", "crisis_lag3", "crisis_lag7",
        "control_total_t", "control_total_lag1",
        "control_density_t", "control_density_lag1",
        "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    ]

    import math

    data = []
    for d in dates:
        lag_dates = [d - timedelta(days=k) for k in lags]  # type: ignore
        if any(ld not in by_date for ld in lag_dates):
            continue

        cur = by_date[d]  # type: ignore
        x = []
        for k in lags:
            prev = by_date[d - timedelta(days=k)]  # type: ignore
            x.append(_safe_float(prev.get("ufo_count", 0)))  # type: ignore
        for k in lags:
            prev = by_date[d - timedelta(days=k)]  # type: ignore
            x.append(_safe_float(prev.get("crisis_count", 0)))  # type: ignore

        prev1 = by_date[d - timedelta(days=1)]  # type: ignore
        x.append(_safe_float(cur.get("control_total", 0)))
        x.append(_safe_float(prev1.get("control_total", 0)))
        x.append(_safe_float(cur.get("control_density_accepted", 0)))
        x.append(_safe_float(prev1.get("control_density_accepted", 0)))

        weekday = d.weekday()
        month = d.month
        x.append(math.sin(2.0 * math.pi * weekday / 7.0))
        x.append(math.cos(2.0 * math.pi * weekday / 7.0))
        x.append(math.sin(2.0 * math.pi * month / 12.0))
        x.append(math.cos(2.0 * math.pi * month / 12.0))

        y = _safe_float(cur.get("ufo_count", 0))
        crisis = _safe_float(cur.get("crisis_count", 0))
        t = 1 if crisis >= crisis_threshold else 0

        data.append({
            "date": d.isoformat(),  # type: ignore
            "x": x,
            "y": y,
            "t": t,
            "crisis_count": crisis,
        })

    return data, feature_names


def _build_cv_folds(n: int, n_folds: int, seed: int) -> List[List[int]]:
    idx = list(range(n))
    rng = make_rng(seed)
    rng.shuffle(idx)
    bins: List[List[int]] = [[] for _ in range(max(1, n_folds))]
    for pos, i in enumerate(idx):
        bins[pos % len(bins)].append(i)
    return [b for b in bins if b]


def _fit_ridge_linear(
    xs: List[List[float]],  # type: ignore
    ys: List[float],  # type: ignore
    alpha: float = 1.0,
    sample_weight: List[float] | None = None,  # type: ignore
) -> List[float]:
    if not xs or not ys:
        return []
    p = len(xs[0]) + 1  # intercept + features
    xtx = [[0.0 for _ in range(p)] for _ in range(p)]
    xty = [0.0 for _ in range(p)]

    for i, (x, y) in enumerate(zip(xs, ys)):
        w = float(sample_weight[i]) if sample_weight else 1.0
        row = [1.0] + [float(v) for v in x]
        for a in range(p):
            xty[a] += w * row[a] * float(y)
            for b in range(p):
                xtx[a][b] += w * row[a] * row[b]

    # 不正则化截距，只正则化斜率项。
    for j in range(1, p):
        xtx[j][j] += float(alpha)

    # 高斯-约当消元（维度很小：~17x17）
    aug = [xtx[i][:] + [xty[i]] for i in range(p)]  # type: ignore
    eps = 1e-12
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < eps:
            continue
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        div = aug[col][col]
        for j in range(col, p + 1):
            aug[col][j] /= div

        for r in range(p):
            if r == col:
                continue
            fac = aug[r][col]
            if abs(fac) < eps:
                continue
            for j in range(col, p + 1):
                aug[r][j] -= fac * aug[col][j]

    coef = [0.0 for _ in range(p)]
    for i in range(p):
        if abs(aug[i][i]) >= eps:
            coef[i] = float(aug[i][p])
    return coef


def _predict_linear(xs: List[List[float]], coef: List[float]) -> List[float]:  # type: ignore
    if not xs or not coef:
        return [0.0 for _ in xs]
    out = []
    for x in xs:
        y = coef[0]
        for c, v in zip(coef[1:], x):
            y += float(c) * float(v)
        out.append(float(y))
    return out


def estimate_nuisance(
    data: List[dict],  # type: ignore
    folds: int,
) -> tuple[List[float], List[float], str]:
    ys = [float(r["y"]) for r in data]
    ts = [int(r["t"]) for r in data]
    xs = [r["x"] for r in data]
    n = len(data)

    if not ys or not ts:
        return [], [], "empty_data"

    fallback_m = mean(ys)
    fallback_e = min(0.97, max(0.03, mean(ts)))
    m_hat = [fallback_m for _ in range(n)]
    e_hat = [fallback_e for _ in range(n)]

    if n < 20:
        return m_hat, e_hat, "mean_fallback"

    if not SKLEARN_READY:
        n_folds = max(2, min(folds, n))
        bins = _build_cv_folds(n, n_folds, seed=SEED + 9)
        for fold_id, test_idx in enumerate(bins, start=1):
            test_set = set(test_idx)
            train_idx = [i for i in range(n) if i not in test_set]
            if not train_idx:
                continue
            x_train = [xs[i] for i in train_idx]
            y_train = [ys[i] for i in train_idx]
            t_train = [float(ts[i]) for i in train_idx]
            x_test = [xs[i] for i in test_idx]

            y_coef = _fit_ridge_linear(x_train, y_train, alpha=1.0)
            y_pred = _predict_linear(x_test, y_coef)
            for j, idx in enumerate(test_idx):
                m_hat[idx] = float(y_pred[j])

            t_coef = _fit_ridge_linear(x_train, t_train, alpha=1.0)
            t_pred = _predict_linear(x_test, t_coef)
            for j, idx in enumerate(test_idx):
                e_hat[idx] = float(t_pred[j])

        e_hat = [min(0.97, max(0.03, x)) for x in e_hat]
        return m_hat, e_hat, "cross_fitted_linear_ridge"

    n_folds = max(2, min(folds, n))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_id = 0
    for train_idx, test_idx in kf.split(xs):
        fold_id += 1
        x_train = [xs[i] for i in train_idx]
        x_test = [xs[i] for i in test_idx]
        y_train = [ys[i] for i in train_idx]
        t_train = [ts[i] for i in train_idx]

        y_model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=SEED + fold_id,
        )
        y_model.fit(x_train, y_train)
        y_pred = y_model.predict(x_test).tolist()  # type: ignore
        for j, idx in enumerate(test_idx):
            m_hat[idx] = float(y_pred[j])

        p_treat = mean(t_train) if t_train else fallback_e
        if len(set(t_train)) >= 2:
            t_model = RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                random_state=SEED + 100 + fold_id,
                class_weight="balanced_subsample",
            )
            t_model.fit(x_train, t_train)
            probs = t_model.predict_proba(x_test)  # type: ignore
            for j, idx in enumerate(test_idx):
                e_hat[idx] = float(probs[j][1])  # type: ignore
        else:
            for idx in test_idx:
                e_hat[idx] = float(p_treat)

    e_hat = [min(0.97, max(0.03, x)) for x in e_hat]
    return m_hat, e_hat, "cross_fitted_rf"


def orthogonal_ate(
    ys: List[float],  # type: ignore
    ts: List[int],  # type: ignore
    m_hat: List[float],  # type: ignore
    e_hat: List[float],  # type: ignore
) -> tuple[float, List[float], List[float], float]:
    y_res = [y - m for y, m in zip(ys, m_hat)]  # type: ignore
    t_res = [float(t) - e for t, e in zip(ts, e_hat)]  # type: ignore
    den = sum(x * x for x in t_res)
    if den <= 1e-9:
        return 0.0, y_res, t_res, den
    ate = sum(a * b for a, b in zip(t_res, y_res)) / den  # type: ignore
    return ate, y_res, t_res, den


def permutation_pvalue(
    y_res: List[float],  # type: ignore
    t_res: List[float],  # type: ignore
    observed_ate: float,
    permutations: int,
) -> tuple[float, list[float]]:
    rng = make_rng(SEED + 77)
    null = []
    for _ in range(permutations):
        perm = t_res[:]
        rng.shuffle(perm)
        den = sum(x * x for x in perm)
        if den <= 1e-9:
            continue
        tau = sum(a * b for a, b in zip(perm, y_res)) / den  # type: ignore
        null.append(tau)
    if not null:
        return 1.0, []
    p = (sum(1 for x in null if abs(x) >= abs(observed_ate)) + 1) / float(len(null) + 1)
    return p, null


def estimate_cate(
    data: List[dict],  # type: ignore
    y_res: List[float],  # type: ignore
    t_res: List[float],  # type: ignore
) -> tuple[List[float], str]:
    if not data:
        return [], "no_data"

    # sklearn 不可用时，使用带权线性正则回归估计个体效应代理。
    if not SKLEARN_READY:
        if len(data) < 20:
            ate_like_den = sum(x * x for x in t_res)
            ate_like = (sum(a * b for a, b in zip(t_res, y_res)) / ate_like_den) if ate_like_den > 1e-9 else 0.0
            return [ate_like for _ in data], "constant_cate_fallback"

        xs = [r["x"] for r in data]
        target = []
        weights = []
        for yr, tr in zip(y_res, t_res):
            adj = tr
            if abs(adj) < 1e-3:
                adj = 1e-3 if adj >= 0 else -1e-3
            target.append(yr / adj)
            weights.append(max(1e-6, tr * tr))

        coef = _fit_ridge_linear(xs, target, alpha=2.0, sample_weight=weights)
        tau = _predict_linear(xs, coef)
        return [float(x) for x in tau], "orthogonal_linear_ridge_proxy"

    # sklearn 可用时采用随机森林代理。
    xs = [r["x"] for r in data]
    target = []
    weights = []
    for yr, tr in zip(y_res, t_res):
        adj = tr
        if abs(adj) < 1e-3:
            adj = 1e-3 if adj >= 0 else -1e-3
        target.append(yr / adj)
        weights.append(max(1e-6, tr * tr))

    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=6,
        random_state=SEED + 201,
    )
    model.fit(xs, target, sample_weight=weights)
    tau = model.predict(xs).tolist()  # type: ignore
    return [float(x) for x in tau], "orthogonal_random_forest_proxy"


def summarize_heterogeneity(tau_hat: List[float]) -> Dict[str, float | None]:
    if len(tau_hat) < 9:
        return {
            "q33": None,
            "q67": None,
            "top_mean": None,
            "bottom_mean": None,
            "spread_top_minus_bottom": None,
        }
    q33 = percentile(tau_hat, 33.0)
    q67 = percentile(tau_hat, 67.0)
    top = [x for x in tau_hat if x >= q67]
    bottom = [x for x in tau_hat if x <= q33]
    top_mean = mean(top) if top else None
    bottom_mean = mean(bottom) if bottom else None
    spread = None
    if top_mean is not None and bottom_mean is not None:
        spread = top_mean - bottom_mean
    return {
        "q33": round(q33, 6),  # type: ignore
        "q67": round(q67, 6),  # type: ignore
        "top_mean": round(top_mean, 6) if top_mean is not None else None,
        "bottom_mean": round(bottom_mean, 6) if bottom_mean is not None else None,
        "spread_top_minus_bottom": round(spread, 6) if spread is not None else None,
    }


def compute_causal_ml_pass(
    ate_positive: bool,
    ate_significant: bool,
    nuisance_model_ready: bool,
    cate_model_ready: bool,
    heterogeneity_estimated: bool,
) -> bool:
    return (
        ate_positive
        and ate_significant
        and nuisance_model_ready
        and cate_model_ready
        and heterogeneity_estimated
    )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.policy)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # type: ignore
        "model": "causal_ml_orthogonal_forest",
        "policy": args.policy,
        "observed_days": len(rows),
        "status": "pending",
        "reason": "",
        "dependencies": {
            "sklearn_ready": SKLEARN_READY,
            "sklearn_error": SKLEARN_ERR if not SKLEARN_READY else "",
        },
        "sample": {
            "n_rows": len(rows),
            "n_samples_for_ml": 0,
            "shock_threshold": None,
            "shock_days": 0,
            "treated_ratio": None,
            "n_features": 0,
            "feature_names": [],
        },
        "estimation": {
            "nuisance_method": "",
            "cate_method": "",
            "orthogonal_ate": None,
            "orthogonal_att": None,
            "ate_p_value": None,
            "ate_interval_note": "ate_placebo_null_ci95 is a placebo/null reference band, not an estimator confidence interval",
            "ate_placebo_null_ci95": None,
            "null_draws": 0,
            "cate_mean": None,
            "cate_std_proxy": None,
            "heterogeneity": {},
        },
        "gates": {
            "ate_positive": False,
            "ate_significant": False,
            "nuisance_model_ready": False,
            "cate_model_ready": False,
            "heterogeneity_estimated": False,
            "causal_ml_passed": False,
        },
    }

    if len(rows) < args.min_observed_days:
        out["reason"] = f"observed_days < {args.min_observed_days}，样本不足"  # type: ignore
    else:
        crisis_nonzero = [_safe_float(r.get("crisis_count", 0)) for r in rows if _safe_float(r.get("crisis_count", 0)) > 0]  # type: ignore
        threshold = compute_shock_threshold(crisis_nonzero)
        data, feature_names = build_dataset(rows, threshold)
        shock_days = sum(1 for r in data if int(r["t"]) == 1)

        out["sample"]["shock_threshold"] = round(threshold, 6)  # type: ignore
        out["sample"]["shock_days"] = shock_days  # type: ignore
        out["sample"]["n_samples_for_ml"] = len(data)  # type: ignore
        out["sample"]["n_features"] = len(feature_names)  # type: ignore
        out["sample"]["feature_names"] = feature_names  # type: ignore
        out["sample"]["treated_ratio"] = (round(shock_days / float(len(data)), 6) if data else None)  # type: ignore

        if len(data) < args.min_observed_days:
            out["reason"] = f"n_samples_for_ml < {args.min_observed_days}，构造特征后样本不足"  # type: ignore
        elif shock_days < args.min_shock_days:
            out["reason"] = f"shock_days < {args.min_shock_days}，处理组样本不足"  # type: ignore
        else:
            ys = [float(r["y"]) for r in data]
            ts = [int(r["t"]) for r in data]
            m_hat, e_hat, nuisance_method = estimate_nuisance(data, args.folds)
            ate, y_res, t_res, den = orthogonal_ate(ys, ts, m_hat, e_hat)
            out["estimation"]["nuisance_method"] = nuisance_method  # type: ignore

            if den <= 1e-9:
                out["status"] = "blocked"  # type: ignore
                out["reason"] = "orthogonal_denominator_too_small（treatment residual variance≈0）"  # type: ignore
            else:
                p_value, null = permutation_pvalue(y_res, t_res, ate, args.permutations)
                cate_hat, cate_method = estimate_cate(data, y_res, t_res)
                hetero = summarize_heterogeneity(cate_hat)

                treated_tau = [tau for tau, t in zip(cate_hat, ts) if t == 1]
                att = mean(treated_tau) if treated_tau else ate
                ci = None
                if null:
                    ci = [round(quantile(null, 0.025), 6), round(quantile(null, 0.975), 6)]  # type: ignore

                cate_std_proxy = None
                if len(cate_hat) >= 2:
                    cate_mean = mean(cate_hat)
                    cate_std_proxy = (sum((x - cate_mean) ** 2 for x in cate_hat) / (len(cate_hat) - 1)) ** 0.5
                else:
                    cate_mean = cate_hat[0] if cate_hat else 0.0

                spread = hetero.get("spread_top_minus_bottom")
                heterogeneity_estimated = spread is not None

                out["estimation"]["cate_method"] = cate_method  # type: ignore
                out["estimation"]["orthogonal_ate"] = round(ate, 6)  # type: ignore
                out["estimation"]["orthogonal_att"] = round(att, 6)  # type: ignore
                out["estimation"]["ate_p_value"] = round(p_value, 6)  # type: ignore
                out["estimation"]["ate_placebo_null_ci95"] = ci  # type: ignore
                out["estimation"]["null_draws"] = len(null)  # type: ignore
                out["estimation"]["cate_mean"] = round(cate_mean, 6)  # type: ignore
                out["estimation"]["cate_std_proxy"] = (round(cate_std_proxy, 6) if cate_std_proxy is not None else None)  # type: ignore
                out["estimation"]["heterogeneity"] = hetero  # type: ignore

                ate_positive = ate > 0
                ate_significant = p_value < 0.05
                nuisance_model_ready = nuisance_method != "mean_fallback"
                cate_model_ready = cate_method not in ("constant_cate_fallback", "no_data")
                causal_ml_passed = compute_causal_ml_pass(
                    ate_positive=ate_positive,
                    ate_significant=ate_significant,
                    nuisance_model_ready=nuisance_model_ready,
                    cate_model_ready=cate_model_ready,
                    heterogeneity_estimated=heterogeneity_estimated,
                )

                out["gates"]["ate_positive"] = ate_positive  # type: ignore
                out["gates"]["ate_significant"] = ate_significant  # type: ignore
                out["gates"]["nuisance_model_ready"] = nuisance_model_ready  # type: ignore
                out["gates"]["cate_model_ready"] = cate_model_ready  # type: ignore
                out["gates"]["heterogeneity_estimated"] = heterogeneity_estimated  # type: ignore
                out["gates"]["causal_ml_passed"] = causal_ml_passed  # type: ignore

                out["status"] = "ok"  # type: ignore
                if causal_ml_passed:
                    out["reason"] = "causal_ml_estimated_strict_pass"  # type: ignore
                elif not (nuisance_model_ready and cate_model_ready):
                    out["reason"] = "causal_ml_fallback_not_eligible_for_pass"  # type: ignore
                else:
                    out["reason"] = "causal_ml_estimated_but_gate_not_met"  # type: ignore

    with OUT_FILE.open("w", encoding="utf-8") as f:  # type: ignore
        json.dump(out, f, ensure_ascii=False, indent=2)  # type: ignore

    print("=== Causal ML (Orthogonal Forest) ===")
    print(f"status: {out['status']}")  # type: ignore
    print(f"reason: {out['reason']}")  # type: ignore
    print(f"shock_days: {out['sample']['shock_days']}")  # type: ignore
    print(f"causal_ml_passed: {out['gates']['causal_ml_passed']}")  # type: ignore
    print(f"[输出] {OUT_FILE}")  # type: ignore


if __name__ == "__main__":
    main()
