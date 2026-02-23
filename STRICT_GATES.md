# STRICT_GATES（统一严格审批闸门）

版本：v1.0（2026-02-21）

## 目标

把研究结论分成可审计等级，避免从“相关”直接跳到“因果”。

- L0: 未发现稳定关联
- L1: 仅时间关联
- L2: 稳定前导关系（预测性）
- L3: 准实验因果证据成立
- L4: 多模型一致且跨样本复现

## 核心闸门（必须全部通过才能声称 L3）

1. G1 数据覆盖闸门
- 规则：`observed_days >= 180`
- 来源：`data/causal_report.json` -> `panel.observed_days`

2. G2 冲击样本闸门
- 规则：默认 `shock_days >= 12`；当冲击源为 `events_v2_crisis_dates`（语义主轨）时，执行 `shock_days >= 5`（`EV2_MIN_DATES`）
- 来源：`data/causal_report.json` -> `panel.n_shocks`

3. G3 连续性闸门
- 规则：`observed_ratio >= 0.85` 且 `max_missing_streak <= 7`
- 来源：`data/causal_report.json` -> `panel.observed_ratio`, `panel.max_missing_streak`

4. G4 来源健康闸门
- 规则：`availability_rate >= 0.9` 且 `failed_sources <= 10%`
- 来源：`data/scraped_news.json` -> `source_health`

5. G5 双档稳定性闸门
- 规则：`overlap_days >= 30`, `shock_agreement >= 0.7`, `crisis_rel_delta <= 0.6`
- 来源：`data/strict_dual_review.json`

5b. G5b 冲击目录锁闸门（当使用 shock catalog 时）
- 规则：若 `panel.shock_source=shock_catalog_dates`，则
  `data/crisis_shock_catalog.json.catalog_signature_sha256`
  必须与 `data/crisis_shock_catalog_lock.json.catalog_signature_sha256` 一致，且 `panel_shock_catalog_key` 在目录中存在。
- 来源：`data/strict_review_snapshot.json` -> `quality.shock_catalog_lock`

6. G6 方向性闸门
- 规则：`best_lag in [1,30]` 且 `best_corr > 0.1`
- 来源：`data/causal_report.json` -> `approval.gates` / `panel`

7. G7 显著窗口闸门
- 规则：`>=2_significant_windows` 为真
- 来源：`data/causal_report.json` -> `approval.gates`

8. G8 反向排除闸门
- 规则：`reverse_not_dominant` 为真
- 来源：`data/causal_report.json` -> `approval.gates`

## 证伪闸门（L3 必须附带）

9. G9 负对照闸门
- 规则：负对照主题（体育/娱乐/气象）不出现同方向显著效应
- 输出：`data/model_did_report.json`

10. G10 安慰剂闸门
- 规则：假冲击日置换检验不过显著
- 输出：`data/model_event_study_report.json`

11. G11 模型一致性闸门
- 规则：DID / 事件研究 / 合成控制方向一致
- 输出：`data/model_did_report.json`, `data/model_event_study_report.json`, `data/model_synth_control_report.json`

12. G12 复现实验闸门
- 规则：同样代码与数据，重复运行结论不翻转
- 输出：`data/strict_review_snapshot.json`

## 判定规则

- 只要 G1~G8 任一失败：最高只能到 L1/L2。
- G1~G8 全通过，但 G9~G11 任一失败：最高 L2。
- G1~G11 全通过，但 G12（跨日复现）未通过：最高 L2（等待跨日复现）。
- G1~G12 全通过：可声明 L3。
- L4 需要额外跨时期、跨数据源复现（实现口径：`external_replication_passed=true`）。

## 关于 lead-lag 闸门（G6）的说明

- `best_lag=0`（同期相关最强）时，`lead_lag_positive` 闸门**不通过**。
- 只有正向 lag（≥1天）的相关系数严格优于 lag=0 时，才视为"危机领先UFO"的前导信号。
- 当前数据（lag0_corr=0.591 > best_positive_corr=0.550）表明同期相关主导，不构成因果前导证据。
