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
- 规则：`shock_days >= 12`
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
- G1~G8 全通过，但 G9~G12 任一失败：最高 L2。
- G1~G12 全通过：可声明 L3。
- L4 需要额外跨时期、跨数据源复现。
