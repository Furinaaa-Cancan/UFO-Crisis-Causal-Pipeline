# control_panels

该目录用于放置“反事实/对照组”面板数据。

## 文件

1. `control_topics.csv`
- 列：`date,topic,count`
- topic 建议包含：`sports`, `entertainment`, `weather`, `technology`。

2. `country_controls.csv`
- 列：`date,country,ufo_policy_news,crisis_index`
- country 至少 3 个国家，日期覆盖建议 >= 180 天。

## 说明

- 这些文件是统一因果识别（DID/Event Study/Synthetic Control）的必需输入。
- 当前仓库内模型脚本会在缺失时输出 `pending` 或 `blocked`，不会伪造因果结论。
