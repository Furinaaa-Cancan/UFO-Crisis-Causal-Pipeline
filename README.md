# 外星人事件关联分析 —— 美国政治危机与UFO新闻的时间关联研究

## 项目简介

本项目系统分析了一个历史规律：**美国每当爆发重大政治丑闻或危机时，往往在数天至数周内，UFO/外星人相关新闻便会大量出现在主流媒体上**。

这一"转移注意力"模式在媒体研究和政治分析领域被称为 **"Wag the Dog"（摇尾巴的狗）效应**，最早源于1998年克林顿莱温斯基丑闻期间的导弹袭击事件。

---

## 已记录的核心案例（2017–2026）

| 政治危机 | 日期 | 间隔 | UFO/外星人事件 | 日期 |
|---------|------|------|--------------|------|
| 弗林认罪（穆勒调查高峰）| 2017-12-01 | **+15天** | NYT头版曝光五角大楼UFO项目AATIP | 2017-12-16 |
| 乌克兰通话丑闻 | 2019-07-25 | **-54天** | Facebook"冲进51区"病毒式传播 | 2019-06-27 |
| 弹劾调查启动 | 2019-09-24 | +6天后 | 海军官方确认UFO视频真实性 | 2019-09-18 |
| COVID危机+特朗普遭批 | 2020-03-01 | **+57天** | 五角大楼正式解密UFO视频 | 2020-04-27 |
| 1月6日调查委员会被阻挠 | 2021-05-28 | **+28天** | 五角大楼发布UAP报告（144起不明事件）| 2021-06-25 |
| 拜登机密文件丑闻曝光 | 2023-01-09 | **+26天** | 中国气球+疯狂击落UFO媒体狂潮 | 2023-02-04 |
| 特朗普被联邦起诉（首次）| 2023-06-09 | **+47天** | 国会UFO听证：格鲁施爆料"非人类遗骸" | 2023-07-26 |
| 爱泼斯坦文件争议爆发 | 2025-12-01 | **+80天** | 特朗普下令公开所有外星人/UFO文件 | 2026-02-19 |

### 经典历史参照
- **1998-08-17** 克林顿电视认罪（莱温斯基丑闻）→ **+3天** → 对阿富汗/苏丹发动导弹袭击（"摇尾巴的狗"）

---

## 项目结构

```
外星人事件/
├── README.md              # 本文件
├── requirements.txt       # Python依赖（requests/bs4/dateutil/lxml/rich）
├── scraper.py             # 严格真实性新闻抓取器（评分+交叉佐证+过滤）
├── causal_analyzer.py     # 因果检验器（置换检验+方向性检验+样本充分性诊断）
├── panel_pipeline.py      # 日常累计管道（抓取+审批+进度）
├── research_unified_pipeline.py # 统一研究总管道（双档+对照构建+严格评审+模型）
├── control_panel_builder.py # 对照面板构建器（topic/country 自动更新）
├── strict_reviewer.py     # 统一严格评审器（输出研究等级与闸门状态）
├── STRICT_GATES.md        # 严格闸门定义（L0-L4）
├── pre_registration.md    # 预注册草案
├── RESEARCH_STRICT_REVIEW.md # 顶刊方法对照与漏洞清单
└── data/
    ├── sources.json       # 来源配置（28个来源，权重1-3，支持 fallback_url）
    ├── events_v2.json     # 主数据集（置信度评级 + 政府主动行为标记，手工核实）
    ├── scraped_news.json  # 爬虫输出（含 source_health/source_stats/rejected_news）
    ├── causal_panel.json  # 长期面板（日度累计）
    ├── causal_report.json # 严格审批报告
    ├── panel_progress.json # 门槛进度
    ├── strict_dual_review.json # strict 双档稳定性评审
    ├── strict_review_snapshot.json # 统一严格评审快照
    ├── model_did_report.json # DID 准实验输出
    ├── model_event_study_report.json # 事件研究动态效应输出
    ├── model_synth_control_report.json # 合成控制简化输出
    ├── research_variable_dictionary.json # 变量字典
    └── control_panels/    # 对照组面板（topic/country + 来源配置）
```

---

## 来源体系（sources.json）

爬虫从 `data/sources.json` 动态加载所有来源，无需修改代码即可增删来源。

| 权重 | 含义 | 代表来源 |
|------|------|----------|
| **3** | 通讯社 / 政府官方 | AP、Reuters、NYT、WaPo、Pentagon DoD、White House |
| **2** | 主流媒体 / 专项媒体 | BBC、The Guardian、NPR、Politico、The Hill、Space.com、Defense One、ABC、CBS |
| **1** | 聚合 / 社区 | Google News（专项关键词）、Reddit r/UFOs、r/politics、r/news |

**字段说明：**
- `active: false` — 暂时禁用该来源（如 VICE 已停刊）
- `fallback_url` — 主 URL 失败时自动重试的备用地址（如 Reuters 迁移后的新路径）
- `weight` — 影响结果排序优先级，政府/通讯社来源权重最高

---

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 抓取最新新闻并交叉验证（需要网络）
```bash
python scraper.py --policy strict
```

常用参数：
```bash
# 双档严格评审（推荐）
python scraper.py --policy strict
python scraper.py --policy strict-balanced

# 放宽过滤（兼容旧参数）
python scraper.py --lenient

# 调整回看窗口 / 关联窗口
python scraper.py --lookback-days 90 --window-days 45

# 调整并发
python scraper.py --max-workers 8
```

> 爬虫会并发抓取 `sources.json` 中所有 `active: true` 的来源，并输出：
> - 各来源抓取摘要表（含成功/失败状态）
> - 来源健康审计（可用率/fallback命中/错误类型/替代建议）
> - 真实性过滤摘要（原始条目数、过滤后条目数、主要拒绝原因）
> - 关联时间窗口内的危机↔UFO关联对（默认60天）
> - 与 `events_v2.json` 历史数据集的交叉验证结果
> - 结果保存至 `data/scraped_news.json`（包含通过条目与拒绝条目）

策略说明：
- `strict`：最严格，危机需满足标题级硬信号 + 国家级政治语境，噪声最低
- `strict-balanced`：高严格度，保留硬信号约束但适度放宽部分标题门槛，避免样本清零
- `lenient`：用于探索，不建议直接作为研究结论依据

### 3. 添加新来源（无需改代码）
编辑 `data/sources.json`，添加一个条目即可：
```json
{
  "name": "你的来源名称",
  "url": "https://example.com/rss",
  "type": "rss",
  "category": "ufo",
  "weight": 2,
  "active": true
}
```
支持的 `type`：`rss`（含 Atom）、`reddit_json`

### 4. 运行因果检验（严格建议）
```bash
python causal_analyzer.py
```
输出将给出：
- 历史配对数据的时间结构显著性（置换检验）
- 实时样本是否满足因果推断所需样本量
- 审慎结论（因果 / 仅相关 / 证据不足）

### 5. 启用长期面板（升级版）
`causal_analyzer.py` 现在会把当前 `scraped_news.json` 自动写入 `data/causal_panel.json`（按日期+policy去重）。

常用命令：
```bash
# 默认：更新面板 + 运行因果检验
python causal_analyzer.py

# 仅分析，不写面板
python causal_analyzer.py --no-update-panel

# 指定面板分析策略与最低样本门槛
python causal_analyzer.py --panel-policy strict-balanced --min-panel-days 180 --min-panel-shocks 12
python causal_analyzer.py --panel-policy strict-balanced --min-panel-observed-ratio 0.85

# 严格审批闸门：未通过则返回非零退出码（适合CI）
python causal_analyzer.py --fail-on-reject
```

面板检验包含：
- lead-lag 相关（危机是否领先UFO热度）
- 置换检验（随机冲击日对照）
- 安慰剂检验（控制议题波动）
- 有效观测覆盖率闸门（避免“日期跨度足够但中间大量缺失”的误判）
- 审批报告自动写入 `data/causal_report.json`（含每条审批门槛通过状态）

### 6. 一键累计管道（推荐）
新增 `panel_pipeline.py`，用于把“唯一有效路径”标准化为日常流程：
```bash
# 每日标准跑法（推荐）
python panel_pipeline.py --policy strict-balanced

# 只更新进度，不重跑抓取/因果分析
python panel_pipeline.py --policy strict-balanced --skip-scrape --skip-causal

# 严格闸门模式（未通过审批则非零退出）
python panel_pipeline.py --policy strict-balanced --enforce-gate

# 指定覆盖率闸门（默认0.85）
python panel_pipeline.py --policy strict-balanced --min-observed-ratio 0.85
```
它会自动写入：
- `data/causal_panel.json`（累计面板）
- `data/causal_report.json`（审批报告）
- `data/panel_progress.json`（距离 180天/12冲击门槛的进度）
- `data/strict_dual_review.json`（strict vs strict-balanced 稳定性评审）

严格双档建议：
```bash
# 同日分别跑两档，累计可比样本
python panel_pipeline.py --policy strict
python panel_pipeline.py --policy strict-balanced
```

### 7. 统一研究总管道（严格评审版）
新增 `research_unified_pipeline.py`，把双档、严格评审、模型输出统一起来：
```bash
# 推荐：双档都跑
python research_unified_pipeline.py

# 仅更新评审和模型（不抓取）
python research_unified_pipeline.py --skip-scrape --skip-causal

# 只跑 strict-balanced
python research_unified_pipeline.py --only-policy strict-balanced

# 只跑 strict，并让模型/评审也按 strict 口径
python research_unified_pipeline.py --only-policy strict --model-policy strict
```
统一输出：
- `data/strict_review_snapshot.json`（当前研究等级 L0-L4）
- `data/model_did_report.json`
- `data/model_event_study_report.json`
- `data/model_synth_control_report.json`

> 注意：模型脚本在样本不足或对照组缺失时会明确返回 `pending/blocked`，不会伪造因果结论。

### 8. 对照组面板构建器
`control_panel_builder.py` 会自动更新：
- `data/control_panels/control_topics.csv`
- `data/control_panels/country_controls.csv`
- `data/control_panels/control_panel_build_report.json`

手动运行：
```bash
python control_panel_builder.py --lookback-days 120
```

### 9. 逻辑回归测试
```bash
python -m unittest discover -s tests -p 'test_*.py'
```
当前测试覆盖：
- `run_day_only` 面板口径是否生效
- 双档评审是否排除旧口径行
- 严格研究等级（L3/L4）判定是否满足闸门逻辑

---

## 抓取真实性机制（v2 严格模式）

`scraper.py` 默认启用严格模式，核心原则是：**低可信来源必须被高可信来源交叉佐证**。

### 评分维度
- 来源权重：`weight=3/2/1` 对应高/中/低基础分
- 来源类型：RSS高于社区源（Reddit）
- URL质量：HTTPS、有效域名、去追踪参数规范化
- 文本质量：标题长度与垃圾标题模式检测
- 时间有效性：日期可解析、非未来、且在回看窗口内
- 主题相关性：UFO/危机关键词命中
- 制度动作信号：法院裁决/听证/行政命令/起诉等“官方动作关键词”命中

### 交叉佐证规则
- 同一事件按标题指纹聚类（claim fingerprint）
- 统计同一事件的“可信来源数”（`RSS 且 weight>=2 且非聚合源`）
- 可信来源越多，真实性分数越高
- 通过审核后会按 claim 聚合为“事件级记录”，避免同一新闻多来源重复计数

### 严格过滤门槛（默认）
- 分数低于阈值直接拒绝
- `weight=1`（聚合/社区）必须有至少 2 个可信来源佐证
- Reddit与聚合源（如 Google News）若缺少可信佐证会被拒绝
- 日期解析失败、未来日期、超出窗口条目直接拒绝
- 关联匹配启用限额：每个危机最多匹配少量高分UFO事件，每个UFO事件也限制复用次数
- 危机事件必须满足“标题级危机/官方动作信号”（如 indictment/impeach/scandal/supreme court ruling 等），仅在摘要命中不通过
- 危机事件标题需出现国家级政治行为体（如 president/white house/congress/doj/trump/biden）
- 评论/专栏与混合串联标题（如 `... And, ...`）会被直接剔除

---

## 统计发现

- **平均间隔**：政治危机爆发后约 **35天** 内出现重大UFO新闻
- **30天内命中率**：约 **50%** 的案例在30天内出现UFO媒体热潮
- **60天内命中率**：约 **87%** 的案例在60天内出现UFO媒体热潮
- **总体趋势**：政治危机严重程度越高，随后UFO新闻的媒体热度也越高（正相关）

---

## 重要声明

本项目为**信息整理与时间线分析**工具，所有数据来源于公开新闻报道和政府文件。时间关联**不等于因果关系**。本项目旨在提供数据层面的观察，供研究参考，不构成任何政治立场声明。
