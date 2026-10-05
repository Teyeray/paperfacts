# PaperFacts 第三路视觉证据设计

> 历史方案，已由 [2026-10-05 自动补证设计](2026-10-05-auto-visual-evidence-design.md) 替代。
> 下文的手动触发、先铺设 Web/Excel 集成和转写匹配方案不再作为实施依据。

状态：行动方案评审稿；用户已认可方向，尚未授权代码实施。

本设计以 PaperFacts `e126e1d1aeea4046511739e1296be3ae06201aa2` 和 Paper_Reader
`12c814712d06d2c9689572c52e5b2b3f2064cc41` 的只读审查为基线。
实现前重新核对 HEAD、工作区和接口，不能把本稿中的现状当作未来版本事实。

## 目标与已确认范围

通过原 PDF 的像素转写，为现有两路事实提供可回查的第三路辅助证据，暴露解析错误和
尚未核实的归属关系；逐步发现两路共同遗漏的区域。这里的“第三路”是提取方法的增加，
并非第三份独立科学来源。

- 用户已同意：先接通现有视觉复核核心，再增加独立区域发现；不直接做三路多数投票。
- 用户已确认：首轮真实验收沿用仓库现有领域与样例。以 TCO 的真实 gold 为主；
  catalysis/battery_cathode 的示例 profile 用于工程兼容检查。
- 推荐并纳入本稿：第一期手动触发、只出旁路报告，保留原两路和主表结果；第二期产生
  独立发现的证据候选，也不自动填主表。
- 尚待用户答复：沿用现有外部模型接口还是只用本地/内网模型；原 PDF/既有运行数据的位置。
  本稿暂按现有 OpenAI-compatible 接口可配置设计，不启动模型、不传输 PDF。部署选择只影响
  client 配置与真实验收，不阻塞纯规则、存储和 fake-client 集成的开发。

## 已核实的现状

1. `workflow.run_document` 已连接 parse、figures、extract、compare、export；没有 validate stage。
2. `validate.py` 已有盲转写提示词、区域构造、值匹配、结果模型；`keys.py` 已有独立 validation key。
   `vlm.enabled=true`、`fill_blanks=true` 目前没有相应生产执行链，接通时不能使旧配置突然触发付费。
3. `figures` 已可读取图表，但使用 parser 的候选框，输出近似读数且不加入主表。
4. `CropStore` 已保存页码、归一化坐标、DPI、图片 SHA；现有文件名未区分 `max_pixels`，需要在
   接通复核时验证并修复由此产生的图片复用问题。
5. Paper_Reader 可借鉴 PDF 文本层、图注/图形几何归属、图片质量门槛和来源绑定；
   OCR 是占位，多 panel 主要合并成整图，不能作为现成子图识别引擎。

## 第一期 数据流与证据语义

```text
现有两路抽取和比较
  → 确定本次核验目标
  → 从原 PDF 裁剪引用区域及必要上下文
  → 图片质量检查
  → 视觉模型盲转写（不传候选数值、原解析文本或比较结论）
  → 现有 matcher 检查数值字符串
  → 独立核验报告 → Web 原图回查 / Excel 独立工作表
```

第一期只核验已存在候选值；`missing` 指只有一路有值。两路都没有的值或样品不是第一期
的检索目标，不能宣称获得全篇独立覆盖。

- 复用 `ValueValidation.verdict`，但 `confirmed` 的显示文字为“数值见于转写”，
  `contradicted` 为“转写未找到该数值”，后者不等于论文否定该值。
- 每条结果另带 `attribution_status=not_checked`，明确样品、列头、单位和条件归属尚未完成
  结构核验。第一期不输出“事实已确认”或“自动采纳”。归属矛盾可由用户对照原图复查。
- `illegible`、`not_checked`、`error` 分别表示不可辨读、未检查、请求/响应失败，不能混为否定票。
- 坏 JSON、被 token 上限截断的回答不得退化成可用于确认的自由文本；复用 lenient parser 时
  必须先作响应完整性检查。
- 多页共同支撑一个候选的情况先标 `not_checked/multi_page_evidence`，展示所有原引用；
  不能只检查第一页就给全条核验通过。跨页表格结构核验推后。
- 原 `LaneExtraction`、`ComparisonReport`、`DatasetPayload` 内容和完成判定保持原语义。
  `figures` 近似读数仍独立展示。核验报告中的两路目标共用同一裁图时只产生一次读图证据。

## 第一期 触发、范围与失败

- 唯一触发意图来自请求级 `validate` 或 `force_validate`。普通上传、run、批处理、run-all、
  普通 force 和离线导出均不隐式触发；`force_validate` 隐含 `validate`。
- CLI 首期提供 `run --validate`、`run --force-validate`；Web 在单篇详情提供“视觉复核”和
  “重新复核”，复用现有文档作业队列。第一期不增加自动批量核验。
- 旧 `vlm.enabled` 兼容读取，但不能单独触发。显式请求建立本次有效设置；
  `fill_blanks` 在 check 模式固定不执行，报告记 `mode=check`、`fill_performed=false`。
  文档清楚说明旧开关在第一期的兼容语义。
- 沿用 `disputed/tables/all` 选择策略，默认 `tables`。按 conflict、ungrounded、ambiguous、
  missing、其余 table/all 排优先级，再按 page、bbox、target_id 稳定排序。
- 推荐默认每文最多 20 个不同裁剪区域，`vlm.max_regions_per_document` 可配置；同区域多值
  复用一次转写，超额目标记 `not_checked/budget_limit`，不得默默丢弃或显示全部核实。
- 沿用 200 DPI、2,000,000 像素上限、现有超时配置及全局请求并发限制。每个模型请求最多两次
  尝试；只重试可重试失败。记录区域数量、调用数、耗时和接口提供的 token 用量，不猜测费用。
- 单区域失败继续其他区域；主链仍完成。未请求显示 skipped；有失败显示独立 stage failed，
  已读部分保留。超过预算、缺 PDF、无引用等须显示覆盖率和原因，而非完整核验。

## 第一期 存储与缓存

- 冻结的 `ValidationReport` 包含 schema version、完整 PDF SHA、profile、三个 key、两路 parse
  hash、实际核验输入摘要、模型设置、结果、未检查原因、覆盖计数、complete 与运行统计。
- 核验输入摘要绑定当前候选的 backend、entity、owner、field、raw value/unit、condition、
  source ids 和比较状态；强制重新抽取产生不同答案时，不能因为 extractor key 相同而复用旧报告。
- 目标身份含 entity 和引用；同名但不同实体的样品、同值不同条件、同值不同出处不会互相覆盖。
- 图片使用现有 `RegionCrop`；页码统一 0-based，坐标统一 `NormalizedBBox`，经 `pdf.py` 的
  PDFium 锁渲染。缓存身份包括所有影响像素的参数；旧 crop 可保留，不能被错误命中。
- 报告路径由 `storage.py` 定义为
  `validations/<profile>/<extractor_key>.<comparison_key>.<validation_key>.<input_digest>.json`。
  原文档目录已由 PDF hash 隔离；所有文件原子写入。
- `validation_key` 按实际 check-only 参数计算，writer、reader、Web 与导出共用入口。
  核验 prompt/预算改变只使核验结果失效；若必须改共享渲染模块造成 figure key 合理变化，
  明确记录并测旧图结果的 stale 行为，不承诺该类修改完全不影响 figure key。
- 旧报告只可明确标记为 stale 供回查；不得显示成当前结果。重试优先复用有效成功响应，
  离线缓存不足返回可解释状态，绝不转为联网。

## 第二期 独立发现与候选

先做有边界的评估，再决定生产接入。用原 PDF 的文本层、图注、矢量/内嵌图像信息发现区域，
保留候选来源为 parser-derived 或 native-PDF，分别报告覆盖率。

优先复用 PDFium/Pillow 和纯几何规则；确需 Paper_Reader 的 PyMuPDF 对象 API 时，先在独立
验证脚本比较增益与依赖成本，再按现有隔离 runner 约定设计接入。不要导入整个 Paper_Reader、
Zotero 工作流、arXiv 下载器或另一套存储系统。

生产化必须覆盖：图注归属、坐标变换、旋转/CropBox、轴与 legend 完整性、低质量弃权；
扫描页文本层缺失要明确降级。增加原文中印刷数值的候选时保留样品/列头/单位/条件出处；
不能唯一映射到现有样品的候选留在未归属区。曲线估读继续单列，不自动回填主表。

子图语义切分、完整表格结构识别、自动仲裁/填表、全篇扫描 OCR 和新领域 profile 不在本次
第一期交付内；第二期评估用失败案例决定是否需要它们，不预先承诺全做。

## 验收边界

- 工程：无外部请求的真实裁图+fake client 集成，原主表与两路语义不变；失败、缓存、作业竞争、
  旧 API、跨 profile、缺 PDF 都有测试。人工构造的错行例即使数值命中，归属仍必须显示未核验。
- 真实：沿用 11 篇 TCO gold；先恢复与 gold SHA 相符的 PDF。现有 4 篇中的 18 个 figure_only
  单元只是重新人工检查的线索，不自动视为新的像素 gold。
- 第一阶段真实评估选不少于 5 篇（至少 1 篇负样本）、30 个可审计区域，包含表格、指数单位、
  重复数值/错行、正文图表矛盾。其中至少 10 个区域留作不调参检查集；缺少材料则报告实际数量。
- 原 `eval/score.py` 保留原口径：figure_only 不属于 required recall。旁路复核不会提高主表分数，
  应测转写准确率、错误证据未被警示的比例、弃权率、覆盖率、原图定位和每区域成本。
- 第二期另测独立发现召回、图注/样品/字段/单位/条件联合正确率、候选精度和增量覆盖，分开统计
  印刷值和曲线估读。没有基线前不承诺准确率数字；真实结果不足以支持推广时保留手动试验状态。
- 自动采纳数必须为 0；来源定位和无副作用是硬门槛。真实评估结果只对指定版本、模型和样本负责。

## 与实施的关系

行动清单见 [实施计划](../plans/2026-10-02-visual-evidence.md)。本稿和计划待用户评审，
当前不安装依赖、不改运行配置、不触碰生产数据、不执行模型、不提交或推送代码。
