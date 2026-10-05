# PaperFacts 第三路视觉证据行动方案

> 历史方案，已由 [2026-10-05 自动补证行动方案（HTML）](2026-10-05-auto-visual-evidence.html) 替代。
> 保留本文用于追溯；请勿照此任务列表实施。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or
> superpowers:executing-plans to implement this plan task-by-task after user approval.
> Steps use checkbox syntax; an unchecked item is planned work, not a completion claim.

**Goal:** 接通手动、可回查的 PDF 视觉复核，随后评估独立区域发现对共同遗漏的增益。

**Architecture:** 保持 MinerU/PaddleOCR-VL 的两路抽取与比较，追加旁路 validate stage。
复用 `validate.py`、`CropStore`、`VisionClient` 和现有作业队列；独立报告与缓存只服务证据复核。
第一期不自动填表，第二期先测独立定位增益再生产化。

**Tech Stack:** Python `>=3.13,<3.14`、Pydantic、现有 PDFium/Pillow、OpenAI-compatible client、
FastAPI、原生 JavaScript、openpyxl、pytest、ruff。第一期不新增产品依赖。

**Spec:** [第三路视觉证据设计](../specs/2026-10-02-visual-evidence-design.md)。

**状态：** 评审稿，仅文档工作；当前代码基线 `e126e1d`，尚未实施。用户已确认首轮沿用仓库
现有领域与样例。模型部署方式、PDF/已有运行数据位置待回复，按设计中的明确边界处理。

## Global Constraints

- 先读根 `AGENTS.md` 与 `CLAUDE.md`；新目录先记录结构、命名和清理约定。
- Python `>=3.13,<3.14`；主包不导入 MinerU/PaddleOCR；源码保持扁平，profile 相关内容不硬编码。
- 页码 0-based，坐标 `NormalizedBBox`；PDFium 调用只经 `pdf.py` 的进程锁。
- 文件路径和原子写入统一经 `storage.py`；模型相关改变进入其对应缓存 key。
- 第一期 mode=check、fill_performed=false，attribution_status=not_checked；自动采纳数为 0。
- 显式 validate/force_validate 才触发，旧 vlm 开关、普通 force 和 run-all 不能隐式增加调用。
- 不更改两路数据与主表，视觉阶段失败不改变主链完成状态；读取、导出均无模型副作用。
- 私有 PDF/响应/裁图放忽略目录，gold 必须注明人工来源；模拟测试不能代替真实评估。
- 删除、密钥/.env/CI/CD、数据库迁移、push/rebase/hard reset、全局安装与生产发布均须单独授权。

## Review Focus

1. 同一个数字在表格另一行出现：只给转写命中，绝不证明样品/列头归属。任务 1、4 覆盖。
2. 同 key 强制抽取却换了候选，或同名样品属于不同 entity：报告不能串用。任务 1、2 覆盖。
3. 同 bbox、DPI 改 max_pixels，旋转页或非零 CropBox：实际图像和定位必须对应。任务 2 覆盖。
4. 当前配置两个 true、普通上传/run-all/force、旧 API：新增模型与 fill 调用必须为 0。任务 3、4 覆盖。
5. 坏 JSON/截断/超时/缺 PDF/多页证据：不伪造成功，已有主表仍可读取和导出。任务 2、3、4 覆盖。

## 第一阶段交付总览

| 顺序 | 交付物 | 验收点 | 依赖 |
|---|---|---|---|
| 0 | 规则、设计与运行基线 | 约定先于实现；基线和输入明确 | 本轮文档 |
| 1 | 核验目标与报告合同 | 状态含义准确，无跨实体覆盖，无自动采纳 | 0 |
| 2 | 可重放的裁图与转写链 | 图片/报告正确失效，坏响应不确认 | 1 |
| 3 | CLI 与实际工作流接通 | 手动触发，失败隔离，主表不变 | 2 |
| 4 | Web/Excel 证据展示 | 可回看原图，旧接口兼容，导出零调用 | 3 |
| 5 | 回归与真实试点评估 | 工程报告和真实准确率报告分开 | 4、PDF、模型 |
| 6 | 独立区域发现评估 | 用净增益决定生产接入范围 | 5 |

## Task 0 规则和基线

**Files:** `AGENTS.md`、本设计/计划；实施时更新 `README.md` 的 vlm 节与 `CLAUDE.md`。

- [x] 建立 `AGENTS.md`，保存设计和行动方案；本项不表示实施已获批准。
- [ ] 实施开始前重新记录 HEAD、工作区、现有测试状态及实际 Settings；日志不得包含凭据。
- [ ] 依照隔离工作约定准备开发 checkout；以这次审查基线为参考，处理新增代码差异后再实施。
- [ ] 先更新文档：显式请求、check-only、旧开关兼容语义、旁路数据和目录生命周期。
- [ ] 先运行 `uv run pytest`；失败应区分基线问题，不能用跳过/改快照掩盖。按环境需要遵循项目
      `uv sync --group dev`，不安装全局依赖。

## Task 1 目标选择和报告合同

**Files:** 修改 `src/paperfacts/validate.py`、`tests/test_validate_core.py`；新增
`tests/test_validation_report.py`。`validate.py` 保持纯规则，不放网络或存储编排。

**Interfaces:** 新增冻结的 `ValidationReport`；新增
`select_validation_targets(report: ComparisonReport, lanes: Mapping[Backend, LaneExtraction],
artifacts: Mapping[Backend, ParsedArtifact], profile: DomainProfile, *, policy: ValidationPolicy)
-> tuple[ValidationTarget, ...]`。扩展 target/result 的 entity、target_id、归属未核验状态；
旧 `value_key` 的兼容调用仍可工作，但新报告不以它作为唯一身份。

- [ ] 写失败用例：policies 选择正确、冲突优先；同名跨 entity、同值不同条件/引用不覆盖；
      单路 missing 可检查，两路共同没有值不伪造目标。
- [ ] 写失败用例：另一行的相同数值只能得到“数值见于转写”，归属仍 not_checked；
      unreadable/error/not_checked 不计确认；多页共同引用不退化为首张通过。
- [ ] 运行 `uv run pytest tests/test_validate_core.py tests/test_validation_report.py -q`，确认测试
      因预期缺失行为失败，再实现最小合同、稳定身份、选择和覆盖计数。
- [ ] 报告绑定 PDF、profile、parse hash、候选与比较状态的 input_digest；明确 complete 与
      covered/unchecked/failed 的差别，保存实际 mode 和 fill_performed。
- [ ] 重跑上述测试，检查通过后独立 review 本任务的状态语义，再进入存储。

## Task 2 裁图、盲转写与重放

**Files:** 修改 `src/paperfacts/crops.py`、`storage.py`、`keys.py`、`config.py`；
新增扁平 `src/paperfacts/validations.py`、`tests/test_validations.py`；扩展
`tests/test_crops.py`、`tests/test_keys_validation.py`、`tests/test_keys_unhashed.py`。

**Interfaces:** `validations.py` 提供
`read_document_validation(document: DocumentInput, settings: Settings, profile: DomainProfile,
lanes: Mapping[Backend, LaneExtraction], report: ComparisonReport, client: VisionClient,
*, force: bool = False) -> ValidationReport`。
内部加载当前 artifacts、校验关联，调用 Task 1 目标选择；复用 `CropStore.crop` 和
`VisionClient.complete_vision`，按设计路径存储，不创建平行 LLM cache。

- [ ] 先测同页同区域多个目标只读一次；同区域换 max_pixels 得到正确新图片；每张 crop 的 hash
      与模型请求完全一致。用生成的含文字/图形 PDF，不能只用空白页证明裁剪正确。
- [ ] 先测相同 key 的新候选/parse 不命中旧报告；profile、model、prompt、预算变化正确失效；
      转写 prompt 不含原候选数值、单位答案、OCR 文本或比较结论。
- [ ] 先测截断、坏 JSON、空白/极小裁图、超时、离线 miss、超 20 个不同区域；按设计拒绝
      不完整回答，并逐项记录失败/未检查原因。检测阈值须在代码常量/配置和测试中一一对应。
- [ ] 运行 `uv run pytest tests/test_crops.py tests/test_keys_validation.py tests/test_validations.py -q`
      观察预期失败，再实现裁图身份修正、质量检查、去重、预算、最多两次请求尝试和原子报告。
- [ ] `storage.py` 新增 validation 报告路径；有效 settings 固定 check-only，key 不记未执行的 fill。
      新增 `vlm.max_regions_per_document=20` 及对应 `PAPERFACTS_VLM_MAX_REGIONS_PER_DOCUMENT`。
- [ ] 更换 crop 缓存身份时保留旧文件，旧文件不得错误复用；只读显示旧结果时明确 stale。
      共享渲染代码影响 figure key 时写明原因并验证，不通过重录无关快照消除差异。
- [ ] 重跑上述测试及 `tests/test_keys_unhashed.py`，确认核验规则/预算不改变两路抽取与比较 key。

## Task 3 接通单篇工作流和 CLI

**Files:** 修改 `src/paperfacts/workflow.py`、`cli.py`、`stored.py`、`__init__.py`、`README.md`；
新增 `tests/test_workflow_validation.py`、`tests/test_cli_validation.py`；扩展 `tests/test_workflow_run.py`。

**Interfaces:** `run_document` 追加关键字参数 `validate: bool = False`、
`force_validate: bool = False`；`PipelineResult` 增加可选 `validation` 报告。
新增 `build_validation_client(settings: Settings) -> OpenAICompatibleClient`，读取 vlm 配置，
复用现有 credential 解析与全局并发；无需新增 secret。

- [ ] 使用 shipped 的两个 true 配置写回归：普通 run、force、batch/run-all 均 0 次核验和 fill；
      `--validate` 才调用；`--force-validate` 只刷新核验证据，不强迫重跑解析/抽取/figures。
- [ ] fake pipeline 测核验发生在初步 compare 后、导出前；未请求 skipped；失败/缺 PDF 不使
      原主链失败，部分证据保留；old parse-only 运行仍可导出。
- [ ] 比较启用前后的 LaneExtraction、ComparisonReport、DatasetPayload，字段和值完全一致；
      `stored.is_finished` 不依赖 validation，run-all 不因缺核验报告不断重排作业。
- [ ] 运行 `uv run pytest tests/test_workflow_validation.py tests/test_cli_validation.py tests/test_workflow_run.py -q`
      验证红灯后，以薄编排连接 Task 2，更新 stage list 和静态已存阶段状态。
- [ ] 重跑上述测试与 `tests/test_workflow_figures.py`，确保图表功能没有被复核开关联动。

## Task 4 Web 和 Excel 展示

**Files:** 修改 `src/paperfacts/web/jobs.py`、`web/app.py`、`web/documents.py`、
`web/static/document.js`、`job.js`、`api.js`、`index.html`、`workbook.py`、`batch.py`；
可新增同级 `web/static/validation.js`。扩展 `tests/test_web_jobs.py`、`test_web_app.py`、
`test_web_profiles_api.py`、`test_workbook.py` 和既有 e2e 场景；新增 `tests/test_web_validation.py`。

**Interfaces:** Job/JobBrief 增加默认 false 的 `validate/force_validate`，沿用
`POST /api/documents/{id}/run` 携带这两个参数；新增只读
`GET /api/documents/{id}/validation` 返回当前 profile 的状态和证据。
`validations.py` 提供 `shown_validation(document_id: str, settings: Settings,
profile: DomainProfile) -> ValidationView`，返回 not_requested/current/stale/partial/error
等明确展示状态；只读函数不得调用模型。workbook 增加默认空的 `validation_rows`。

- [ ] 先测旧请求省略参数行为不变；核验请求不能复用不核验的活动任务；重复显式请求复用满足
      请求的作业；profile 切换与页面迟到响应不会串显示，GET 永远不发模型请求。
- [ ] 增加单篇按钮与独立证据区：候选值、转写匹配状态、归属未核验、原图、转写全文、页码和
      定位；只显示用户可理解的状态，hash/模型参数放详情。模型/论文内容必须转义。
- [ ] 重复付费使用单独“重新复核”动作；沿用图表重跑交互。普通上传页首期不增加默认勾选项。
- [ ] Excel 增“视觉复核”sheet；原论文/样品数据单元格和旧 sheet 含义不变。export-only
      使用已有报告，有无报告均可导出，缺失报告不临时调用模型。
- [ ] 运行 `uv run pytest tests/test_web_validation.py tests/test_web_jobs.py tests/test_web_app.py tests/test_web_profiles_api.py tests/test_workbook.py -q`
      按先红后绿完成，再运行 `PYTHONPATH=src uv run --with playwright pytest -m e2e`。
      若浏览器运行时缺失，先说明实际缺口；不得把跳过计作浏览器验收通过。

## Task 5 工程回归和真实试点

**Files:** 更新 `eval/README.md`；新增 `eval/score_validation.py`、`tests/test_eval_validation.py`；
新增 fixture 目录前先在其 README 规定来源/命名。评估产物放
`output/visual-evidence/<YYYY-MM-DD>-<run-id>/`，私有材料不进 git。

- [ ] 跑 `uv run pytest`、`uv run ruff check src tests runners`、
      `uv run ruff format --check src tests runners`；相关浏览器测试通过，必要时记录覆盖率与项目
      80% 目标。对失败定位原因，不修改原 gold、跳过或放宽阈值来获得绿灯。
- [ ] 从用户提供位置只读定位原 PDF/已有解析与模型缓存；核对 gold SHA。缺 PDF 只影响真实验收，
      报告明确“工程已验，真实识别未验”，不把 fake client 结果当准确率。
- [ ] 选至少 5 篇、30 区域、1 篇负样本；覆盖错行同数、单位指数、表头/脚注、正文图冲突。
      至少 10 区域不参与调参。先人工看原图冻结 gold，再调用模型，避免答案泄漏。
- [ ] 在模型部署方式和材料位置明确后做显式小规模试跑；保存 code/gold revision、模型、配置、
      crop hash、prompt/response、时延/token 与所有失败。未知价格不估算实际账单。
- [ ] `score_validation.py` 单独输出匹配正确率、错误漏警示率、弃权、覆盖、原图定位和调用开销；
      保留旧 scorer 中 figure_only=soft 的规则，原主表分数应保持基线。
- [ ] 复核硬门槛：0 自动采纳、0 错位来源引用、0 普通流程新增 VLM 调用、失败不毁主表。
      真实效能不足则保留手动试验状态并归类错误，不能仅因测试通过宣称产品准确性达标。
- [ ] 独立 reviewer 检查最终差异与验收证据；修复后只重跑受影响检查及项目要求的终验，
      交付完整结果给用户。部署、push、私有材料清理不包含在此步骤。

## Task 6 第二期 独立区域发现评估

这是第一期之后的评估任务；结果决定后续生产改动，不提前扩充 `Backend` 或改 compare。

**Files:** 首先在 `eval/` 编写独立候选发现实验脚本与对应合成测试；如需外部依赖，记录理由、
版本与隔离方式后采用项目 runner 模式。生产文件清单由实验结论收敛，不预先新增整套子系统。

- [ ] 对同一批冻结 PDF 分别记录 parser 候选与 native-PDF 候选；独立组不得把 parser bbox
      当作唯一发现入口。报告来源和逐页扫描范围，不能把第三次读取同一框计作发现增益。
- [ ] 借鉴 Paper_Reader 的图注归属、区域合并、嵌入图像补充与质量检查；优先验证已有
      PDFium/Pillow 是否足够，必要时比较隔离 PyMuPDF 原型。没有实测增益就不新增依赖。
- [ ] 人工补充区域 gold，测候选召回、图注归属、轴/legend 完整率、误检和有效增量；
      再测印刷值的样品/字段/单位/条件联合正确率。曲线估读单独统计。
- [ ] 只把有来源、质量明确的候选接入待复核区；未知样品保持未归属，不填现有两路，不产生 agree。
- [ ] 给出生产接入或停止的明确建议及失败案例；如需子图分割、表格结构化或自动填表，
      基于这些案例另写有边界的后续方案，由用户选择范围。

## 执行建议与交付

推荐在本会话顺序实施任务 1→2→3；任务 4 的界面和评估工具可在合同固定后分工，最终由独立
reviewer 复核整体。前几项接口耦合较强，不适合一开始让多个 agent 同时改 workflow/storage/keys。

本轮交付为规则、设计、计划三个文档。下一步是用户评审方案，补充模型部署方式与材料位置；
只有明确要求开始后才实施。每个任务完成时报告改动、验证及遗留边界，不默认 commit/push/deploy。
