# 设计镜像：查看器放大 + DeepSeek 浅色系 + 文档页 Tab IA

Date: 2026-10-06
Status: Approved（全部决策见决策记录）；实现计划与代码草稿见
[`.pi/plans/2026-10-06-viewer-zoom-light/plan.md`](../../../.pi/plans/2026-10-06-viewer-zoom-light/plan.md)（含令牌映射表、
e2e 重钉行号清单、scout 上下文链接）。本文件是该设计在 `docs/superpowers/specs/` 的镜像，按 AGENTS.md 目录惯例存放。

## Intent

用户提出三项相关前端改动：(1) PDF 查看器增加放大功能；(2) 配色改为浅色系，参考 DeepSeek 主页；(3) 文档页中间栏信息
架构重排——结果表作为主视图（默认态），图中读数、事实对照、样品记录成为三个可选 tab 页。另有连带要求：查看器窗格的
收起按钮向左上角边栏（rail）收起按钮的图标按钮方言看齐。交付形态：3 个独立 PR，按序合入。

## 决策记录（用户已拍板）

| 决策点 | 结论 |
|---|---|
| 放大交互 | 混合式：−/百分比/+ 控件 + Ctrl/Cmd+滚轮；≤200% 纯 CSS 宽度缩放，>200% 切 dpi=220 重渲染；422/失败静默回退 CSS 缩放 |
| 缩放持久化 | `getState().zoomIndex`（抗重克隆）+ localStorage `paperfacts.viewer-zoom`（跨会话） |
| accent 色 | **#3964fe**（DeepSeek 聊天端令牌；白字对比 4.7:1 过关，主页 #4d6bfe 仅 4.3:1 压线） |
| 深色模式 | 保留三档切换，仅 accent 锚点与新浅色同源微调（#5686fe 方向，实现时实测） |
| KPI 瓦片 | 留在「事实对照」tab 顶部（模板中本就在 evidence 区块顶部，零移动） |
| 处理日志 | 保持 details 折叠块，置于 tab 面板区之后（任何 tab 拉到底可见）；运行中自动展开照旧 |
| Tab 命名 | 用户原词：结果表（默认）/ 图中读数 / 事实对照 / 样品记录 |
| Tab 状态 | 纯 UI 状态（`state.tab`），不进 URL、不进 localStorage；切文档重置为结果表，任务完成重载保留 |
| 收起按钮 | rail-toggle 同款 32×32 ghost 图标按钮（panel-right SVG、动态标签、aria-expanded），位置仍在窗格头部 |
| 拆分 | 3 个 PR：① 放大+收起按钮图标化（PR #49）→ ② 浅色令牌 → ③ Tab IA |

## Behavior（摘要）

- **PR1 放大**：查看器窗格头部 `− [125%] +` 控件；阶梯 `[100,125,150,175,200,250,300,400]%`；Ctrl/Cmd+滚轮同步缩放；
  点百分比回 100%；缩放在任务轮询/完成重渲染后保持并跨会话记住；>200% 时 img 切 `?dpi=220` 保持清晰；dpi 请求失败
  （422/网络）静默回退 CSS 缩放一次，防循环；视位按滚动比例保持；`pdfAvailable=false` 时不渲染控件；缩放不引起窗格外
  页面滚动；横向滚动收在新 `.page-viewport`（`overflow:auto`），`.viewer`/`.page` 的 `overflow:hidden` 保留不动。
- **PR2 配色**：全站浅色令牌换 DeepSeek 系（底 #f9f8f8、alpha-black 发丝线、圆角 10/8、环形+弥散阴影、accent
  #3964fe）；派生令牌自动跟随；深色三档照常；`--ink-3` 三表面对比 ≥4.5:1，accent-as-text 审计（必要时
  `--accent-strong`）；lane/状态色/字体栈不动；无 webfont。
- **PR3 Tab**：中间栏顶部 tab 条（role=tablist），默认「结果表」，其余三面板 `hidden`，纯显隐无请求；fact 深链自动激活
  「事实对照」；空结果格点击切「样品记录」并滚动；处理日志在面板区下方，运行中自动展开不打断当前 tab；任务完成重载保留
  当前 tab，切文档回结果表；URL hash 仍只含 fact（不新增 `#/section` 语法）；sections.js 的滚动侦测整体删除，
  重写为 tabs.js。

完整代码草稿、令牌映射表、e2e 重钉清单与验证命令见计划文件；ISC（ISC-1..ISC-20、ISC-A-1..4）亦在计划文件的
「Ideal State Criteria」一节，此处不重复。

## Implementation status

- PR1（放大 + 收起按钮图标化）：#49，已合入 main（squash 38526a3）。
- PR2（浅色令牌）与 PR3（Tab IA）：待实施，按计划顺序执行。
