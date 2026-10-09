# 靶材成分（component）论文级单元格为空：原因与决策

Date: 2026-10-08
Status: 问题一（配对）已随 PR #54 实现；问题二选项 A 在分支 `feat/component-many-targets` 上实现，待负责人批准全库重抽后合入。

## 反馈

同事反馈：若干论文在「事实对照」里能看到靶材成分的标注，但结果表的靶材栏为空。举例
`Chen et al. - 2025 - Indium-less and indium-free … sputtered tin oxide` (`8977655673fa6d9a`)、
`om0035 59..65.pdf`（AZO/Ag/AZO、ZnO/Ag/ZnO）。

PR #54 最初的假设是模型把论文级字段放进了样品里，比较与数据集阶段又把它过滤掉。审阅指出该路径在真实流水线里
不可达：清洗层（`records.response_to_records`）已经把放错层级的字段丢弃并记入 `dropped`；passage 模式（生产默认）
的字段问题本身决定层级，论文级答案从不进样品。本文记录在生产数据上核实到的真实原因。

## 证据（生产 `data/docs/8977655673fa6d9a`，抽取 `db19a50e70fe`，比较 `5aa6f20197e0`）

两路的 `facts/*.json` 都把 `component` 正确记在 `paper.fields`，`dropped` 里没有它：

| 通道 | value_raw | unit_raw | condition |
|---|---|---|---|
| MinerU | `In2O3:SnO2 = 90:10` | `wt%` | — |
| MinerU | `Sb2O5:SnO2 = 5:95` | `wt%` | — |
| PaddleOCR-VL | `In2O3:SnO2 = 90:10 wt%` ×2 | `wt%` | `ITO ceramic sputtering target` / `…(ITO layer of the bilayer)` |
| PaddleOCR-VL | `Sb2O5:SnO2 = 5:95 wt%` ×2 | `wt%` | `ATO ceramic sputtering target` / `…(ATO layer of the bilayer)` |

比较报告：6 条 `paper/component`，全部 `missing`（2 条 only in mineru，4 条 only in paddleocr_vl）。
数据集质量行：`component` → `multiple_conditions`，「同一解析通道记录了多种测量条件，无法唯一确定」，`paper_row.component = null`。

在 `origin/main` 代码上用这两份 lane 文件离线重放 `compare_lanes`，结果与生产一致。

## 两个叠加的问题

### 问题一：单位写在文本里与写在旁边，比较不配对（代码缺陷）

`TextRules.compare` 只比 `value_raw` 的文本。PaddleOCR-VL 把 `wt%` 写进了引文，MinerU 只写在 `unit_raw`，于是
`In2O3:SnO2=90:10` ≠ `In2O3:SnO2=90:10wt%`，第二阶段「相等值跨条件配对」也配不上。配置里该字段的示例本身就是
`'ITO 90:10 wt%'`，模型两种写法都会出现。

**决策：修。** 一个字符串，所有地方共用：`FieldValue.quote`（`records.py`）给出「引文 + 单位，单位只写一次，空白折叠」——
引文里已经写着该单位（`text.has_unit`：按 `loose_key` 的折叠判断，所以 `wt.%`、`wt %` 都算 `wt%`；在文本中间也算，`2.5 at% fluorine` 带着 `at%`；字母紧贴的不算，`ITO 10 nm` 不带单位 `m`）就原样保留，否则把 `unit_raw` 接在后面。
`TextRules.compare`、`TextRules.cell`、`TextRules.prefer`、`decide` 的审计备注、`compare._set_pairs` 与
`decide._elements`（列表字段的元素键与元素文本）读的都是这同一个属性，因此比较与单元格不可能对同一对值给出不同判断；
单位不同（`wt%` 对 `at%`）自然是不同文本，不需要单独的分支。重放结果：2 条 `agree`（备注「conditions worded
differently」）+ PaddleOCR-VL 重复引文 2 条 `missing`。

一处行为变化需知悉：一路写了 `unit_raw`、另一路没写（`ITO 90:10`+`wt%` 对 `ITO 90:10`+无）在 main 上判 `agree`，
现在判 `conflict`，因为两路的单元格本就是两个不同的字符串；生产数据里尚未见到这种情况。

`normalize.py`、`kinds.py`、`compare.py`、`decide.py` 均在抽取或比较指纹内，合入后所有已存抽取与比较换名；LLM 缓存按
请求载荷命中，离线重放零 miss 即可重新推导。

### 问题二：单值字段遇到多靶材（选项 A，已实现，待批准）

`component` 在 `profiles/tco.json` 里曾是 `cardinality: one`（默认），`description_zh` 明说「保留唯一组成文本，不拆选多个靶材」。
这篇论文用了 ITO 与 ATO 两个靶，两路都如实给出两个组成，`decide` 在一个通道里看到两个不同值后依规则拒绝填格。
问题一修复后该论文的靶材栏仍为空（`multiple_conditions`），空是「拒绝猜测」，不是丢失。

| 选项 | 效果 | 代价 |
|---|---|---|
| A. `component` 改为 `cardinality: "many"` | 靶材栏显示「In2O3:SnO2 = 90:10 wt%；Sb2O5:SnO2 = 5:95 wt%」；om0035 类论文同理 | `cardinality` 是 PROMPT+VERDICT 属性，改后提示词变化，每篇每路重问一次 `component`（其余请求命中缓存）；`description_zh` 改写；B0/B1 的钉子（指纹、提示词快照、请求载荷、CLI 金样、工作簿快照）重录 |
| B. 保持单值，页面显示原因 | 结果表论文级空格旁显示质量行的原因（如「两个靶材，未填」） | 多靶材论文的靶材栏仍空；仅前端改动 |

**决策：A。** 实现（分支 `feat/component-many-targets`）：

- `profiles/tco.json`：`component.cardinality = "many"`，`description` 加一句「几个靶材就几条」，`description_zh` 改为
  「溅射靶材的化学组成；论文用了几种靶材就记几条，每条一种靶材的组成」。
- 前端不改：`many` 列本就由 `table.js` / `tsv.js` / 工作簿按列的 `cardinality` 以「；」连接显示。
- `description` 另加一句「never split one target's composition into its constituents」：B1 的 GZO 用例里 MinerU 把
  `3 wt.%`、`97 wt.%` 各记一条，正是列表字段会诱发的拆分。
- **严格列表**（字段属性 `strict_list: true`，角色 VERDICT，默认 `false`，只能配 `cardinality: many`；TCO 的 `component` 开启）：
  两路都作答的列表，只有两路都读到的元素才算数。它是字段的属性而不是 composition 这一「种类」的属性：电池例子里的
  前驱体也是 composition 列表，却应取并集（一路读到 CoSO4、另一路没读到，是两种盐）。实现在比较层，不在单元格层：
  `compare._compare_records` 把只有一路读到的元素报为 `ambiguous`（未定，而非仅仅没读到），只是重复了两路都读到的元素
  （一路把同一靶材在两种条件措辞下各引一次，正是该论文的样子）的仍按普通列表报 `missing`；不按顺序把两路的剩余元素
  配成 `conflict`——顺序不同时那是凭空配对，而且 `conflict` 会让监督模型白白打分（`decide_many` 不采用其裁定）。
  `decide_many` 走既有的 `troubled` 拒绝路径，不加分支。于是「一种组成两种写法」（`ITO 90:10` 对 `In2O3:SnO2 = 90:10`）、
  以及「第一种靶材一致、第二种写法不同」都留空（`ambiguous`），而不是列出两个或三个靶材；两路都读到两个靶材时仍为
  `agree` 列表。代价：一路只读到一个靶材的双靶材论文被拒绝而非列出——与单值字段一贯的保守立场相同。文本类列表
  （试剂、表征手段）不受影响，仍取并集。
- 重录的钉子，均只动 `component` 一行：`tests/fixtures/prompts/snapshot.json`、`tests/fixtures/payloads/b0.json`（两条
  `component` 请求的 `cache_key` 变化）、`tests/fixtures/cli_prompts/tco.txt`、`tests/fixtures/workbook/tco_workbook.json`
  （字段说明表的单位列「文本（多值）」与说明）、`tests/test_tco_fingerprints_pinned.py`（抽取/比较指纹、文件 sha、content_hash；`strict_list` 只动比较指纹）、
  `tests/test_profile_load.py` 的 `EDITED_AFTER_B0`。检索与识图指纹不动。
- 该论文离线重放：比较 2 条 `agree` + 2 条 `missing`，`paper_row.component = ["In2O3:SnO2 = 90:10 wt%", "Sb2O5:SnO2 = 5:95 wt%"]`，判定 `agree`。

对现有金标的影响：B0/B2 金标里的 4 个 `component` 格都不是多靶材论文。有守卫时它们的结果与单值时一致（GZO 与 ITO 的
两种写法照旧留空：原来是 `multiple_values`/`conflict`，现在是 `ambiguous`），没有守卫则会变成「两个靶材」的错误列表。
`eval/score.py` 对列表格逐元素计分，所以 `component` 的计数在此改动前后不可直接比较；下次对 B0/B2 评分时按此解读。

合入前需负责人安排重抽：passage 模式下只有 `component` 这一个字段问题的请求变了
（`tests/fixtures/payloads/b0.json` 的 50 条请求中只有 2 条换了 `cache_key`），其余问题仍命中 LLM 缓存，所以
`deploy.sh --rerun` 的代价是每篇论文每路各一次 `component` 提问，而不是全库重抽。

## 测试

- `tests/test_records.py`：`FieldValue.quote` 的几种写法（单位在旁、在文本里、`wt.%`、`wt %`、紧贴数字、无单位、空白折叠、字母边界）。
- `tests/test_compare.py`：同一组成两种单位写法 → `paper/component agree`；`wt.%` 对 `wt%` → `agree`；文本同、单位异 → `conflict`。
- `tests/test_dataset.py`：两种写法进论文行，单元格为 `In2O3:SnO2 = 90:10 wt%`，判定 `agree`；`wt.%` 在文本里对 `wt%`
  在旁边，比较与单元格同判 `agree`。
- `tests/test_records.py` 的「论文级字段出现在样品下即丢弃」保持不变：清洗层的不变量仍成立。
- 选项 A：`tests/test_dataset.py` 两靶材双路 → 两元素 `agree`；一靶材两种写法 → 留空 `conflict`；
  `tests/test_production_cases_b1.py` GZO 仍留空；`tests/test_cardinality_many.py` `strict_list` 按字段声明、须为列表；`tests/test_compare.py` 严格列表的 `agree`/`ambiguous`/`missing` 行；一靶材共享、一靶材两种写法 → 留空 `ambiguous`。
