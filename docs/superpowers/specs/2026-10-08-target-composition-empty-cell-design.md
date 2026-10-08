# 靶材成分（component）论文级单元格为空：原因与决策

Date: 2026-10-08
Status: Proposed. 问题一（配对）随 PR #54 实现；问题二（单值字段遇多靶材）待拍板。

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

**决策：修。** `normalize.without_own_unit(value_raw, unit_raw)`：引文末尾若是 `unit_raw` 所指的单位（允许字符间有空格，
字母紧贴的不算），去掉它；文本中间的单位不动，只剩单位的文本不动。`TextRules.compare` 先各自去掉自带单位再 `same_text`，
文本相同但 `unit_raw` 不同判 `conflict`（wt% 对 at%）；`TextRules.cell` 输出「去单位文本 + 单位」，两路单元格因此同串，
且单元格仍写着 `wt%`。重放结果：2 条 `agree`（备注「conditions worded differently」）+ PaddleOCR-VL 重复引文 2 条 `missing`。

`normalize.py` 与 `kinds.py` 均在抽取、比较指纹内，合入后所有已存抽取与比较换名；LLM 缓存按请求载荷命中，
离线重放零 miss 即可重新推导。

未改：`cardinality: many` 的文本字段走 `element_key(spec, value_raw)`（`compare._set_pairs`、`decide` 的并集单元格），
同样的单位写法差异在列表字段上仍不配对；无生产用例，留作后续。

### 问题二：单值字段遇到多靶材（设计决策，待定）

`component` 在 `profiles/tco.json` 里是 `cardinality: one`（默认），`description_zh` 明说「保留唯一组成文本，不拆选多个靶材」。
这篇论文用了 ITO 与 ATO 两个靶，两路都如实给出两个组成，`decide` 在一个通道里看到两个不同值后依规则拒绝填格。
问题一修复后该论文的靶材栏仍为空（`multiple_conditions`），空是「拒绝猜测」，不是丢失。

| 选项 | 效果 | 代价 |
|---|---|---|
| A. `component` 改为 `cardinality: "many"` | 靶材栏显示「In2O3:SnO2 = 90:10 wt%；Sb2O5:SnO2 = 5:95 wt%」；om0035 类论文同理 | `cardinality` 是 PROMPT+VERDICT 属性，改后提示词变化，TCO 全库重新抽取（非离线重放）；`description_zh` 需改写；列表字段也需要问题一的「未改」部分 |
| B. 保持单值，页面显示原因 | 结果表论文级空格旁显示质量行的原因（如「两个靶材，未填」） | 多靶材论文的靶材栏仍空；仅前端改动 |

建议 A，但这是领域决定，需负责人拍板后再动代码。

## 测试

- `tests/test_normalize_text.py`：`without_own_unit` 的去留边界（末尾、带空格、紧贴数字去；文本中间、紧贴字母、只有单位留）。
- `tests/test_compare.py`：同一组成两种单位写法 → `paper/component agree`；文本同、单位异 → `conflict`。
- `tests/test_dataset.py`：两种写法进论文行，单元格为 `In2O3:SnO2 = 90:10 wt%`，判定 `agree`。
- `tests/test_records.py` 的「论文级字段出现在样品下即丢弃」保持不变：清洗层的不变量仍成立。
