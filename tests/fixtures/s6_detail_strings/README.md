# The two TCO detail strings S6 changes

S6 (special cases → profile attributes) replaced every field-name special case in `src/` with a profile
attribute. Every TCO output stays byte-identical except these two stored, human-readable texts, which named the
TCO domain and are now domain-free. Neither is part of any prompt or request payload, so no LLM cache entry
moves. `scripts/diff_derived.py` in S15a is expected to report exactly these two, and nothing else, modulo the
new `profile_fingerprint` fields.

| # | Where it is stored | Before | After |
|---|---|---|---|
| 1 | `LaneExtraction.dropped` (`extract.py`, passage mode), once per sample-level field when the inventory says the paper deposits no film of its own (`no_tco_film`) | `<field>: the paper deposits no TCO film of its own, so it was not asked about` | `<field>: the inventory found no in-scope sample, so it was not asked about` |
| 2 | The `__selection__` quality row's `detail` (`dataset.py`), when no sample could be matched and the paper row keeps only paper-level fields | `未提取到可匹配样品；论文行仅保留唯一的靶材字段` | `未提取到可匹配样品；论文行仅保留唯一的论文级字段` |

Not changed, though it moved: the missing-condition note on `transmittance` (`原文提取结果未注明透光率波长或波段`)
now comes from the profile's `missing_condition_note_zh`, and the scientific number format of `resistance` and
`resistivity` from `display_format`; both render the same bytes as before.

Tests: `tests/test_special_cases.py` (string 2, the generic note, the display format) and
`tests/test_extract_passages.py::test_a_paper_depositing_no_tco_film_is_asked_only_paper_level_fields` (string 1).
