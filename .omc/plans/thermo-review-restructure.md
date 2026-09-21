# Thermo-nuclear review follow-through (2026-09-20)

Source: strict maintainability review of improve/pipeline-quality (01328fd..b8182cd). Verdict: changes
required. Every item below preserves behaviour. Proof per phase: full suite green, and where a
fingerprinted module changes, a corpus rebuild from cache must reproduce the served verdict mix exactly
(14 papers: agree 525 / single_source 345 / conflict 0 / ambiguous 7 / multiple_conditions 88).

## Phase 0 — blockers (defects)
- B4 table.js: target-row TSV emits one leading column too few. Build the blanks from LEADING.length.
- B1 dataset.py `_DESCRIPTIONS`: a second field table in Python. Move to config.json `description_zh`
  per field, parse into FieldSpec, exclude from cache keys, one helper builds field columns for JSON and
  the Excel field sheet.
- B2 dataset.py `_condition_key`: third definition of "same condition". Use normalize_key like compare.py.
- B3 config: `llm.concurrency` missing from config.json; delete the three absent-key accessors.

## Phase 1 — workflow judo (delete code)
- J1 `compare_document` re-extracts lanes `run_document` already holds → accept `lanes`.
- J3 `_drain_abandoned_lanes` + `abandoned` bookkeeping → collect all futures' outcomes, raise the first
  in BACKENDS order, log the rest.
- J4 `Library.document()` fabricates a PDF path → use `layout.source_pdf(sha)` (where it would be).

## Phase 2 — vote model + decomposition (extract.py 838 → ~600)
- J2 citations belong to the full identity, not the rank — REJECTED after implementation: the two accepting
  cases of the guarded union are one wider than the full key (a lone entry per pass merges across wordings)
  and one narrower (opposite-order conditions must not merge), so no model keyed on the full identity
  alone passes both pinned tests. The rank-scoped `_Tally` model stays; the code moved unchanged.
- M1 move the cross-pass vote (`merge_passes`, `_values`, keys, `_deduplicate`/`_merge_repeats`) to a new
  module `voting.py`; add it to `extraction_code_fingerprint`; CLAUDE.md/`__init__.py` module order.

## Phase 3 — boundaries + frontend decomposition
- T1 `DatasetPayload` / `FieldColumn` / `CorpusPayload` pydantic models replace `dict[str, Any]` across
  dataset.as_dict, Library.dataset/corpus, and the three endpoints; `from_dict` becomes validation.
- M2 split table.js into `fieldpicker.js` and `tsv.js`; results renderer stays; CLAUDE.md module list.

## Phase 4 — sentinel and decision ladder
- J5 one `ReasoningEffort` type with an explicit INHERIT for the inventory override; README note.
- T2 `_decide` as an ordered list of named checks — tried and REJECTED in review (more concepts, implicit
  ordering contract). Kept: `_matching_blocked(scope)` replaces the caller's string, `_commit` names the
  happy path; the ladder stays linear.

## Order and gates
Phase 0 → 1 → 2 → 3 → 4. Each phase: executor (opus) → code-reviewer → fix → commit → rebuild check.
Phases 1 and 0 run in parallel (disjoint files). Nothing is pushed.

## Status (2026-09-20, evening)
All phases done. B1-B4, J1, J3, J4, J5, M1, M2, T1 landed; J2 and T2's dispatcher rejected on evidence.
Every phase reproduced the corpus verdict mix exactly (525/345/0/7/88).
