# Reviewer's roadmap

*A reading order for someone who wants to understand this repository — the original PaperFacts
architecture first, then the visual validation stage this fork adds — and check that the addition does
what it claims. Budget: about two hours for the whole thing, forty minutes for the fork alone if you
already know PaperFacts.*

Every step says what to open, what to look for, and the question you should be able to answer before
moving on. Commands assume `uv sync --group dev` has run.

---

## Part A — The original architecture in ten minutes

You can skip this part if you already know the upstream repository.

### A1. The thesis: disagreement is the product

Open `README.md`, read **Why two parsers**. Two structurally independent parsers (MinerU's classical
pipeline, PaddleOCR-VL's vision model) turn one PDF into two Markdowns; one LLM extracts facts from each
with identical prompts; the comparison labels each fact AGREE / CONFLICT / AMBIGUOUS / MISSING. The tool's
value is not the numbers but the *located, reviewable disagreements* between two readings.

> You should be able to answer: why would asymmetry between the two lanes (a different prompt, a different
> retrieval rule) destroy the measurement?

### A2. The pipeline, stage by stage

Open `src/paperfacts/__init__.py`. Its docstring lists every module in pipeline order; that list is the map.
Then read `workflow.run_document` (`workflow.py`, near the bottom) — the *only* orchestration path; the CLI
and the web job both call it.

| Stage | Module | One line |
|---|---|---|
| parse | `parsers.py` → `adapters.py` | run a parser (subprocess or HTTP), turn its native output into `SourceBlock`s with normalised bboxes |
| extract | `passages.py` → `extract.py` → `records.py` | retrieve the blocks a question needs (deterministic), ask the LLM, clean and validate the answer |
| ground | `grounding.py` | does the quoted text occur in the block it cites? flags, never drops |
| compare | `matching.py` → `compare.py` | pair samples across lanes, then pair values field by field, verdict by tolerance |
| export | `dataset.py` | one conservative value per (sample, field), refusals named |

> Answer: where does a number get *converted* (unit, scientific notation)? Not in the model — find the
> line in `records.py` that makes that structurally impossible.

### A3. The four guardrails

`README.md` → **What a blank cell means**. Schema/type cleaning, scope enforcement, citation validation,
grounding. The first three drop with an audited reason; grounding only flags. Grounding is the one that
makes "traceable" true rather than asserted.

### A4. The rules that shape every change

`CLAUDE.md`, all of it — it is short. The four that will matter for reviewing the fork:

1. `src/paperfacts/` never imports a model library; heavy models are subprocesses or HTTP services.
2. Both lanes are treated identically everywhere.
3. The model quotes; the code converts.
4. Cache keys are content fingerprints of exactly their own inputs (`keys.py`); anything at its built-in
   baseline is left out of the key so an unedited checkout keeps its filenames.

### A5. Configuration in three layers

`config.py` docstring: built-in constants → `config.json` → `PAPERFACTS_*` environment. Every setting has
all three. Secrets only in `.env`.

---

## Part B — The fork: what was added and why

Read in this order. Each file's module docstring is written to be read first.

### B1. The argument (15 min)

`docs/vlm-validation.md`, §1–§3. Come back to §4–§7 after the code.

> Answer before moving on: (a) what failure can the two-lane design not see? (b) why is the model asked to
> *transcribe* rather than to *confirm*? (c) what does it mean that the selection rule is "lane-blind"?

### B2. The new stage itself (20 min)

`src/paperfacts/validate.py`. Read top to bottom; it is organised in the order the data flows.

| Section | Read for |
|---|---|
| module docstring | the three rules |
| `Verdict`, `Reason`, the stored models | five verdicts kept apart; `ValueValidation.key` |
| `value_key`, `owner_of`, `select_targets` | the lane-blind rule; why the owner is looked up in the lane rather than parsed from the scope string |
| `region_for`, `CropStore` | union of cited blocks on one page; rendered once, kept under `crops/` |
| `parse_reading`, `transcription_fold`, `adjudicate` | lenient about the JSON wrapper, strict about digits; **the verdict is `grounding.is_grounded`** |
| `validate_lanes`, `_validate_one` | concurrency, one failed request = one `error` verdict, all failed = raise |

Then the tests that pin each of those: `tests/test_validate.py`. The parametrised tables under
`test_a_value_the_model_read_is_confirmed` / `..._is_contradicted` are the fastest way to see what the
matcher accepts and refuses.

> Check: find the test that proves the prompt never contains the value under check.

### B3. The two seams into existing code (15 min)

**The client.** `src/paperfacts/llm.py`: `VisionClient` (a separate protocol — read the docstring for why),
`complete_vision`, `vision_payload`, `vision_cache_key`. The design point is the cache key: the image is
stood in by its sha256, so the same crop is the same key but a megabyte of base64 is never hashed or stored.
Tests: `tests/test_llm_vision.py`.

**The render.** `src/paperfacts/pdf.py`: `render_region` renders the whole page and crops with
`NormalizedBBox.to_pixels` — the same mapping the web viewer uses — so the crop is exactly the rectangle
the viewer draws. `max_pixels` shrinks an oversized crop here rather than on the endpoint. Still behind the
pdfium lock. Tests: `tests/test_pdf_region.py`.

### B4. Where verdicts act — and where they must not (15 min)

`src/paperfacts/dataset.py`, `_decide`. Read the module docstring first, then the three commented blocks
inside `_decide` (contradicted set aside; `resolved`; the `trusted` filter). Everything else in the
function is unchanged from upstream.

Then `tests/test_dataset_validation.py`, which is organised as *entry point, then the neighbouring shape it
does not apply to*:

- conflict + one confirmed + one contradicted → `vlm_resolved`; both confirmed → still `conflict`; survivor
  unchecked → still `conflict`
- agree + both contradicted → `vlm_contradicted` (the case the stage exists for)
- ungrounded + confirmed → trusted; ungrounded + illegible → still `ungrounded`; no citation → never rescued
- a verdict keyed to the wrong owner → ignored

> Check: convince yourself a forged `confirmed` verdict cannot make the dataset commit a value that has no
> citation. (It is the last test in the "appeal" block.)

### B5. Keys, config, storage (10 min)

- `keys.py` → `validation_key`. It is its own key, and the tests in `tests/test_keys_validation.py` prove it
  is independent of the other two and that endpoint/concurrency settings are *not* in it.
- `config.py` → the `vlm.*` constants and `Settings.vlm_*`; `require_vlm_api_key` falls back to the LLM key.
  Note the one deliberate exception, `vlm_enabled`: shipped `true`, baseline `false`, pinned in
  `test_config_file.py::test_the_shipped_configuration_agrees_with_the_dataclass_defaults`.
- `storage.py` → `validation_path` (three keys), `crops_dir`/`crop_path`, and `dataset_json_path` growing an
  optional third key. Read the comment on the last one: an installation without a VLM keeps every filename.

### B6. Wiring (10 min)

- `workflow.py`: `build_vlm_client`, `read_validation`, `validate_document`, `_validate_stage`,
  `stage_names()` gaining `validate`, `export_document` reading a stored validation offline.
  Tests: `tests/test_workflow_validate.py`; the stage-order test in `tests/test_workflow_run.py` shows the
  skipped mark when the VLM is off.
- `cli.py`: `paperfacts validate`, `--policy`. `report.py`: `render_validation`.
- `web/documents.py`, `web/app.py`: `Library.validation`, `GET /api/documents/{id}/validation`, the
  three-key-then-two-key dataset lookup. `web/static/facts.js`: `ownerOf` + `verdictFor` rebuild
  `value_key` from a comparison row — the field order is a contract with `validate.value_key`.

### B7. Deployment and docs (5 min)

- `deploy/compose.yaml`: `qwen-vlm-server` on GPU 7, vLLM's OpenAI-compatible server. `deploy/.env.example`
  §3 for the knobs. `deploy/host/start_qwen_vlm.sh` for the bare-metal route.
- `README.md`: the diagram, **Why a third reader**, the `vlm` configuration section, **How the page is read
  back**, the decision table's two new rows, the data directory, the workbook's new column.
- `CLAUDE.md` → **Visual validation**: the conventions a future change must keep.

---

## Part C — Running it

```bash
uv sync --group dev
uv run pytest                         # ~1560 tests; the 11 parser-integration cases skip without --run-parser
uv run ruff check src tests runners && uv run ruff format --check src tests runners
uv run paperfacts fields              # the field table loaded
```

Then, with an API key in `.env` and a paper that has already been compared:

```bash
uv run paperfacts validate paper.pdf               # disputed values only
uv run paperfacts validate paper.pdf --policy all  # every value: the "both lanes wrong" measurement
uv run paperfacts serve                            # badges in 事实对照, a 视觉确认 tile, the crop on hover
```

Look at `data/docs/<sha>/crops/` — those PNGs are exactly what the model was shown — and at
`validations/<ek>.<ck>.<vk>.json`, where every verdict sits beside its transcription.

---

## Part D — Questions to bring back

Things this fork decided that the group may want to decide differently. None is hard to change.

1. **Default model.** `qwen3-vl-32b-instruct` on Model Studio; the exact served name should be checked
   against the workspace console. `deploy/` serves the 8B by default (`VALIDATION_MODEL`); the 32B fits an
   80 GB card and matches the hosted default, making pilot and on-prem verdicts directly comparable.
2. **Shipped `vlm.enabled = true`.** The fork's whole point, but it means a `run` needs the VLM endpoint to
   answer. `PAPERFACTS_VLM_ENABLED=false` restores upstream behaviour byte for byte.
3. **The `resolved` shape.** Requires every surviving side to be confirmed. A looser rule (one confirmed, one
   contradicted, others unchecked) would resolve more conflicts and trust the reader more.
4. **The `disputed` policy's reach.** It includes every MISSING (one-sided) value, which on the corpus is
   most of the requests. Dropping MISSING from the default would make the stage cheaper and leave
   single-source values unchecked.
5. **Scope.** Characters in a region only. Attribution errors (right number, wrong sample) need table-
   structure reasoning and a different prompt; a third full lane needs `compare_lanes` redesigned. Both
   are arguable next steps once the "both lanes wrong" rate is measured.
6. **The multiplication fold.** `transcription_fold` closes one spacing gap grounding refuses to close. Watch
   the `contradicted` verdicts on real papers for other spelling gaps between parser and model (thin
   spaces, `−` vs `-`, `·` vs `.`) before adding another fold — and add it to both sides.
