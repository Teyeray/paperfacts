# PaperFacts — working conventions

Extract traceable, sample-level measurements from scientific PDFs. Start with [README.md](README.md);
this file is the part that is easy to get wrong.

## Environment

- Python is pinned to `>=3.13,<3.14` (MinerU caps below 3.14, paddlepaddle has no cp314 wheels).
- `uv sync --group dev` for the main package. Never `pip install` into the venv.
- macOS only: if `import paperfacts` suddenly fails, an external tool has marked `.venv` hidden and
  Python 3.13 then skips its `.pth` files. Run `scripts/dev_fix_venv.sh`.

## Dependency isolation — the rule that shapes everything

- `src/paperfacts/` is pure Python and **never** imports `mineru` or `paddleocr`. Their dependency trees
  conflict (two different `cv2`, torch vs paddlepaddle), so they cannot share an environment.
- Each parser is a PEP 723 script in `runners/`, run with `uv run --locked --script`. Lockfiles are
  committed; after editing a dependency header run `uv lock --script runners/<name>.py`.
- A runner only dumps the parser's native output plus `meta.json`. Converting that into `SourceBlock`s is
  an adapter in the main package: a pure function, tested against recorded fixtures.
- `meta.json`'s structure is defined in `models.py`. Runners do not import the main package, so
  the contract is held by tests (real-output fixtures plus the `--run-parser` integration suite). Changing
  a meta field means changing the model and the fixtures in the same commit.
- Runner scripts must not be named `mineru.py` or `paddle.py`: uv puts the script's directory on
  `sys.path[0]`, where they would shadow the real packages.

## Layout

- `src/paperfacts/` is flat: one module per pipeline stage, listed in order in `__init__.py`. `web/` is
  the only sub-package. Do not add re-exporting `__init__` files or nest packages; import from the module
  that defines a name.
- Every exception class is in `errors.py`.

## Configuration

- Everything that is not a secret and not the domain lives in `config.json` at the repository root. The domain
  (groups, field table, condition keywords, prompt wording) is a profile, `profiles/<name>.json`, picked by
  `profile`; a `config.json` that still has `fields` or `condition_keywords` is refused. `config.py` reads `config.json` once, validates it with errors that name the key and the file, and layers
  `PAPERFACTS_*` environment variables over it. Built-in constants are the third layer underneath, and they
  are the **baseline** the cache keys treat as "unedited" -- never change one to change a default; change
  `config.json`.
- Secrets only in `.env` (gitignored, loaded without overriding what the environment already has) and only
  the API key. `config.json` has nowhere to put a key, which is the point.
- A new setting means: a key in `config.json`, a field on `Settings`, a `PAPERFACTS_*` override, and a line
  in the README. If it changes what the model is asked, it also goes into `extractor_key`; if it changes a
  verdict, into `comparison_key`. The one exception with no override, `comparison.ambiguous_match_confidence`,
  is listed in the README as file-only; do not add a second without saying why.

## Code conventions

- Comments and docstrings are **English**, and explain *why*. A comment that restates the code is deleted,
  not translated. Identifiers are English.
- Every model is `frozen=True`. Update with `model_copy(update=...)`; never mutate in place.
- A bounding box is always a `NormalizedBBox` in `[0, 1]` page coordinates. Native coordinates enter
  through its `from_*` factories and nowhere else.
- Backend literals are `"mineru"` and `"paddleocr_vl"`. Source ids are `{backend}_p{page}_b{order}`, pages
  0-based.
- All on-disk paths and atomic writes come from `storage.py`, and so do the path rules over a stored
  document (`stored_pdf`, `is_runnable`, `stored_document`). A document's full sha256, display name and
  origin live only in `identity.json`, written the moment the directory is created.
- **PDFium is not thread-safe.** Every pypdfium2 call goes through `pdf.py`, serialised behind its
  process-wide lock. Concurrent opens corrupt its global state, after which every subsequent open fails
  with "Data format error" until the process restarts.
- `workflow.run_document` is the single orchestration path; the CLI and the web job both call it. Business
  logic lives in `workflow.py` — `web/` only does the document library, background jobs and HTTP mapping.
  Directory batches and offline re-export (`run_batch`, `discover_pdfs`, `export_document`) are `batch.py`,
  which runs each document through `run_document`.
- Errors: parsers raise `ParserError(backend, stage, detail)`. Adapters map unknown labels to `unknown`
  while keeping `raw_label`, and skip malformed boxes with a warning — never silently, never fatally.
- Logging is the standard library, logger name = module name.
- `ruff check` + `ruff format`, line length 120.

## Extraction

- Both lanes use the same prompts, model, temperature and retrieval. Any asymmetry there contaminates the
  disagreement signal, which is the whole measurement.
- Two modes, both in `extract.py`: `document` asks for the whole paper at once, `passage` asks which samples
  exist and then one question per field over the blocks `passages.py` retrieved for it. Retrieval is
  deterministic code, never a model call. Passage is the default; `.omc/research/extraction-modes.md` has
  the measurement that decided it.
- A sample-level value the model cannot place on a sample goes to `LaneExtraction.unattributed`: kept,
  grounded and shown, but compared with nothing. Never attach it to a plausible neighbour. Sample ids are
  keyed by `records.sample_key` everywhere samples meet (attribution, the pass vote, exact cross-lane
  pairing, both modes' `records.clean_samples`); never by `normalize_key`, which deletes Greek letters and
  folds a case-distinguished suffix. The two exceptions are explicit, never inferred: a paper with exactly
  one sample owns every unplaced value, and a value the model flags `applies_to_all_samples` (the paper states it for the whole series) is written onto
  every sample with `series=True`. A value stated for a named subset ("all films deposited at 100 °C") is
  placed by the model, once per sample id of that subset, and only when the excerpts or the sample list say
  exactly which samples form it; otherwise it stays unplaced (passage mode) or is left out (document mode,
  which has nowhere to keep an unplaced sample-level value). Code never infers a subset.
- The model quotes; the code converts. `ExtractionResponse` has no `value`/`unit` field, so unit
  conversion cannot happen in the model even by accident.
- Five guardrails on the response: schema and type cleaning, scope enforcement (a paper-level field may
  not be attached to a sample), and citation validation — all in `records.py` — the plausible range
  (`valid_range`, judged on the converted value by `normalize.drop_implausible`), plus grounding
  (`grounding.py`), where the quoted text must occur in the block it cites. The first four drop the value
  with an audited reason; grounding only flags, never drops.
- Cache keys live in `keys.py`. `extractor_key(options)` is the only extraction key: it hashes one frozen
  `ExtractionOptions` (model, mode and every sampling/retrieval setting). The workflow builds it once with
  `ExtractionOptions.from_settings` and passes it into `extract_lane`, and readers use
  `extractor_key_for(settings)`, so writer and reader cannot disagree -- never spell the settings out a
  second time. It also hashes the field schema *minus* the verdict-only cells (tolerances, categories,
  condition preferences, display text), the prompts, and the source of the extraction modules (`extract.py`,
  `records.py`, `fields.py`, `profile.py`, `units.py`, `text.py`, `adapters.py`, `prompts.py`, `normalize.py`,
  `grounding.py`, `voting.py`, `continuation.py`); passage mode adds its two prompts plus `retrieval_fingerprint`
  (the keywords, `passages.py`, `continuation.py`, `units.py` and `text.py`). `comparison_key` hashes the whole field schema including
  tolerances, categories, condition preferences, `normalize.py`, `compare.py`, `matching.py`, `decide.py`, `dataset.py` and the matching prompt.
  A tolerance edit therefore re-keys comparisons only. Anything that is at its built-in
  baseline is left out of the material, so an unedited checkout keeps the filenames it has. Changing any of them invalidates the right cache automatically; do not add a
  hand-maintained version number. The LLM cache is keyed by request payload, so a code-only change
  re-derives records for free as long as the rendered document and prompts stay byte-identical.
- Presentation stays out of hashed modules: the Excel layout is `workbook.py`, not `dataset.py` (which only
  assembles the rows, a set of verdicts). `workbook.py` and `readings.py` are in no key list, and
  `tests/test_keys_unhashed.py` holds that.

## Figures

- `figures.py` is the opt-in `figures` stage (starts after parse, runs beside the extraction lanes, joined
  before export): a vision model reads property-vs-condition charts selected by deterministic code (the
  whole-figure caption names a film field by its keywords; panels go to captions by geometry). It is
  paper-level, not a lane: its readings are approximate (±10 % / ±20 %), never create or identify a sample
  (chart x snaps to ticks), never fill a dataset cell and never join the two-lane comparison. They live in
  their own file, the 图中读数 sheet (`workbook.write_dataset(figure_rows=...)`), `GET /api/documents/{id}/figures`
  and their own web section; `dataset.py` and `decide.py` must not import `figures.py`. A failure in it marks only its own
  stage failed, and `--force` never re-reads charts (`--force-figures` does). Its prompt lives in `figures.py`, not `prompts.py`, so
  tuning it never renames stored extractions; `figure_key` in `keys.py` covers it.
- Where readings are stored and which are shown (`shown_figures`, `read_document_figures`, `FiguresView`,
  `figure_rows`) is `readings.py`, not `figures.py`: `figures.py`'s source is hashed into `figure_key`, and
  moving storage or display code there would rename every stored reading.
- Vision requests go through `llm.complete_vision` on a `VisionClient`, never the extraction client;
  crops come from `pdf.render_region`.

## Testing

- `uv run pytest` — no models, no network, no real papers. Temporary PDFs are generated with pypdfium2.
- `uv run pytest --run-parser` — integration; needs both parser environments and their weights.
- `tests/fixtures/corpus/` records every numeric value string and sample id of the real corpus with how
  `parse_number` and `sample_key` read them. A change to either that moves a corpus reading fails
  `test_corpus_strings.py`; if it is intended, re-run `tests/fixtures/corpus/generate.py <data_root>`,
  review the JSON diff, and add the string to `INTENDED_VALUE_CHANGES` with the reason.
- Coverage target ≥ 80% (`--cov=paperfacts`).
- The frontend has no JS test runner; `tests/e2e/web_races.py` drives it in headless Chromium against a seeded
  library with a stub job (`PYTHONPATH=src uv run --with playwright python tests/e2e/web_races.py`, or
  `PYTHONPATH=src uv run --with playwright pytest -m e2e`). A plain pytest run deselects the `e2e` marker, and
  without Playwright it skips. A frontend change to routing, polling or layout should keep it passing.

## Web interface

- No build step: ES modules plus CSS custom properties, no framework, no external fonts (the server may be
  offline). Modules are `state`, `api`, `html`, `router`, `library`, `document`, `table`, `fieldpicker`, `tsv`,
  `corpus`, `facts`, `figures`, `samples`, `job`, `viewer`; `app.js` is only the entry point.
- Async ownership: the router bumps `state.generation` on every navigation to another view. Every load, poll
  and finish handler notes it before its first `await` and draws nothing once it has changed; do not add a
  per-feature "is this still the current document" check instead. Polling retries with backoff and a loop is
  owned by a token, so it cannot run twice.
- Progress is the server's: `DocumentSummary.stages` (every `stage_names()` stage) and `runnable`. The
  frontend never rebuilds a stage list of its own.
- A results table is a list of columns `{header, head, html(item), text(item)}` (`table.js`); the rendered
  rows and the clipboard copy are both built from that one list.
- Controls that re-render their own table carry a `data-focus` key and the re-render goes through
  `keepFocus`, so keyboard focus survives. Clickable rows and cells are focusable and act on Enter/Space.
- The HTTP edge (`web/app.py`'s one middleware): Basic auth compared as UTF-8 bytes, a same-origin check
  on every non-GET request (`Sec-Fetch-Site` decides alone when present; Origin/Referer against Host is
  only the fallback), and frame/nosniff headers on every response. The upload size is the upload route's
  own first step: a declared `Content-Length` is the fast path, the bytes that arrive are counted anyway,
  so a chunked body is fine. `/api/jobs` is briefs without logs; finished jobs are pruned to the newest 200.
- Background jobs run on `web.max_parallel_documents` workers, never two on the same document; a worker
  takes the oldest queued job whose document is free. A `Job` is a frozen value in a lock-guarded dict,
  replaced whole on every transition, so a poller never sees a half-applied state. Submitting the same
  document twice while it is active returns the same job. A job's log is attributed by a context variable:
  every pool in the pipeline is `threads.ContextThreadPoolExecutor`, which runs each task in a copy of the
  caller's context; a plain `ThreadPoolExecutor` would drop its records from the log.
- One server per data root. The per-document and per-parser locks are process-local, so two servers (say,
  under two profiles) over one `data_root` can parse the same document at once. A server's profile is read
  once; `pipeline_runner` refuses a job once the file's content hash on disk differs from the loaded one.
- Parallel documents are bounded twice: `parsers.py` holds one lock per parser around the actual parse
  (not around a cache hit), and every model request takes a slot of `llm.IN_FLIGHT` around the HTTP call
  only -- never while waiting on a future or a backoff, which is what keeps the nested pools deadlock-free.
- Lane colours are fixed: blue for MinerU, orange for PaddleOCR-VL. Chart readings belong to neither lane
  and use the neutral colour. Status colours always accompany text,
  never carry meaning alone.
- The UI copy is Chinese; code comments are English.
- The selected fact is in the URL (`#/doc/<id>/fact/<n>`) so a link survives a reload.

## Deployment

`deploy/` targets a Linux GPU server. GPU ids are set per service by env var, default 0. Development
happens on macOS and is pushed to GitHub; the server pulls it. The server may be operated over ssh
(`ssh yangrm@ssh.yangruiming.org`, through a cloudflared tunnel that drops connections now and then):
production runs from `~/Projects/paperfacts` under `systemctl --user` and is updated only with
`scripts/deploy.sh`; experiments run in the `~/Projects/paperfacts-dev` worktree against its own copy of the
data. Run anything longer than a minute there with `nohup`, and judge whether it ran from the files it
writes, never from `pgrep -f`/`pkill -f`, whose pattern matches the ssh command's own shell.
