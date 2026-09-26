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
- Domain modules: `text.py` (a leaf: text folding, `clean_unit`), `units.py` (built-in converters and retrieval
  patterns, the `UnitRegistry` of declared units), `fields.py` (`FieldSpec` and its roles), `profile.py`
  (`DomainProfile` and the slots: value types and their defaults, hashed), `profile_loader.py` (reading and
  checking a profile file, `field_spec`, `load_units`, the regex checks: unhashed, since everything it decides
  reaches a key as a value). Presentation: `ui_copy.py`, `workbook.py`, `columns.py`, `readings.py`. `batch.py`
  is directory runs and offline export; `stored.py` is what is stored for a document and whether it is current.
  `units.py` and `passages.py` must not import `normalize.py` (that is why `text.py` exists). `readers.py` holds
  the range, interval and date readers built on `normalize.read_number`; `KindContext`/`NO_CONTEXT` live in
  `records.py` so `normalize.py` can use them without importing `kinds.py`.
- What a field's kind decides (reading a value, when two lanes agree, what a dataset cell holds, the kind's note in
  a field line) is one row per kind in `kinds.py`; normalisation, comparison, the dataset cell and the prompts
  ask `kinds.rules_for(spec)` and never branch on `spec.kind` (profile validation and the CLI listing still do). `records.py` and `passages.py` sit below it and read `fields.DIGIT_KINDS` instead. A dataset column
  (`columns.FieldColumn`) carries `kind` and `cardinality` so the workbook (`format_cell`), `table.js` and `tsv.js`
  format a cell by its column, never by the value's shape: a `many` list joined with "; " ("；" on the page), a
  boolean TRUE/FALSE (是/否), an interval as two workbook columns `<name> 下限` / `<name> 上限`.
- No module converts or retrieves with a unit table of its own: every conversion goes through the
  `UnitRegistry` of the profile it runs under (`profile.units`), including in tests and fixture generators.

## Configuration

- Everything that is not a secret and not the domain lives in `config.json` at the repository root. The domain
  (groups, field table, condition keywords, prompt wording, units, display copy) is a profile,
  `profiles/<name>.json`, picked by `profile` / `PAPERFACTS_PROFILE` / `--profile`; a `config.json` that still has
  `fields` or `condition_keywords` is refused. `config.py` reads `config.json` once, validates it with errors that
  name the key and the file, and layers `PAPERFACTS_*` environment variables over it. Built-in constants are the
  third layer underneath, and they are the **baseline** the cache keys treat as "unedited" -- never change one to
  change a default; change `config.json` or the profile. The same holds for the slot defaults (`PromptSlots` in
  `profile.py`) and the attribute defaults (`FieldSpec` in `fields.py`): both modules are hashed and an attribute
  at its default is left out of the key material.
- A profile is loaded once per entry point (`profile_loader.load_profile`, cached per resolved path) and passed
  explicitly as a `DomainProfile`; no module holds a domain table (`tests/test_no_domain_globals.py`). Python string constants
  and `web/static/` stay domain-free (`tests/test_domain_free.py`): a domain word belongs in a profile slot or in
  `ui`. Every profile error is a `ConfigError` naming the file and the key; `parse_profile` checks every section
  and every field before raising one error with a line per problem.
- One server per `data_root`; a server serves every profile under `profiles/`, the configured one as the default
  (see Web interface).
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
  keyed by `records.sample_key` within their entity type everywhere samples meet (attribution, the pass vote, exact cross-lane
  pairing, both modes' `records.clean_samples`); never by `normalize_key`, which deletes Greek letters and
  folds a case-distinguished suffix. The two exceptions are explicit, never inferred: a paper with exactly
  one sample owns every unplaced value, and a value the model flags `applies_to_all_samples` (the paper states it for the whole series) is written onto
  every sample with `series=True`. A value stated for a named subset ("all films deposited at 100 °C") is
  placed by the model, once per sample id of that subset, and only when the excerpts or the sample list say
  exactly which samples form it; otherwise it stays unplaced (passage mode) or is left out (document mode,
  which has nowhere to keep an unplaced sample-level value). Code never infers a subset.
- Entity types: up to five (`profile.entities`, the first is primary; a profile without `entities` has one implicit
  entity, `"sample"`). Each has its own inventory, sample list, field system prompt, matching
  (`ComparisonReport.matchings[entity]`, scopes `<entity>:<a>|<b>`) and dataset rows; paper-level fields are asked
  with the primary entity's prompt. Entities meet only through a `reference` field. Passage mode only: refused by
  `workflow.check_mode` where a profile is loaded to run, only asserted in `ExtractionOptions.from_settings`. Rows
  and sample-level quality rows carry `entity` whenever `profile.declared_entities` is non-empty (one included),
  because `FieldSpec.entity` and the gold name it; never key that on `len(profile.entities) > 1`.
- A reference is grounded by resolution, not text: `grounding.ground_lane(..., profile=)` resolves it by
  `sample_key` among the lane's samples of the referenced entity, and `workflow.read_lane` re-grounds on every
  read, so no stored verdict survives. An id two differently spelled samples share resolves to nothing
  (ambiguous), never to the first. `records.KindContext` (a lane's samples, each entity's matched pairs, the
  dataset's row ids) is a required argument of the kind rows' `read`/`compare`/`cell`, `normalize_field`,
  `compare_values` and `decide`/`decide_cell`, because a forgotten one is silently wrong; a caller with no reference
  in play passes `NO_CONTEXT`. Voting (`deduplicate`, `merge_passes`) keys a reference's quote by `sample_key` via
  the required `reference_fields`.
- A number, date or interval quote longer than `fields.MAX_NUMBER_QUOTE` characters is dropped at cleaning and
  refused by `normalize.read_value` before parsing (the number reader is quadratic in a digit run); both measure
  the stored `value_raw`. `read_number` keeps a backstop that leaves room for the bound `read_value` prefixes.
- The model quotes; the code converts. `ExtractionResponse` has no `value`/`unit` field, so unit
  conversion cannot happen in the model even by accident.
- Five guardrails on the response: schema and type cleaning, scope enforcement (a paper-level field may
  not be attached to a sample), and citation validation — all in `records.py` — the plausible range
  (`valid_range`, judged on the converted value by `normalize.drop_implausible`), plus grounding
  (`grounding.py`), where the quoted text must occur in the block it cites. The first four drop the value
  with an audited reason; grounding only flags, never drops.
- Prompts are profile-driven: the templates in `prompts.py` are domain-free and every domain word is a
  `PromptSlots` slot, rendered in one pass (a slot is never rescanned; computed markers are finished text
  first). The TCO profile's slots reproduce the measured prompts byte for byte, pinned by
  `tests/fixtures/prompts/snapshot.json` (sha-pinned) and `tests/fixtures/payloads/b0.json`; never re-record
  either to make a change pass.
- Internal names: the paper-level record is `paper` (`PaperRecord`, comparison scope and quality-row id
  `"paper"`) and the no-samples verdict `no_samples`, in code, stored files and the web for every profile. The
  model sees the profile's `paper_key` / `no_samples_key` (TCO: `target` / `no_tco_film`), mapped onto them by
  `records.response_models` aliases; the top-level response classes keep their names because validation errors
  (JSON mode: top-level class plus key path, `tests/test_records.py` pins them) reach the model in repair
  requests. A report holds `matchings: {entity: SampleMatching}` and must hold `"sample"` (the implicit entity);
  read it with `report.sample_matching()`. Files written before that rename are read through aliases
  (`LaneExtraction`) and before-validators (`ComparisonReport`, `DatasetPayload`); `tests/fixtures/b0_formats` and `b1_formats` hold
  them. Every persisted model ignores unknown keys, so a rename without typed fixtures of every persisted type
  would load empty records silently.
- Cache keys live in `keys.py`. `extractor_key(options)` is the only extraction key: it hashes one frozen
  `ExtractionOptions` (profile, model, mode and every sampling/retrieval setting). The workflow builds it once
  with `ExtractionOptions.from_settings(settings, profile)` and hands the same object to both lanes, and readers
  use `extractor_key_for(settings, profile)`, so writer and reader cannot disagree -- never spell the settings
  out a second time. `comparison_key` takes a `ComparisonOptions` the same way. Lanes, reports and datasets
  record the profile fingerprint; a mismatch raises `ProfileMismatchError`, never a mixed comparison.
- What reaches which key follows from roles, never from a hand list. Every `FieldSpec` attribute declares its
  `FieldRole` set in its dataclass metadata (`tests/test_field_roles.py`): PROMPT and CLEANING attributes, the
  groups (name, level) and the declared units form the extraction schema; VERDICT adds to that for the
  comparison; RETRIEVAL (keywords) plus the profile's `retrieval` go into `retrieval_fingerprint` (passage mode
  only); FIGURE attributes of `figure_readable` fields plus the `figures` slots into `figure_key`; DISPLAY
  reaches no key. Prompt slots are hashed by value as the rendered system prompts. The profile's file name,
  `title_zh`, `maturity`, `ui` and every `label_zh` are display. A new attribute is a decision about its roles,
  made where it is declared. Anything at its built-in baseline is left out of the material, so an unedited
  checkout keeps the filenames it has; do not add a hand-maintained version number. The LLM cache is keyed by
  request payload, so a code-only change re-derives records for free as long as the rendered document and
  prompts stay byte-identical; prove it with an offline replay (`--offline` / `PAPERFACTS_LLM_OFFLINE=1`,
  zero misses) plus `scripts/diff_derived.py`.
- Hashed module sources, by fingerprint (`keys.py` is the truth; the docs follow it):
  - extraction code: `extract`, `fields`, `profile`, `units`, `text`, `voting`, `records`, `adapters`,
    `prompts`, `normalize`, `readers`, `grounding`, `continuation`, `kinds`;
  - retrieval (passage mode): `passages`, `continuation`, `units`, `text`, `fields`;
  - normalization (comparison): `normalize`, `readers`, `units`, `text`, `kinds`;
  - comparison code: `compare`, `matching`, `dataset`, `decide`, `kinds`, `fields`, `profile`;
  - figure code: `figures`, `normalize`, `readers`, `passages`, `units`, `text`, `fields`, `profile`.
  Editing any of them re-keys. A module that holds a default the keys omit must be hashed. Besides the rendered
  system prompts, `extractor_key` hashes every prompt slot not at its default (except the `matching_*` ones, which
  `comparison_key` hashes), because a slot may reach only a user prompt.
- The workbook removes control characters from every string before openpyxl appends it (openpyxl raises on
  them), and the loader refuses a control character or more than 40 characters in any `label_zh`, which names a
  sheet.
- Presentation and orchestration stay out of hashed modules: the Excel layout is `workbook.py`, not
  `dataset.py` (which only assembles the rows, a set of verdicts); the column labels and descriptions are
  `columns.py` and are never stored with a table; display copy defaults are `ui_copy.py`, not `profile.py`; reading
  a profile file is `profile_loader.py`. `workbook`, `columns`, `readings`, `ui_copy`, `profile_loader`, `llm`,
  `config`, `cli`, `workflow`, `batch`, `profile_view` and `profile_check` are in no key list, and `tests/test_keys_unhashed.py`
  holds that. Do not move display, storage or loading code into a hashed module. The prompt preview
  (`profile_view.prompt_sections`, what `paperfacts prompts` prints and `/api/profiles/<name>/prompts` returns) is
  assembled in `profile_view.py`, never in `prompts.py`; `tests/fixtures/cli_prompts/` pins its output.

## Figures

- `figures.py` is the opt-in `figures` stage (starts after parse, runs beside the extraction lanes, joined
  before export): a vision model reads property-vs-condition charts selected by deterministic code (the
  whole-figure caption names a `figure_readable` field by its keywords; panels go to captions by geometry).
  Readings are stored per profile, `figures/<profile>/<figure_key>.json`. It is
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
  offline). Modules are `state`, `api`, `html`, `router`, `profiles`, `profile`, `library`, `document`, `table`,
  `fieldpicker`, `tsv`, `corpus`, `facts`, `figures`, `samples`, `job`, `viewer`, `check`; `app.js` is only the
  entry point.
  `profile.js`'s `renderDefinition(root, definition, …)` draws any profile definition (the read-only page,
  `(#/p/<name>)/profile`, and the check page's preview); its fields table takes its columns from the definition's
  attributes, never a hand list.
- Profiles in the page: the router reads `#/p/<name>/…` (no prefix = the default, which is `null` in the frontend,
  never its name) into `state.profileName`; a profile change is a new view (bumps the generation, reloads the
  rail, drops `/fact/n` and the filter). Every per-profile request goes through `profileApi(profile, path)` /
  `profileHref` (`api.js`) with the profile the load captured before its first await; plain `api()` refuses
  any path outside its profile-free allowlist, `undefined` throws, a POST always names the profile when the
  page knows the default's name, and a response whose `X-PaperFacts-Profile` differs is refused. `state.profile`
  (the labels) is only set by `adoptProfile` inside a generation-guarded load that awaited `profileView(profile)`
  beside its data, so labels and data always come from one profile. A job of another profile on the open paper is
  only noted, never polled or drawn (`jobInProfile`).
- Async ownership: the router bumps `state.generation` on every navigation to another view. Every load, poll
  and finish handler notes it before its first `await` and draws nothing once it has changed; do not add a
  per-feature "is this still the current document" check instead. The one sanctioned exception is
  `viewShows(id, profile)` (`state.js`), for a handler that must outlive a finish reload, which bumps the generation
  while its request is out (a rerun, a bulk run, a profile view's retry). Polling retries with backoff and a loop is
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
- `POST /api/profile-check` takes untrusted profile JSON: 256 KiB counted as it arrives and read within 15 s (408)
  before one of the two check slots is taken (429), `?field=` an identifier (422), and parsed and rendered only in
  `profile_check.run_check`'s child process (options on its stdin, never argv; 10 s wall clock, CPU/FSIZE limits,
  a RLIMIT_AS memory limit on Linux only -- macOS refuses it, so there the wall clock and CPU limit bound it -- its
  answer read to at most 16 MiB, empty environment; 503 when it timed out or could not start, 502 when it crashed or
  gave no answer), because validation folds unit spellings through `text._HTML_SUB`, which backtracks
  polynomially and holds the GIL, and fills unbounded caches (`units._compiled`). Never validate pasted text in the
  server process, never pass it to `load_profile` or a key function, and never use its name as a path;
  `tests/test_profile_check.py` snapshots every package cache and the data and profile trees around a check.
- Background jobs run on `web.max_parallel_documents` workers, never two on the same document; a worker
  takes the oldest queued job whose document is free. A `Job` is a frozen value in a lock-guarded dict,
  replaced whole on every transition, so a poller never sees a half-applied state. Submitting the same
  document twice under one profile while it is active returns the same job. A job's log is attributed by a context variable:
  every pool in the pipeline is `threads.ContextThreadPoolExecutor`, which runs each task in a copy of the
  caller's context; a plain `ThreadPoolExecutor` would drop its records from the log.
- One server per data root. The per-document and per-parser locks are process-local, so two servers over one
  `data_root` can parse the same document at once. A server serves several profiles through
  `web/registry.py`'s `ProfileRegistry`, built once in `create_app`: the default is the one the settings select
  (fatal when it does not load or its mode cannot ask it), every other `profiles/*.json` that `web.profiles`
  allows is loaded with `load_run_profile(..., to_run=False)` (a failure, or a link that loads as another
  profile, lists it in `invalid`, file name only, never fatal), keyed by `profile.name`. Each served profile has
  its own `Library`, except one `check_mode` refuses, whose keys cannot be computed (`library is None`). A route
  whose answer depends on a profile takes `?profile=` through the `Served` / `ProfileLibrary` dependencies
  (absent = the default, so old URLs are unchanged; malformed 422, unknown 404, invalid 409, no library 409) and
  its response carries `X-PaperFacts-Profile`; the parse artifact and page images are profile-free and read
  through the default's library. `Job.profile` names the
  profile; a resubmission is deduped on (document, profile), but `_take` stays busy by document, so one document
  never has two running jobs whatever their profiles (they share `parsed/`, `raw/`, `identity.json`). Each
  profile is read once; `pipeline_runner` refuses a job once *its* profile file's bytes on disk (re-resolved from
  the path it was named by) differ from the ones loaded. That sha256 is `profile.loaded_file_sha256`, kept
  beside the profile, never in `content_hash` or a key.
- Parallel documents are bounded twice: `parsers.py` holds one lock per parser around the actual parse
  (not around a cache hit), and every model request takes a slot of `llm.IN_FLIGHT` around the HTTP call
  only -- never while waiting on a future or a backoff, which is what keeps the nested pools deadlock-free.
- Lane colours are fixed: blue for MinerU, orange for PaddleOCR-VL. Chart readings belong to neither lane
  and use the neutral colour. Status colours always accompany text,
  never carry meaning alone.
- The UI copy is Chinese; code comments are English.
- The selected fact is in the URL (`(#/p/<profile>)/doc/<id>/fact/<n>`) so a link survives a reload.

## Deployment

`deploy/` targets a Linux GPU server. GPU ids are set per service by env var, default 0. Development
happens on macOS and is pushed to GitHub; the server pulls it. The server may be operated over ssh
(`ssh yangrm@ssh.yangruiming.org`, through a cloudflared tunnel that drops connections now and then):
production runs from `~/Projects/paperfacts` under `systemctl --user` and is updated only with
`scripts/deploy.sh`; experiments run in the `~/Projects/paperfacts-dev` worktree against its own copy of the
data. Run anything longer than a minute there with `nohup`, and judge whether it ran from the files it
writes, never from `pgrep -f`/`pkill -f`, whose pattern matches the ssh command's own shell.
