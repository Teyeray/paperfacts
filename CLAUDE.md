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

- Everything that is not a secret lives in `config.json` at the repository root, including the field table.
  `config.py` reads it once, validates it with errors that name the key and the file, and layers
  `PAPERFACTS_*` environment variables over it. Built-in constants are the third layer underneath, and they
  are the **baseline** the cache keys treat as "unedited" -- never change one to change a default; change
  `config.json`.
- Secrets only in `.env` (gitignored, loaded without overriding what the environment already has) and only
  the API key. `config.json` has nowhere to put a key, which is the point.
- A new setting means: a key in `config.json`, a field on `Settings`, a `PAPERFACTS_*` override, and a line
  in the README. If it changes what the model is asked, it also goes into `extractor_key`; if it changes a
  verdict, into `comparison_key`. The three exceptions with no override (`fields`, `condition_keywords`,
  `comparison.ambiguous_match_confidence`) are listed in the README as file-only; do not add a fourth
  without saying why.

## Code conventions

- Comments and docstrings are **English**, and explain *why*. A comment that restates the code is deleted,
  not translated. Identifiers are English.
- Every model is `frozen=True`. Update with `model_copy(update=...)`; never mutate in place.
- A bounding box is always a `NormalizedBBox` in `[0, 1]` page coordinates. Native coordinates enter
  through its `from_*` factories and nowhere else.
- Backend literals are `"mineru"` and `"paddleocr_vl"`. Source ids are `{backend}_p{page}_b{order}`, pages
  0-based.
- All on-disk paths and atomic writes come from `storage.py`. A document's full sha256, display name and
  origin live only in `identity.json`, written the moment the directory is created.
- **PDFium is not thread-safe.** Every pypdfium2 call goes through `pdf.py`, serialised behind its
  process-wide lock. Concurrent opens corrupt its global state, after which every subsequent open fails
  with "Data format error" until the process restarts.
- `workflow.run_document` is the single orchestration path; the CLI and the web job both call it. Business
  logic lives in `workflow.py` — `web/` only does the document library, background jobs and HTTP mapping.
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
  grounded and shown, but compared with nothing. Never attach it to a plausible neighbour. The two
  exceptions are explicit, never inferred: a paper with exactly one sample owns every unplaced value, and a
  value the model flags `applies_to_all_samples` (the paper states it for the whole series) is written onto
  every sample with `series=True`.
- The model quotes; the code converts. `ExtractionResponse` has no `value`/`unit` field, so unit
  conversion cannot happen in the model even by accident.
- Five guardrails on the response: schema and type cleaning, scope enforcement (a paper-level field may
  not be attached to a sample), and citation validation — all in `records.py` — the plausible range
  (`valid_range`, judged on the converted value by `normalize.drop_implausible`), plus grounding
  (`grounding.py`), where the quoted text must occur in the block it cites. The first four drop the value
  with an audited reason; grounding only flags, never drops.
- Cache keys live in `keys.py`. `extractor_key` hashes the model, the field schema, the prompts, the
  sampling settings and the source of `extract.py`, `records.py` and `adapters.py`; passage mode adds its
  two prompts plus `retrieval_fingerprint` (the keywords, `passages.py` and `continuation.py`). `comparison_key` hashes
  tolerances, `normalize.py`, `compare.py`, `matching.py`, `dataset.py` and the matching prompt. Anything that is at its built-in
  baseline is left out of the material, so an unedited checkout keeps the filenames it has. Changing any of them invalidates the right cache automatically; do not add a
  hand-maintained version number. The LLM cache is keyed by request payload, so a code-only change
  re-derives records for free as long as the rendered document and prompts stay byte-identical.

## Testing

- `uv run pytest` — no models, no network, no real papers. Temporary PDFs are generated with pypdfium2.
- `uv run pytest --run-parser` — integration; needs both parser environments and their weights.
- Coverage target ≥ 80% (`--cov=paperfacts`).

## Web interface

- No build step: ES modules plus CSS custom properties, no framework, no external fonts (the server may be
  offline). Modules are `state`, `api`, `html`, `router`, `library`, `document`, `table`, `fieldpicker`, `tsv`,
  `corpus`, `facts`, `samples`, `job`, `viewer`; `app.js` is only the entry point.
- Background jobs are a single worker. A `Job` is a frozen value in a lock-guarded dict, replaced whole on
  every transition, so a poller never sees a half-applied state. Submitting the same document twice while
  it is active returns the same job.
- Lane colours are fixed: blue for MinerU, orange for PaddleOCR-VL. Status colours always accompany text,
  never carry meaning alone.
- The UI copy is Chinese; code comments are English.
- The selected fact is in the URL (`#/doc/<id>/fact/<n>`) so a link survives a reload.

## Deployment

`deploy/` targets a Linux GPU server and is restricted to **GPUs 4–7**; do not widen that. Development
happens on macOS, is pushed to GitHub, and pulled on the server — do not try to operate the server over
ssh from here.
