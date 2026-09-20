# PaperFacts

Extract structured, **verifiably traceable** measurements from scientific PDFs.

Give it a batch of papers and a list of target fields — sputtering power, gas flow, sheet resistance,
transmittance — and it returns one record per sample, where every number can be traced back to the exact
page and bounding box it was read from.

```
paper.pdf
  ├─ MinerU        ──→ Markdown + source map ──→ LLM extraction ──→ facts A
  └─ PaddleOCR-VL  ──→ Markdown + source map ──→ LLM extraction ──→ facts B
                                                        │
                                          sample matching, normalisation,
                                             field-by-field comparison
                                                        │
                        ┌───────────────┬───────────────┼───────────────┐
                      AGREE          CONFLICT        AMBIGUOUS        MISSING
                   both lanes      values differ    can't decide    one lane only
```

## Why two parsers

A single PDF parser plus a language model will happily produce a confident, well-formatted table of
numbers that are subtly wrong — a column misread, a row shifted, a value attached to the wrong sample —
and nothing in the output says which ones. The failure is silent, which is the worst property a data
pipeline can have.

PaperFacts runs two structurally independent parsers over the same paper and extracts from each with an
identical prompt and model. Where they agree, the number survived two different layout analyses and two
different OCR passes. Where they disagree, you have a specific, located, reviewable question instead of a
uniform wall of unearned confidence. Disagreement is the product.

## Provenance is checked, not asserted

"Every value is traceable" is easy to claim and easy to get wrong. Four mechanisms make it true, and each
one exists because the failure it catches was observed in a real run:

| Guardrail | What it catches |
|---|---|
| **Schema and type cleaning** | Fields outside the target schema; numeric fields holding words like `"minimum"` or `"n.a."` |
| **Scope enforcement** | A paper-level field attached to one sample, or the reverse — a film's dopant concentration reported as the sputtering target's composition |
| **Citation validation** | Block ids the model invented, or ids from parts of the document it was never shown |
| **Grounding** | The quoted text cannot be found in the block it cites — a real id attached to a value that did not come from it |

Grounding is the one that matters most, and the one usually missing. Without it, "traceable to a page and
a bounding box" only means the model named a real block. Matching tolerates formatting differences — the
same number reaches the model as `$( 4 0 \times 1 0 \mathrm { c m }$` from one parser and `(40 × 10 cm`
from the other — while staying strict about digits. Values that fail are kept but flagged, never silently
dropped.

The model is also constrained structurally rather than by instruction: the JSON schema it must return has
`value_raw` and `unit_raw` and no `value` or `unit` field, so it has nowhere to put a converted number.
All unit conversion, scientific-notation parsing and tolerance comparison happen in ordinary, testable
Python.

## Install

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev                        # the main package: pure Python, seconds
uv sync --script runners/mineru_runner.py  # MinerU environment (torch, ~1-2 GB on first run)
uv sync --script runners/paddle_runner.py  # PaddleOCR-VL environment (paddlepaddle)
```

The two parsers live in **separate environments on purpose**: they both ship a `cv2` (opencv-python vs
opencv-contrib-python) and they pull in torch and paddlepaddle respectively, so they cannot be installed
together. Each is a [PEP 723](https://peps.python.org/pep-0723/) script under `runners/` with its own
lockfile, run via `uv run --script`. The main package never imports either one; it only reads the files
they write. Model weights download on first use; set `MINERU_MODEL_SOURCE=modelscope` on networks where
Hugging Face is slow.

On Apple silicon, hand PaddleOCR-VL's vision stage to MLX — in-process CPU inference takes hours per
paper, MLX takes a minute or two:

```bash
uvx --python 3.13 --from "mlx-vlm>=0.3.11" mlx_vlm.server --port 8111   # leave running
export PAPERFACTS_PADDLE_VL_BACKEND=mlx-vlm-server
export PAPERFACTS_PADDLE_VL_SERVER_URL=http://localhost:8111/
export PAPERFACTS_PADDLE_VL_MODEL_NAME=PaddlePaddle/PaddleOCR-VL-1.6
```

Extraction needs an OpenAI-compatible LLM; any endpoint works, and the default in `config.json` is an
Alibaba Cloud Model Studio (百炼) workspace serving `deepseek-v4.1-flash` through its `compatible-mode/v1`
address. The key is read from `PAPERFACTS_LLM_API_KEY`, then `DEEPSEEK_API_KEY`, then
`PAPERFACTS_LLM_API_KEY_FILE`, then a `deepseek_api_key` file in the repository root (gitignored).

## Use

```bash
uv run paperfacts run paper.pdf            # parse both lanes, extract both, compare
uv run paperfacts batch template_files --output data/exports/template_files.xlsx
uv run paperfacts serve                    # the web interface on http://127.0.0.1:8000
uv run paperfacts fields                   # list the field table loaded from config.json
```

`run` is the whole pipeline; the individual stages are available separately and share the same caches:

```bash
uv run paperfacts parse   paper.pdf --backend both -v   # → data/docs/<sha>/parsed/
uv run paperfacts overlay paper.pdf --backend both      # draw block boxes on page images, to check by eye
uv run paperfacts extract paper.pdf                     # → data/docs/<sha>/facts/
uv run paperfacts compare paper.pdf                     # → data/docs/<sha>/comparisons/
```

### Automated Excel datasets

`run` now finishes by saving `data/docs/<sha>/dataset.xlsx`. To process every PDF in a directory
(including subdirectories and `.PDF` extensions) and automatically combine the results:

```bash
uv run paperfacts batch template_files --output data/exports/template_files.xlsx
```

The workbook contains five sheets:

| Sheet | Contents |
|---|---|
| 论文数据 | One row per unique PDF, one value per field; the twenty field names are stable column names |
| 样品数据 | All samples after merging the two parser sources, one row per sample |
| 字段说明 | Field definitions and canonical units |
| 数据质量 | Final field decisions, measurement conditions, merged citations and reasons for blank cells |
| 运行记录 | Success/failure per paper and the extraction/comparison versions |

The paper row selects the sample with the most usable fields, then the most two-lane agreements,
then a stable sample-ID tie break. **It never combines different samples' measurements into one row.**
This chooses the most complete sample, not the paper's highest-performing sample. Other samples remain
in the sample sheet. Target properties are paper-level and shared only when their extracted value is unique.

Numeric cells use the canonical units in `字段说明`; missing values are empty, never zero. Grounded,
single-source values are allowed and marked `single_source` in the quality sheet. Conflicts, ambiguous
sample matches, multiple values or conditions, ranges, bounds and rectangular dimensions stay blank.
Approximate values and measurements with ± uncertainty retain their center value with a quality note.
The main tables have no per-source value columns. Use the quality sheet to restrict a training set to
two-lane agreements if required; completeness alone is not a quality score.

Identical PDF content is processed once. Each completed or failed paper checkpoints the workbook
atomically; re-running the same command reuses the caches and rebuilds the table without appending
duplicate rows. Failed papers appear in `运行记录`, processing continues, and the command exits with status 1
if any paper failed. The table contains extracted values only; values absent from the paper or found only
in plots remain missing.

To regenerate the Excel workbook from current cached extractions/comparisons, with no parser or LLM calls:

```bash
uv run paperfacts export template_files --output data/exports/template_files.xlsx
```

`export` requires caches matching the current model, field schema and comparison rules; it reports outdated
or absent results as failures. On Apple silicon, start and configure the MLX service described under
**Install** before `batch` processes uncached PDFs.

### The web interface

```bash
uv run paperfacts serve                    # local
uv run paperfacts serve --host 0.0.0.0     # reachable from other machines
```

The home page is the corpus table: one row per processed paper, its selected sample across the field
columns, with a link into each document and a 「下载全部 Excel」 button for the whole library.

The document library's 「处理全部未完成」 button queues every document that has a PDF and is not yet
compared under the current keys, one job each, in library order; a document already being processed
keeps the job it has.

Published on the internet — a cloudflared tunnel, a shared server — it needs a password:

```bash
echo 'PAPERFACTS_WEB_PASSWORD=...' >> .env   # the username is web.username in config.json
```

With `PAPERFACTS_WEB_PASSWORD` set, every route, `/api` included, answers 401 until a browser or a client
sends HTTP Basic credentials, so an open tunnel cannot upload PDFs or spend tokens. Unset, the app is open,
which is what running it on a laptop wants. The password is a secret and lives only in `.env`; the username
is configuration and lives in `config.json`.

Drop a PDF on the left and processing starts, with live per-stage progress. The result is a fact-by-fact
comparison table, AGREE / CONFLICT / AMBIGUOUS / MISSING counts, and a page viewer: click any fact and
both lanes' source blocks light up on the rendered page — blue for MinerU, orange for PaddleOCR-VL. The
selected fact is part of the URL, so a link to one disputed number is shareable. Below the table are each
lane's raw sample records, from the paper's own wording through to the normalised value and back to the
block it came from. Papers processed from the command line appear in the library too, though only
uploaded ones carry their PDF and can be re-rendered on another machine. The API is documented at
`/api/docs`.

Above the comparison sits the results table: one row per sample after both lanes are merged, one column
per field, plus a first row for the paper-level target values. A green cell was confirmed by both lanes and
an amber one by a single lane — each says so in words as well as in colour — and a blank cell is a value
the pipeline refused to guess, with the reason on hover. Clicking a cell lights up the blocks it was merged
from. Two endpoints serve it: `GET /api/documents/<id>/dataset` returns the table as JSON, stamped with the
current extractor and comparison keys so a stale one is never shown, and `GET /api/documents/<id>/dataset.xlsx`
downloads the workbook behind the 「下载 Excel」 button.

### How the model is asked

Handing a fifteen-thousand-token paper to a model and asking for twenty fields and every sample at once makes
it lose its place: it cites a block that merely discusses the number, and it never mentions fields the paper
states only in passing. So by default the question is split up.

**Passage mode** (the default) asks which samples the paper reports, then asks about one field at a time,
showing only the blocks retrieved for that field. Retrieval is ordinary code, not a model: keyword and unit
matching, ranked, capped at eight blocks. Both lanes get identical retrieval rules, so the comparison still
measures the parsers and not the retrieval.

```bash
uv run paperfacts run paper.pdf --mode document   # the older whole-paper question
```

On the three papers in this repository, against whole-document mode:

| | document | passage |
|---|---|---|
| Values found | 48 | 88 |
| Agreeing facts | 17 | 35 |
| Conflicts | 4 | 0 |
| Values failing grounding | 4.2% | 3.4% |
| Prompt tokens | 64K | 208K |

The conflicts did not disappear into silence: both lanes now read the same evidence, so where one lane's
text garbled a number the other lane simply does not have it, and it is reported as MISSING with both
readings visible rather than as a single conflicting pair. The cost is roughly three times the prompt
tokens, because the field questions share overlapping context.

A value the model cannot place on any sample -- a paper-level claim such as "transmittance above 80% from
500 to 2500 nm" -- is kept and shown as **unattributed** rather than attached to a plausible sample. When
both lanes hold the same unplaced value it is paired and compared like any other (scope `unattributed`);
a value only one lane could not place stays out of the comparison, because the other lane may hold it on
a sample where it is already reported. An unplaced value is visible; a misplaced one is not.

### Repeated extraction

Language models are not deterministic even at temperature 0: repeated extractions of the same paper
occasionally gain or lose a value. With a single pass that noise is indistinguishable from genuine
parser disagreement, which is the signal this tool exists to measure. Extract each lane several times and
keep only what a majority of passes agree on:

```bash
uv run paperfacts run paper.pdf --passes 3    # 3x the LLM calls, 3x the cost
```

Off by default. Each surviving value records the fraction of passes that produced it, so a 2/3 value stays
visibly weaker than a 3/3 one. In passage mode the sample inventory is asked once and only the field
questions repeat, so every pass sees the same sample ids; the vote is on the number and unit, never on the
wording of the measurement condition, and a paper that reports the same number under two conditions keeps
both. Measured on three papers: a second pass reproduces 75–90 % of a first pass's values, so two passes are
a reproducibility filter at twice the model cost, not a way to find more (`.omc/research/reasoning-effort.md`).

## Caching

Nothing is recomputed unless something it depends on changed, and each cache is keyed by a content hash
of exactly its own inputs:

| Cache | Key | Invalidated by |
|---|---|---|
| Parser output | `raw/<backend>/meta.json` exists | `--force` |
| Extraction | model + prompts + field schema + document rendering | changing any of them |
| Comparison | field tolerances + normalisation source | changing a tolerance or a rule |
| LLM requests | the entire request payload | nothing — identical requests are free |

So adjusting a numeric tolerance recomputes the comparison without paying for extraction again, and
cannot serve a stale verdict either. Re-running a finished paper costs nothing.

## Configuration

Two files at the repository root. **`config.json` holds everything that is not a secret**, and is meant to
be edited:

```jsonc
{
  "data_root": "data",
  "server":     { "host": "127.0.0.1", "port": 8000, "max_upload_mb": 200,
                  "page_dpi": { "default": 110, "min": 50, "max": 220 } },
  "web":        { "username": "paperfacts" },          // the password is PAPERFACTS_WEB_PASSWORD in .env
  "llm":        { "base_url": "https://<workspace>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                  "model": "deepseek-v4.1-flash",     // any OpenAI-compatible endpoint and model
                  "timeout_s": 600, "context_tokens": 60000, "temperature": 0.0, "max_tokens": 16384,
                  "reasoning_effort": null,                    // null | none | low | medium | high
                  "inventory_reasoning_effort": null,          // null = inherit reasoning_effort
                  "retry_attempts": 4, "retry_backoff_s": 2.0 },
  "extraction": { "mode": "passage", "passes": 1, "candidate_limit": 8 },
  "comparison": { "ambiguous_match_confidence": 0.6 },
  "parsers":    { "mineru_url": null, "paddle_url": null, "paddle_render_dpi": 200,
                  "paddle_vl_backend": null, "paddle_vl_server_url": null, "paddle_vl_model_name": null,
                  "subprocess_timeout_s": 3600, "http_timeout_s": 900, "uv_bin": "uv" },
  "overlay":    { "dpi": 150 },
  "condition_keywords": ["sample", "substrate", "deposition", "..."],
  "fields":     [ /* the table below */ ]
}
```

**`.env` holds the secrets**, and is gitignored. Copy the template and fill in one line:

```bash
cp .env.example .env     # then set PAPERFACTS_LLM_API_KEY
```

The key is looked for in `PAPERFACTS_LLM_API_KEY`, then `DEEPSEEK_API_KEY`, then the file named by
`PAPERFACTS_LLM_API_KEY_FILE`, then `deepseek_api_key` in the repository root. It is never read from
`config.json`, and `config.json` has nowhere to put it.

Every scalar setting in `config.json` also has a `PAPERFACTS_*` environment variable that wins over it,
which is how one machine points at its own services without editing the shared file: `PAPERFACTS_DATA_ROOT`,
`PAPERFACTS_MINERU_URL`, `PAPERFACTS_PADDLE_URL`, `PAPERFACTS_PADDLE_RENDER_DPI`,
`PAPERFACTS_PADDLE_VL_BACKEND`, `PAPERFACTS_PADDLE_VL_SERVER_URL`, `PAPERFACTS_PADDLE_VL_MODEL_NAME`,
`PAPERFACTS_LLM_BASE_URL`, `PAPERFACTS_LLM_MODEL`, `PAPERFACTS_LLM_TIMEOUT_S`,
`PAPERFACTS_LLM_CONTEXT_TOKENS`, `PAPERFACTS_LLM_TEMPERATURE`, `PAPERFACTS_LLM_MAX_TOKENS`,
`PAPERFACTS_LLM_REASONING_EFFORT`, `PAPERFACTS_LLM_INVENTORY_REASONING_EFFORT`,
`PAPERFACTS_LLM_CONCURRENCY`, `PAPERFACTS_LLM_RETRY_ATTEMPTS`, `PAPERFACTS_LLM_RETRY_BACKOFF_S`,
`PAPERFACTS_EXTRACTION_MODE`,
`PAPERFACTS_EXTRACTION_PASSES`, `PAPERFACTS_CANDIDATE_LIMIT`, `PAPERFACTS_SERVER_HOST`,
`PAPERFACTS_SERVER_PORT`, `PAPERFACTS_MAX_UPLOAD_MB`, `PAPERFACTS_PAGE_DPI`, `PAPERFACTS_PAGE_DPI_MIN`,
`PAPERFACTS_PAGE_DPI_MAX`, `PAPERFACTS_OVERLAY_DPI`, `PAPERFACTS_SUBPROCESS_TIMEOUT_S`,
`PAPERFACTS_HTTP_TIMEOUT_S`, `PAPERFACTS_UV_BIN`, `PAPERFACTS_WEB_USERNAME`, `PAPERFACTS_WEB_PASSWORD`.
`PAPERFACTS_CONFIG` points at a different configuration file altogether.

`llm.reasoning_effort` is how much hidden reasoning the endpoint is asked for before it answers, sent as the
OpenAI-shaped `reasoning_effort` parameter; `null` omits the parameter entirely, which is the shipped
default. On this endpoint's `deepseek-v4.1-flash`, `"none"` makes a paper about six times faster but loses
recall: on the same twelve-page paper it found 77 values and 7 samples where the default found 112 values and
8 samples (`.omc/research/reasoning-effort.md`). Set it to `"none"` for a quick first pass over a large
batch, and leave it unset for the numbers you keep. It changes what the model is asked, so changing it
writes a new `extractor_key` and re-extracts. `llm.inventory_reasoning_effort` gives passage mode's one
inventory question its own effort, inheriting `llm.reasoning_effort` when left `null`: that question alone
spends 11k-17k hidden reasoning tokens per lane, about 70% of a run's completion tokens, while the field
questions after it reason in tens to hundreds, so turning it down is most of the wall-clock for one
question's worth of recall risk. Both lanes always get the same value, and it too writes a new
`extractor_key` (in passage mode only, since document mode never asks the question).

`llm.concurrency` is how many of one lane's per-field questions wait on the endpoint at once (default 4);
the two lanes themselves always run as a pair, so at most twice that many requests are open. It is the one
knob here that changes only *when* requests are sent, never what they contain, so it stays out of both
`extractor_key` and `comparison_key`: raising or lowering it never re-extracts and never re-compares. Set it
to 1 to send every question strictly one after another, which is what the pipeline did before it existed.

Three settings are file-only, because a single environment variable is the wrong shape for them:
`fields`, `condition_keywords` and `comparison.ambiguous_match_confidence`.

### The fields to extract

`config.json`'s `fields` list **is** the schema. Each entry drives the description the model is given, the
keywords retrieval searches for, the unit everything is converted to, and how close two numbers have to be
to count as the same fact:

```jsonc
{
  "name": "sheet_resistance",
  "group": "film",                          // target = paper-level, process / film = per sample
  "kind": "numeric",                        // numeric | composition | text
  "description": "Sheet resistance of the film (Ω/sq).",
  "keywords": ["sheet resistance", "sheet resistivity", "Rs", "R_s"],
  "canonical_unit": "Ω/sq",
  "rel_tol": 0.02,                          // |a-b| <= max(rel_tol * max(|a|,|b|), abs_tol)
  "abs_tol": 0.0,
  "condition_hint": null,                   // what to record alongside, e.g. a wavelength
  "bare_number": "reject"                   // reject | assume_canonical | percent_or_fraction
}
```

A `text` field may add `"categories"`, the closed set of answers it accepts, written the way the output
should spell them: `"categories": ["DC", "RF", "pulsed DC", "DC+RF", "HiPIMS"]` on `mode`. A quoted value is
reduced to the tokens those names contain, so "DC and RF magnetron co-sputtering" and "DC and RF" both
resolve to `DC+RF` and stop being judged two different modes, while "DC" and "RF" stay apart. A value naming
no category is compared as ordinary text, never rounded to the nearest one. `categories` changes only
verdicts, so it is folded into `comparison_key` and leaves `extractor_key` alone: adding one re-compares the
stored facts instead of re-extracting them.

Adding a field is one entry. A `canonical_unit` must be one the converters know
(`Ω/sq`, `Ω·cm`, `nm`, `min`, `inch`, `%`, `℃`, `cm`, `W`, `sccm`, `rpm`) or startup fails rather than guessing.
`paperfacts fields` lists the table the package actually loaded. Editing the table changes `extractor_key`, so
affected papers are re-extracted and nothing stale is served.

## Data layout

One directory per document, holding every intermediate state:

```text
data/docs/<first 16 hex of sha256>/
├── identity.json                   full sha256, display name, origin
├── source.pdf                      the uploaded PDF (web uploads only)
├── raw/<backend>/                  parser's native output + meta.json
├── parsed/<backend>.md             every block's text behind its <!-- source: id --> marker, the same
│                                   rendering the extraction model reads
├── parsed/<backend>.artifact.json  the complete artifact: blocks with page + bbox, page geometry
├── facts/<backend>.<key>.json      one lane's sample-level extraction
├── comparisons/<key>.<key>.json    the two-lane comparison report
├── datasets/<key>.<key>.json       the consolidated per-sample table the web UI reads
├── dataset.xlsx                   consolidated paper/sample tables, written automatically by run
├── overlays/<backend>/page_*.png   bbox overlays
└── pages/<dpi>dpi/                 page renders for the web viewer
```

Every block carries `page` plus a bounding box normalised to `[0, 1]`, and every extracted value cites the
block ids it was read from, so any value can be mapped back to a rectangle on a page.

## Server deployment

On a GPU server the parsers run as long-running services and the main package talks to them over HTTP,
producing byte-identical output to the subprocess path. Compose files, Dockerfile, host scripts and GPU
assignments are in **[`deploy/`](deploy/README.md)**. The client side is two variables:

```bash
export PAPERFACTS_MINERU_URL=http://localhost:8002
export PAPERFACTS_PADDLE_URL=http://localhost:8080
uv run paperfacts run paper.pdf
```

Running the subprocess path directly on Linux also works and picks up CUDA automatically; restrict it
with `CUDA_VISIBLE_DEVICES` as usual.

## Results on real papers

Three transparent-conductive-oxide papers, DeepSeek `deepseek-chat` at temperature 0, parsers on an
Apple M5:

| | SnO₂:Ta, 10 pp. | GZO/ITO, 12 pp. | ATO, 6 pp. |
|---|---|---|---|
| MinerU blocks, kept for the prompt | 70 of 123 | 64 of 128 | 39 of 71 |
| PaddleOCR-VL blocks, kept | 82 of 176 | 78 of 191 | 43 of 91 |
| Prompt tokens per lane | 17.9K / 16.6K | 12.3K / 12.7K | 4.9K / 4.7K |
| Samples found, both lanes | 6 | 7 | 4 |
| Sample matching | exact, no model call | exact, no model call | exact, no model call |
| Facts compared | 2 agree, 2 conflict, 2 missing | 5 agree, 3 missing | 10 agree, 2 conflict |
| Values failing grounding | 1 | 0 | 1 |

Filtering page furniture and the bibliography removes 40–55% of blocks and about a fifth of the prompt
tokens before the model sees anything.

On a two-page comparison both parsers produced **identical block counts and type distributions**,
differing only in how they labelled running heads. Paragraph-level layout analysis is not where these two
disagree; text recognition, table structure, and how far the model is willing to read are.

The disagreements were informative rather than noisy, and every guardrail earned its place on real input:

- **Grounding caught a cross-block quote.** The model reported a target composition of
  `95% SnO2 and 5% Sb2O3` citing one block — but the sentence straddles two adjacent blocks, and only the
  second was cited. The value is real, so a quote crossing the junction between the cited block and its
  same-page neighbour counts as grounded; a quote lying entirely inside the neighbour still does not.
- **Scope enforcement caught a mislabelled measurement.** One lane reported a film's Ta dopant
  concentration (`0.74 at.%`) as the sputtering target's `component`. Four such values were dropped with
  an audited reason.
- **Repeated extraction quantified the noise.** Run three times, the `40 × 10 cm` target size and a
  `>80%` transmittance each appeared in only **one pass out of three**. Single-pass runs had been
  including or excluding them arbitrarily, and that flicker was being counted as parser disagreement.
- **The rules refused to guess.** One lane read a Scherrer crystallite size of `15.6 to 16.3 nm` as a film
  thickness where the other read `2 μm`; they were paired by numeric proximity, found incompatible in both
  value and condition, and returned AMBIGUOUS rather than letting either through.

## Limitations

- **Values that exist only in figures cannot be extracted.** A sheet-resistance curve that appears only in
  a plot is correctly reported as MISSING, not invented.
- **Rectangular target dimensions** such as `40 × 10 cm` are converted using the first number; a size-typed
  field would be needed to represent both.
- **The two lanes share one extractor**, so a mistake made by the language model itself — attributing a
  value to the wrong sample — correlates across lanes and AGREE will not catch it. Parser error and
  extractor error have to be counted separately when evaluating.
- **Attribution disagreements** where both lanes extracted the value but neither could place it on a
  sample are compared under the `unattributed` scope; the remaining case — placed in one lane,
  unattributed in the other — is still reported as a MISSING on the placed side.
- The field schema is currently twenty fields aimed at sputtered TCO films (`src/paperfacts/fields.py`).
  Adding a field is one table entry; the prompt, normalisation and tolerances follow from it.

## Development

```bash
uv run pytest                                              # unit tests; no models, no network
uv run pytest --run-parser                                 # integration; needs both parser environments
uv run ruff check src tests runners && uv run ruff format --check src tests runners
```

Tests never touch a real model or a real LLM: parser output comes from recorded fixtures of genuine runs,
and the LLM is a fake that also serves as an assertion surface for prompt content.

| Path | Contents |
|---|---|
| `src/paperfacts/` | One flat module per pipeline stage (see the package docstring), plus `web/` |
| `runners/` | The two PEP 723 parser scripts and their lockfiles |
| `deploy/` | Linux GPU server deployment |
| `tests/` | pytest suite and recorded parser fixtures |

Conventions are in [CLAUDE.md](CLAUDE.md).
