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

Extraction needs an OpenAI-compatible LLM; the default is DeepSeek `deepseek-chat`. The key is read from
`PAPERFACTS_LLM_API_KEY`, then `DEEPSEEK_API_KEY`, then `PAPERFACTS_LLM_API_KEY_FILE`, then a
`deepseek_api_key` file in the repository root (gitignored).

## Use

```bash
uv run paperfacts run paper.pdf            # parse both lanes, extract both, compare
uv run paperfacts serve                    # the web interface on http://127.0.0.1:8000
```

`run` is the whole pipeline; the individual stages are available separately and share the same caches:

```bash
uv run paperfacts parse   paper.pdf --backend both -v   # → data/docs/<sha>/parsed/
uv run paperfacts overlay paper.pdf --backend both      # draw block boxes on page images, to check by eye
uv run paperfacts extract paper.pdf                     # → data/docs/<sha>/facts/
uv run paperfacts compare paper.pdf                     # → data/docs/<sha>/comparisons/
```

### The web interface

```bash
uv run paperfacts serve                    # local
uv run paperfacts serve --host 0.0.0.0     # reachable from other machines
```

Drop a PDF on the left and processing starts, with live per-stage progress. The result is a fact-by-fact
comparison table, AGREE / CONFLICT / AMBIGUOUS / MISSING counts, and a page viewer: click any fact and
both lanes' source blocks light up on the rendered page — blue for MinerU, orange for PaddleOCR-VL. The
selected fact is part of the URL, so a link to one disputed number is shareable. Below the table are each
lane's raw sample records, from the paper's own wording through to the normalised value and back to the
block it came from. Papers processed from the command line appear in the library too, though only
uploaded ones carry their PDF and can be re-rendered on another machine. The API is documented at
`/api/docs`.

### Repeated extraction

Language models are not deterministic even at temperature 0: repeated extractions of the same paper
occasionally gain or lose a value. With a single pass that noise is indistinguishable from genuine
parser disagreement, which is the signal this tool exists to measure. Extract each lane several times and
keep only what a majority of passes agree on:

```bash
uv run paperfacts run paper.pdf --passes 3    # 3x the LLM calls, 3x the cost
```

Off by default. Each surviving value records the fraction of passes that produced it, so a 2/3 value stays
visibly weaker than a 3/3 one.

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

Everything is environment variables; there is no config file.

| Variable | Default | Meaning |
|---|---|---|
| `PAPERFACTS_DATA_ROOT` | `./data` | Where artifacts are written |
| `PAPERFACTS_MINERU_URL` | — | MinerU HTTP service; unset means run `runners/` as a subprocess |
| `PAPERFACTS_PADDLE_URL` | — | PaddleOCR-VL HTTP service; same |
| `PAPERFACTS_PADDLE_RENDER_DPI` | `200` | Page rasterisation DPI; must match between subprocess and HTTP |
| `PAPERFACTS_PADDLE_VL_BACKEND` | — | Hand the vision stage to an external server (`mlx-vlm-server`, `vllm-server`) |
| `PAPERFACTS_PADDLE_VL_SERVER_URL` | — | That server's URL |
| `PAPERFACTS_PADDLE_VL_MODEL_NAME` | — | Model name it serves |
| `PAPERFACTS_LLM_BASE_URL` | `https://api.deepseek.com` | Any OpenAI-compatible endpoint |
| `PAPERFACTS_LLM_MODEL` | `deepseek-chat` | Extraction model |
| `PAPERFACTS_LLM_API_KEY` | — | Key; see the resolution order above |
| `PAPERFACTS_LLM_CONTEXT_TOKENS` | `60000` | Context budget; a paper that exceeds it fails fast instead of being truncated |
| `PAPERFACTS_EXTRACTION_PASSES` | `1` | Majority-vote passes per lane |
| `PAPERFACTS_SUBPROCESS_TIMEOUT_S` | `3600` | Parser subprocess timeout |
| `PAPERFACTS_HTTP_TIMEOUT_S` | `900` | Parser HTTP timeout |
| `PAPERFACTS_LLM_TIMEOUT_S` | `300` | LLM request timeout |

## Data layout

One directory per document, holding every intermediate state:

```text
data/docs/<first 16 hex of sha256>/
├── identity.json                   full sha256, display name, origin
├── source.pdf                      the uploaded PDF (web uploads only)
├── raw/<backend>/                  parser's native output + meta.json
├── parsed/<backend>.md             Markdown with <!-- source: id --> markers
├── parsed/<backend>.sources.json   block list: page, bbox, type, offsets
├── parsed/<backend>.artifact.json  the complete artifact
├── facts/<backend>.<key>.json      one lane's sample-level extraction
├── comparisons/<key>.<key>.json    the two-lane comparison report
├── overlays/<backend>/page_*.png   bbox overlays
└── pages/<dpi>dpi/                 page renders for the web viewer
```

Every block carries `page` plus a bounding box normalised to `[0, 1]`, and the Markdown satisfies
`markdown[block.markdown_start:block.markdown_end] == block.content`, so any span of text can be mapped
back to a rectangle on a page.

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
  `95% SnO2 and 5% Sb2O3` citing one block — but the sentence straddles two blocks, and only the second
  was cited. The value is real; the citation was not complete, and it is flagged rather than presented as
  traceable.
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
- **Attribution disagreements are reported as two MISSINGs**, one per lane, rather than as a single
  labelled conflict.
- The field schema is currently nine fields aimed at sputtered TCO films (`src/paperfacts/extraction/fields.py`).
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
| `src/paperfacts/` | Models, parser clients, adapters, extraction, normalisation, comparison, storage, CLI, web |
| `runners/` | The two PEP 723 parser scripts and their lockfiles |
| `deploy/` | Linux GPU server deployment |
| `tests/` | pytest suite and recorded parser fixtures |

Conventions are in [CLAUDE.md](CLAUDE.md).
