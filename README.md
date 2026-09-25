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

This file is the whole manual: installing it on a Mac, deploying it on the GPU server, using the web
interface, every command, every configuration key, and what a blank cell means. `CLAUDE.md` is internal
working conventions for people editing the code, and nothing here depends on reading it.

## Why two parsers

A single PDF parser plus a language model will happily produce a confident, well-formatted table of
numbers that are subtly wrong — a column misread, a row shifted, a value attached to the wrong sample —
and nothing in the output says which ones. The failure is silent, which is the worst property a data
pipeline can have.

PaperFacts runs two structurally independent parsers over the same paper and extracts from each with an
identical prompt, model and retrieval. Where they agree, the number survived two different layout
analyses and two different OCR passes. Where they disagree, you have a specific, located, reviewable
question instead of a uniform wall of unearned confidence. Disagreement is the product.

The model is constrained structurally rather than by instruction: the JSON schema it must return has
`value_raw` and `unit_raw` and no `value` or `unit` field, so it has nowhere to put a converted number.
All unit conversion, scientific-notation parsing and tolerance comparison happen in ordinary, testable
Python. The model quotes; the code converts.

## Quick start on a Mac

Requires Apple silicon, Python 3.13 and [uv](https://docs.astral.sh/uv/). Python is pinned to
`>=3.13,<3.14`: MinerU caps below 3.14 and paddlepaddle publishes no 3.14 wheels.

```bash
uv sync --group dev                        # the main package: pure Python, seconds
uv sync --script runners/mineru_runner.py  # MinerU environment (torch, ~1-2 GB on first run)
uv sync --script runners/paddle_runner.py  # PaddleOCR-VL environment (paddlepaddle)
```

The two parsers live in **separate environments on purpose**: they both ship a `cv2` (opencv-python vs
opencv-contrib-python) and they pull in torch and paddlepaddle respectively, so they cannot be installed
together. Each is a [PEP 723](https://peps.python.org/pep-0723/) script under `runners/` with its own
committed lockfile, run via `uv run --script`. The main package never imports either one; it only reads
the files they write. Model weights download on first use; set `MINERU_MODEL_SOURCE=modelscope` on
networks where Hugging Face is slow.

Then the API key. Extraction needs an OpenAI-compatible LLM endpoint; the shipped default is an Alibaba
Cloud Model Studio (百炼) workspace serving `deepseek-v4.1-flash` through its `compatible-mode/v1`
address:

```bash
cp .env.example .env     # then set PAPERFACTS_LLM_API_KEY
```

The key is looked for in `PAPERFACTS_LLM_API_KEY`, then `DEEPSEEK_API_KEY`, then the file named by
`PAPERFACTS_LLM_API_KEY_FILE`, then a `deepseek_api_key` file in the repository root. All of those are
gitignored, and `config.json` has nowhere to put a key, which is the point.

### Starting everything

```bash
scripts/dev_up.sh                # the usual way to work
scripts/dev_up.sh --host 0.0.0.0 --port 8765    # arguments are passed to `paperfacts serve`
```

`dev_up.sh` does three things in order. It clears a hidden flag that keeps appearing on `.venv`; it
starts an `mlx-vlm` server on port 8111 for PaddleOCR-VL's vision stage unless something already listens
there, waiting up to 60 seconds and refusing to continue if it never comes up; and it starts
`paperfacts serve` with `PAPERFACTS_PADDLE_VL_BACKEND`, `PAPERFACTS_PADDLE_VL_SERVER_URL` and
`PAPERFACTS_PADDLE_VL_MODEL_NAME` filled in for whatever `.env` left unset. The MLX step is not a nicety:
PaddleOCR-VL's vision model on in-process CPU inference takes hours per paper, and a minute or two
through MLX.

```bash
scripts/dev_down.sh              # stops the mlx-vlm server it started
```

Stop the web server with Ctrl-C in its own terminal; `dev_down.sh` only handles the background process,
whose pid it reads from `data/mlx_vlm_server.pid`.

On this machine some external tool periodically marks the whole `.venv` as hidden, and Python 3.13 then
skips its `.pth` files, so `import paperfacts` suddenly raises `ModuleNotFoundError`; `scripts/dev_fix_venv.sh`
clears the flag without a reinstall, and `dev_up.sh` runs it for you.

To drive the MLX path by hand instead, the three variables are:

```bash
uvx --python 3.13 --from "mlx-vlm>=0.3.11" mlx_vlm.server --port 8111   # leave running
export PAPERFACTS_PADDLE_VL_BACKEND=mlx-vlm-server
export PAPERFACTS_PADDLE_VL_SERVER_URL=http://localhost:8111/
export PAPERFACTS_PADDLE_VL_MODEL_NAME=PaddlePaddle/PaddleOCR-VL-1.6
```

## Deploying on the Linux GPU server

On the server the two parsers run as long-running services and the main package talks to them over HTTP,
producing byte-identical output to the subprocess path used on a Mac. Everything needed is under
`deploy/`: `compose.yaml` for the three containers, `mineru.Dockerfile` for the one image that has to be
built locally, `vllm_config.yaml` for the VLM's memory and concurrency, and `host/*.sh` for the
bare-metal route when Docker is unavailable. `deploy/README.md` is the long form; this is the shape of it.

**Updating the running web service.** On the current workstation the web UI, the PaddleOCR-VL vLLM lane
and the cloudflared tunnel run as `systemctl --user` units (`paperfacts.service`, `pf-vllm-paddle.service`,
`paperfacts-tunnel.service`), and the venv is an editable install, so a deployment is pull, test, restart,
verify: `scripts/deploy.sh --pull --rerun` does exactly that and then re-runs, from the LLM cache, every
document whose stored results the new cache keys displaced. `scripts/deploy.sh --help` lists the rest
(`--check` reports whether the running service is stale). It refuses to restart only while a job is queued or
running (`GET /api/jobs`; `--force` overrides), since a restart drops those; a document that was never run or
failed for good does not block it. The web password is read from `.env` the way the service reads it, and
reaches `curl` on stdin, never on its command line.

**The current server has a single GPU (id 0).** All three services default to it; GPU ids are set per
service by env var (Docker Compose's `device_ids`, or `CUDA_VISIBLE_DEVICES` for the host scripts), so a
multi-GPU host can spread them out instead.

| GPU | Runs | Port |
|---|---|---|
| 0 | `mineru-router` (one `mineru-api` worker per visible GPU), `paddleocr-vl-api` and the `paddleocr-vlm-server` vLLM service it calls, sharing the card | 8002 `POST /file_parse`; 8080 `POST /layout-parsing`; vLLM on 8118, local only |

Docker Compose pins the allocation with `device_ids`; the host scripts pin it with `CUDA_VISIBLE_DEVICES`
and `deploy/host/_common.sh` checks it's actually set before anything loads, so a missing id exits
immediately instead of being discovered after a model has half-loaded. Do not publish port 8118: it
is the VLM's raw OpenAI-compatible endpoint, meant only for the API layer beside it.

```bash
cd deploy
cp .env.example .env
docker compose build mineru-router   # 20-40 min, pre-downloads MinerU weights into the image; use tmux
docker compose up -d
curl http://localhost:8002/health
curl http://localhost:8080/health
```

Cold starts are slow by nature — each MinerU worker loads three model sets, and vLLM loads weights and
compiles a CUDA graph in roughly five minutes — so read `docker compose logs -f` before concluding that
something is wrong.

### Pointing the orchestrator at the services

The main package needs exactly two variables. Both set means HTTP to the services; neither set means run
the `runners/` scripts as subprocesses, which is how a Mac works.

```bash
export PAPERFACTS_MINERU_URL=http://localhost:8002
export PAPERFACTS_PADDLE_URL=http://localhost:8080
```

From another machine, replace `localhost` with the server's hostname and open 8002 and 8080 in the
firewall.

### The web app on the server

`.env` on the server holds two secrets: the LLM API key, and `PAPERFACTS_WEB_PASSWORD`. Set the password
before binding to anything but localhost.

```bash
echo 'PAPERFACTS_LLM_API_KEY=...'  >> .env
echo 'PAPERFACTS_WEB_PASSWORD=...' >> .env     # the username is web.username in config.json
uv run paperfacts serve --host 0.0.0.0 --port 8000
```

With `PAPERFACTS_WEB_PASSWORD` set, every route, `/api` included, answers 401 until a browser or client
sends HTTP Basic credentials, so an open tunnel cannot upload PDFs or spend tokens. Unset, the app is
open, which is what a laptop wants. Run it long-lived in tmux.

Either way, a request that changes something (every POST) is refused with 403 when the browser says it
came from another site, so a page elsewhere cannot use a logged-in browser to queue work. The browser's
`Sec-Fetch-Site` decides when it is sent (`same-origin` or `none` pass), so a proxy that rewrites `Host`
does not lock out the UI; only without it are `Origin` or `Referer` compared with `Host` (or
`X-Forwarded-Host`). A request naming no origin at all (curl, `scripts/deploy.sh`) is not a browser and
passes. Every response forbids framing and MIME sniffing. An upload carries one PDF within
`server.max_upload_mb`: a declared `Content-Length` over it is refused unread, and the bytes that arrive
are counted, so a chunked upload without a length works and a larger one is refused as it passes the limit.

### How code reaches the server

Development happens on the Mac, is committed and pushed to GitHub, and pulled on the server. **Do not try
to operate the server over ssh from the dev machine.** Someone with a session on the server pulls and
restarts it there.

## Using the web interface

`http://<host>:8000`. The interface is in Chinese; this section names the strings on screen.

**The rail on the left** is the document library, with the drop zone above it. Drag a PDF in, or press
「选择文件」, and processing starts on upload; 「忽略缓存，全部重跑」 next to the drop zone forces every
stage to run again. Each library entry shows the paper's name and four badges: 一致 (both lanes agreed),
冲突 (the lanes read different values), 不确定 (the pipeline could not decide) and 缺失 (only one lane
found it), plus one dot per pipeline stage with a done/total count. Several PDFs can be dropped or picked
at once; each is uploaded as its own request. On a narrow screen the list folds into 「文档列表」.
「处理全部未完成」 queues every document that can run and is not already finished (exported) under the
current keys, one job each, in library order; a document already queued or running simply gets its
existing job back, so pressing it twice costs nothing. Documents with neither a PDF nor a cached parse
are skipped with a reason. Up to `web.max_parallel_documents` (default 3) documents run at once, started in
the order they were queued; a document is never worked on twice at the same time, so 「强制重跑」 on a
paper that is already running waits for that run to end.

**The home page** is 论文结果总表: one row per processed paper, showing the sample that paper selected
across the field columns, with a link into each document and a 「下载全部 Excel」 button for the whole
library.

**A document page** reads top to bottom.

1. The header carries the display name, the document id, 「强制重跑」 and 「重新处理」.
2. The stage list and its progress: `parse:mineru`, `parse:paddleocr_vl`, `figures` (读图, skipped unless
   switched on), `extract:mineru`, `extract:paddleocr_vl`, `compare`, `export`.
3. KPI tiles: the 一致 / 冲突 / 不确定 / 缺失 counts.
4. 结果表（按样品） — the deliverable.
5. 图中读数, only when the paper's charts were read; see [Reading figures](#reading-figures).
6. 事实对照 and the page viewer beside it.
7. 样品记录, collapsed.

### 结果表（按样品）

One row per sample after both lanes are merged, one column per field, plus a 靶材（论文级） row for the
paper-level target values. The row chosen as the paper's row is marked ★ 论文行.

A cell that carries a value shows the number and the field's canonical unit, and a small badge saying how
it was decided: **双路** when both lanes agreed, or the lane's own name — **MinerU** or **PaddleOCR-VL** —
when only one lane had it. Clicking a value highlights, on the rendered page, the blocks it was merged
from. A value the paper stated once for a whole sample series says so in the tooltip rather than in a
badge.

**An empty cell is a refusal, not a gap.** The pipeline declined to commit a value and the reason is in
the cell's tooltip and its accessible name, so a screen reader gets it without a hover. Clicking or
pressing Enter on an empty cell jumps down to the two lanes' records for that sample and field, so the
refusal can be checked against what each lane actually read.

Above the table: 选择字段 chooses which of the twenty columns to show, 显示空字段 adds back the columns
that are empty for every row, 复制表格 copies the table as TSV for a spreadsheet, and 下载 Excel
downloads this document's workbook.

### 事实对照 and the page viewer

The comparison table is one row per compared fact: 状态, 样品, 字段, 条件, the MinerU reading, the
PaddleOCR-VL reading, and 说明. Click any row and both lanes' source blocks light up on the rendered page
in the viewer beside it — blue for MinerU, orange for PaddleOCR-VL, always with text as well as colour.
The selected fact is part of the URL (`#/doc/<id>/fact/<n>`), so a link to one disputed number survives a
reload and can be sent to someone else.

### 样品记录

Collapsed by default: each lane's raw sample records, from the paper's own wording through to the
normalised value and back to the block it came from. A value tagged 全系列 was stated once for the whole
sample series and written onto each sample by the code, not read separately for this one.

Papers processed from the command line appear in the library too, though only uploaded ones carry their
PDF and can be re-rendered on another machine. The HTTP API is documented at `/api/docs`.

## Command line

```bash
uv run paperfacts run paper.pdf            # parse both lanes, extract both, compare, write the workbook
uv run paperfacts run paper.pdf --figures  # the same, and read the paper's charts with the vision model
uv run paperfacts batch template_files --output data/exports/template_files.xlsx
uv run paperfacts batch template_files --jobs 4   # four papers at once; default web.max_parallel_documents
uv run paperfacts serve                    # the web interface on http://127.0.0.1:8000
uv run paperfacts fields                   # list the field table the package actually loaded
```

| Command | Purpose |
|---|---|
| `run <pdf>` | Parse, extract, compare and save `dataset.xlsx` for one paper |
| `batch <pdf or dir>` | Recursively process every PDF and write one workbook for all of them |
| `export <pdf or dir>` | Rebuild that workbook from cached results, with no parser and no LLM calls |
| `parse <pdf>` | Parse into Markdown with provenance markers, a block list and the full artifact |
| `extract <pdf>` | Extract sample-level records from parsed Markdown. Needs `parse` |
| `compare <pdf>` | Match samples across lanes and compare their fields. Needs `extract` (which implies `parse`) |
| `overlay <pdf>` | Draw block boxes onto page images, to check provenance by eye. Needs `parse` |
| `serve` | Serve the web interface |
| `fields` | Print the loaded field table, so an edit to `config.json` can be checked at a glance |

The flags worth knowing:

- `--force` ignores caches and redoes that step. On `extract` it re-calls the LLM, which costs money; on
  `parse` it re-runs the parser, which costs minutes of GPU time.
- `--passes N` extracts each lane N times and keeps only what a majority of passes produced. N times the
  calls, N times the cost.
- `--mode document|passage` picks how the model is asked; see below.
- `--figures` / `--no-figures` on `run` and `batch` switches the figures stage on or off for this run,
  over `figures.enabled`; `--force-figures` re-reads the charts without redoing anything else.
- `--backend mineru|paddleocr_vl|both` on `parse`, `extract` and `overlay` runs one lane or both.
- `--output` / `-o` names the Excel workbook for `batch` and `export`.
- `--jobs N` / `-j N` on `batch` processes N papers at once (default `web.max_parallel_documents`, 3);
  `--jobs 1` is the old one-after-another run.
- `--data-root` overrides the data directory; `--verbose` / `-v` prints INFO logs.
- `overlay` also takes `--dpi` and `--pages 0,3,4` (0-based).

`batch` walks subdirectories and accepts `.PDF` as well as `.pdf`. Identical PDF content is processed
once. Every completed or failed paper checkpoints the workbook atomically, so re-running the same command
reuses the caches and rebuilds the table without appending duplicate rows. Failed papers are listed in the
运行记录 sheet, processing continues past them, and **the command exits with status 1 if any paper
failed.**

With `--jobs` above 1 the papers overlap, but the workbook, the failure list and the summary are in input
order, so the table is the one a serial run writes; each progress line is prefixed with its paper
(`[3/28 x.pdf parse:mineru] done ...`). Stopping one (Ctrl-C, or a workbook that cannot be written) starts
no new paper and prints `[batch] failed stopping; waiting for N running papers to reach a stage boundary`:
each running paper finishes the stage it is in -- a request already paid for is worth caching -- and stops
there, marked `skipped`. That can take as long as one stage (minutes for a first parse or extraction).
Stopped papers are neither rows nor failures, and the caches make the next run resume where this one
stopped. `export` always reads one paper at a time, since it only reads the caches.

Three things keep parallel papers from multiplying the load:

- **Parsing is one paper per parser at a time.** The server has one GPU, and two papers sent to the same
  parser service only compete for its memory; MinerU and PaddleOCR-VL may parse two different papers side
  by side, and with a parser service configured the two lanes of one paper parse side by side too. On a workstation the two runner subprocesses share one lock instead, because both model sets do
  not fit in its memory at once. A cached parse never waits.
- **Model requests share `llm.max_in_flight`** (default 8) across every paper, lane and stage in the
  process, so the Model Studio rate limit sees at most that many open requests however many papers run.
- **Each paper writes only its own directory**, the LLM cache is written atomically under unique temp
  names, and the batch workbook is checkpointed under a lock in input order.

The defaults are conservative on purpose. 3 papers at once is enough to keep 8 requests in flight while
one paper waits for a parser; more papers than that mostly queue for the GPU or the limit. 8 in flight is
twice one paper's two lanes at `llm.concurrency` 4, and a Model Studio workspace answering 429 costs a
retry out of a fixed budget where a queued request costs only time. Raise `llm.max_in_flight` first if
the endpoint takes it, then `--jobs`. One setting serves both the web queue and `batch` rather than two,
because both are bounded by the same GPU and the same endpoint; `--jobs` overrides it for one run. The
limits are per process: a `batch` run beside the web server has its own.

`export` is offline. It requires cached extractions and comparisons matching the current model, field
schema and comparison rules, and reports outdated or absent results as failures rather than silently
producing an old table. On a Mac, bring the MLX service up before `batch` touches an uncached PDF.

## Configuration

Two files at the repository root. **`config.json` holds everything that is not a secret** and is meant to
be edited; **`.env` holds the API key and the web password** and is gitignored. Three layers decide a
value: the built-in constants in `config.py`, then `config.json`, then the environment (including what
`.env` puts there). Never edit a built-in constant to change a default — they are the baseline the cache
keys treat as "unedited", so editing one renames every cached file. Change `config.json`.

### `llm`

| Key | Meaning |
|---|---|
| `base_url` | Any OpenAI-compatible endpoint; a trailing slash is stripped |
| `model` | Model name sent with every request. Default `deepseek-v4.1-flash` |
| `timeout_s` | Per-request timeout. Default 600, because this model reasons before it answers |
| `context_tokens` | The window the prompt is planned against. Default 200000 |
| `temperature` | 0 to 2. Default 0.0 |
| `max_tokens` | Completion budget, hidden reasoning included; must be below `context_tokens`. Default 65536 |
| `concurrency` | How many of one lane's field questions are in flight at once. Default 4 |
| `max_in_flight` | How many model requests, text and vision, the whole process has on the wire at once. Default 8 |
| `reasoning_effort` | `null` \| `"none"` \| `"low"` \| `"medium"` \| `"high"` |
| `inventory_reasoning_effort` | `null`/`"inherit"` \| `"omit"` \| `"none"`…`"high"` |
| `retry_attempts` | Default 4. `Retry-After` from the endpoint is honoured, up to 120 s |
| `retry_backoff_s` | Default 2.0 |

`reasoning_effort` is how much hidden reasoning the endpoint is asked for before it answers, sent as the
OpenAI-shaped `reasoning_effort` parameter. `null` omits the parameter entirely, which is the shipped
default and the measured best answer. It changes what the model is asked, so changing it writes a new
`extractor_key` and re-extracts.

`inventory_reasoning_effort` gives passage mode's one inventory question its own effort. That question
alone spends 11k–17k hidden reasoning tokens per lane, about 70 % of a run's completion tokens, while the
field questions after it usually reason in the low hundreds. It has three answers, and they are three
different requests:

- `null` or `"inherit"` — send the inventory question exactly as the client builds it, carrying whatever
  `reasoning_effort` says. This is the shipped value.
- `"omit"` — send that one question with no `reasoning_effort` parameter at all, while every field
  question still carries the client's. This is the only way to say that.
- `"none"`, `"low"`, `"medium"`, `"high"` — the effort to send for that question.

Both lanes always get the same value, and anything but the baseline writes a new `extractor_key`, in
passage mode only, since document mode never asks the question.

`concurrency` is the one knob here that changes only *when* requests are sent, never what they contain, so
it stays out of both cache keys: raising or lowering it never re-extracts and never re-compares. The two
lanes always run as a pair, so one paper has at most twice this many requests open, and `max_in_flight`
caps the sum over every paper running. Set it to 1 to send every question strictly one after another.

`max_in_flight` is the ceiling over everything else: every lane of every document running at once, the
figures stage and the sample matching all take a slot before a request goes out and give it back when the
answer arrives. A cache hit takes none, so a resumed run replays at disk speed. The slot covers the HTTP call
only, never a retry's backoff, so a request waiting out a 429 does not hold up the others. 8 is two
documents' worth of lanes at the default `concurrency`; it is a guess at what one Model Studio workspace
serves without answering 429, kept low because a throttled request costs a retry out of a fixed budget
while a queued one costs only time. The limit is per process: the web server and a `batch` run at the same
time each have their own. Like `concurrency`, it is in neither cache key.

### `extraction`

| Key | Meaning |
|---|---|
| `mode` | `"passage"` (default) or `"document"` |
| `passes` | Extract each lane this many times and keep the majority. Default 1 |
| `candidate_limit` | How many blocks matched only by a unit a field question may show; blocks naming the field always come. Default 8 |

**Passage mode** asks which samples the paper reports, then asks about one field at a time, showing only
the blocks retrieved for that field. Retrieval is ordinary code, not a model call: every block naming the
field by one of its keywords, plus the best blocks matched only by a unit up to `candidate_limit`. A table's
caption, and the other half of a paragraph a page or column break cut in two, come along with whichever
half was picked. Both lanes get identical retrieval rules, so the comparison still measures the parsers
and not the retrieval. **Document mode** hands the whole paper over and asks
for everything at once; on a fifteen-thousand-token paper the model loses its place, cites blocks that
merely discuss a number, and never mentions fields the paper states in passing.

`passes` votes on `(field, number, unit)` per rank — never on the wording of the measurement condition, so
a paraphrase between passes is not counted as disagreement — and the sample inventory is asked once per
lane, so every pass sees the same sample ids. A paper that reports the same number under two conditions
keeps both. Each surviving value records the fraction of passes that produced it, so a 2/3 value stays
visibly weaker than a 3/3 one. Measured on three papers, a second pass reproduces **75–90 %** of a first
pass's values at temperature 0. Two passes are therefore a reproducibility filter at twice the model cost,
not a way to find more.

Sample ids are keyed by one rule wherever samples meet: placing a value on a sample, merging passes, and
pairing the two lanes before the model is asked. Spaces, hyphens, underscores and punctuation are dropped
(only two adjacent numbers keep a boundary) and word case is folded, so `O2-100 sccm` and `O₂ 100sccm`, or
`WOx` and `WO_x`, are one sample. Greek letters, decimals, a leading or `=`-sign and the case of a trailing
single-letter suffix are kept, so `α-ITO` and `β-ITO`, or `ITO-a` and `ITO-A`, stay two. A LaTeX `\alpha`
(or `\varepsilon`) counts as the letter, so both lanes key the one sample alike.

### `figures`

Reading property-vs-condition charts with a vision model; see [Reading figures](#reading-figures).

| Key | Meaning |
|---|---|
| `enabled` | Run the `figures` stage. Default `false`: it costs about a minute of the vision model per chart |
| `model` | The vision model. Default `qwen3.7-plus`, the only one measured accurate enough; it uses the `llm` endpoint and key |
| `max_per_document` | At most this many chart panels are read per paper. Default 12 |
| `dpi` | DPI the chart is cropped from the page at. Default 200 |
| `max_pixels` | Largest crop area sent; bigger crops are shrunk here, not by the endpoint. Default 2000000 |
| `timeout_s` | Per request. Default 300; one failed request is retried once |

### `parsers`, `server`, `web`, `overlay`, `comparison`

| Key | Meaning |
|---|---|
| `parsers.mineru_url` / `parsers.paddle_url` | Empty means run the `runners/` script as a subprocess; set means call that service over HTTP |
| `parsers.paddle_render_dpi` | DPI pages are rasterised at for PaddleOCR-VL. Default 200. The subprocess and HTTP paths must agree or their pixel coordinates are not comparable |
| `parsers.paddle_vl_backend` / `paddle_vl_server_url` / `paddle_vl_model_name` | Hand PaddleOCR-VL's vision stage to an external server, as `dev_up.sh` does with MLX |
| `parsers.subprocess_timeout_s` | Default 3600: a first subprocess run downloads weights |
| `parsers.http_timeout_s` | Default 900. Per request; a timeout, a connection error or a 5xx is retried twice with a 5 s / 10 s backoff before the parse fails (PaddleOCR-VL retries the one page; MinerU re-sends the whole paper, but not after a read timeout, when the service is most likely still parsing it) |
| `parsers.uv_bin` | The `uv` executable used to launch the runner scripts |
| `server.host` / `server.port` | Defaults `127.0.0.1` and 8000 |
| `server.max_upload_mb` | Default 200 |
| `server.page_dpi.default` / `.min` / `.max` | Page renders for the viewer. Defaults 110, 50, 220 |
| `web.username` | HTTP Basic username. Default `paperfacts`. The password is never here |
| `web.max_parallel_documents` | Documents processed at once, by the web job queue and by `batch` (unless `--jobs` says otherwise). Default 3 |
| `overlay.dpi` | Default 150 |
| `comparison.ambiguous_match_confidence` | Below this, a sample match is AMBIGUOUS rather than accepted. Default 0.6 |
| `condition_keywords` | The words that mark a measurement condition worth recording |
| `data_root` | Where everything is written. Default `data` |

### Environment overrides

Every scalar setting also has a `PAPERFACTS_*` variable that wins over the file, which is how one machine
points at its own services without editing the shared file:

`PAPERFACTS_DATA_ROOT`, `PAPERFACTS_REPO_ROOT`, `PAPERFACTS_UV_BIN`, `PAPERFACTS_MINERU_URL`,
`PAPERFACTS_PADDLE_URL`, `PAPERFACTS_PADDLE_RENDER_DPI`, `PAPERFACTS_PADDLE_VL_BACKEND`,
`PAPERFACTS_PADDLE_VL_SERVER_URL`, `PAPERFACTS_PADDLE_VL_MODEL_NAME`, `PAPERFACTS_SUBPROCESS_TIMEOUT_S`,
`PAPERFACTS_HTTP_TIMEOUT_S`, `PAPERFACTS_LLM_BASE_URL`, `PAPERFACTS_LLM_MODEL`,
`PAPERFACTS_LLM_TIMEOUT_S`, `PAPERFACTS_LLM_CONTEXT_TOKENS`, `PAPERFACTS_LLM_TEMPERATURE`,
`PAPERFACTS_LLM_MAX_TOKENS`, `PAPERFACTS_LLM_REASONING_EFFORT`,
`PAPERFACTS_LLM_INVENTORY_REASONING_EFFORT`, `PAPERFACTS_LLM_CONCURRENCY`, `PAPERFACTS_LLM_MAX_IN_FLIGHT`,
`PAPERFACTS_LLM_RETRY_ATTEMPTS`, `PAPERFACTS_LLM_RETRY_BACKOFF_S`, `PAPERFACTS_EXTRACTION_MODE`,
`PAPERFACTS_EXTRACTION_PASSES`, `PAPERFACTS_CANDIDATE_LIMIT`, `PAPERFACTS_SERVER_HOST`,
`PAPERFACTS_SERVER_PORT`, `PAPERFACTS_MAX_UPLOAD_MB`, `PAPERFACTS_PAGE_DPI`, `PAPERFACTS_PAGE_DPI_MIN`,
`PAPERFACTS_PAGE_DPI_MAX`, `PAPERFACTS_OVERLAY_DPI`, `PAPERFACTS_WEB_USERNAME`, `PAPERFACTS_WEB_MAX_PARALLEL_DOCUMENTS`,
`PAPERFACTS_WEB_PASSWORD`, `PAPERFACTS_FIGURES_ENABLED` (`true`/`false`), `PAPERFACTS_FIGURES_MODEL`,
`PAPERFACTS_FIGURES_MAX_PER_DOCUMENT`, `PAPERFACTS_FIGURES_DPI`, `PAPERFACTS_FIGURES_MAX_PIXELS`,
`PAPERFACTS_FIGURES_TIMEOUT_S`.

`PAPERFACTS_CONFIG` points at a different configuration file altogether. An empty string counts as unset,
and a value that will not parse as a number names the variable in the error.

Three settings are **file-only**, because a single environment variable is the wrong shape for them:
`fields`, `condition_keywords` and `comparison.ambiguous_match_confidence`.

Secrets live only in `.env`: `PAPERFACTS_LLM_API_KEY` and `PAPERFACTS_WEB_PASSWORD`. `.env` is loaded
without overriding what the environment already holds.

### The field table

`config.json`'s `fields` list **is** the schema. Each entry drives the description the model is given, the
keywords retrieval searches for, the unit everything is converted to, and how close two numbers have to be
to count as the same fact.

```jsonc
{
  "name": "sheet_resistance",
  "group": "film",                          // target = paper-level; process / film = per sample
  "kind": "numeric",                        // numeric | composition | text
  "description": "Sheet resistance of the film (Ω/sq).",
  "label": "方阻",                            // Chinese column header; display only
  "description_zh": "所选样品的薄膜方块电阻。",   // Chinese explanation; display only
  "keywords": ["sheet resistance", "sheet resistivity", "Rs", "R_s"],
  "canonical_unit": "Ω/sq",
  "rel_tol": 0.02,                          // |a-b| <= max(rel_tol * max(|a|,|b|), abs_tol); both >= 0
  "abs_tol": 0.0,
  "condition_hint": null,                   // what to record alongside, e.g. a wavelength
  "bare_number": "reject",                  // reject | assume_canonical | percent_or_fraction (only with "%")
  "valid_range": {"max": 500},              // optional plausible range in canonical_unit; min and/or max
  "condition_preference": ["400-800", "550"] // optional: which measurement fills the dataset cell
}
```

`description` is the English sentence the model is told to look for. `label` and `description_zh` are
display only — the web table prints the label above the column and the description in its tooltip, and the
Excel 字段说明 sheet prints both — so editing either changes neither cache key. `keywords` steers
passage-mode retrieval and nothing else. A `text` field may add `categories`, the closed set of answers it
accepts, spelled the way the output should spell them: with `"categories": ["DC", "RF", "pulsed DC",
"DC+RF", "HiPIMS"]` on `mode`, both "DC and RF magnetron co-sputtering" and "DC and RF" resolve to `DC+RF`
and stop being judged two different modes, while "DC" and "RF" stay apart. A value naming no category is
compared as ordinary text, never rounded to the nearest one. `categories` changes only verdicts, so adding
one re-compares the stored facts instead of re-extracting them.

A numeric field may declare `valid_range`, the plausible values in its `canonical_unit`, with either end
open. The model is told the range with its field question, and a value whose converted number still falls
outside it is dropped with the reason in the lane's `dropped` audit (the web's 清洗记录). It is meant for
the confusions a unit cannot catch: the spin-coating rpm of an absorber read as the substrate rotation, the
thickness of a wafer or a glass substrate read as the electrode's. The shipped table caps `rotation_speed`
at 100 rpm and `thickness` at 5000 nm and floors `transmittance` at 60 %. A value that
cannot be converted is kept, since there is no number to judge. A range changes the prompt and which values
survive, so it moves both cache keys; a field without one keeps the keys it had.

A `canonical_unit` must be one the converters know (`Ω/sq`, `Ω·cm`, `nm`, `min`, `inch`, `%`, `℃`, `cm`,
`W`, `sccm`, `rpm`, `Pa`) or startup fails, naming the field and the file, rather than guessing. Adding a field is one table entry; the prompt,
normalisation and tolerances follow from it. `rel_tol` and `abs_tol` only decide verdicts, so editing one
re-compares the stored facts instead of re-extracting them. Tolerances may not be negative, and
`percent_or_fraction` is only accepted on a `%` field. `uv run paperfacts fields` prints what was actually
loaded.

A sample often has one field measured several ways -- transmittance averaged over 400-800 nm, at 550 nm,
over 400-1800 nm -- and the dataset has one cell for it. The cell takes the measurement stated in the same
block as the rest of the sample's row; failing that, the first entry of `condition_preference` that picks
exactly one condition. An entry names the numbers a condition states, so `"400-800"` matches "average
400–800 nm" and "from 400 to 800 nm" alike. When one entry matches several conditions in a lane, the one
that says average / avg / mean / AVT is taken and the others (a peak, a minimum, an unlabelled range) are set
aside; without such an average the next entry is tried. The shipped transmittance preference is 400-800,
380-780, 400-700, 550, then 400-1100 nm, last so that a paper stating both keeps the 550 nm value it has
always committed. If none of that settles it the cell stays empty as
`multiple_conditions`. Every measurement stays in the facts either way. The preference changes only which
cell is committed, so editing it re-compares without re-extracting.

The twenty-three shipped fields are aimed at sputtered transparent-conductive-oxide films:

| Field | 中文名 | Group | Kind | Unit |
|---|---|---|---|---|
| `component` | 靶材成分 | target | composition | — |
| `resistance` | 靶材电阻率 | target | numeric | Ω·cm |
| `density` | 靶材密度 | target | numeric | % |
| `inch` | 靶材尺寸 | target | numeric | inch |
| `sputtering_time` | 溅射时间 | process | numeric | min |
| `sputtering_power` | 溅射功率 | process | numeric | W |
| `mode` | 溅射模式 | process | text (categories) | — |
| `ar_flow_rate` | Ar 流量 | process | numeric | sccm |
| `o2_flow_rate` | O2 流量 | process | numeric | sccm |
| `h2_flow_rate` | H2 流量 | process | numeric | sccm |
| `o2_ratio` | O2 比例 | process | numeric | % |
| `h2_ratio` | H2 比例 | process | numeric | % |
| `working_pressure` | 工作气压 | process | numeric | Pa |
| `target_substrate_distance` | 靶基距 | process | numeric | cm |
| `substrate_axis_distance` | 基片偏轴距 | process | numeric | cm |
| `substrate_temperature` | 基片温度 | process | numeric | ℃ |
| `annealing_temperature` | 退火温度 | process | numeric | ℃ |
| `annealing_time` | 退火时间 | process | numeric | min |
| `rotation_speed` | 转速 | process | numeric | rpm |
| `sheet_resistance` | 方阻 | film | numeric | Ω/sq |
| `resistivity` | 电阻率 | film | numeric | Ω·cm |
| `transmittance` | 透光率 | film | numeric | % |
| `thickness` | 厚度 | film | numeric | nm |

`transmittance` carries a `condition_hint` asking for the wavelength or spectral range; `density` and
`transmittance` read a bare number as a percent or a fraction, and every other numeric field rejects a
number with no unit rather than assuming one.

## Reading figures

Many papers give a sample's sheet resistance or resistivity only as a marker on a chart, "Rs vs O2 flow",
where neither text lane can see it. The `figures` stage, off by default, crops such charts out of the page
and asks a vision model (`qwen3.7-plus`) to read them. It starts after parsing, runs beside the two
extraction lanes and is joined before export; switch it on with `figures.enabled`,
`PAPERFACTS_FIGURES_ENABLED=true` or `--figures`. `--force` does not re-read charts and `--force-figures`
re-reads only them, since each costs minutes of a different model.

- **Which charts.** Figure and caption blocks that sit together on a page are split among the captions
  that start "Fig." / "Figure" / "FIGURE" by geometry: each panel goes to the nearest such caption on the
  side that caption was written on. MinerU captions a panel of a multi-panel figure with the neighbouring
  panels' labels ("(a) (c)"), which name nothing, and hangs the figure's caption under whichever panel it
  was attached to, so reading order alone would hand panels to the next figure. A figure is read when that caption names a
  film property (sheet resistance, resistivity, transmittance, thickness) by one of the field's retrieval
  keywords, and then every panel is asked about separately, up to `figures.max_per_document` panels per
  paper. The boxes are MinerU's, or PaddleOCR-VL's when there is no MinerU parse. A panel that turns out to
  be a spectrum or an XRD pattern is refused by the model and yields nothing.
- **What a reading is.** The model reports each marker's y in the axis's own unit, multiplier included
  ("25" on an axis titled "[10^2 Ω/sq]"), and the code converts it to the field's canonical unit. Every
  reading is **approximate**, labelled ±10 % on a linear axis and ±20 % on a log axis or a chart with four
  or more series: the measured p90 error was 6 % on ordinary charts and 13.6 % over all of them
  (`.omc/research/figure-reading-accuracy.md`).
- **What a reading is not.** The chart's x is shown for the reader only: the models round it to the nearest
  tick label, so it never creates a sample or decides which sample a point is. Readings never fill a cell
  of 结果表 or of the 论文数据 / 样品数据 sheets and never take part in the two-lane comparison; `dataset.py`
  does not even import the stage. They have their own sheet, 图中读数, their own endpoint
  (`GET /api/documents/{id}/figures`, read straight from the readings file), and their own section on the
  document page, where clicking one outlines the chart on the page.
- **Cost and failure.** About a minute per chart, two for a crowded one; requests overlap
  `llm.concurrency` at a time, time out after `figures.timeout_s` and are retried once. A failure marks only
  the `figures` stage failed: the paper is still extracted, compared and exported. A request that failed,
  a reply cut off at the token limit (never cached) and an answer that could not be used are asked again
  on the next run -- the last with the cache bypassed -- while the answered panels replay from the LLM cache.
- **Stored.** `figures/<figure_key>.json` per document. `figure_key` hashes the vision model and its
  sampling, `figures.dpi`, `figures.max_pixels`, `figures.max_per_document`, the film fields' descriptions,
  keywords and units, and the source of `figures.py`, `normalize.py` and `passages.py`. Stored readings are
  shown and exported even when the stage is switched off for a later run. With nothing under the current
  key, the newest older file is shown and marked stale (旧版本读数); readings citing figure blocks the
  current parse no longer has are marked too, and both notes appear in the stage detail.

## Caching, and why filenames carry keys

Nothing is recomputed unless something it depends on changed, and each cache is keyed by a content hash of
exactly its own inputs. The hashes are the `<key>` in the filenames under a document directory.

| Cache | Keyed on | Invalidated by |
|---|---|---|
| Parser output | nothing; `raw/<backend>/meta.json` exists or it does not | `--force` |
| Extraction (`extractor_key`) | the model and its sampling settings (one `ExtractionOptions`, built the same way by the writer and every reader), the field schema minus the tolerances, categories, condition preferences and display text, the prompts, the document rendering, and the source of `extract.py`, `records.py`, `fields.py`, `adapters.py`, `prompts.py`, `normalize.py`, `grounding.py`, `voting.py` and `continuation.py`; passage mode adds its two prompts, `candidate_limit`, `context_tokens`, the inventory effort, and a retrieval fingerprint over the keywords, `passages.py` and `continuation.py` | changing any of them |
| Comparison (`comparison_key`) | the whole field schema including the tolerances, the categories, the condition preferences, and the source of `normalize.py`, `compare.py`, `matching.py`, `dataset.py`, `decide.py` and the matching prompt | changing a tolerance or a rule |
| Figure readings (`figure_key`) | the vision model and its sampling, the crop settings, the per-paper limit, the film fields, and the source of `figures.py`, `normalize.py` and `passages.py` | changing any of them |
| LLM requests | the entire request payload (a chart's image by its sha256) | nothing — an identical request is free |

Extractions, comparisons and consolidated tables also record the parse they came from (a hash of the
artifact's blocks). After a re-parse, a stored lane, comparison or table of the old parse is a miss (not
served, and the paper is not finished) and is derived again: source ids
are positional, so the old citations would point at whatever block now has that ordinal. Re-deriving is
free from the LLM cache whenever the rendered prompts are byte-identical. Files written before the hash was
recorded have none and are read as before.

Only an answer that validated is cached. A JSON reply cut off at `max_tokens` is an error, an invalid answer
costs one repair request and is never written, and an invalid answer already in the cache is asked again
rather than replayed. A sample matching that failed (the model answered badly twice) is shown for that run
but not stored, so the next run asks again instead of serving the failure until `--force`. Neither is that
run's consolidated table (`datasets/…json`, only `dataset.xlsx` is written): the stored table is what marks
a paper finished, so 「处理全部未完成」 and `deploy.sh --rerun` pick the paper up again.

The same holds for one field question in passage mode that gets no valid answer (invalid twice, or cut off):
it costs that field, not the lane. The lane is stored with the question in `failed_questions` and its other
fields intact; the comparison and table of that run are not stored, and the next run extracts the lane again,
which re-asks only that question (every other answer replays from the cache). The inventory question and
transport failures still fail the lane.

So adjusting a numeric tolerance recomputes the comparison without paying for extraction again, and cannot
serve a stale verdict either. Re-running a finished paper costs nothing. And because the model's own
answers are cached by request payload, **a code-only change re-derives records for free** as long as the
rendered document and the prompts stay byte-identical: rebuilding comparisons and tables for the whole
14-paper corpus after a key change took 8 seconds (every model answer is a cache hit).

Anything sitting at its built-in baseline is left out of the key material, so an unedited checkout keeps
the filenames it has. There is no hand-maintained version number anywhere, and there should not be one.

### The data directory

```text
data/
├── llm_cache/                          model answers, keyed by request payload
├── exports/paperfacts.xlsx             the default batch workbook
└── docs/<first 16 hex of sha256>/
    ├── identity.json                   full sha256, display name, origin
    ├── source.pdf                      the uploaded PDF (web uploads only)
    ├── raw/<backend>/                  the parser's native output plus meta.json
    ├── parsed/<backend>.md             every block behind its <!-- source: id --> marker: exactly
    │                                   what the extraction model reads
    ├── parsed/<backend>.artifact.json  blocks with page and bbox, plus page geometry
    ├── facts/<backend>.<extractor_key>.json          one lane's sample-level extraction
    ├── comparisons/<extractor_key>.<comparison_key>.json   the two-lane comparison report
    ├── datasets/<extractor_key>.<comparison_key>.json      the consolidated table the web UI reads
    ├── figures/<figure_key>.json       values read off charts by the opt-in figures stage
    ├── dataset.xlsx                    this paper's workbook, written automatically by `run`
    ├── overlays/<backend>/page_*.png   bbox overlays from `overlay`
    └── pages/<dpi>dpi/                 page renders for the web viewer
```

Every block carries `page` plus a bounding box normalised to `[0, 1]`, and every extracted value cites the
block ids it was read from, so any value maps back to a rectangle on a page. An export made under
different settings lands beside the old one instead of overwriting it.

## The Excel workbook

`run` writes `data/docs/<sha>/dataset.xlsx` for one paper; `batch` and `export` write one workbook for a
whole directory; the web UI serves the same thing behind 「下载 Excel」 and 「下载全部 Excel」. Six
sheets:

| Sheet | Contents |
|---|---|
| 论文数据 | One row per unique PDF: the selected sample's values, one column per field |
| 样品数据 | Every sample after merging the two lanes, one row each, same columns |
| 字段说明 | 字段, 中文名, 层级, 标准单位, 中文说明, 单值与缺失规则 |
| 数据质量 | 文档ID, 文件名, 样品ID, 字段, 最终决策, 输出值, 标准单位, 条件, 合并证据来源, 证据来源通道, 系列级, 说明 |
| 图中读数 | Chart readings, empty unless the figures stage ran: 图, 页码, 图块来源, 子图, 字段, 系列, 横轴（仅供参考）, 读数（近似值）, 标准单位, 精度, 图中原始读数, 纵轴刻度, 图注, 说明 |
| 运行记录 | 文档ID, 文件名, 状态, 合并后样品数, 抽取版本, 比较版本, 说明 |

论文数据 and 样品数据 both begin with 文档ID, 文件名, 样品ID, 样品标签, 样品及测量条件, 可用字段数 and
双路一致字段数 before the twenty field columns.

数据质量 is where the provenance is: 最终决策 is `agree` or `single_source` for a committed value and the
refusal name otherwise, 合并证据来源 lists the block ids behind it, **证据来源通道** says which lanes
supplied it, and **系列级** marks a value the paper stated once for the whole sample series.

The paper row selects the sample with the most usable fields, then the most two-lane agreements, then a
stable sample-id tie break. **It never combines different samples' measurements into one row.** That
chooses the most complete sample, not the best-performing one; the others stay in 样品数据. Target
properties are paper-level and shared across samples only when their extracted value is unique. Use 数据质量
to restrict a training set to two-lane agreements if you need to — completeness alone is not a quality
score.

Numeric cells use the canonical units named in 字段说明. Missing values are empty, never zero.

## What a blank cell means

Four guardrails stand between the model's answer and a committed number, and each exists because the
failure it catches was observed on real input:

| Guardrail | What it catches |
|---|---|
| Schema and type cleaning | Fields outside the target schema; numeric fields holding words like `"minimum"` or `"n.a."` (a number word from one to twelve that is the whole value, or is followed only by the value's own unit -- `"four"` or `"four-inch"` quoted with the unit `inch` -- is read as 4; `"one of the samples"`, `"five to ten"`, `"one-third"`, `"ten-fold"` are not numbers) |
| Scope enforcement | A paper-level field attached to one sample, or the reverse — a film's dopant concentration reported as the sputtering target's composition |
| Citation validation | Block ids the model invented, or ids from parts of the document it was never shown |
| Grounding | The quoted text cannot be found in the block it cites — a real id attached to a value that did not come from it |

The first three drop the value with an audited reason. Grounding only flags; it never drops. Grounding is
the one that matters most and the one usually missing: without it, "traceable to a page and a bounding
box" only means the model named a real block. It tolerates formatting differences — the same number
reaches the model as `$( 4 0 \times 1 0 \mathrm { c m }$` from one parser and `(40 × 10 cm` from the other
— while staying strict about digits: a quoted `5` is not found inside `0.5` or `5.2`, nor `10` inside
`10⁻⁴`. A quote that straddles the junction between the cited block and
its same-page neighbour counts as grounded. A quote lying entirely inside the neighbour still does not.

Two more rules shape the table. **Series fan-out**: when the model states that a value holds for every
listed sample, the code writes it onto each of them and marks it 系列级, rather than leaving it
unattributed. Both modes do this the same way; document mode asks for such a value once, under the target.
A sample the model lists without a usable id keeps its values as unattributed, and a sample listed twice is
kept once; both are recorded in the lane's audit. **The single-sample rule**: a cell is committed only when exactly one value survives for
that sample and field. Everything else is a refusal, and the refusal has a name:

| Decision | What happened |
|---|---|
| `agree` | Both lanes produced the same value. Committed |
| `single_source` | One lane produced it, grounded and cited. Committed |
| `conflict` | The lanes produced different values. Once a condition is chosen, only a conflict involving a candidate at that condition counts: differing 400-1100 nm averages do not refuse a cell whose preferred 550 nm values agree |
| `ambiguous` | The lanes could not be decided between, or the sample match fell below `ambiguous_match_confidence` |
| `ungrounded` | No evidence both located in the text and carrying a valid citation |
| `multiple_conditions` | One lane recorded the field under several measurement conditions, so no single value is the answer |
| `multiple_values` | One lane recorded several different values under the same condition, or several candidates were never confirmed across lanes |
| `non_scalar` | Every candidate is a range, a bound, or a rectangular dimension such as `40 × 10 cm`; no unique scalar exists |

Two things are not refusals. A bound or range beside a scalar under the chosen condition (`>80 %` next to
`80.6 %`) is set aside with a note and the scalar decides the cell. The condition is chosen over every
candidate, bounds included, and a bound is never set aside to make room for another condition: a bound at the
condition the row's block or `condition_preference` picks makes the cell `non_scalar`, and when nothing picks
a condition the cell stays `multiple_conditions` rather than letting the scalar's condition win by default. A
preference entry that matches two states of the film in one lane (550 nm as-deposited and annealed) ends the
search rather than falling through to a later entry. And several condition texts in one lane that all give
the very same number ("100 nm, by TEM cross-section", "100 nm, not reduced by the forming gas") are one
measurement, committed with the texts joined -- unless the conditions name different numbers: 85 % at 450 nm
and 85 % at 600 nm stay two measurements, and so do 100 nm as-deposited and 104 nm after annealing.

A cell is `agree` only when both lanes' final candidates are within the field's tolerance and no pair across
the lanes quotes conditions naming different numbers; where a lane quoted several conditions, the lanes must
name the very same numbers. Two lanes that measured different things are `multiple_conditions`, however close
their values.

Approximate values and measurements with ± uncertainty keep their centre value and carry a note, including
`(4.5 ± 0.2) × 10⁻⁴`. A spelling with no single safe reading is refused and compared as ambiguous rather
than guessed: a ratio such as `1:4` or `10/10`, a pair that does not ascend (`10-4` is as likely 10⁻⁴
without its caret as a range), a range whose exponent is written once (`1.2-1.5 × 10⁻³`), bounds in two
different units, two values joined by "and", a list (`30, 40`), a number with its own unit before another
number (`550 nm: 85%`, `140 nm ATO/25 nm ITO`), or scientific notation with other numbers beside it. One
quantity written in two of its units, larger first, is one value: `3 h 30 min` is 210 min, `1 min 30 s` is
1.5 min. A range
keeps its midpoint whether or not each bound repeats the unit (`80%–85%`, `500 °C to 530 °C`). A condition
after the value (`550 nm at 80%`), a name before `=` (`O2/(Ar+O2) = 5%`) and the digits of a formula or a
unit exponent (`H2`, `cm^-3`) are set aside with a note, never read as the value. On a field whose bare
number may be a fraction, only a value below 1 is read as one: a bare `1` is 1 %, not 100 %. A value
the model cannot place on any sample — a paper-level claim such as "transmittance above 80 % from 500 to
2500 nm" — is kept and shown as **unattributed** rather than attached to a plausible neighbour. When both
lanes hold the same unplaced value it is paired and compared like any other; a value only one lane could
not place stays out of the comparison, because the other lane may hold it on a sample where it is already
reported. An unplaced value is visible; a misplaced one is not.

Values that exist only in a figure cannot be extracted, and a sheet-resistance curve that appears only in
a plot is correctly reported as MISSING, not invented. The 48 `multiple_conditions` refusals in the
current corpus were read one by one and are all correct: per-layer versus total thickness in bilayer and
graded films, two targets with their two modes and diameters, transmittance quoted for two wavelength
bands. No word-level rule separates those from a paraphrase, so the blank cell with its reason on hover is
the right output for them.

The two lanes share one extractor, so a mistake made by the language model itself — attributing a value to
the wrong sample — correlates across lanes and AGREE will not catch it. Parser error and extractor error
have to be counted separately when evaluating.

## Measurements that decided the defaults

All from `.omc/research/reasoning-effort.md`, run against `deepseek-v4.1-flash` on the Alibaba Cloud
endpoint, passage mode, parsers cached.

| Question | Answer | Evidence |
|---|---|---|
| `llm.reasoning_effort` | Leave unset | One 12-page paper: unset 10 min 30 s, 112 values, 8 samples; `"none"` 1 min 39 s but only 77 values and 7 samples; `"low"` 18 min 50 s and no better than unset |
| `llm.inventory_reasoning_effort` | Leave unset | Same paper family: unset found 10 samples in each lane and matched all 10; `"low"` found 6 and 8 and matched 4; `"none"` found 6 and 6. Less reasoning merges the as-deposited and annealed films into one sample |
| `llm.concurrency` | 4 | 42 live model calls finished in 3 min 42 s against roughly three times that serially. It changes only when requests are sent, so it costs no re-extraction |
| `extraction.passes` | 1, opt in to more | A second pass reproduces 75–90 % of a first pass's values at temperature 0. That is the noise floor for any prompt experiment, not a recall gain |
| Corpus verdict mix | — | 14 papers, 169 samples, 2760 sample×field cells. After the latest pairing change: 603 agree, 268 single-source, 0 conflicts; the comparison itself holds 765 agree, 400 missing, 9 ambiguous rows. Most cells are missing, and that is the honest answer |

The research note also records what did **not** work and was reverted: a numeric-signature condition key
(it cannot tell "550 nm" from "550 nm, annealed"), and a descriptive-reference prompt rule (no measurable
gain, and the run-to-run noise floor above swallowed the difference). Read it before proposing a prompt
change, and run the baseline twice before crediting or blaming a wording.

## Troubleshooting

**`import paperfacts` raises `ModuleNotFoundError` on the Mac.** An external tool has marked `.venv`
hidden, and Python 3.13 then skips its `.pth` files. Run `scripts/dev_fix_venv.sh`. No reinstall is
needed, and Linux does not have this problem.

**Every request answers HTTP 500 `BalanceError: There are no suitable services`.** The account is out of
balance, or the endpoint is having a transient outage. It is not the paper, the payload or the
concurrency — all three have been bisected. Nothing runs until balance returns, and the resume is free
because the answers already bought are cached. One observed outage lasted five minutes.

**PaddleOCR-VL fails with a 401 and "Repository Not Found" against the MLX server.** The pipeline asked
the server for `PaddleOCR-VL-1.6-0.9B`, which is not a Hugging Face repository. Set
`PAPERFACTS_PADDLE_VL_MODEL_NAME=PaddlePaddle/PaddleOCR-VL-1.6`. The name must be set wherever the MLX
backend is used; `dev_up.sh` sets it for you.

**A document has no PDF.** A paper processed on another machine still has its stored parse for both
lanes, and extraction, comparison and export never open the PDF, so it can be re-run. The web app says
so: 「处理全部未完成」 skips only documents with neither a PDF nor a cached parse, and asking to re-run
one of those answers 409 with "re-upload it to process it". Only uploaded documents can have their pages
re-rendered for the viewer.

**A forced re-parse is expensive.** `--force` on `parse` re-runs both parsers: on this Mac, MinerU took
24.5 s and PaddleOCR-VL 153.6 s for a 12-page paper with the MLX server up, and hours without it.
`--force` on `extract` re-buys every model answer. Prefer letting the cache keys invalidate what actually
changed.

**A service will not come up on the server.** Cold starts are minutes, not seconds. Read
`docker compose logs -f` before restarting anything, and check GPU usage with `nvidia-smi`.

## Development

```bash
uv run pytest                                              # unit tests; no models, no network, no real papers
uv run pytest --cov=paperfacts                             # coverage target is 80%
uv run pytest --run-parser                                 # integration; needs both parser environments and their weights
uv run ruff check src tests runners && uv run ruff format --check src tests runners
# the web frontend in a real browser (navigation races, polling, layout, keyboard); deselected by default
uv run --with playwright python -m playwright install chromium   # once
PYTHONPATH=src uv run --with playwright pytest -m e2e      # or: python tests/e2e/web_races.py [--only NAME]
```

Line length is 120. Tests never touch a real model or a real LLM: parser output comes from recorded
fixtures of genuine runs, temporary PDFs are generated with pypdfium2, and the LLM is a fake that doubles
as an assertion surface for prompt content.

| Path | Contents |
|---|---|
| `src/paperfacts/` | One flat module per pipeline stage, listed in order in `__init__.py`, plus `web/` |
| `runners/` | The two PEP 723 parser scripts and their committed lockfiles |
| `deploy/` | Linux GPU server deployment |
| `tests/` | The pytest suite and the recorded parser fixtures |
| `.omc/research/` | The measurements behind the defaults |

Use `uv sync --group dev`; never `pip install` into the venv. After editing a runner's dependency header,
run `uv lock --script runners/<name>.py`. `CLAUDE.md` holds the internal conventions for contributors —
nothing a user or an operator needs.
