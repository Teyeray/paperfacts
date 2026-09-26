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
interface, every command, every configuration key, writing a domain profile for a new field of research, and
what a blank cell means. `CLAUDE.md` is internal
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
failed for good does not block it. Every setting it needs is read the way the service reads it: the web
password from `.env`; the web user and the paddle lane's settings from `.env` (`PAPERFACTS_WEB_USERNAME`,
`PAPERFACTS_PADDLE_VL_*`, quoted or not) and otherwise from `config.json`. The credentials reach `curl` on
stdin, never on its command line.

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

Development happens on the Mac, is committed and pushed to GitHub, and pulled on the server. Code is never
edited on the server: production is updated only by `scripts/deploy.sh --pull` run there (over ssh or in a
session on the machine), and experiments run in a separate worktree against their own copy of the data, never
against production's `data/`.

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
uv run paperfacts fields                   # list the field table of the profile a run would load
uv run paperfacts profiles                 # list profiles/: name, maturity, field counts, content hash, title
uv run paperfacts profiles --check profiles/my_domain.json   # validate a profile while writing it
uv run paperfacts prompts --profile tco --field thickness          # what the model is asked, no model call
```

| Command | Purpose |
|---|---|
| `run <pdf>` | Parse, extract, compare and save `exports/<profile>.xlsx` for one paper |
| `batch <pdf or dir>` | Recursively process every PDF and write one workbook for all of them |
| `export <pdf or dir>` | Rebuild that workbook from cached results, with no parser and no LLM calls |
| `parse <pdf>` | Parse into Markdown with provenance markers, a block list and the full artifact |
| `extract <pdf>` | Extract sample-level records from parsed Markdown. Needs `parse` |
| `compare <pdf>` | Match samples across lanes and compare their fields. Needs `extract` (which implies `parse`) |
| `overlay <pdf>` | Draw block boxes onto page images, to check provenance by eye. Needs `parse` |
| `serve` | Serve the web interface |
| `fields` | Print the profile's field table (`--profile` for another), so an edit can be checked at a glance |
| `profiles` | List the profiles in `profiles/` with their maturity, paper/sample field counts, content hash prefix and title. `--check PATH` validates one file instead: it prints the profile's line and any warnings then `ok`, or every error it finds, one `error:` line each, and exits 1. The listing reads `profiles/` without `config.json`, so it works while that file is broken |
| `prompts` | Print the rendered inventory, per-field, extraction and matching system prompts of a profile (`--profile`), exactly as the model gets them, each labelled with the mode that sends it (passage mode never sends the extraction prompt; document mode sends only it). `--field NAME` prints the per-field system prompt, that field's line, and the question's framing with `<sample list>` and `<excerpts>` in place of what a run fills in. No model is called |

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
- `--profile NAME_OR_PATH` on `run`, `batch`, `export`, `extract`, `compare` and `serve` runs the command
  under another domain profile than `profile` in `config.json` (or `PAPERFACTS_PROFILE`). Workbooks are named
  after the profile, so a profile file given by path whose name is also a different `profiles/<name>.json` is
  refused unless the two files are byte-identical, and the name `paperfacts` (the pre-profile workbook) is
  reserved. On `serve`, `--profile` picks the **default** profile: the server serves every profile under
  `profiles/` (see "A second profile beside the first"), and the default answers every request that names none.
  Each profile is read once: `/api/health` reports the default's name and hash (and, under `profiles`, every
  served profile's), `/api/profile` gives the page its title and copy (the header shows the title, and the
  paper-level record is named the profile's way), and after any edit to a profile's file on disk -- display text
  included, or a symlink pointed at another file -- every new job under that profile is refused until the server
  is restarted; `/api/health`'s `profile_on_disk_changed` (the default) and `profiles.<name>.on_disk_changed`
  say so first. A file that cannot be read (deleted, or caught mid-save) refuses the job with its own message.
  Run **one server per data root**: two servers over the same `data_root` can parse the same document at the
  same time.
- `--jobs N` / `-j N` on `batch` processes N papers at once (default `web.max_parallel_documents`, 3);
  `--jobs 1` is the old one-after-another run.
- `--offline` on `run` and `batch` answers every model request from the LLM cache and fails on a miss; see
  [Offline replay](#offline-replay).
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
| `offline` | Kept for compatibility; leave it `false` and switch replay on per run with `--offline` or `PAPERFACTS_LLM_OFFLINE=1` (see [Offline replay](#offline-replay)). Default `false` |

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
| `web.profiles` | The profiles under `profiles/` a server serves beside its default, as a list of names (`["battery_cathode"]`); `null` serves every one that loads, `[]` only the default. The default profile is always served. A name with no file is listed among the invalid profiles. Default `null` |
| `overlay.dpi` | Default 150 |
| `comparison.ambiguous_match_confidence` | Below this, a sample match is AMBIGUOUS rather than accepted. Default 0.6 |
| `data_root` | Where everything is written. Default `data` |
| `profile` | The domain profile: a name, read from `profiles/<name>.json`, or a path to a profile file. It holds the groups, fields, condition keywords and domain wording, and every run, batch and export reads them from it, and `serve` makes it the default of the profiles it serves (`--profile NAME_OR_PATH` overrides it for one command). Default `tco` |

`config.json` no longer holds `fields` or `condition_keywords`: they live in the profile (`fields` and
`retrieval.condition_keywords`). A `config.json` that still has either is refused with an error naming the key,
the file and the profile file to edit instead, so an old copy never looks as if its table were read.

### Environment overrides

Every scalar setting also has a `PAPERFACTS_*` variable that wins over the file, which is how one machine
points at its own services without editing the shared file:

`PAPERFACTS_DATA_ROOT`, `PAPERFACTS_REPO_ROOT`, `PAPERFACTS_PROFILE`, `PAPERFACTS_UV_BIN`, `PAPERFACTS_MINERU_URL`,
`PAPERFACTS_PADDLE_URL`, `PAPERFACTS_PADDLE_RENDER_DPI`, `PAPERFACTS_PADDLE_VL_BACKEND`,
`PAPERFACTS_PADDLE_VL_SERVER_URL`, `PAPERFACTS_PADDLE_VL_MODEL_NAME`, `PAPERFACTS_SUBPROCESS_TIMEOUT_S`,
`PAPERFACTS_HTTP_TIMEOUT_S`, `PAPERFACTS_LLM_BASE_URL`, `PAPERFACTS_LLM_MODEL`,
`PAPERFACTS_LLM_TIMEOUT_S`, `PAPERFACTS_LLM_CONTEXT_TOKENS`, `PAPERFACTS_LLM_TEMPERATURE`,
`PAPERFACTS_LLM_MAX_TOKENS`, `PAPERFACTS_LLM_REASONING_EFFORT`,
`PAPERFACTS_LLM_INVENTORY_REASONING_EFFORT`, `PAPERFACTS_LLM_CONCURRENCY`, `PAPERFACTS_LLM_MAX_IN_FLIGHT`,
`PAPERFACTS_LLM_RETRY_ATTEMPTS`, `PAPERFACTS_LLM_RETRY_BACKOFF_S`, `PAPERFACTS_LLM_OFFLINE`, `PAPERFACTS_EXTRACTION_MODE`,
`PAPERFACTS_EXTRACTION_PASSES`, `PAPERFACTS_CANDIDATE_LIMIT`, `PAPERFACTS_SERVER_HOST`,
`PAPERFACTS_SERVER_PORT`, `PAPERFACTS_MAX_UPLOAD_MB`, `PAPERFACTS_PAGE_DPI`, `PAPERFACTS_PAGE_DPI_MIN`,
`PAPERFACTS_PAGE_DPI_MAX`, `PAPERFACTS_OVERLAY_DPI`, `PAPERFACTS_WEB_USERNAME`, `PAPERFACTS_WEB_MAX_PARALLEL_DOCUMENTS`,
`PAPERFACTS_WEB_PROFILES` (comma-separated names), `PAPERFACTS_WEB_PASSWORD`, `PAPERFACTS_FIGURES_ENABLED` (`true`/`false`), `PAPERFACTS_FIGURES_MODEL`,
`PAPERFACTS_FIGURES_MAX_PER_DOCUMENT`, `PAPERFACTS_FIGURES_DPI`, `PAPERFACTS_FIGURES_MAX_PIXELS`,
`PAPERFACTS_FIGURES_TIMEOUT_S`.

`PAPERFACTS_CONFIG` points at a different configuration file altogether. An empty string counts as unset,
and a value that will not parse as a number names the variable in the error.

One setting is **file-only**, with no `PAPERFACTS_*` variable: `comparison.ambiguous_match_confidence`. (The field table and condition keywords were the other two; they are the profile's
now, and `PAPERFACTS_PROFILE` picks the profile.)

Secrets live only in `.env`: `PAPERFACTS_LLM_API_KEY` and `PAPERFACTS_WEB_PASSWORD`. `.env` is loaded
without overriding what the environment already holds.

### The field table

The profile's `fields` list (`profiles/tco.json` for the shipped one; [Domain profiles](#domain-profiles) has
every attribute and how to write a profile of your own) **is** the schema. Each entry drives the description the model is given, the
keywords retrieval searches for, the unit everything is converted to, and how close two numbers have to be
to count as the same fact.

```jsonc
{
  "name": "sheet_resistance",
  "group": "film",                          // target = paper-level; process / film = per sample
  "kind": "numeric",                        // numeric | composition | text | boolean | date | interval | reference
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
compared as ordinary text, never rounded to the nearest one. Text is equal across spacing, case and a hyphen or
period one parser dropped ("rfmagnetron sputtering" is "rf-magnetron sputtering", so both are `RF`) unless both
values name a category, and then only the categories count. `categories` changes only verdicts, so adding
one re-compares the stored facts instead of re-extracting them.

A numeric field may declare `valid_range`, the plausible values in its `canonical_unit`, with either end
open. The model is told the range with its field question, and a value whose converted number still falls
outside it is dropped with the reason in the lane's `dropped` audit (the web's 清洗记录). It is meant for
the confusions a unit cannot catch: the spin-coating rpm of an absorber read as the substrate rotation, the
thickness of a wafer or a glass substrate read as the electrode's. The shipped table caps `rotation_speed`
at 100 rpm and `thickness` at 5000 nm and floors `transmittance` at 60 %. A value that
cannot be converted is kept, since there is no number to judge. A range changes the prompt and which values
survive, so it moves both cache keys; a field without one keeps the keys it had. A numeric field with no
`canonical_unit` (a count, such as the battery profile's `cycle_number`) may declare one too; it is judged on
the number as parsed.

#### Ranges, bounds and a unit written in the value

Two more numeric attributes decide how a quoted value is read. `range_policy` (`midpoint`, the default,
`reject`, `lower` or `upper`) decides what a range quoted as one value ("10-20") becomes in the lanes and in the
comparison: its midpoint, no value, or its lower or upper end (a calcination "at 450-500 °C" reported by the
temperature it reached: `upper`). Under `lower` / `upper` the chosen end also fills the **dataset cell**, with
the note 原文为区间 a–b，按字段配置取上限/下限: an end is a number the paper printed. Under `midpoint` and
`reject` a range never fills a cell: a midpoint is a number nobody measured. Only a **clean range** has an end,
and the lanes and the cell use the one definition of it (`readers.read_range`): a range the general number
reader reads as one -- two ascending numbers, both plain or both in scientific notation -- and after it nothing
but a unit of the field ("450-500", "450 °C to 500 °C", "1.2e-4 - 1.5e-4 Ω cm"); an approximation ("~450-500")
may precede it. Anything else is refused under `lower` / `upper` in the lanes as in the cell: a bound ("> 450-500",
"below 1.2e-4 - 1.5e-4"), a condition ("450-500 °C for 2 h"), a parenthesis ("450-500 (600)") or another unit
("450-500 K" on a ℃ field).

A unit written inside the quote ("1.5e-4 Ω·cm", "450-500 °C", ">80 mW") is checked by one rule
(`normalize.unit_of_value`) in the lanes, the cell and an interval's range or bound: units are compared as the
profile's unit registry converts them, never as spellings, so "Ω cm", "Ω-cm" and "ohm cm" are all Ω·cm. One that
converts exactly as `unit_raw` does (a header's power of ten included) changes nothing. One that converts
otherwise is the more specific statement and the number is converted from it: "1.5e-4 Ω·cm" under `unit_raw`
"mΩ·cm" is 1.5e-4 Ω·cm, and "1.2-1.5 Ω·cm" under a header "×10^-4 Ω·cm" does not take the header's power of ten.
With no `unit_raw` the written unit is the unit ("0.6%" is 0.6 %, not a fraction). One the registry cannot read
for the field ("K" or "oC" on a built-in ℃ field; a profile adds spellings with a declared unit) keeps the cell
empty, and the range or bound out. A bare range on a `percent_or_fraction` field is a fraction only when all of it is below 1, so both
ends read in one unit ("0.8-1.2" is 0.8-1.2 %). A bound (">80 %", or "80" quoted out of "above 80 %") is no
range under any policy, and a descending pair or a range whose exponent is written once (`1.2-1.5 × 10⁻³`) is
refused under every one. `after_clause` (`refuse`, the default, or `condition`) decides a value quoted with an "after ..."
clause: by default "100 nm after annealing" is refused, since it describes another state of the sample; under
`condition` ("92.5% after 100 cycles" for a capacity retention) the number is read and the clause is appended
to the value's `condition` (`; `-joined when the model already gave one), so "after 50 cycles" and "after 100
cycles" stay separate measurements in the comparison and the dataset cell. Both are cleaning and verdict
rules, so changing either moves both keys.

A quoted unit is compared with its spaces removed (`clean_unit`), so a profile's unit aliases that differ only
by spaces ("mAh g-1" and "mAhg-1") are one spelling, and declaring both is refused as a duplicate. A profile's
top-level `ignored_unit_suffixes` lists the words a paper may write after a unit to say whose quantity it is:
the TCO profile lists the chamber gases (`Ar`, `O2`, `N2`, `H2`, `He`, `Kr`, `Xe`, `air`), so "1.1 Pa Ar" and
"3 mTorr (O2)" read as pressures. A profile that lists none sets nothing aside.

The prompt wording that used to carry TCO examples is profile text too, each with a neutral default: rule 2's
table-header examples (`prompt.scaled_header_examples`, `prompt.plain_header_example`), what a number outside a
field's plausible range usually is (`prompt.implausible_origin`), and the chart prompt's axis and tick-label
examples (`figures.symbol_axis_example`, `figures.x_label_examples`).

A `canonical_unit` must be one the built-in converters know (`Ω/sq`, `Ω·cm`, `nm`, `min`, `inch`, `%`, `℃`,
`cm`, `W`, `sccm`, `rpm`, `Pa`) or one the profile declares (see [Declared units](#declared-units)), or loading
the profile fails, naming the field and the file, rather than guessing. Adding a field is one table entry; the prompt,
normalisation and tolerances follow from it. `rel_tol` and `abs_tol` only decide verdicts, so editing one
re-compares the stored facts instead of re-extracting them. Tolerances may not be negative, and
`percent_or_fraction` is only accepted on a `%` field. `uv run paperfacts fields` prints the profile's
table.

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

`transmittance` carries a `condition_hint` asking for the wavelength or spectral range, and a `condition_rule`
that tells the model to always fill it. A field with a `condition_rule` must also give
`missing_condition_note_zh`, the note a dataset cell whose value came without the condition gets
(`原文提取结果未注明透光率波长或波段` for `transmittance`): the note is stored in the verdict, so it is profile text
the comparison key covers, never generated from the display-only `label`; `density` and
`transmittance` read a bare number as a percent or a fraction, and every other numeric field rejects a
number with no unit rather than assuming one.

## Domain profiles

Everything PaperFacts knows about one field of research is in one JSON file, `profiles/<name>.json`: which
fields to extract, how they are grouped, the domain wording of every prompt, the words that find how a sample
was made, any units the built-in tables lack, and the Chinese display copy. The code holds the rules around
that wording (quote verbatim, cite only ids you were shown, never guess a subset) and no word about any domain.
Two profiles ship:

| Profile | `maturity` | What it is |
|---|---|---|
| `tco` | `production` | Sputtered transparent-conductive-oxide films: 4 paper-level target fields and 19 sample-level fields. Its wording is byte-for-byte the prompts the corpus was measured with |
| `battery_cathode` | `example` | Lithium-ion battery cathode materials: 13 sample-level fields, no paper-level group, four declared units (`mAh/g`, `C`, `V`, and `K` added to `℃`). Written as the template for a new domain; it has not been measured against a gold set |
| `catalysis` | `example` | Heterogeneous catalysis, CO2 hydrogenation to methanol: two [entity types](#entity-types), catalysts (6 fields) and the reaction tests run on them (9 fields, one a [reference](#reference-fields) naming the test's catalyst), plus 3 paper-level fields. The worked example of every structural feature: paper-level [lists](#list-fields) with and without categories, a [date, a yes/no and an interval](#yes-or-no-date-and-interval-fields), a range read by its upper end, three declared units (`m2/g`, `MPa`, `mL/(g·h)`). No real catalysis paper has been hand-checked against it; its tests are a synthetic end-to-end run and a synthetic gold set for the scorer (`tests/fixtures/catalysis_gold/`) |

`maturity` is `production` or `example` (the default when it is left out). It is display only: the web header
shows 示例配置 beside the title of an `example` profile, and `paperfacts profiles` lists it. Promote a profile to
`production` once its output has been checked against hand-read papers (`eval/` has the scorer and the format).

**Selecting one.** `profile` in `config.json` (default `tco`), `PAPERFACTS_PROFILE`, or `--profile NAME_OR_PATH`
on one command. A bare name is `profiles/<name>.json` in the repository; a value containing `/` or ending in
`.json` is a path, and a relative one is resolved against the directory the command runs in, not the repository
(`--profile profiles/tco.json` works from the checkout's root only; the bare `tco` works from anywhere).
The file is read and validated once per process, so a running server sees an edit only after a restart
(`scripts/deploy.sh` restarts when anything under `profiles/` is newer than the running process).

**What the file holds.** The top-level keys are `format` (always 1), `name` (must equal the file name without
`.json`, and match `^[a-z][a-z0-9_]{0,39}$`), `title_zh`, `maturity`, `description_zh`, `groups`, `prompt`,
`figures`, `retrieval`, `units`, `ignored_unit_suffixes`, `ui`, `fields`, `entities`, and a free `$comment`. `format`,
`name`, `groups`, `prompt`, `retrieval` and `fields` are required. An unknown key anywhere is refused with the
list of valid ones, and every error names the file and the key.

| Key | Holds |
|---|---|
| `groups` | `{name, level, label_zh, entity}` each. `level` is `paper` (one record per paper, e.g. TCO's sputtering `target`) or `sample` (one value per sample). At least one sample-level group; paper-level groups may be none. The name is shown to the model in every field line (`group: film`); `label_zh` is display only. `entity` names the [entity type](#entity-types) a sample group's fields describe: required on every sample group of a profile with `entities`, refused otherwise and on a paper group |
| `entities` | Optional: 1 to 5 [entity types](#entity-types), `{name, label_zh, prompt, retrieval}` each. Without it the profile has one implicit entity, `sample` |
| `fields` | The field table ([below](#field-attributes-and-what-they-do)). Declaration order is question order and column order. At least one sample-level field; more than 40 logs a cost warning, more than 100 is refused |
| `prompt` | The prompt slots ([below](#prompt-slots)) |
| `figures` | The chart-reading slots: `subject`, `property_noun`, `chart_definition`, `axis_example`, and optionally `symbol_axis_example` and `x_label_examples`. Required exactly when some field is `figure_readable`, refused otherwise |
| `retrieval` | `condition_keywords`, the words that mark a block describing how samples were made, and `condition_unit_pattern`, a regular expression (at most 500 characters, matched case-insensitively on lower-cased text) for a number in a condition's unit. Passage mode's inventory question is shown the blocks either one finds |
| `units` | Units the built-in tables do not have ([below](#declared-units)) |
| `ignored_unit_suffixes` | Words a paper writes after a unit to say whose quantity it is (TCO: the chamber gases, so "1.1 Pa Ar" reads as a pressure). At most 50; none by default |
| `ui` | Chinese copy: `paper_level_label_zh` (the paper-level record in a column header or a fact's scope), `paper_level_short_zh` (the same where only a word fits), `entity_label_zh` (what one sample is called), `no_samples_message_zh` (shown in place of the sample table when the inventory found no in-scope sample). Each defaults to a neutral wording |

### What a profile can and cannot express

A profile changes the words, never the shape of the answer. The shape is fixed in code:

- **Up to five kinds of sample, linked by references.** A profile without `entities` has one list of samples per
  paper, all of the same kind (a film, a cathode material), each one row. With [entity types](#entity-types) each
  kind (a catalyst, a reaction test) has its own list, rows and matching, and a [reference field](#reference-fields)
  links a sample of one kind to one sample of another (the catalyst a test ran on). Only passage mode can ask
  about them.
- **Two levels.** A field is paper-level (one record per paper) or sample-level (one value per sample of its
  entity). There is no third level: nothing per layer within a sample, per measurement within a sample, or per
  figure.
- **Seven kinds of field.** `numeric` (a number converted to one canonical unit), `composition` (a ratio or
  formula, compared as normalised text), `text` (optionally a closed set of `categories`), `boolean` (yes or no,
  stated in words), `date` (read to ISO at the precision written), `interval` (two ends or a one-sided bound, in a
  unit) and `reference` (a sample of another entity type). No table or curve is a value.
- **Lists of text only.** A `text` or `composition` field may hold several values at once (`cardinality: many`:
  each precursor, each characterization technique), at either level; its cell is the union of what either lane
  grounded. Every other field is single-valued: a paper-level one holds one value for the whole paper, and a
  quantity that differs between samples must be sample-level.
- **A range is a midpoint, an end, an interval, or nothing.** A value quoted as a range ("10-20") becomes its
  midpoint, its lower or upper end (`range_policy: lower` / `upper`) or, under `reject`, no value; only an end
  fills a dataset cell, and a bound (">80 %") fills none. A field whose value *is* a range is an `interval`.
- **A censored value is a bound, not a number.** "IC50 > 10 µM" fills no `numeric` cell, since it is a bound;
  declare a quantity that papers often report censored as an `interval` field, whose cell holds `[10, open]`.
- **Charts are property-vs-condition only.** The opt-in figure reading reads a y value per marker off a chart
  whose caption names a `figure_readable` field; spectra, micrographs, maps and schematics are not read.
- **The prompts are English.** The templates around the slots are English, so slots are written in English;
  only the display copy (`title_zh`, `label`, `ui`, ...) is Chinese.

Still not supported: a list of numbers, dates or references (`many` is text and composition only); a
many-to-many or multi-hop link (a reference names one sample); nesting or order between samples (a layer stack);
a value stated for a whole series across entity types; figures bound to an entity; more than five entity types;
entity types in document mode; the home table (corpus view) beyond the primary entity. When a domain does not
fit, narrow it until it does rather than stretch a slot: pick the entities the gold data is about, move a
per-layer quantity into one field per layer that matters (`etl_thickness`, `absorber_thickness`), model a relation
that belongs to no one entity as a third entity with two references, and leave curves and spectra to the charts
or out of scope. What still does not fit needs a code change, not a profile.

### Writing a profile for a new domain

1. **Copy the example.** `cp profiles/battery_cathode.json profiles/perovskite.json`, then set `name` to
   `perovskite`, and write `title_zh` and `description_zh`. Leave `maturity` at `example`. A domain with more than
   one kind of sample starts from `profiles/catalysis.json` instead.
2. **Groups.** Declare at least one `sample` group. Add a `paper` group only for facts that belong to the paper as
   a whole and can never differ between its samples (TCO's sputtering target); a value the paper states once for
   a whole sample series needs no paper group, because the model flags it `applies_to_all_samples` and the code
   writes it onto every sample. When a paper reports two kinds of sample, each with its own values (catalysts and
   the tests run on them), declare them as [entity types](#entity-types), give each its own sample group, its own
   `sample_definition` and `sample_list_heading`, and link them with a [reference field](#reference-fields); keep
   to one entity (no `entities` at all) when there is one kind.
3. **The three required slots.** `domain_subject` (what the papers are about, one phrase), `sample_definition`
   (what counts as one sample, and what never does) and `field_scope` (which component every field describes,
   and what to leave out). They carry most of the domain; the next section says why each exists. Every other
   slot has a neutral default, but the defaults are generic, and domain examples in them measurably help.
4. **Fields.** One entry per column. Write `description` for the model (it is the whole definition the model
   gets), `keywords` for retrieval (the names a paper uses for the quantity, not its unit), `canonical_unit`,
   tolerances, and a `valid_range` for any quantity another component of the paper is likely to be mistaken
   for. `label` and `description_zh` are what the web page and the workbook print. Pick the kind by what the
   paper writes: a yes/no statement is `boolean`, a date `date`, a range that is itself the value (a cycling
   window) `interval`, several values that hold at once `cardinality: many`, and a range quoted for one number
   `range_policy`.
5. **Units.** If a `canonical_unit` is not one of the twelve built-in ones, declare it under `units`.
6. **Check it.** `uv run paperfacts profiles --check profiles/perovskite.json` validates without a model or a
   configuration file: it prints the profile's line (name, maturity, paper/sample field counts, content hash,
   title) and any warnings then `ok`, or every error it found, one `error:` line each, and exits 1. It makes the
   checks a run makes too: the name `paperfacts` is reserved, and a file named like a repository profile but
   differing from it is refused, since their workbooks would overwrite each other.
7. **Read what the model will be asked.** `uv run paperfacts prompts --profile perovskite` prints the inventory,
   field, extraction and matching system prompts exactly as sent; `--field NAME` prints the per-field system
   prompt, that field's line, and the question around it with `<sample list>` and `<excerpts>` standing for what
   a run fills in. A profile with `figure_readable` fields also gets the chart-reading question the figures stage
   sends per chart panel (only when `figures.enabled`), with `<caption>` for the figure's caption and every
   figure-readable field listed where a run lists only those the caption names. No model is called. Read rule 4,
   5, 6 and 10 of each prompt with the slots in place.
8. **Run a few papers.** `uv run paperfacts run paper.pdf --profile perovskite`, or a second server (below).
   Iterate on display text and tolerances freely -- they re-key nothing or only the comparison, which is
   recomputed from stored extractions at no model cost -- and batch prompt or field edits, which re-extract.
9. **Measure before promoting.** Hand-check a few papers into a gold directory of their own (`eval/README.md`
   has the format), score them with `eval/score.py --profile profiles/perovskite.json --gold <dir>`, and only
   then set `maturity` to `production`. The scorer reads the profile through the package, and without `--keys`
   it scores each paper's newest dataset built under that profile (by the fingerprint the dataset records).

### Prompt slots

Each slot is plain text of 1 to 2000 characters, inserted verbatim in a single pass (a slot may not contain a
`{marker}` the templates fill). The failure each one exists to prevent is why it is worth writing well; the
lessons come from the prompt comments and `.omc/research/`.

| Slot | Required | Where it goes | What goes wrong without a good one |
|---|---|---|---|
| `domain_subject` | yes | The first line of every extraction prompt | The model does not know which of the paper's materials the questions are about |
| `sample_definition` | yes | Rule 4 (document) and rule 1 (inventory) | On a paper that makes no sample of the domain, the inventory builds samples out of the paper's protagonist instead -- perovskite films and whole devices on a paper that only bought its ITO glass -- and both lanes agree, so the comparison cannot catch it (`feedback-e-field-confusion.md`) |
| `field_scope` | yes | Rule 6 of the document and field prompts | A field described as "of the film" collects the absorber's thickness, the spin-coater's rpm and the device's transmittance, because each is a film too (`feedback-e-field-confusion.md`) |
| `fact_noun`, `sample_plural`, `sample_singular` | | Headers and the inventory | Wording only |
| `sample_unit`, `sample_examples`, `sample_id_example`, `condition_noun`, `condition_examples` | | Rule 4 / inventory rules 1-3 | The inventory's granularity, what makes two samples different: too coarse and the as-deposited and the annealed film become one sample; ids built from no condition cannot be paired across lanes |
| `unit_examples` | | Rule 2 | Units rewritten or dropped instead of quoted as written |
| `scaled_header_examples`, `plain_header_example` | | Rule 2 | A table headed `ρ × 10^4 (Ω cm)` loses its power of ten: 6.8 is stored as 6.8 Ω·cm, 10⁴ too high, and both lanes agree (`gold-eval-untuned.md`, cause 2). The model copies the factor into `unit_raw` and the code applies it |
| `paper_key`, `paper_level_rule` | | The document prompt's JSON shape and rule 5 | A paper-level value attached to one sample, or a film's dopant reported as the target's composition. `paper_key` is the JSON key the model writes (TCO `target`, default `paper`); `paper_level_rule` is generated from the paper groups when left out, with its own wording for a profile that has none |
| `no_samples_key`, `no_samples_clause`, `no_samples_condition`, `samples_present_condition` | | Inventory rule 6 | A paper that makes nothing in scope (a device paper on purchased ITO glass) is still asked every sample-level question and harvests the absorber's values. The flag is honoured only with an empty sample list, and must be false "when the excerpts simply do not say": one wrong `true` would empty a lane (`review-core.md`, A16) |
| `subset_examples` | | The subset rule in both modes | A value stated for part of the series ("all films deposited at 100 °C") becomes one value with no sample: the gold set lost 36 substrate temperatures that way. The rule places it on each named sample, and only when the text says which samples form the subset |
| `whole_series_examples` | | Rule 10 (document), rule 5 (field) | A value stated once for every sample is reported against no sample and lost, instead of being flagged `applies_to_all_samples` |
| `partial_collective_example` | | Field rule 5 | A collective noun covering most but not all samples ("the sputtered films" when one is not) is flagged as whole-series, and the value lands on a sample it does not belong to |
| `multi_condition_example` | | Rule 7 | Two measurements of one quantity (transmittance at 550 nm and averaged) merged into one, or one dropped |
| `implausible_origin` | | Every field line with a `valid_range` | Told what an out-of-range number usually is, the model checks before quoting it (TCO: "a different layer, process step or quantity") |
| `sample_list_heading` | | The field question's sample list ("Samples this paper reports:") | Wording only; an entity type names its own list with it ("Reaction tests") |
| `matching_condition_examples`, `matching_value_examples`, `matching_justification_example` | | The sample-matching prompt | Samples paired across the lanes by similar values rather than by the condition that defines them |

A field's `condition_rule` fills rule 8 ("For `transmittance` always fill `condition` with the wavelength or
spectral range"): without it a transmittance arrives with no wavelength and the dataset cell cannot say which
measurement it holds. Internally, and in stored files and the web API, the paper-level record is `paper` and
the no-samples verdict `no_samples` for every profile; only the JSON keys the model sees come from `paper_key`
and `no_samples_key` (TCO keeps `target` and `no_tco_film`, the keys its corpus was extracted with). Files
written before this rename, which say `target` / `no_tco_film`, a `"target"` comparison scope and a single
`matching`, still load: the old names are read aliases, and the next run re-derives everything under the new
names from the LLM cache.

### Field attributes and what they do

| Attribute | Default | Role | Meaning |
|---|---|---|---|
| `name` | required | prompt, figure | Identifier (`^[a-z][a-z0-9_]{0,39}$`); the model answers with it. Reserved: `document_id`, `filename`, `sample_id`, `sample_label`, `conditions`, `available_fields`, `agree_fields`, `field`, `target`, `paper`, `unattributed`, `samples`, `entity` |
| `group` | required | prompt | One of the profile's groups; the field's `level` (paper or sample) is the group's, derived, never written |
| `kind` | required | prompt | `numeric`, `composition`, `text`, `boolean`, `date`, `interval` or `reference` ([Yes/no, date and interval fields](#yes-or-no-date-and-interval-fields), [Reference fields](#reference-fields)) |
| `description` | required | prompt, figure | What the model is told to look for |
| `keywords` | `[]` | retrieval, figure | The names a paper uses for the quantity; passage-mode retrieval and chart selection match them as whole tokens |
| `canonical_unit` | none | prompt, figure | The unit every value converts to; must be built-in or declared, with a retrieval pattern |
| `label`, `description_zh` | `""` | display | Column header and its explanation |
| `rel_tol`, `abs_tol` | 0 | verdict | Two values agree when they differ by at most `rel_tol` times the larger magnitude, or by `abs_tol`, whichever is larger |
| `condition_hint` | none | prompt | What to record alongside the value, e.g. a wavelength |
| `condition_rule` | none | prompt | Rule 8: the condition this field must always carry. Needs `condition_hint` and `missing_condition_note_zh` |
| `missing_condition_note_zh` | none | verdict | The note on a dataset cell whose value came without its condition |
| `bare_number` | `reject` | cleaning, figure | `reject`, `assume_canonical`, or `percent_or_fraction` (only with `%`) |
| `categories` | `[]` | verdict | A text field's closed set of answers |
| `valid_range` | none | prompt, cleaning | `{min, max}`, either end open, in `canonical_unit`: told to the model, and a converted value outside it is dropped |
| `condition_preference` | `[]` | verdict | Which measurement fills the dataset cell when a sample has several |
| `range_policy` | `midpoint` | cleaning, verdict | `midpoint`, `reject`, `lower` or `upper`: what a range quoted as one value becomes; an end (`lower` / `upper`) also fills the dataset cell. Numeric only |
| `after_clause` | `refuse` | cleaning, verdict | `refuse` or `condition`: what "92.5% after 100 cycles" becomes. Numeric only |
| `figure_readable` | `false` | figure | Whether a chart's y axis may be read for this field; numeric with a unit only |
| `display_format` | `plain` | display | `plain` or `scientific` in the workbook. Numeric only |
| `cardinality` | `one` | prompt, verdict | `one` or `many`: a list of values that hold at once (the precursors of a sample, the techniques a paper applies). Text or composition only, at either level; refused together with `figure_readable`, `condition_preference`, `condition_rule` and every numeric attribute. See [List fields](#list-fields) |
| `prompt_categories` | derived | prompt | Never written: a `many` field's `categories`, named in its field line; empty for every other field, so a single-valued field's `categories` stay verdict only |
| `entity` | derived | prompt, cleaning, verdict | Never written: the entity type of the field's group, none for a paper-level field and in a profile without `entities` |
| `references` | none | prompt, cleaning, verdict | Required with `kind: reference` and refused otherwise: the other [entity type](#entity-types) whose sample the field names (the catalyst a reaction test ran on). The field is sample-level of a declared entity, `one`, with no unit, categories or `condition_rule`. Its question shows that entity's sample list too; a value is grounded when its id resolves to one of the lane's samples of that entity; two lanes agree when that entity's matching pairs the samples they name; the cell is the `sample_id` of the referenced row |

### Entity types

A profile may declare several kinds of sample, each an **entity type**: heterogeneous catalysis has catalysts
(composition, loading, calcination) and reaction tests (temperature, conversion) as two lists in one paper.

```jsonc
"entities": [
  { "name": "catalyst", "label_zh": "催化剂",
    "prompt": { "sample_definition": "One catalyst is one prepared material ...", "sample_plural": "catalysts",
                "sample_singular": "catalyst", "sample_list_heading": "Catalysts" },
    "retrieval": { "condition_keywords": ["calcined", "impregnated"] } },
  { "name": "test", "label_zh": "反应测试",
    "prompt": { "sample_definition": "One test is one set of reaction conditions ...",
                "sample_list_heading": "Reaction tests" } }
],
"groups": [
  { "name": "study", "level": "paper" },
  { "name": "preparation", "level": "sample", "entity": "catalyst" },
  { "name": "reaction", "level": "sample", "entity": "test" }
]
```

- **Declaring.** 1 to 5 entities; `name` is an identifier, unique, and neither `paper` nor `unattributed`; the
  first is the **primary** one. Every sample group names one, and every entity needs a sample group. An entity's
  `prompt` may override only `sample_definition`, `field_scope`, `sample_plural`, `sample_singular`,
  `sample_unit`, `sample_examples`, `sample_id_example`, `condition_noun`, `condition_examples`,
  `no_samples_clause`, `no_samples_condition`, `samples_present_condition`, `subset_examples`,
  `whole_series_examples`, `partial_collective_example`, `multi_condition_example`, `sample_list_heading` and the
  `matching_*` slots; with several entities each must give its own `sample_definition`. Its `retrieval` replaces
  either key of the profile's. Everything else is the profile's.
- **Asking.** Each lane asks one inventory per entity, with that entity's slots and retrieval. A sample-level
  field is asked with its entity's field system prompt and sample list; a paper-level field with the primary
  entity's. An entity whose inventory reports no sample of its own skips only its own fields; the lane's
  `no_samples` is true only when every entity has none.
- **Identity.** A sample is `(entity, sample id)`: a value is placed, a series value fanned out and a vote cast
  among the samples of its field's entity only. Unplaced values share one unattributed list.
- **Comparing.** Samples are matched per entity (`ComparisonReport.matchings[entity]`), with that entity's matching
  prompt, and compared under scopes `<entity>:<a>|<b>`; the implicit entity keeps `sample:`. The counts add up
  over every entity, and any failed matching leaves the run incomplete.
- **Rows.** Each entity has its own rows, holding its own fields plus the paper-level decisions; the paper row is
  chosen among the primary entity's rows. Every row carries `entity` whenever the profile declares entity types, a
  single one included (the gold set and the page find a row's fields by it). The workbook and the page divide
  only with several entity types: one data sheet per entity, named after its `label_zh` (at most 40 characters, no
  control characters), a 实体 column on the 字段说明 and 数据质量 sheets, and one results table per entity on the
  page. A single declared entity keeps the one 样品数据 sheet and no 实体 column.
- **Linking.** A [reference field](#reference-fields) of one entity names a sample of another. The worked example
  is `profiles/catalysis.json`: catalysts, and reaction tests that each name their catalyst.
- **Not supported.** Document mode (refused when a run loads the profile: `paperfacts profiles --check` notes
  it, `prompts` and `fields` still print it); links between entities other than a `reference` field; nesting or order (a layer stack);
  many-to-many or multi-hop relations; a series value across entities; figures bound to an entity; more than
  five entities.

### List fields

`"cardinality": "many"` makes a text or composition field a list: several values that hold at once, such as each
precursor of a sample. A categorical list is `kind: text` with `categories` and `cardinality: many`:

```jsonc
{ "name": "characterization_techniques", "group": "study", "kind": "text", "cardinality": "many",
  "categories": ["XRD", "XPS", "TEM", "SEM", "BET"],
  "description": "Each characterization technique the paper applies to its catalysts." }
```

- **Prompt.** The field line adds "Several values may hold at once: report each as its own entry.", and with
  categories "Name each with one of: XRD, XPS, …." Nothing else in the prompts changes.
- **Comparison.** The lanes' values pair as a set, by element: the category a value names, or else its text
  with Unicode folded, spacing dropped and a hyphen or period dropped unless a digit follows (and case folded, for
  `text` but not `composition`), so OCR's `Ni(NO3)2 · 6H2O` and `co-precipitation` are one element with
  `Ni(NO3)2·6H2O` and `coprecipitation`. This is stricter than a single-valued field's text equality, which drops
  Greek letters: `α-Al2O3` and `γ-Al2O3` are two elements. What
  only one lane read is `missing` on the other. A list never reports `conflict`.
- **Dataset cell.** The **union** of the elements either lane grounded and cited, after the usual refusals
  (`unanswered`, `missing`, the sample-match `ambiguous`, a troubled comparison, `unreviewed`). With categories
  an element is the category it names, in the categories' order, and a quote naming none -- including one naming
  two, "XRD and XPS" -- is refused as an element with a note; without, elements keep their first-seen order, MinerU
  first. The cell is `agree` when both lanes hold every element and `single_source` otherwise, and its 数据质量
  detail names each element's lanes, so the union never hides which lane an element rests on. A cell left with no
  element is `non_scalar`. One `ambiguous` element row -- an element one lane holds under a sample whose match
  failed -- refuses the whole list as `ambiguous`, not just that element: a union without the doubted element,
  or with a stray one, would read as a complete answer. A non-empty list counts as one available field.
- **Display.** Joined with "; " in Excel and the clipboard copy, and with "；" on the web page; 字段说明 marks the
  column 多值. The eval scorer scores a list per element (eval/README.md).

### Yes-or-no, date and interval fields

Three kinds read a value that is not one number or one text. `profiles/catalysis.json` has one of each.

```jsonc
{ "name": "pre_reduced", "group": "reaction", "kind": "boolean",
  "description": "Whether the catalyst was reduced (activated in H2) before this test." }
{ "name": "received_date", "group": "study", "kind": "date",
  "description": "Date the journal received the manuscript, as printed on the paper." }
{ "name": "temperature_window", "group": "reaction", "kind": "interval", "canonical_unit": "℃",
  "rel_tol": 0.01, "abs_tol": 2.0, "valid_range": {"min": 0, "max": 600},
  "description": "The temperature range this test covered, both ends, or the one-sided bound the paper states (°C)." }
```

- **`boolean`.** The model quotes the words that state it ("pre-reduced in H2", "without reduction") and says
  whether they affirm or deny it in a `holds` key. The code never reads negation itself. An answer without
  `holds` is dropped at cleaning, with the reason in the lane's audit (清洗记录). The lanes agree when their
  `holds` agree. The cell is TRUE/FALSE in Excel and 是/否 on the page. A true pass and a false pass never vote as
  one value. The `holds` key is added to the answer format only for a profile that has a boolean field, so every
  other profile is asked exactly what it was asked before.
- **`date`.** The model quotes the date as written; the code reads it to ISO at the precision written ("2021",
  "2021-03", "2021-03-12"). Month names and abbreviations and year-first numbers ("2021/03/12") are read. Refused,
  as an ambiguous value: a two-digit year, an all-numeric date that is not year first ("03/04/2021" is March or
  April), a range, and a year before 1800 or after 2100. The lanes agree when the ISO dates are equal; when one
  is a prefix of the other (a month against a day) the precision differs and the pair is ambiguous.
- **`interval`.** A value that *is* a range: a cycling window, a temperature range a test covered. The model
  quotes both ends and the unit ("200–300 °C"), or a one-sided bound (">80 %", "at most 5 nm"). Each end is
  converted to `canonical_unit`, and the cell is `[low, high]` with an open end empty: "80" quoted out of
  "above 80 °C" is `[80, open]`, exactly as ">80 °C" quoted whole. A bare number, a descending range, and anything
  but one clean range or bound (a condition, a parenthesis, another unit) are refused. `valid_range` is checked on
  each finite end. The lanes agree when both ends are within the field's tolerance and an open end meets an open
  end. The workbook gives an interval two numeric columns, `<name> 下限` and `<name> 上限`; the clipboard copy
  writes `low–high` and the page `≥ low` / `≤ high` for a bound.
- **What each kind accepts.** `canonical_unit`, `valid_range`, `rel_tol` and `abs_tol` go with `numeric` and
  `interval`. `bare_number`, `range_policy`, `after_clause`, `display_format` and `figure_readable` go with
  `numeric` only, and `categories` with `text` only. `condition_rule` goes with every kind but `boolean` and
  `reference`. Anything else is refused at load, naming the field.

### Reference fields

A reference names a sample of another [entity type](#entity-types): the catalyst a reaction test ran on. It is the
one link between entity types.

```jsonc
{ "name": "catalyst", "group": "reaction", "kind": "reference", "references": "catalyst",
  "description": "The catalyst this reaction test was run on.", "keywords": ["catalyst", "ran", "over"] }
```

- **Declaring.** `references` names another declared entity; the field is sample-level of a different entity,
  single-valued, and has no unit, categories or `condition_rule`.
- **Asking.** The field line tells the model to copy the id from the referenced entity's list, and the question
  shows that list ("Catalysts this paper reports:") after the field's own. A lane whose referenced inventory
  listed nothing is not asked.
- **Grounding is resolution.** The value is an id, not a quote from an excerpt. It is grounded when it names one
  of the lane's samples of that entity by `sample_key` ("cat-1" names "Cat 1"). This is redone every time a lane is
  read, like every other grounding verdict. An id two differently spelled samples of that entity share is refused
  as ambiguous rather than given to either. Passes vote on the id by `sample_key` too, so "Cat-1" and "cat 1" are
  one answer.
- **Comparing.** The two lanes agree when the referenced entity's matching pairs the two samples they name. When
  either of them is paired with any other sample, they are two samples and it is a `conflict`, since matching is
  one to one. When neither is paired, or one names no listed sample, it is `ambiguous`.
- **The cell** is the `sample_id` of the dataset row the referenced sample became, so it names a row of that
  entity's sheet (字段说明 calls the column `<entity label>样品ID`). An id no listed sample has leaves the cell
  empty as `ungrounded`.
- **Gold.** A gold reference cell holds the `id` of the gold sample it names ([eval/README.md](eval/README.md)).

### Which edit re-keys what

A stored result is named by the keys of exactly what it depends on (see
[Caching](#caching-and-why-filenames-carry-keys)), and every field attribute's role decides which keys it reaches.
Re-keying extraction means the next run re-asks the model (paid) unless the rendered requests are unchanged;
re-keying the comparison recomputes it from the stored extractions, for free.

| Edit | `extractor_key` | `comparison_key` | `figure_key` |
|---|---|---|---|
| Display: `label`, `description_zh`, `display_format`, a group's `label_zh`, `title_zh`, `description_zh`, `maturity`, `ui`, `$comment`, the file name | — | — | — |
| Verdict: `rel_tol`, `abs_tol`, `categories` (of a `many` field: also extraction, as its `prompt_categories`), `condition_preference`, `missing_condition_note_zh` | — | yes | — |
| Prompt and cleaning: `name`, `group`, `kind`, `description`, `canonical_unit`, `condition_hint`, `condition_rule`, `valid_range`, `bare_number`, `range_policy`, `after_clause`, `cardinality`, a group's name or level, the order of the fields | yes | yes | only for a `figure_readable` field's `name`, `description`, `canonical_unit`, `bare_number` |
| `keywords` | passage mode | — | for a `figure_readable` field |
| `retrieval` | passage mode | — | — |
| `units`, `ignored_unit_suffixes` | yes | yes | yes |
| A `prompt` slot | yes, except the three `matching_*` slots (a slot only the inventory or field prompt uses: passage mode only) | only `sample_plural`, `condition_noun` and the `matching_*` slots | — |
| `figures` slots, `figure_readable` | — | — | yes |
| `entities` (names, order, overridden slots, retrieval), a group's `entity` | yes (retrieval: passage mode) | yes | — |

Display edits are therefore safe on a live library. The file name is in no key, but it names the workbooks
(`exports/<name>.xlsx`) and the readings directory (`figures/<name>/`), so a renamed profile writes new ones
beside the old. Two profiles with identical non-display content share every key and every stored file.

### Declared units

```jsonc
"units": {
  "mAh/g": { "aliases": { "mAh/g": 1, "mAh g-1": 1, "mAh g^-1": 1, "Ah/kg": 1, "Ah/g": 1000 }, "case_sensitive": true },
  "C":     { "aliases": { "C": 1 }, "case_sensitive": true, "retrieval": "\\d\\s*c\\b(?!\\s*°)" },
  "V":     { "aliases": { "V": 1, "mV": 0.001 }, "case_sensitive": true },
  "℃":     { "extends_builtin": true, "aliases": { "K": { "factor": 1, "offset": -273.15 } }, "exclude": ["C"] }
}
```

- **Aliases** (`aliases`). Each spelling maps to a factor, or to `{factor, offset}`: a value quoted in it becomes
  `value * factor + offset` in the canonical unit. 1 to 50 spellings; factors finite and above 0. A new unit lists
  its own spelling with factor 1. A power of ten from a table header is applied first, then the offset.
- **Offsets** are for temperature only (a canonical unit of `℃` or `K`), since K to ℃ is the one conversion a
  factor cannot do, and are never applied to a bare number.
- **Case.** Spellings are compared case-insensitively unless `case_sensitive` is true -- set it whenever a
  prefix matters (`mS` against `MS`, `mAh` against `MAh`). Two spellings that are the same once folded are
  refused.
- **Spaces.** A spelling is compared with its spaces removed, as a quoted unit is, so `mAh g-1` also reads
  `mAhg-1`; declaring both is refused as a duplicate.
- **Retrieval.** Every canonical unit needs a pattern that finds a number in it in running text. By default it
  is derived from the spellings: a digit, then the spelling lower-cased with its spaces optional, then a word
  boundary when it ends in a letter or digit (so `V` does not match "Vis"). Give `retrieval` yourself when that
  is too loose -- a bare `C` would match "°C". It is matched case-insensitively, on lower-cased text.
- **Extending a built-in.** A built-in unit (`Ω/sq`, `Ω·cm`, `nm`, `min`, `inch`, `%`, `℃`, `cm`, `W`, `sccm`,
  `rpm`, `Pa`) can only gain spellings, with `extends_builtin` set to true; its own converter is always asked first, so
  an extension never changes how a spelling it already reads converts. The TCO conventions stay: under `tco`,
  573 K is ambiguous; under `battery_cathode`, it is 299.85 ℃.
- **Excluding built-in spellings** (`exclude`, on an extension only). The built-in tables are TCO's conventions,
  and a spelling can mean something else in another domain: to TCO a bare "C" after a number is degrees, to a
  battery group it is a C-rate. List such spellings (matched case-insensitively) and the built-in converter
  refuses them and its retrieval pattern stops finding them after a number, so under `battery_cathode` "1 C" is
  neither a temperature nor a reason to show a block to a temperature question. Each must be a spelling the
  built-in reads; an extension that only excludes may leave `aliases` out.

### Cost

Every field is one question per lane in passage mode. For one paper, uncached:

    LLM calls ≈ 2 lanes × (E inventories + passes × F) + Σ M  [+ repairs]  [+ chart panels]

- `F` is the number of fields asked in that lane, at most the profile's field count `N`: a field no block of the
  lane mentions is not asked, and neither are the sample-level fields when the inventory says the paper has no
  in-scope sample.
- The inventory is asked once per lane per entity type (`E`, 1 without `entities`) whatever
  `extraction.passes` is.
- Each `M` is 0 or 1, one per entity type: sample matching asks the model only when both lanes have samples of
  that entity left after pairing identical ids.
- A question whose answer fails validation costs one repair request, at most.
- With the figures stage on, each chart panel is one vision request (at most `figures.max_per_document`,
  retried once on failure).
- Document mode is `2 × passes + M` (a profile without entity types only).

So TCO's 23 fields cost at most 49 calls per paper at one pass, the battery example's 13 at most 29, and the
catalysis example's 18 fields over two entity types at most 2 × (2 + 18) + 2 = 42. Most of the
completion tokens are the inventory's reasoning (see `llm.inventory_reasoning_effort`), so the bill grows more
slowly than the call count. A re-run is free: every answer is cached by request payload.

### A second profile beside the first

One server serves every profile under `profiles/`. The one `profile` / `PAPERFACTS_PROFILE` / `--profile`
selects is the **default**: it must load or the server does not start, and every request that names no profile
is answered under it, so every link and bookmark from before keeps its meaning. `PAPERFACTS_PROFILE=battery_cathode uv run paperfacts serve`
serves the same profiles with `battery_cathode` as the default; `web.profiles` narrows the others (see Configuration). Every other `profiles/*.json` is
loaded beside it at start-up; one that does not load is listed with its errors (`GET /api/profiles`, `invalid`)
and never stops the others, and so is a file that is a link to another profile's file (it loads as that
profile). `web.profiles` names the ones to serve beside the default when not all of them should be (every run
costs tokens). A profile with entity types under `extraction.mode: document` is listed and described but not
runnable: its keys cannot be computed under that mode, so every read and write under it answers 409.

The HTTP interface names the profile with `?profile=<name>` on every route whose answer depends on one:
`/api/profile`, `/api/documents` (upload, run, run-all, and every per-document read but the parse artifact and
the page images, which are the document's under every profile), `/api/dataset` and `/api/dataset.xlsx`. Absent
means the default, and every such response names the profile that answered in an `X-PaperFacts-Profile` header,
so a caller that forgot the parameter can tell. `GET /api/profiles` lists the served profiles (with how many documents each has finished,
whether each is runnable and whether its file changed on disk) and the invalid files; `GET /api/profiles/<name>`
is one profile's full read-only definition (groups, entities, every field attribute, declared units, retrieval);
`GET /api/profiles/<name>/prompts?field=` is what `paperfacts prompts` prints, from the same function. A
document summary names in `profiles_done` every profile it is finished under.

A document is shared: its PDF, parses and identity belong to it under every profile, so the server never runs
two jobs on one document at once, whatever their profiles; a job under a second profile waits for the first.
Submitting a document again under the same profile while it is queued or running returns the same job. Results
need no separating: every derived file is named by keys that follow the profile's content, workbooks and chart
readings by the profile's name. Two profiles of identical content share their results by design: both list a
finished document in `profiles_done` and both show its results, but its workbook download
(`/api/documents/<id>/dataset.xlsx`) is named after the profile a run was made under and is missing under the
other until a run under it writes one.

Run one server per data root: the per-document and per-parser locks live in one process, so two servers over
one `data_root` can parse the same document at once. Profiles may still share a data root outside a server --
`paperfacts batch papers/ --profile battery_cathode` over the library a server uses, while it is idle -- and
then share the parses and the LLM cache. A server reads its profiles once; a file added to `profiles/` is served
after a restart, and after a served file changes on disk its jobs are refused until the restart
(`/api/health` reports it per profile).

### Proving a refactor free

A change meant to alter no answer -- moving code, renaming, touching a hashed module -- still re-keys, and the
proof that it cost nothing is an offline re-derivation on a copy of the data:

```bash
PAPERFACTS_LLM_OFFLINE_REPORT=misses.json uv run paperfacts batch papers/ --offline --data-root /copy/of/data
python scripts/diff_derived.py --data-root /copy/of/data --old <ek>.<ck> --new <ek>.<ck>
uv run python eval/score.py --data-root /copy/of/data --keys <ek>.<ck>
```

`offline misses: 0` means every request the new code sends is one the old code already sent
([Offline replay](#offline-replay)); `diff_derived.py` then compares every document's facts, comparison and
dataset under the old keys against the new, ignoring only the keys, fingerprints and usage counters; and the
scorer's cells against the gold set must not move. Never run this against production's `data/`.

### Where a profile's output goes

| Path | Per profile? |
|---|---|
| `raw/`, `parsed/`, `pages/`, `overlays/`, `identity.json`, `llm_cache/` | Shared by every profile |
| `facts/`, `comparisons/`, `datasets/` | Named by keys, which differ between profiles |
| `figures/<profile>/<figure_key>.json` | Per profile (TCO also reads the older flat `figures/<figure_key>.json`; `scripts/migrate_figures.py` moves them) |
| `docs/<id>/exports/<profile>.xlsx`, `exports/<profile>.xlsx` | Per profile; the pre-profile `dataset.xlsx` and `exports/paperfacts.xlsx` are left where they are, and the name `paperfacts` is reserved |

Lanes, comparison reports and datasets record the profile's fingerprint; comparing two lanes extracted under
different profiles, or consolidating a report under another, is refused rather than mixed.

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
  field the profile marks `figure_readable` (in the TCO profile: sheet resistance, resistivity,
  transmittance, thickness) by one of the field's retrieval keywords, and then every panel is asked about separately, up to `figures.max_per_document` panels per
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
- **Stored.** `figures/<profile>/<figure_key>.json` per document: two profiles with the same chart slots and
  figure fields share a `figure_key`, and each keeps its own file. The TCO profile also reads the flat
  `figures/<figure_key>.json` files stored before readings were kept per profile, until a data root is migrated
  once with `uv run python scripts/migrate_figures.py --data-root data --apply` (without `--apply` it prints the
  plan): each flat file moves into the directory of the profile it records, `tco` when it records none, and is
  stamped with it; a file whose destination already exists is left for you to look at. `figure_key` hashes the vision model and its
  sampling, `figures.dpi`, `figures.max_pixels`, `figures.max_per_document`, the profile's `figures` slots, the
  `figure_readable` fields' names, descriptions, keywords, units and bare-number policies, the declared units,
  and the source of `figures.py`, `normalize.py`, `readers.py`, `passages.py`, `units.py`, `text.py`, `fields.py` and
  `profile.py`. Stored readings are
  shown and exported even when the stage is switched off for a later run. With nothing under the current
  key, the newest older file is shown and marked stale (旧版本读数); readings citing figure blocks the
  current parse no longer has are marked too, and both notes appear in the stage detail.

### Offline replay

`--offline` on `run` or `batch` (or `PAPERFACTS_LLM_OFFLINE=1` for any command) answers every model request,
text and vision, from the LLM cache and never sends one. It is how a refactor proves it re-derives the
corpus for zero model calls. It is a switch for one invocation, not a standing setting. `llm.offline` in
`config.json` is still read, for compatibility, but a replay left switched on in the shared file would make
every later run fail. `serve` refuses to start under it. `--force` and `--force-figures` cannot be combined
with it, because a forced request skips the cache and could only miss.

A request the cache cannot answer raises `LlmOfflineMiss` and is logged as `llm offline miss key=… user=…`.
A miss is never an outcome. It is not a failed panel, a failed figures stage or a failed sample matching.
The paper fails, and nothing derived from the miss is written: no readings file, no comparison, no
dataset. `batch` goes on with the other papers. `run` and `batch` end with `offline misses: N`, whether they
finished or failed. When `PAPERFACTS_LLM_OFFLINE_REPORT` names a file, the misses are also written there as
JSON. Each entry has three parts:

- `key`: the first 16 hex digits of the cache key.
- `kind`: `json`, `repair` (the follow-up to an answer that failed validation) or `vision`.
- `user`: the first 120 characters of the question.

This lets two replays be compared by request.

N is a **lower bound**. It counts the misses reached before each paper stopped, and the first miss hides
the ones behind it:

- A missed inventory question means the field questions are never built.
- A missed lane means sample matching never runs.
- A failed lane stops the figures stage before its remaining panels are asked.

`0` is the proof. Any other count means "at least this many", and the set can differ between two replays
of the same change.

A question that needed a repair replays too. When an answer fails validation, it is also kept under its own
cache entry, `<key>.rejected.json`, which an online run never reads. Offline replay serves the accepted
answer when there is one, exactly as online. Otherwise it serves the rejected answer, but only if that
answer still fails validation. The repair request is then rebuilt from the rejected answer and pydantic's
error, and the cache answers that too. A rejected answer that now validates (a loosened schema) is a miss,
because an online run would ask again.

Three limits follow:

- **Same `uv.lock`.** The repair request quotes pydantic's error text, including its documentation URL with
  the pydantic version. A different pydantic moves every repair request's key, and every repaired question
  misses.
- **Caches recorded before rejected answers were kept.** On such a cache, every question that needed a
  repair misses until one online run has recorded its rejected answer. That run asks the model again, and
  the model may answer differently.
- **Permanent misses.** A reply cut off at `max_tokens`, or a request that ended in an HTTP error, was
  never cached. It misses on every replay, so a corpus that has one never reaches zero.

## Caching, and why filenames carry keys

Nothing is recomputed unless something it depends on changed, and each cache is keyed by a content hash of
exactly its own inputs. The hashes are the `<key>` in the filenames under a document directory.

| Cache | Keyed on | Invalidated by |
|---|---|---|
| Parser output | nothing; `raw/<backend>/meta.json` exists or it does not | `--force` |
| Extraction (`extractor_key`) | the model and its sampling settings (one `ExtractionOptions`, built the same way by the writer and every reader); the profile's field attributes with the PROMPT or CLEANING role, its groups and its declared units; the rendered system prompts; every prompt slot not at its default, except the `matching_*` ones; the document rendering; and the source of `extract.py`, `records.py`, `fields.py`, `profile.py`, `units.py`, `text.py`, `adapters.py`, `prompts.py`, `normalize.py`, `readers.py`, `grounding.py`, `voting.py`, `continuation.py` and `kinds.py`. Passage mode adds its two prompts, `candidate_limit`, `context_tokens`, the inventory effort, and a retrieval fingerprint over the keywords, the profile's `retrieval` section and unit patterns, and `passages.py`, `continuation.py`, `units.py`, `text.py` and `fields.py` | changing any of them |
| Comparison (`comparison_key`) | the same field attributes plus the VERDICT ones (tolerances, categories, condition preferences, `missing_condition_note_zh`), the groups and units, `ambiguous_match_confidence`, the matching prompt and its `matching_*` slots not at their default, and the source of `normalize.py`, `readers.py`, `units.py`, `text.py`, `kinds.py`, `compare.py`, `matching.py`, `dataset.py`, `decide.py`, `fields.py` and `profile.py` | changing a tolerance or a rule |
| Figure readings (`figure_key`) | the vision model and its sampling, the crop settings, the per-paper limit, the `figure_readable` fields and the chart slots, and the source of `figures.py`, `normalize.py`, `readers.py`, `passages.py`, `units.py`, `text.py`, `fields.py` and `profile.py` | changing any of them |
| LLM requests | the entire request payload (a chart's image by its sha256) | nothing — an identical request is free |

The workbook layout (`workbook.py`), where chart readings are stored and which are shown (`readings.py`), the
profile's display copy defaults (`ui_copy.py`), and the orchestration and transport (`workflow.py`, `batch.py`,
`llm.py`, `config.py`, `cli.py`) are in no key: editing them renames no stored file. Nor is the profile's file
name or any of its display text; [Domain profiles](#which-edit-re-keys-what) has the whole table.

Extractions, comparisons and consolidated tables also record the parse they came from (a hash of the
artifact's blocks). After a re-parse, a stored lane, comparison or table of the old parse is a miss (not
served, and the paper is not finished) and is derived again: source ids
are positional, so the old citations would point at whatever block now has that ordinal. Re-deriving is
free from the LLM cache whenever the rendered prompts are byte-identical. Files written before the hash was
recorded have none and are read as before -- which for a consolidated table means it is served even after a
re-parse. Re-export once after upgrading (`paperfacts export data/docs`, offline and free) to record the
hashes in every stored table.

Only an answer that validated is cached. A JSON reply cut off at `max_tokens` is an error, an invalid answer
costs one repair request and is never written as the answer (it is kept apart only for
[offline replay](#offline-replay)), and an invalid answer already in the cache is asked again rather than
replayed. A sample matching that failed (the model answered badly twice) is shown for that run
but not stored, so the next run asks again instead of serving the failure until `--force`. Neither is that
run's consolidated table (`datasets/…json`, only `exports/<profile>.xlsx` is written): the stored table is what marks
a paper finished, so 「处理全部未完成」 and `deploy.sh --rerun` pick the paper up again.

The same holds for one field question in passage mode that gets no valid answer (invalid twice, or cut off):
it costs that field, not the lane. The lane is stored with the question in `failed_questions` and its other
fields intact; that field's cells are `unanswered` in both lanes; the run's workbook marks the paper
`incomplete` in 运行记录 (and `run`/`batch` say so); the comparison and table of that run are not stored, and the next run extracts the lane again,
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
├── exports/<profile>.xlsx              the default batch workbook
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
    ├── figures/<profile>/<figure_key>.json   values read off charts by the opt-in figures stage
    ├── exports/<profile>.xlsx          this paper's workbook, written automatically by `run`
    ├── overlays/<backend>/page_*.png   bbox overlays from `overlay`
    └── pages/<dpi>dpi/                 page renders for the web viewer
```

Every block carries `page` plus a bounding box normalised to `[0, 1]`, and every extracted value cites the
block ids it was read from, so any value maps back to a rectangle on a page. An export made under
different settings lands beside the old one instead of overwriting it.

## The Excel workbook

`run` writes `data/docs/<sha>/exports/<profile>.xlsx` for one paper (`<profile>` is the domain profile's
name, `tco` by default; workbooks from before profiles, `dataset.xlsx` and `exports/paperfacts.xlsx`, are left
where they are); `batch` and `export` write one workbook for a
whole directory; the web UI serves the same thing behind 「下载 Excel」 and 「下载全部 Excel」, downloaded as
`<profile>-<id>.xlsx` and `<profile>-corpus.xlsx`. Six
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

A profile with several [entity types](#entity-types) replaces 样品数据 with one sheet per entity, named
`<label_zh>数据` (催化剂数据, 反应测试数据), each holding the paper-level columns and its own fields; 字段说明 and
数据质量 gain a 实体 column. A list cell is joined with "; ", a yes/no cell is TRUE/FALSE, and an interval fills two
numeric columns, `<name> 下限` and `<name> 上限`. Control characters in any text are removed on the way in.

数据质量 is where the provenance is: 最终决策 is `agree` or `single_source` for a committed value and the
refusal name otherwise, 合并证据来源 lists the block ids behind it, **证据来源通道** says which lanes
supplied it, and **系列级** marks a value the paper stated once for the whole sample series. A paper-level
value's row has 样品ID `paper`.

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
Grounding also notes a bound the block writes right before the quote (above, over, more/greater/higher than, exceeding,
at least, below, less/lower than, up to, at most, `>`, `≥`, `<`, `≤`; not `~`, and not "under", which papers use for a condition): `90`
quoted out of "above 90 %" is then read exactly as a quoted "above 90 %" and fills no dataset cell.

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
| `unanswered` | One lane's question about this field got no valid answer. Refused in both lanes, so the other lane's value never passes as single-source; the next run asks that question again |
| `multiple_conditions` | One lane recorded the field under several measurement conditions, so no single value is the answer |
| `multiple_values` | One lane recorded several different values under the same condition, or several candidates were never confirmed across lanes |
| `non_scalar` | Every candidate is a range (under `range_policy` `midpoint` or `reject`), a bound, or a rectangular dimension such as `40 × 10 cm`; no unique scalar exists |

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
duration written in two of its units, larger first, is one value: `3 h 30 min` is 210 min, `1 min 30 s` is
1.5 min, in the comparison and the dataset cell alike (`1 h 90 min` is refused). Only a duration is a sum:
elsewhere a second unit restates the value (`0.5 Pa 3.75 mTorr`), and that is refused. A power of ten in a
table header is copied by the model into `unit_raw` and applied by the code in the convention the header
wrote: leading the unit (`ρ (10^-4 Ω cm)`, `×10^-4 Ω·cm`, `ρ × 10^-4 Ω·cm`) the cell is multiplied by it;
on the quantity with the unit bracketed apart (`ρ × 10^4 (Ω cm)`) it is divided, so a cell of 6.8 is
6.8 × 10⁻⁴ Ω·cm either way. A header that says neither (`Ω·cm × 10^-4`, a factor bracketed alone beside the
symbol as in `ρ (×10^-4) (Ω cm)`, `ρ × 10^4` with no unit), or a cell that carries its own power of ten as
well, is refused. A range
keeps its midpoint whether or not each bound repeats the unit (`80%–85%`, `500 °C to 530 °C`). A condition
after the value (`550 nm at 80%`, `400 °C for 2 h`, `500 °C under N2`; `at`, `for`, `during`, `under`, and only
when a number stays before it, so `deposited for 10 min` still reads 10; a condition holding the field's own
quantity when the value does not, such as annealing time `400 °C for 2 h`, is refused; `after` introduces
another state of the sample, so `85% after 10 cycles` is refused), a name before `=` (`O2/(Ar+O2) = 5%`) and the digits of a formula or a
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
| `profiles/` | The domain profiles; see [Domain profiles](#domain-profiles) |
| `runners/` | The two PEP 723 parser scripts and their committed lockfiles |
| `deploy/` | Linux GPU server deployment |
| `tests/` | The pytest suite and the recorded parser fixtures |
| `.omc/research/` | The measurements behind the defaults |

Use `uv sync --group dev`; never `pip install` into the venv. After editing a runner's dependency header,
run `uv lock --script runners/<name>.py`. `CLAUDE.md` holds the internal conventions for contributors —
nothing a user or an operator needs.
