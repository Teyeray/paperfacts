# PaperFacts Deployment (Linux 8-GPU Server)

This directory does exactly one thing: **run the two parser services** so the main package
(the orchestrator) can call them over HTTP.

## Architecture

```text
paperfacts orchestrator (pure python, no torch / paddle installed)
     │
     │  HTTP
     ├──→ mineru-router            :8002   POST /file_parse       GPU 4, 5
     │       └─ internally: one mineru-api worker per GPU, load-balanced by the router
     │
     └──→ paddleocr-vl-api         :8080   POST /layout-parsing   GPU 6
             └─ internal HTTP ──→ paddleocr-vlm-server (vLLM service for the 0.9B VLM) GPU 6
```

The main package `src/paperfacts` only depends on pure-python packages like pydantic / httpx /
pypdfium2, and **never imports mineru or paddleocr**. Each parser lives in its own environment
(on a dev machine, that's the isolated venvs used by the PEP 723 scripts
`runners/mineru_runner.py` and `runners/paddle_runner.py`; on the server, it's the
long-running services deployed here). Both paths produce an identical `meta.json` structure,
so upstream code never needs to know which mode it's running in.

There are two deployment routes:

- **Route A (recommended)**: Docker Compose. Reproducible, cleanly isolated, upgrades are just
  an image tag bump.
- **Route B (fallback)**: bare-metal host install. Use only when docker access isn't
  available.

---

## Prerequisites

| Item | Requirement | Check command |
|---|---|---|
| NVIDIA driver | Supports **CUDA 12.6 or newer** (driver ≥ 560). Hard requirement of the official PaddleOCR-VL image. | `nvidia-smi` (the CUDA Version shown top-right is the driver's upper bound) |
| GPU | ids 4, 5, 6, 7 available and idle | `nvidia-smi` |
| Docker | Engine 20.10+, **docker compose v2** (`docker compose`, not `docker-compose`) | `docker compose version` |
| nvidia-container-toolkit | installed and configured as a docker runtime | see below |
| Disk | at least 80GB free (MinerU image + two official Paddle images + model weights) | `df -h /var/lib/docker` |

Verify GPUs are usable inside a container (**test GPU 4 only — do not touch 0-3**):

```bash
docker run --rm --gpus '"device=4"' nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

Seeing that GPU's info means nvidia-container-toolkit is working. If you get
`could not select device driver "nvidia"`, the toolkit isn't installed or the docker daemon
hasn't been restarted since installing it.

---

## Route A: Docker Compose (recommended)

```bash
cd deploy

# 1) Prepare environment variables (image tags, torch wheel index, etc — all documented inline)
cp .env.example .env

# 2) Build the MinerU image
#    MinerU has no official prebuilt image, so this must be built locally.
#    This step pre-downloads the pipeline model weights (several GB) into the image; the
#    first build usually takes 20-40 minutes depending on bandwidth to HuggingFace. Run it in
#    tmux.
docker compose build mineru-router

# 3) Bring up all three services
#    The two PaddleOCR images are official prebuilt images — pulled, not built.
docker compose up -d

# 4) Watch startup progress (the VLM has a cold start of a few minutes; health shows
#    "starting" until then)
docker compose ps
docker compose logs -f
```

Verify:

```bash
curl http://localhost:8002/health      # MinerU router
curl http://localhost:8080/health      # PaddleOCR-VL API
```

Check GPU usage — **only GPUs 4, 5, 6 should show any usage; 7 must be idle, and 0-3 must have
none of our processes**:

```bash
nvidia-smi
```

Stop / restart:

```bash
docker compose down          # stop and remove containers (models live in the image, not lost)
docker compose restart       # restart only, no rebuild
```

### What each file does

| File | Purpose |
|---|---|
| `compose.yaml` | Orchestration for the three services; GPU allocation is pinned here |
| `.env.example` | Environment variable template; `cp` to `.env` before use |
| `mineru.Dockerfile` | MinerU service image (built locally) |
| `mineru.Dockerfile.dockerignore` | Empties the build context so `template_files/` and `data/` aren't sent to the docker daemon |
| `vllm_config.yaml` | vLLM startup args for the VLM service (memory fraction, concurrency) |
| `host/` | Scripts for Route B |

---

## Route B: bare-metal host install (no docker access)

Creates two mutually isolated venvs under `~/.paperfacts/envs/`. **Isolation is required**:
MinerU depends on torch, PaddleOCR depends on paddlepaddle, and their bundled CUDA runtime
libraries (cudnn / nccl) frequently conflict in version.

```bash
# One-time install (creates venvs + installs dependencies + downloads models + generates the
# pipeline config)
bash deploy/host/setup.sh
```

Once installed, start all three services. They're all long-running foreground processes —
**run each in its own tmux window**:

```bash
tmux new -s paperfacts

# Window 1: MinerU (GPU 4,5 → :8002)
bash deploy/host/start_mineru.sh

# Window 2: PaddleOCR-VL's VLM service (GPU 6 → :8118, local only)
bash deploy/host/start_paddle_vlm.sh

# Window 3: PaddleOCR-VL's API layer (GPU 6 → :8080) — must start after window 2 is ready
bash deploy/host/start_paddle_api.sh
```

`start_paddle_api.sh` curls the VLM's `/health` before starting and exits with a clear error
if it's not reachable, instead of surfacing a wall of confusing openai connection exceptions.

Verification is the same as Route A (`curl localhost:8002/health`, `curl localhost:8080/health`).

### About `deploy/host/paddleocr_vl_pipeline.yaml`

This file is generated by `setup.sh` and isn't checked into version control, since its
content depends on the installed paddleocr version. Generation flow:

1. `paddlex --get_pipeline_config PaddleOCR-VL-1.6 --save_path <temp dir>` copies paddlex's
   built-in pipeline config out (a plain file copy, no model weights needed);
2. the script rewrites the VLM backend fields to point at the local vLLM service.

**Fields worth checking by hand** (after generation, `cat` the file once):

```yaml
SubModules:
  VLRecognition:
    model_name: PaddleOCR-VL-1.6-0.9B     # must exactly match the VLM service's --model_name
    genai_config:
      backend: vllm-server                 # changed from native to vllm-server
      server_url: http://127.0.0.1:8118/v1 # must include the /v1 suffix
```

Field names follow the parameter names in `paddleocr doc_parser --help` (corresponding to
`--vl_rec_backend` / `--vl_rec_server_url`). The above has been verified against paddleocr
3.7.0 / paddlex 3.7.2; **if a future version renames these fields, treat the actually
generated yaml as the source of truth** — `setup.sh`'s rewrite step will fail loudly instead
of writing silently broken output if it can't find the `genai_config` block it expects.

Keeping `model_name` in sync isn't pedantry: the API layer sends the pipeline's `model_name`
as the OpenAI request's `model` field, while vLLM's `served-model-name` comes from the
`--model_name` it was started with. A mismatch means every request gets a 404. That's why
`start_paddle_vlm.sh` reads the model name directly from this yaml by default, rather than
hardcoding a constant.

---

## GPU allocation table

The server has 8 GPUs total. **PaperFacts may only use 4, 5, 6, 7 — 0-3 belong to other
projects and must never be touched.**

| GPU | Purpose | Owner | Memory policy |
|---|---|---|---|
| 0-3 | **Someone else's** — do not use | — | — |
| 4 | MinerU pipeline worker #1 | `mineru-api`, spawned by `mineru-router` | one worker per GPU, allocated on demand |
| 5 | MinerU pipeline worker #2 | same as above | same as above |
| 6 | PaddleOCR-VL (both layers share this GPU) | `paddleocr-vlm-server` + `paddleocr-vl-api` | vLLM capped at `gpu-memory-utilization: 0.5`, leaving the rest for the layout detection model PP-DocLayoutV2 (1-2GB) |
| 7 | **Unused, currently free** | — | — |

Route A enforces this via compose's `device_ids`; Route B enforces it via
`CUDA_VISIBLE_DEVICES` in each script, and `require_allowed_gpus` in
`deploy/host/_common.sh` validates the id before startup — a typo exits immediately instead
of being discovered only after a model has loaded and grabbed someone else's GPU.

---

## Configuring the orchestrator

The main package only reads two environment variables (see `src/paperfacts/config.py`):

```bash
export PAPERFACTS_MINERU_URL=http://localhost:8002
export PAPERFACTS_PADDLE_URL=http://localhost:8080
```

- **Both set** → calls the long-running services deployed here over HTTP (production mode)
- **Neither set** → falls back to running parsers as subprocesses via
  `runners/mineru_runner.py` / `runners/paddle_runner.py` (dev machine mode — how it runs on
  a Mac)

When the orchestrator runs on a different machine, replace `localhost` with this server's
IP/hostname, and make sure the firewall allows 8002 and 8080.
Note: **do not expose 8118 externally** — that's the VLM's raw OpenAI-compatible endpoint,
meant only for the local API layer.

---

## Troubleshooting

### 1. Driver too old

Symptoms:
- container fails to start with `CUDA driver version is insufficient for CUDA runtime version`;
- or the official Paddle images fail to start at all.

Fix:

- **MinerU**: set `TORCH_INDEX_URL` in `.env` to install an older CUDA build of torch, then
  rebuild:
  ```bash
  # .env
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126   # or cu121
  ```
  ```bash
  docker compose build --no-cache mineru-router
  ```
- **PaddleOCR-VL**: the official image requires CUDA 12.6+ driver support, with no room to
  downgrade. Either upgrade the driver, or check whether the vendor offers a lower-CUDA tag
  via `API_IMAGE_TAG_SUFFIX` / `VLM_IMAGE_TAG_SUFFIX` in `.env`.
- If upgrading the driver truly isn't an option, use Route B instead and point
  `PAPERFACTS_PADDLE_INDEX` at an older CUDA index such as
  `https://www.paddlepaddle.org.cn/packages/stable/cu118/`.

### 2. Out of memory (OOM)

The VLM reports `No available memory for the cache blocks` at startup, or the API layer
reports a CUDA OOM:

Edit `deploy/vllm_config.yaml`:

```yaml
gpu-memory-utilization: 0.4   # lower it, leaving more room for the layout detection model
max-num-seqs: 32              # lower concurrency, reducing peak KV cache usage
max-model-len: 8192           # add this on a small GPU (default is 16384)
```

Restart after editing: `docker compose restart paddleocr-vlm-server` (on Route B, just re-run
`start_paddle_vlm.sh`).

Conversely, if GPU 6 has memory to spare and you want more throughput, raise `max-num-seqs`
to 128 or 256.

### 3. Offline / air-gapped machines

- **PaddleOCR-VL**: both tags in `.env` default to the `-offline` suffix, meaning model
  weights are already baked into the image and the container starts without any network
  access. Just move the images over with `docker save` / `docker load`.
- **MinerU**: the image build already runs `mineru-models-download`, and the weights live
  inside the image at `/opt/mineru/models`; at runtime, `MINERU_MODEL_SOURCE=local` only
  reads local files. So **the build must happen on a machine with internet access**, then
  transfer with `docker save paperfacts-mineru:3.4.5 | ...`.
- **Route B**: `setup.sh` needs internet access (to install packages and download models). On
  an air-gapped machine, first install `~/.paperfacts/envs/` on a networked machine with the
  same architecture, then copy the whole directory over, along with `~/mineru.json` (it
  records absolute model paths, which must match between the two machines, or be edited by
  hand).

### 4. Port conflicts

`docker compose up` reports `address already in use`:

```bash
ss -ltnp | grep -E ':(8002|8080)'
```

Changing ports requires updating two places consistently:

- the host side of `ports` in `compose.yaml` (e.g. `"18002:8002"` — the number after the
  colon is the in-container port; leave that alone);
- `PAPERFACTS_MINERU_URL` / `PAPERFACTS_PADDLE_URL` in `.env`.

On Route B, override directly with `PAPERFACTS_MINERU_PORT` / `PAPERFACTS_PADDLE_PORT` /
`PAPERFACTS_VLM_PORT`.

### 5. API layer can't reach the VLM service (Route A only)

The official PaddleOCR-VL image's built-in `pipeline_config_vllm.yaml` looks up the VLM
container by **service name**, so the service in `compose.yaml` **must not be renamed** (it
must stay `paddleocr-vlm-server`). To check what the built-in config says:

```bash
docker compose run --rm --entrypoint cat paddleocr-vl-api /home/paddleocr/pipeline_config_vllm.yaml
```

If its `server_url` points at a different hostname, rename the VLM service in `compose.yaml`
to match.

### 6. MinerU worker reports `ModuleNotFoundError: No module named 'six'`

mineru 3.4.5's `mineru/model/utils/pytorchocr/data/imaug/operators.py` does `import six`
directly, but it isn't declared as a dependency. Both `mineru.Dockerfile` and `host/setup.sh`
already install `six>=1.16` explicitly, so this shouldn't happen normally. If it does in a
hand-rolled environment, `pip install six` fixes it.

### 7. `/health` never comes up

Cold starts are inherently slow:
- each MinerU worker loads three model sets (layout / OCR / table), taking a few minutes;
- vLLM loads weights and compiles a CUDA graph, roughly 5 minutes (that's what
  `start_period: 300s` in compose accounts for).

Check the logs before concluding something's wrong: `docker compose logs -f mineru-router`
(on Route B, check the tmux window instead).


## Web UI (optional)

The main package's web UI is a separate concern from the parser services: the parser services
run via compose, while the web UI is started from the environment running `paperfacts` (on
the host, after `uv sync`):

```bash
export PAPERFACTS_MINERU_URL=http://localhost:8002
export PAPERFACTS_PADDLE_URL=http://localhost:8080
export DEEPSEEK_API_KEY=...                     # or put the key in a `deepseek_api_key` file at the repo root
uv run paperfacts serve --host 0.0.0.0 --port 8000
```

Recommended to run long-lived in tmux. The web UI is intended for local-network use only and
is protected by HTTP Basic auth once `PAPERFACTS_WEB_PASSWORD` is set in the server's `.env` (the
username is `web.username` in config.json); every route, `/api` included, answers 401 without it. Set it
before exposing the port, and still keep a reverse proxy with TLS in front of it.
Uploaded PDFs and all derived artifacts live under `data/docs/<sha>/`, sharing the same
directory layout as documents processed from the command line.
