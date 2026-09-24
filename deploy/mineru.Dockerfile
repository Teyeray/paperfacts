# PaperFacts -- MinerU 3.4.5 (pipeline backend) service image
#
# Why we build our own:
#   MinerU doesn't publish an official, ready-to-use prebuilt image. The Dockerfile in their
#   repo is based on `vllm/vllm-openai:v0.21.0` (CUDA 13.0) -- heavy (tens of GB) and requires
#   a very recent NVIDIA driver, while we only need the pipeline backend (no vLLM, no CUDA 13).
#   So this starts from python:3.13-slim and installs only what the pipeline backend needs.
#
# Why no CUDA base image is needed:
#   The Linux torch wheel on PyPI **bundles** its own CUDA runtime libraries (cudart / cuDNN /
#   cuBLAS, etc.), so the host only needs the NVIDIA driver + nvidia-container-toolkit; the
#   container itself doesn't need the CUDA Toolkit installed.
#
# Build:
#   cd deploy && docker compose build mineru-router
#   # or without compose:
#   docker build -f deploy/mineru.Dockerfile -t paperfacts-mineru:3.4.5 .
#
# Note: this Dockerfile doesn't COPY any file from the repo (the service only runs MinerU's
# own mineru-router). The paired deploy/mineru.Dockerfile.dockerignore empties the build
# context, so large directories like template_files/ and data/ aren't sent to the docker
# daemon for nothing.

ARG PYTHON_IMAGE=python:3.13-slim
FROM ${PYTHON_IMAGE}

# --- Build args ------------------------------------------------------------------

# Extra wheel index for torch. Empty by default = use PyPI's default torch (currently a recent
# CUDA version).
# When the host driver is too old to support it, pass an older CUDA wheel index, e.g.:
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
# Uses --extra-index-url rather than --index-url, so the remaining dependencies still resolve
# from PyPI.
ARG TORCH_INDEX_URL=""

# MinerU version. The main package's runners/mineru_runner.py PEP 723 dependency header pins
# the same version -- the two must stay in sync, or the same PDF could parse differently on a
# dev machine (subprocess) versus the server (HTTP).
ARG MINERU_VERSION=3.4.5

# Where to download model weights during build: huggingface / modelscope.
# On machines with slow access to HF, switch to modelscope (the matching MODELSCOPE_CACHE
# environment variable is also set below).
ARG MINERU_MODEL_DOWNLOAD_SOURCE=huggingface

# --- Environment variables ---------------------------------------------------------

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# The model cache directory is **deliberately** placed at /opt/mineru/models instead of the
# default ~/.cache.
# Why: compose mounts a named volume at /root/.cache for runtime caching. Although docker
# initializes an empty named volume from the image contents, if anyone ever swaps it for a
# bind mount or reuses an old volume, the several GB of weights pre-downloaded into the image
# would be shadowed entirely. Putting them under /opt removes that risk completely.
ENV HF_HOME=/opt/mineru/models/hf \
    MODELSCOPE_CACHE=/opt/mineru/models/modelscope

WORKDIR /app

# --- System dependencies ------------------------------------------------------------
#
#   libgl1          OpenCV(cv2)'s libGL.so.1 -- MinerU's layout/table models rely entirely on
#                   cv2 for image preprocessing
#   libglib2.0-0    another cv2 runtime dependency (libgthread-2.0.so.0)
#   fonts-noto-*    without these, rendering CJK text in intermediate visualization images
#                   shows tofu boxes; fontconfig handles font indexing
#   curl            used by compose's healthcheck to hit /health
#   ca-certificates needed to validate TLS certs when downloading from HuggingFace / PyPI
#                   during the build
#
# libglib2.0-0 was renamed to libglib2.0-0t64 on Debian trixie (13), while bookworm (12) still
# uses the old name. Since python:3.13-slim's base may shift upstream, both names are tried
# here.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        libgl1 \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        curl \
        ca-certificates; \
    apt-get install -y --no-install-recommends libglib2.0-0t64 \
      || apt-get install -y --no-install-recommends libglib2.0-0; \
    rm -rf /var/lib/apt/lists/*

# --- Python dependencies ---------------------------------------------------------
#
# Using uv instead of pip: resolving and downloading is an order of magnitude faster, which
# matters a lot for multi-GB packages like torch.
# `--system` installs into the image's global python (no need for another venv layer inside
# the container).
#
# Deliberately avoiding BuildKit's `RUN --mount=type=cache` and `# syntax=` directives here:
# the former requires a newer Dockerfile frontend, and the latter fetches a frontend image
# over the network, which would hang the build outright on offline/air-gapped machines.
# Trading a bit of rebuild speed for determinism.
RUN pip install --no-cache-dir uv

# `six>=1.16` **must be added explicitly**: mineru 3.4.5's
# mineru/model/utils/pytorchocr/data/imaug/operators.py does `import six` directly, but
# mineru's dependency list doesn't declare it. Running the pipeline backend in a clean
# environment (like this image) raises ModuleNotFoundError: No module named 'six'. This is a
# confirmed upstream bug; drop this once it's fixed upstream.
RUN set -eux; \
    EXTRA_ARGS=""; \
    if [ -n "${TORCH_INDEX_URL}" ]; then \
        echo "Using extra wheel index: ${TORCH_INDEX_URL}"; \
        EXTRA_ARGS="--extra-index-url ${TORCH_INDEX_URL}"; \
    fi; \
    uv pip install --system --break-system-packages --no-cache ${EXTRA_ARGS} \
        "mineru[pipeline]==${MINERU_VERSION}" \
        "six>=1.16"; \
    python -c "import six; print('six ok')"; \
    command -v mineru-router; \
    command -v mineru-models-download

# --- Model pre-download (placed in the last layer) ---------------------------------
#
# Why it's last: this is the largest, slowest layer in the whole image (several GB), and no
# change above this line should invalidate it; conversely, switching only the model source
# means re-running just this layer.
#
# mineru-models-download does two things:
#   1) downloads all weights the pipeline backend needs into $HF_HOME (or $MODELSCOPE_CACHE);
#   2) writes the actual on-disk model root path into the `models-dir` field of
#      /root/mineru.json.
# At runtime, MINERU_MODEL_SOURCE=local uses this JSON file to locate the weights, with no
# network access at all.
#
# The valid values for `-m` have been verified against this mineru 3.4.5 environment:
# pipeline|vlm|all; we only use the pipeline backend (the only one that outputs fine-grained
# layout + bbox), hence pipeline.
RUN set -eux; \
    mineru-models-download -s "${MINERU_MODEL_DOWNLOAD_SOURCE}" -m pipeline; \
    test -f /root/mineru.json; \
    cat /root/mineru.json

# Runtime defaults to local models only. compose also sets this explicitly; this is the
# fallback (also takes effect with a plain docker run).
ENV MINERU_MODEL_SOURCE=local

EXPOSE 8002

# Deliberately **no** ENTRYPOINT / CMD:
# startup args (--host / --port / --local-gpus) are entirely declared in compose, so changing
# GPU allocation doesn't require rebuilding the image.
# When using docker run directly, supply them yourself, e.g.:
#   docker run --rm --gpus '"device=0"' --ipc=host -p 8002:8002 \
#     paperfacts-mineru:3.4.5 mineru-router --host 0.0.0.0 --port 8002 --local-gpus auto
#
# HEALTHCHECK is also not defined here; it's already in compose, to avoid two definitions
# drifting out of sync.
