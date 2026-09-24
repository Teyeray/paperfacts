#!/usr/bin/env bash
# PaperFacts -- one-time setup script for Route B (bare-metal host install)
#
# When to use this: no docker access (or nvidia-container-toolkit simply isn't installed).
# If docker is available, use Route A (deploy/compose.yaml) instead -- it's far more
# reproducible.
#
# What this script does:
#   1. Uses uv to create two **mutually isolated** venvs:
#        ~/.paperfacts/envs/mineru   installs mineru[pipeline]==3.4.5
#        ~/.paperfacts/envs/paddle   installs paddlepaddle-gpu + paddleocr + vLLM
#      Why isolation is required: MinerU depends on torch, PaddleOCR depends on paddlepaddle,
#      and their CUDA runtime libraries (cudnn / nccl) frequently conflict in version --
#      installing them together eventually breaks. The main paperfacts package itself is pure
#      python and doesn't need to coexist with either environment.
#   2. Pre-downloads MinerU's pipeline model weights (so runtime can be fully offline
#      afterwards).
#   3. Generates and rewrites PaddleOCR-VL's pipeline config, pointing its VLM stage at the
#      local vLLM service.
#
# This script only installs environments and runs no inference, so it doesn't touch the GPU;
# actual GPU assignment happens in the three start_*.sh scripts (default GPU 0, overridable
# per service).
#
# Usage:
#   bash deploy/host/setup.sh
#
# Safe to re-run: an already-built venv is reused, an already-downloaded model isn't
# re-downloaded, and an existing pipeline config is not overwritten by default (set
# PAPERFACTS_FORCE_PIPELINE_CONFIG=1 to force regeneration).
#
# Tunable environment variables:
#   PAPERFACTS_ENV_ROOT            venv root directory, default ~/.paperfacts/envs
#   PAPERFACTS_PYTHON              python version, default 3.13 (MinerU requires <3.14; paddle
#                                   currently only ships cp313 wheels)
#   PAPERFACTS_PADDLE_INDEX        wheel index for paddlepaddle-gpu, default the CUDA 12.6 one
#   PAPERFACTS_PADDLE_PIPELINE     pipeline name, default PaddleOCR-VL-1.6
#   PAPERFACTS_VLM_URL             VLM service address written into the pipeline config,
#                                  default http://127.0.0.1:8118/v1
#   PAPERFACTS_MINERU_MODEL_SOURCE model download source, huggingface (default) or modelscope
#   PAPERFACTS_PADDLE_PIPELINE_CONFIG  where the pipeline config is written to disk,
#                                      default deploy/host/paddleocr_vl_pipeline.yaml.
#                                      Defined in _common.sh (PF_PIPELINE_CONFIG); this script
#                                      and the two start_paddle_*.sh scripts share the same
#                                      value.
#   PAPERFACTS_FORCE_PIPELINE_CONFIG=1  force pipeline config regeneration

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

PYTHON_VERSION="${PAPERFACTS_PYTHON:-3.13}"
PADDLE_INDEX="${PAPERFACTS_PADDLE_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}"
PADDLE_PIPELINE="${PAPERFACTS_PADDLE_PIPELINE:-PaddleOCR-VL-1.6}"
VLM_URL="${PAPERFACTS_VLM_URL:-http://127.0.0.1:8118/v1}"
MINERU_MODEL_SOURCE_ARG="${PAPERFACTS_MINERU_MODEL_SOURCE:-huggingface}"

# ---------------------------------------------------------------------------
# 0. Preflight checks
# ---------------------------------------------------------------------------

command -v uv > /dev/null 2>&1 \
    || die "uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh, then restart your shell."

command -v nvidia-smi > /dev/null 2>&1 \
    || log "Warning: nvidia-smi not found. The environment can still be installed, but will fall back to CPU at runtime (too slow to be usable)."

log "Repo root : ${PF_REPO_ROOT}"
log "venv root : ${PF_ENV_ROOT}"
mkdir -p "${PF_ENV_ROOT}"

# ---------------------------------------------------------------------------
# 1. MinerU environment
# ---------------------------------------------------------------------------

log "==== [1/4] Creating / updating the MinerU environment: ${PF_MINERU_ENV} ===="

# --allow-existing is **required**: uv 0.11 errors out immediately if the target is already a
# venv (verified exit code 2), and combined with this script's set -e, a second run of
# setup.sh would abort right here.
# With this flag, an existing environment is reused in place and the following uv pip install
# only applies incremental updates -- this is what makes the script safe to re-run (torch/vLLM
# are several GB each; they shouldn't be reinstalled from scratch every time).
# To really rebuild from scratch, manually rm -rf "${PF_MINERU_ENV}" first.
uv venv --python "${PYTHON_VERSION}" --allow-existing "${PF_MINERU_ENV}"

# `six>=1.16` must be installed explicitly: mineru 3.4.5's
# mineru/model/utils/pytorchocr/data/imaug/operators.py does `import six` directly, but six
# isn't declared in its dependency list. Running the pipeline backend in a clean venv raises
# ModuleNotFoundError: No module named 'six'. This is an upstream bug; drop this line once
# it's fixed.
#
# Note: VIRTUAL_ENV is how uv is told which environment to install into here; no need for
# `source activate`.
VIRTUAL_ENV="${PF_MINERU_ENV}" uv pip install \
    "mineru[pipeline]==3.4.5" \
    "six>=1.16"

log "Verifying six is importable (guards against the upstream bug reappearing)"
"${PF_MINERU_ENV}/bin/python" -c "import six; print('six', six.__version__)"

log "==== [2/4] Pre-downloading MinerU pipeline models (several GB; slow the first time) ===="
# `-m` accepts pipeline|vlm|all (verified against mineru 3.4.5). We only use the pipeline
# backend -- it's the only one that outputs fine-grained layout + bbox data, which is a
# prerequisite for PaperFacts' provenance tracking.
# After downloading, the model root path is written into ~/mineru.json; at runtime,
# MINERU_MODEL_SOURCE=local uses it to locate the weights. Already-downloaded models are
# skipped (huggingface_hub does its own cache validation), so re-running this is safe.
"${PF_MINERU_ENV}/bin/mineru-models-download" -s "${MINERU_MODEL_SOURCE_ARG}" -m pipeline

[[ -f "${HOME}/mineru.json" ]] \
    || die "${HOME}/mineru.json was not created after the model download; MINERU_MODEL_SOURCE=local won't work. Check the output of the previous step."
log "MinerU model config: ${HOME}/mineru.json"

# ---------------------------------------------------------------------------
# 2. PaddleOCR environment
# ---------------------------------------------------------------------------

log "==== [3/4] Creating / updating the PaddleOCR environment: ${PF_PADDLE_ENV} ===="

uv venv --python "${PYTHON_VERSION}" --allow-existing "${PF_PADDLE_ENV}"

# Order matters here -- these three steps can't be merged or reordered:
#
#   (a) paddlepaddle-gpu is only available from Baidu's own index; PyPI has no matching CUDA
#       12.6 wheel. `--index` **adds** an index (with higher priority than PyPI) rather than
#       replacing it, so the remaining packages can still resolve from PyPI.
log "  (a) Installing paddlepaddle-gpu==3.2.1 (index: ${PADDLE_INDEX})"
VIRTUAL_ENV="${PF_PADDLE_ENV}" uv pip install \
    --index "${PADDLE_INDEX}" \
    "paddlepaddle-gpu==3.2.1"

#   (b) paddleocr must be installed after paddlepaddle, otherwise it pulls a CPU-only
#       paddlepaddle from PyPI and clobbers the previous step.
#       Version pinned to <3.8: the pipeline config structure and genai client interface in
#       3.7.x are the ones we've verified against.
log "  (b) Installing paddleocr[doc-parser]>=3.7,<3.8"
VIRTUAL_ENV="${PF_PADDLE_ENV}" uv pip install "paddleocr[doc-parser]>=3.7,<3.8"

#   (c) vLLM and its companion plugin are installed via paddleocr's own command, because it
#       needs to pick a vLLM version compatible with PaddleOCR-VL (the model requires
#       vllm >= 0.11.1), which is easy to get wrong when installed by hand.
log "  (c) Installing vLLM genai server dependencies"
"${PF_PADDLE_ENV}/bin/paddleocr" install_genai_server_deps vllm

# ---------------------------------------------------------------------------
# 3. Generate and patch PaddleOCR-VL's pipeline config
# ---------------------------------------------------------------------------

log "==== [4/4] Generating pipeline config: ${PF_PIPELINE_CONFIG} ===="

if [[ -f "${PF_PIPELINE_CONFIG}" && "${PAPERFACTS_FORCE_PIPELINE_CONFIG:-0}" != "1" ]]; then
    log "Already exists, skipping generation (to force regeneration: PAPERFACTS_FORCE_PIPELINE_CONFIG=1 bash $0)"
else
    # paddlex copies its built-in <pipeline>.yaml into the directory given by --save_path.
    # This step needs no model weights -- it's a plain file copy.
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "${tmp_dir}"' EXIT

    "${PF_PADDLE_ENV}/bin/paddlex" \
        --get_pipeline_config "${PADDLE_PIPELINE}" \
        --save_path "${tmp_dir}"

    generated="$(find "${tmp_dir}" -maxdepth 2 -name '*.yaml' | head -n 1)"
    [[ -n "${generated}" ]] || die "paddlex --get_pipeline_config produced no yaml; check whether the pipeline name '${PADDLE_PIPELINE}' is correct."
    log "paddlex generated: ${generated}"

    cp "${generated}" "${PF_PIPELINE_CONFIG}"
fi

# Whether newly generated or already existing, point the VLM backend fields at the local vLLM
# service and validate the result.
#
# Fields being changed (names verified against the paddlex 3.7.2 source; see GenAIConfig /
# paddleocr doc_parser --help):
#   SubModules.VLRecognition.genai_config.backend     native → vllm-server
#   SubModules.VLRecognition.genai_config.server_url  vLLM's OpenAI-compatible address,
#                                                      **must include the /v1 suffix**
# The corresponding CLI flags are --vl_rec_backend / --vl_rec_server_url.
# If a future paddleocr version renames these fields, treat the actually generated yaml as the
# source of truth (see README "Troubleshooting").
#
# Uses a text-level substitution rather than rewriting via yaml.dump: the built-in config has
# many per-category threshold comments (like `0: 0.5  # abstract`), and a full dump would drop
# all of them, making future manual review impossible.
"${PF_PADDLE_ENV}/bin/python" - "${PF_PIPELINE_CONFIG}" "${VLM_URL}" <<'PY'
import re
import sys

import yaml

config_path, server_url = sys.argv[1], sys.argv[2]

with open(config_path, encoding="utf-8") as fh:
    text = fh.read()

new_block = (
    "    genai_config:\n"
    "      backend: vllm-server\n"
    f"      server_url: {server_url}\n"
)
# Matches the "    genai_config:" line plus every subsequent line indented deeper than it.
pattern = re.compile(r"^ {4}genai_config:\n(?: {6}\S.*\n)+", re.MULTILINE)
patched, count = pattern.subn(new_block, text)
if count != 1:
    sys.exit(
        f"[ERROR] Found {count} genai_config block(s) in {config_path} (expected 1)."
        " The upstream config structure may have changed; manually check the vl_rec_backend / vl_rec_server_url fields against the README."
    )

with open(config_path, "w", encoding="utf-8") as fh:
    fh.write(patched)

# Read back and validate: confirms the yaml is still well-formed, and prints model_name so it
# can be cross-checked against the start script.
config = yaml.safe_load(patched)
vl = config["SubModules"]["VLRecognition"]
assert vl["genai_config"]["backend"] == "vllm-server", vl
assert vl["genai_config"]["server_url"] == server_url, vl
print(f"[paperfacts] Pipeline config updated: backend=vllm-server server_url={server_url}")
print(f"[paperfacts] VLM model name declared in the pipeline: {vl['model_name']}")
PY

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

cat >&2 <<EOF

[paperfacts] ============================ Setup complete ============================

  MinerU env     : ${PF_MINERU_ENV}
  Paddle env     : ${PF_PADDLE_ENV}
  pipeline config: ${PF_PIPELINE_CONFIG}

  Next, start the three services in order (recommended: one tmux window each):

    bash ${PF_HOST_DIR}/start_mineru.sh        # GPU 0  → :8002
    bash ${PF_HOST_DIR}/start_paddle_vlm.sh    # GPU 0  → :8118 (internal)
    bash ${PF_HOST_DIR}/start_paddle_api.sh    # GPU 0  → :8080

  Note: start_paddle_api.sh depends on start_paddle_vlm.sh being up first -- before starting,
  it curls the /health endpoint for the server_url in the pipeline config (default
  http://127.0.0.1:8118/health), and exits with an error right away if that's unreachable.

  On the orchestrator side:
    export PAPERFACTS_MINERU_URL=http://localhost:8002
    export PAPERFACTS_PADDLE_URL=http://localhost:8080

[paperfacts] ==================================================================

EOF
