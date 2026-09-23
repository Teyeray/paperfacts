#!/usr/bin/env bash
# PaperFacts -- start PaddleOCR-VL's VLM inference service (Route B: bare-metal host install)
#
# ============================================================================
# Hard GPU constraint: this server has 8 GPUs; PaperFacts may only use ids 4, 5, 6, 7.
# This script uses GPU 6, **shared with start_paddle_api.sh**:
#   this service (vLLM hosting the 0.9B VLM) is capped at 50% GPU memory by
#   deploy/vllm_config.yaml, leaving the rest for the API layer's layout detection model
#   PP-DocLayoutV2 (only needs 1-2GB).
#   GPU 4, 5 → MinerU (start_mineru.sh)
#   GPU 7    → currently unused
# ============================================================================
#
# Usage (recommended to run long-lived in a tmux window; must be started before
# start_paddle_api.sh):
#   bash deploy/host/start_paddle_vlm.sh
#
# This service exposes an **OpenAI-compatible interface** (vLLM's standard server), not
# PaddleOCR's own business interface -- that's in start_paddle_api.sh. So:
#   - it only listens on 8118, for the local API layer only; no need to expose it externally
#     (don't open it in the firewall)
#   - health check is GET /health, model listing is GET /v1/models
#
# Tunable environment variables:
#   PAPERFACTS_VLM_HOST    listen address, default 127.0.0.1 (local API layer only; don't
#                          change to 0.0.0.0)
#   PAPERFACTS_VLM_PORT    listen port, default 8118
#   PAPERFACTS_VLM_GPUS    GPU to use, default 6 (must be within 4-7)
#   PAPERFACTS_VLM_MODEL   model name, defaults to reading from the pipeline config (see
#                          "model name must match" below)
#   PAPERFACTS_VLM_CONFIG  vLLM args file, default deploy/vllm_config.yaml

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

VLM_HOST="${PAPERFACTS_VLM_HOST:-127.0.0.1}"
VLM_PORT="${PAPERFACTS_VLM_PORT:-8118}"
VLM_GPUS="${PAPERFACTS_VLM_GPUS:-6}"
VLM_CONFIG="${PAPERFACTS_VLM_CONFIG:-${PF_DEPLOY_DIR}/vllm_config.yaml}"

require_allowed_gpus "${VLM_GPUS}"
warn_if_port_busy "${VLM_PORT}"

[[ -d "${PF_PADDLE_ENV}" ]] || die "Environment ${PF_PADDLE_ENV} not found; run ${PF_HOST_DIR}/setup.sh first"
[[ -f "${VLM_CONFIG}" ]] || die "vLLM args file ${VLM_CONFIG} not found"

# --- Model name must match the pipeline config -------------------------------
#
# How it works (verified against the paddlex 3.7.2 source):
#   - when this service starts, vLLM's served-model-name defaults to the value of
#     --model_name;
#   - on the API layer side, GenAIClientPredictor sends
#     SubModules.VLRecognition.model_name from the pipeline config as the OpenAI request's
#     `model` field.
#   If the two don't match, vLLM returns 404 model not found for every request, with a
#   less-than-obvious error message.
# So by default this reads the model name directly from the pipeline config instead of
# hardcoding a constant.
VLM_MODEL="${PAPERFACTS_VLM_MODEL:-}"
if [[ -z "${VLM_MODEL}" && -f "${PF_PIPELINE_CONFIG}" ]]; then
    VLM_MODEL="$(pipeline_field "${PF_PIPELINE_CONFIG}" SubModules VLRecognition model_name)"
    log "Read model name from ${PF_PIPELINE_CONFIG}: ${VLM_MODEL}"
fi
# Fallback value for when the pipeline config hasn't been generated yet (matches setup.sh's
# default PaddleOCR-VL-1.6 pipeline).
VLM_MODEL="${VLM_MODEL:-PaddleOCR-VL-1.6-0.9B}"

# --- Choose the genai server entrypoint ---------------------------------------
#
# The official image uses `paddleocr genai_server ...`, but that subcommand only exists in
# newer versions; we've verified that paddleocr 3.7.0's CLI does **not** have a genai_server
# subcommand. The equivalent entrypoint is `paddlex_genai_server`, installed by paddlex (both
# take identical arguments, since both come from get_arg_parser in
# paddlex/inference/genai/server.py).
# We probe for both here and use whichever is available, so the script doesn't break after a
# paddleocr upgrade.
if "${PF_PADDLE_ENV}/bin/paddleocr" genai_server --help > /dev/null 2>&1; then
    GENAI_CMD=("${PF_PADDLE_ENV}/bin/paddleocr" genai_server)
elif [[ -x "${PF_PADDLE_ENV}/bin/paddlex_genai_server" ]]; then
    GENAI_CMD=("${PF_PADDLE_ENV}/bin/paddlex_genai_server")
else
    die "Neither 'paddleocr genai_server' nor paddlex_genai_server is available; re-run ${PF_HOST_DIR}/setup.sh (it runs paddleocr install_genai_server_deps vllm)"
fi

export CUDA_VISIBLE_DEVICES="${VLM_GPUS}"

log "Starting PaddleOCR-VL's vLLM service: ${VLM_HOST}:${VLM_PORT}, GPU ${VLM_GPUS}"
log "Model: ${VLM_MODEL}    vLLM args: ${VLM_CONFIG}"
log "Health check: curl http://127.0.0.1:${VLM_PORT}/health"
log "(On first startup, loading weights and compiling the CUDA graph takes a few minutes; unreachable for up to 5 minutes is normal)"

exec "${GENAI_CMD[@]}" \
    --model_name "${VLM_MODEL}" \
    --host "${VLM_HOST}" \
    --port "${VLM_PORT}" \
    --backend vllm \
    --backend_config "${VLM_CONFIG}"
