#!/usr/bin/env bash
# PaperFacts -- start the visual validation model (Route B: bare-metal host install)
#
# ============================================================================
# Hard GPU constraint: this server has 8 GPUs; PaperFacts may only use ids 4, 5, 6, 7.
# This script uses GPU 7, the whole card, nothing else runs there:
#   GPU 4, 5 -> MinerU (start_mineru.sh)
#   GPU 6    -> PaddleOCR-VL (start_paddle_vlm.sh + start_paddle_api.sh)
#   GPU 7    -> this service
# ============================================================================
#
# Serves an open-weight Qwen3-VL behind vLLM's OpenAI-compatible server, on port 8090. The
# orchestrator points at it with:
#   export PAPERFACTS_VLM_BASE_URL=http://<server>:8090/v1
#   export PAPERFACTS_VLM_MODEL=<the model name below>
# The name is part of validation_key, so it must be exactly what is served here.
#
# vLLM is installed into its own venv by this script's first run (uv, PyPI); it must not share an
# environment with either parser, whose dependency trees conflict with each other already.
#
# Tunable environment variables:
#   PAPERFACTS_QWEN_VLM_HOST   listen address, default 0.0.0.0 (the orchestrator may be elsewhere)
#   PAPERFACTS_QWEN_VLM_PORT   listen port, default 8090
#   PAPERFACTS_QWEN_VLM_GPUS   GPU to use, default 7 (must be within 4-7)
#   PAPERFACTS_QWEN_VLM_MODEL  Hugging Face repository, default Qwen/Qwen3-VL-8B-Instruct
#   PAPERFACTS_QWEN_VLM_VLLM   vLLM version spec, default "vllm>=0.11"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

QVLM_HOST="${PAPERFACTS_QWEN_VLM_HOST:-0.0.0.0}"
QVLM_PORT="${PAPERFACTS_QWEN_VLM_PORT:-8090}"
QVLM_GPUS="${PAPERFACTS_QWEN_VLM_GPUS:-7}"
QVLM_MODEL="${PAPERFACTS_QWEN_VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
QVLM_VLLM="${PAPERFACTS_QWEN_VLM_VLLM:-vllm>=0.11}"
QVLM_ENV="${PF_ENV_ROOT}/qwen-vlm"

require_allowed_gpus "${QVLM_GPUS}"
warn_if_port_busy "${QVLM_PORT}"

command -v uv > /dev/null 2>&1 || die "uv is required to create the vLLM environment (https://docs.astral.sh/uv/)"

if [[ ! -x "${QVLM_ENV}/bin/vllm" ]]; then
    log "Creating the vLLM environment at ${QVLM_ENV} (first run only; downloads several GB)"
    uv venv --python 3.12 "${QVLM_ENV}"
    VIRTUAL_ENV="${QVLM_ENV}" uv pip install "${QVLM_VLLM}"
fi

export CUDA_VISIBLE_DEVICES="${QVLM_GPUS}"

log "Starting the validation model: ${QVLM_HOST}:${QVLM_PORT}, GPU ${QVLM_GPUS}"
log "Model: ${QVLM_MODEL}"
log "Health check: curl http://127.0.0.1:${QVLM_PORT}/health"
log "(First start downloads the weights and compiles the CUDA graph; ten minutes is not unusual)"

exec "${QVLM_ENV}/bin/vllm" serve "${QVLM_MODEL}" \
    --served-model-name "${QVLM_MODEL}" \
    --host "${QVLM_HOST}" \
    --port "${QVLM_PORT}" \
    --max-model-len 16384 \
    --max-num-seqs 32 \
    --limit-mm-per-prompt image=1 \
    --gpu-memory-utilization 0.90
