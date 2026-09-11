#!/usr/bin/env bash
# PaperFacts -- start PaddleOCR-VL's pipeline / API layer (Route B: bare-metal host install)
#
# ============================================================================
# Hard GPU constraint: this server has 8 GPUs; PaperFacts may only use ids 4, 5, 6, 7.
# This script uses GPU 6, shared with start_paddle_vlm.sh:
#   this layer only runs the layout detection model PP-DocLayoutV2 (1-2GB); all vision
#   recognition is forwarded to the vLLM service.
#   GPU 4, 5 → MinerU (start_mineru.sh)
#   GPU 7    → currently unused
# ============================================================================
#
# Usage (recommended to run long-lived in a tmux window; **start_paddle_vlm.sh must be started
# first**):
#   bash deploy/host/start_paddle_api.sh
#
# External interface (this is the one the orchestrator calls):
#   POST http://<host>:8080/layout-parsing
#   GET  http://<host>:8080/health
#
# Tunable environment variables:
#   PAPERFACTS_PADDLE_HOST             listen address, default 0.0.0.0
#   PAPERFACTS_PADDLE_PORT             listen port, default 8080
#   PAPERFACTS_PADDLE_GPUS             GPUs to use, default 6 (must be within 4-7)
#   PAPERFACTS_PADDLE_PIPELINE_CONFIG  pipeline config, default
#                                      deploy/host/paddleocr_vl_pipeline.yaml
#                                      (defined as PF_PIPELINE_CONFIG in _common.sh; this is
#                                       the same path setup.sh generates, so don't change only
#                                       one side)
#   PAPERFACTS_SKIP_VLM_CHECK=1        skip the VLM connectivity check before startup

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

PADDLE_HOST="${PAPERFACTS_PADDLE_HOST:-0.0.0.0}"
PADDLE_PORT="${PAPERFACTS_PADDLE_PORT:-8080}"
PADDLE_GPUS="${PAPERFACTS_PADDLE_GPUS:-6}"

# Check the GPU whitelist first: this is a hard constraint and should be reported before
# checking whether the environment is even installed.
require_allowed_gpus "${PADDLE_GPUS}"
require_venv_bin "${PF_PADDLE_ENV}" paddlex
warn_if_port_busy "${PADDLE_PORT}"

[[ -f "${PF_PIPELINE_CONFIG}" ]] \
    || die "Pipeline config ${PF_PIPELINE_CONFIG} not found; run ${PF_HOST_DIR}/setup.sh first (it generates and patches this file)"

# --- Check the VLM service is reachable before starting -----------------------
#
# Why check: during pipeline initialization, GenAIClient immediately connects to server_url;
# if the VLM isn't up yet, paddlex throws a long, hard-to-read chain of openai connection
# exceptions during loading. Probing with curl first lets us give a plain-language message
# instead.
if [[ "${PAPERFACTS_SKIP_VLM_CHECK:-0}" != "1" ]]; then
    vlm_url="$(pipeline_field "${PF_PIPELINE_CONFIG}" SubModules VLRecognition genai_config server_url)"
    if [[ -z "${vlm_url}" ]]; then
        log "Warning: the pipeline config has no genai_config.server_url; the VLM will run in-process (very slow)."
        log "      setup.sh should normally have rewritten this to backend=vllm-server. Please check that file."
    else
        # server_url looks like http://127.0.0.1:8118/v1; the health check endpoint is /health
        # on the same origin.
        health_url="${vlm_url%/v1}/health"
        log "Checking the VLM service: ${health_url}"
        if ! curl -fsS --max-time 5 "${health_url}" > /dev/null 2>&1; then
            die "Can't reach the VLM service at ${health_url}. Run ${PF_HOST_DIR}/start_paddle_vlm.sh in another window first and wait for it to become ready (about 5 minutes on first start). To skip this check: PAPERFACTS_SKIP_VLM_CHECK=1"
        fi
        log "VLM service is ready"
    fi
fi

export CUDA_VISIBLE_DEVICES="${PADDLE_GPUS}"

log "Starting PaddleOCR-VL API: ${PADDLE_HOST}:${PADDLE_PORT}, GPU ${PADDLE_GPUS}"
log "Pipeline config: ${PF_PIPELINE_CONFIG}"
log "Health check: curl http://127.0.0.1:${PADDLE_PORT}/health"

exec "${PF_PADDLE_ENV}/bin/paddlex" \
    --serve \
    --pipeline "${PF_PIPELINE_CONFIG}" \
    --host "${PADDLE_HOST}" \
    --port "${PADDLE_PORT}"
