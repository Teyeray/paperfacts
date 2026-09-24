#!/usr/bin/env bash
# PaperFacts -- start the MinerU router (Route B: bare-metal host install)
#
# Usage (recommended to run long-lived in a tmux window):
#   bash deploy/host/start_mineru.sh
#
# How it works:
#   mineru-router doesn't do inference itself; it spins up one mineru-api worker per
#   **visible** GPU (actually running `python -m mineru.cli.fast_api`, using this venv's
#   interpreter, independent of PATH), then round-robins POST /file_parse requests across
#   those workers.
#   `--local-gpus auto` probes in this order: read CUDA_VISIBLE_DEVICES first, and only fall
#   back to torch's own detection if that's unset.
#   So once CUDA_VISIBLE_DEVICES is set below, it starts one worker per listed GPU (default
#   is a single GPU 0).
#   To pin them explicitly you can also use `--local-gpus <ids>` (in host mode these are
#   physical ids).
#
# External interface:
#   POST http://<host>:8002/file_parse   (multipart PDF upload)
#   GET  http://<host>:8002/health
#
# Tunable environment variables:
#   PAPERFACTS_MINERU_HOST   listen address, default 0.0.0.0
#   PAPERFACTS_MINERU_PORT   listen port, default 8002
#   PAPERFACTS_MINERU_GPUS   GPUs to use, default 0
#   PAPERFACTS_ENV_ROOT      venv root directory, default ~/.paperfacts/envs

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MINERU_HOST="${PAPERFACTS_MINERU_HOST:-0.0.0.0}"
MINERU_PORT="${PAPERFACTS_MINERU_PORT:-8002}"
MINERU_GPUS="${PAPERFACTS_MINERU_GPUS:-0}"

# Check a GPU was actually specified before checking whether the environment is installed.
require_gpus "${MINERU_GPUS}"
require_venv_bin "${PF_MINERU_ENV}" mineru-router
warn_if_port_busy "${MINERU_PORT}"

# local = only read weights already downloaded locally, never touch the network.
# The weights path is recorded in the models-dir field of ~/mineru.json, written by setup.sh's
# download step. If this machine's $HOME is shared and you want the config elsewhere, set
# MINERU_TOOLS_CONFIG_JSON to an absolute path.
export MINERU_MODEL_SOURCE=local
export CUDA_VISIBLE_DEVICES="${MINERU_GPUS}"

[[ -f "${MINERU_TOOLS_CONFIG_JSON:-${HOME}/mineru.json}" ]] \
    || die "mineru.json not found (MINERU_MODEL_SOURCE=local needs it to locate the models); run ${PF_HOST_DIR}/setup.sh first"

log "Starting mineru-router: ${MINERU_HOST}:${MINERU_PORT}, GPU ${MINERU_GPUS}"
log "Health check: curl http://127.0.0.1:${MINERU_PORT}/health"
log "(On first startup each worker has to load the layout/OCR/table models; /health being unreachable for a few minutes is normal)"

exec "${PF_MINERU_ENV}/bin/mineru-router" \
    --host "${MINERU_HOST}" \
    --port "${MINERU_PORT}" \
    --local-gpus auto
