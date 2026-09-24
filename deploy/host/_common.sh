#!/usr/bin/env bash
# Shared path resolution and safety checks for the scripts under deploy/host/.
#
# This file cannot be executed directly; it must be sourced:
#   source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

set -euo pipefail

# --- Paths ---------------------------------------------------------------------

# Directory this file lives in = <repo>/deploy/host
PF_HOST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# <repo>/deploy
PF_DEPLOY_DIR="$(cd "${PF_HOST_DIR}/.." && pwd)"
# Repo root
PF_REPO_ROOT="$(cd "${PF_DEPLOY_DIR}/.." && pwd)"

# Where each parser's isolated venv lives. Override with PAPERFACTS_ENV_ROOT (e.g. to put it on
# a larger disk).
PF_ENV_ROOT="${PAPERFACTS_ENV_ROOT:-${HOME}/.paperfacts/envs}"
PF_MINERU_ENV="${PF_ENV_ROOT}/mineru"
PF_PADDLE_ENV="${PF_ENV_ROOT}/paddle"

# Path to PaddleOCR-VL's pipeline config.
#
# This **must be defined in exactly one place**: setup.sh generates/rewrites it,
# start_paddle_vlm.sh reads the model name from it, and start_paddle_api.sh passes it to
# paddlex --serve. If the three scripts each hardcoded their own default, a user who sets
# PAPERFACTS_PADDLE_PIPELINE_CONFIG would end up with setup.sh writing to the default location
# while the start scripts read from the overridden one, permanently stuck on
# "please run setup.sh first".
PF_PIPELINE_CONFIG="${PAPERFACTS_PADDLE_PIPELINE_CONFIG:-${PF_HOST_DIR}/paddleocr_vl_pipeline.yaml}"

# --- Helpers ---------------------------------------------------------------------

log() {
    printf '[paperfacts] %s\n' "$*" >&2
}

die() {
    printf '[paperfacts][ERROR] %s\n' "$*" >&2
    exit 1
}

# Confirm a GPU id was actually specified before starting a service against it.
# Usage: require_gpus "4,5"
require_gpus() {
    local requested="$1"
    [[ -n "${requested}" ]] || die "CUDA_VISIBLE_DEVICES is empty; a GPU must be specified explicitly."
    log "GPU check passed: CUDA_VISIBLE_DEVICES=${requested}"
}

# Confirm a venv has already been created by setup.sh and contains the given executable.
# Usage: require_venv_bin "${PF_MINERU_ENV}" mineru-router
require_venv_bin() {
    local venv="$1" bin_name="$2"
    [[ -d "${venv}" ]] || die "Environment ${venv} not found; run ${PF_HOST_DIR}/setup.sh first"
    [[ -x "${venv}/bin/${bin_name}" ]] || die "${venv}/bin/${bin_name} does not exist or is not executable; re-run ${PF_HOST_DIR}/setup.sh"
}

# Read a value from the pipeline config yaml along a key path, printed to stdout.
#
# Usage:
#   pipeline_field "${PF_PIPELINE_CONFIG}" SubModules VLRecognition model_name
#   pipeline_field "${PF_PIPELINE_CONFIG}" SubModules VLRecognition genai_config server_url
#
# If any level of the path is missing, or the resolved value is None, this prints an empty
# string and returns normally -- callers can just check with `[[ -z "$x" ]]`, no need to
# inspect an exit code. Only a missing file or invalid yaml returns non-zero.
#
# Uses the python inside the paddle venv (that's the one with PyYAML; the host's system python
# may not have it installed).
pipeline_field() {
    local config_path="$1"
    shift
    [[ -f "${config_path}" ]] || die "Pipeline config ${config_path} not found"
    [[ -x "${PF_PADDLE_ENV}/bin/python" ]] \
        || die "${PF_PADDLE_ENV}/bin/python not found; run ${PF_HOST_DIR}/setup.sh first"

    "${PF_PADDLE_ENV}/bin/python" - "${config_path}" "$@" <<'PY'
import sys

import yaml

config_path, keys = sys.argv[1], sys.argv[2:]
with open(config_path, encoding="utf-8") as fh:
    node = yaml.safe_load(fh)

for key in keys:
    if not isinstance(node, dict):
        node = None
        break
    node = node.get(key)
    if node is None:
        break

print("" if node is None else node)
PY
}

# Port-in-use warning (warns only, doesn't exit: the previous process may just not have fully
# exited yet).
warn_if_port_busy() {
    local port="$1"
    if command -v ss > /dev/null 2>&1; then
        if ss -ltn "sport = :${port}" 2> /dev/null | grep -q LISTEN; then
            log "Warning: port ${port} is already in use; startup will most likely fail. Use 'ss -ltnp | grep :${port}' to see what's holding it."
        fi
    fi
}
