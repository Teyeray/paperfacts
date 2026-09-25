#!/usr/bin/env bash
# macOS dev machine: bring up everything the web interface needs, in order, with one command.
#
# 1. Clear the hidden flag some external tool keeps setting on .venv (see dev_fix_venv.sh).
# 2. Start the mlx-vlm server for PaddleOCR-VL's vision stage unless something already listens on its
#    port. Without it the parse runs on the CPU and takes hours per paper; with it, a minute or two.
# 3. Point PaddleOCR-VL at that server, unless .env already does, and start `paperfacts serve`.
#
# Any arguments are passed to `paperfacts serve` (e.g. --host 0.0.0.0 --port 8765).
set -euo pipefail
# macOS only: this starts an Apple-silicon mlx-vlm server and ls's for a macOS lsof. The Linux
# deployment (systemd --user + a vLLM PaddleOCR-VL lane) is scripts/deploy.sh.
if [ "$(uname -s)" != "Darwin" ]; then
    echo "dev_up.sh is macOS-only; on Linux run scripts/deploy.sh --help" >&2
    exit 1
fi
cd "$(dirname "$0")/.."

# .env first, so a value set there wins over the defaults below (the app itself never overrides an
# existing environment variable, so anything exported here would otherwise beat .env).
if [ -f .env ]; then set -a; . ./.env; set +a; fi

MLX_PORT="${PAPERFACTS_MLX_PORT:-8111}"
MLX_LOG="${PAPERFACTS_MLX_LOG:-data/mlx_vlm_server.log}"
MLX_PID_FILE="${PAPERFACTS_MLX_PID_FILE:-data/mlx_vlm_server.pid}"

"$(dirname "$0")/dev_fix_venv.sh" >/dev/null

# macOS lsof syntax; the Linux deployment under deploy/ has its own scripts.
if lsof -nP -iTCP:"$MLX_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "mlx-vlm server already listening on :$MLX_PORT"
else
    mkdir -p "$(dirname "$MLX_LOG")"
    echo "starting mlx-vlm server on :$MLX_PORT (log: $MLX_LOG)"
    nohup uvx --python 3.13 --from "mlx-vlm>=0.3.11" mlx_vlm.server --port "$MLX_PORT" >"$MLX_LOG" 2>&1 &
    echo $! >"$MLX_PID_FILE"
    for _ in $(seq 1 60); do
        lsof -nP -iTCP:"$MLX_PORT" -sTCP:LISTEN >/dev/null 2>&1 && break
        sleep 1
    done
    if ! lsof -nP -iTCP:"$MLX_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        # Starting serve anyway would parse on the CPU for hours without saying so.
        echo "mlx-vlm server did not come up on :$MLX_PORT within 60 s; see $MLX_LOG" >&2
        exit 1
    fi
    echo "mlx-vlm server pid $(cat "$MLX_PID_FILE"); stop it with scripts/dev_down.sh"
fi

# Defaults only for what .env left unset.
export PAPERFACTS_PADDLE_VL_BACKEND="${PAPERFACTS_PADDLE_VL_BACKEND:-mlx-vlm-server}"
export PAPERFACTS_PADDLE_VL_SERVER_URL="${PAPERFACTS_PADDLE_VL_SERVER_URL:-http://localhost:$MLX_PORT/}"
export PAPERFACTS_PADDLE_VL_MODEL_NAME="${PAPERFACTS_PADDLE_VL_MODEL_NAME:-PaddlePaddle/PaddleOCR-VL-1.6}"

exec uv run paperfacts serve "$@"
