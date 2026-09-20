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
cd "$(dirname "$0")/.."

MLX_PORT="${PAPERFACTS_MLX_PORT:-8111}"
MLX_LOG="${PAPERFACTS_MLX_LOG:-data/mlx_vlm_server.log}"

"$(dirname "$0")/dev_fix_venv.sh" >/dev/null

if lsof -nP -iTCP:"$MLX_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "mlx-vlm server already listening on :$MLX_PORT"
else
    mkdir -p "$(dirname "$MLX_LOG")"
    echo "starting mlx-vlm server on :$MLX_PORT (log: $MLX_LOG)"
    nohup uvx --python 3.13 --from "mlx-vlm>=0.3.11" mlx_vlm.server --port "$MLX_PORT" >"$MLX_LOG" 2>&1 &
    for _ in $(seq 1 60); do
        lsof -nP -iTCP:"$MLX_PORT" -sTCP:LISTEN >/dev/null 2>&1 && break
        sleep 1
    done
fi

# .env is loaded by the app itself and wins over these; they only fill in what it leaves unset.
export PAPERFACTS_PADDLE_VL_BACKEND="${PAPERFACTS_PADDLE_VL_BACKEND:-mlx-vlm-server}"
export PAPERFACTS_PADDLE_VL_SERVER_URL="${PAPERFACTS_PADDLE_VL_SERVER_URL:-http://localhost:$MLX_PORT/}"
export PAPERFACTS_PADDLE_VL_MODEL_NAME="${PAPERFACTS_PADDLE_VL_MODEL_NAME:-PaddlePaddle/PaddleOCR-VL-1.6}"

exec uv run paperfacts serve "$@"
