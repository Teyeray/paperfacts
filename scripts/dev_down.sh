#!/usr/bin/env bash
# Stop the mlx-vlm server that scripts/dev_up.sh started (its pid is recorded in data/). The web server
# is stopped with Ctrl-C in its own terminal; this only handles the background process.
set -euo pipefail
cd "$(dirname "$0")/.."
MLX_PID_FILE="${PAPERFACTS_MLX_PID_FILE:-data/mlx_vlm_server.pid}"
if [ ! -f "$MLX_PID_FILE" ]; then
    echo "no pid file at $MLX_PID_FILE; nothing started by dev_up.sh is running"
    exit 0
fi
pid="$(cat "$MLX_PID_FILE")"
if kill "$pid" 2>/dev/null; then
    echo "stopped mlx-vlm server pid $pid"
else
    echo "mlx-vlm server pid $pid was not running"
fi
rm -f "$MLX_PID_FILE"
