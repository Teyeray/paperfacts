#!/usr/bin/env bash
# macOS dev machine only: some external tool on this machine periodically recursively marks
# the entire .venv as hidden, and Python 3.13 skips hidden .pth files -- which shows up as
# `import paperfacts` suddenly raising ModuleNotFoundError (even though the .pth file clearly
# exists). uv's own sync/run doesn't do this, and Linux doesn't have this problem.
# Run this script when the symptom appears; no reinstall needed.
set -euo pipefail
# macOS only: chflags does not exist on Linux. See scripts/deploy.sh for the Linux deployment.
if [ "$(uname -s)" != "Darwin" ]; then
    echo "dev_fix_venv.sh is macOS-only (it clears the macOS hidden flag on .venv); on Linux run scripts/deploy.sh --help" >&2
    exit 1
fi
cd "$(dirname "$0")/.."
if [ -d .venv ]; then
    chflags -R nohidden .venv
    echo "Cleared the hidden flag on .venv"
fi
uv run --no-sync python -c "import paperfacts; print('import paperfacts ok')"
