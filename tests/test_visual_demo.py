"""The documented one-command demo uses generated PDF and fixed local replies only."""

import json
import subprocess
import sys
from pathlib import Path


def test_demo_runs_offline_and_keeps_reviewable_outputs(tmp_path):
    script = Path(__file__).resolve().parents[1] / "eval/visual_demo.py"
    root = tmp_path / "demo"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(root)], capture_output=True, text=True, timeout=45
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((root / "summary.json").read_bytes())
    assert summary["synthetic_only"] is True
    assert summary["repeat_requests"] == 0
    assert summary["adopted"] == 2
    assert summary["refused"] >= 1
    assert summary["stale_adopted"] == 0
    assert (root / "run/export-current/dataset.xlsx").is_file()
    again = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(root)], capture_output=True, text=True, timeout=10
    )
    assert again.returncode != 0 and "already exists" in again.stderr
