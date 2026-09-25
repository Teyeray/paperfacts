"""The browser checks in ``web_races.py``, reachable from pytest: ``pytest -m e2e``.

Deselected in a plain run (``conftest.py`` deselects the ``e2e`` marker unless ``-m`` names it), and skipped
when Playwright is not installed, so the default suite stays free of browsers. With Playwright::

    uv run --with playwright python -m playwright install chromium   # once
    PYTHONPATH=src uv run --with playwright pytest -m e2e

The script runs as a subprocess: it owns its event loop, its server thread and its command line.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

SCRIPT = Path(__file__).with_name("web_races.py")
SRC = Path(__file__).resolve().parents[2] / "src"


def test_the_frontend_survives_its_races_and_keeps_its_layout():
    pytest.importorskip("playwright")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(SRC), os.environ.get("PYTHONPATH"))))}

    result = subprocess.run(
        [sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=900, check=False
    )

    assert result.returncode == 0, result.stdout + result.stderr
