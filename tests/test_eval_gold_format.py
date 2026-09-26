"""The gold set spells the paper-level record ``"paper"`` since round 2 (spec §1.6).

``eval/score.py`` still reads a legacy ``"target"`` gold file, with a note, but the committed gold set and the
baselines scored from it use the current id, so a baseline and a fresh score of it name the same cells.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parent.parent / "eval"
GOLD_FILES = sorted(path for path in (EVAL / "gold").glob("*.json") if not path.name.startswith("."))


def test_the_gold_set_is_present():
    assert len(GOLD_FILES) == 11


@pytest.mark.parametrize("path", GOLD_FILES, ids=lambda path: path.stem)
def test_no_gold_file_keeps_the_legacy_paper_key(path: Path):
    gold = json.loads(path.read_text(encoding="utf-8"))

    assert "target" not in gold
    assert isinstance(gold.get("paper"), dict)


@pytest.mark.parametrize("name", ["gold-B0.json", "gold-B2.json"])
def test_the_baselines_label_paper_level_cells_paper(name: str):
    cells = json.loads((EVAL / "baselines" / name).read_text(encoding="utf-8"))

    assert "target" not in {cell["sample"] for cell in cells} | {cell["row"] for cell in cells}
    assert sum(cell["sample"] == "paper" for cell in cells) == 7
