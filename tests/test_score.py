"""``eval/score.py`` under the profile: the same cells as B0's config.json table, and the dataset chosen by keys.

``tests/fixtures/score_b0/`` holds a synthetic gold set, two datasets and what B0's score.py made of them.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from paperfacts.config import DEFAULT_REPO_ROOT
from paperfacts.fields import FieldSpec

SCRIPT = DEFAULT_REPO_ROOT / "eval" / "score.py"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "score_b0"
TCO = DEFAULT_REPO_ROOT / "profiles" / "tco.json"

_spec = importlib.util.spec_from_file_location("paperfacts_score_under_profile", SCRIPT)
assert _spec is not None and _spec.loader is not None
score = importlib.util.module_from_spec(_spec)
# dataclasses look their module up in sys.modules while the class body is built.
sys.modules[_spec.name] = score
_spec.loader.exec_module(score)


def run(tmp_path: Path, *extra: str) -> tuple[list[dict], str]:
    out, cells = tmp_path / "report.md", tmp_path / "cells.json"
    assert score.main(["--gold", "gold", "--out", str(out), "--json", str(cells), *extra]) == 0
    return json.loads(cells.read_text(encoding="utf-8")), out.read_text(encoding="utf-8")


def test_the_tco_profile_scores_every_cell_as_b0_did(tmp_path, monkeypatch):
    # The recording names its datasets by relative path, so the report's source list only matches from there.
    monkeypatch.chdir(FIXTURE)

    cells, text = run(
        tmp_path,
        "--profile",
        str(TCO),
        "--dataset",
        "0000000000000001=dataset_1.json",
        "--dataset",
        "0000000000000002=dataset_2.json",
    )

    assert cells == json.loads((FIXTURE / "cells.json").read_text(encoding="utf-8"))
    assert text == (FIXTURE / "report.md").read_text(encoding="utf-8")
    # The paper-level cells are reported under "paper", although these gold files still say "target".
    assert {cell["sample"] for cell in cells if cell["field"] == "component"} == {"paper"}


def test_paper_level_is_the_groups_level_not_its_name(tmp_path):
    profile = json.loads(TCO.read_text(encoding="utf-8"))
    profile["groups"][0]["name"] = "paper"
    for field in profile["fields"]:
        if field["group"] == "target":
            field["group"] = "paper"
    renamed = tmp_path / "renamed.json"
    renamed.write_text(json.dumps(profile), encoding="utf-8")

    specs = score.load_specs(renamed)
    gold = json.loads((FIXTURE / "gold" / "0000000000000001.json").read_text(encoding="utf-8"))
    dataset = json.loads((FIXTURE / "dataset_1.json").read_text(encoding="utf-8"))

    assert specs["component"].level == "paper"
    assert score.score_document(specs, gold, dataset) == score.score_document(score.load_specs(TCO), gold, dataset)


def test_a_gold_file_under_the_legacy_paper_key_scores_as_under_the_current_one(capsys):
    specs = score.load_specs(TCO)
    legacy = json.loads((FIXTURE / "gold" / "0000000000000001.json").read_text(encoding="utf-8"))
    current = {("paper" if key == "target" else key): value for key, value in legacy.items()}
    dataset = json.loads((FIXTURE / "dataset_1.json").read_text(encoding="utf-8"))

    assert "target" in legacy
    assert score.score_document(specs, legacy, dataset) == score.score_document(specs, current, dataset)
    assert "legacy key 'target'" in capsys.readouterr().err


def write_dataset(root: Path, doc: str, keys: str, value: float, mtime: int) -> Path:
    path = root / "docs" / doc / "datasets" / f"{keys}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"paper_row": {}, "sample_rows": [{"sample_id": "x", "thickness": value}]}))
    os.utime(path, (mtime, mtime))
    return path


def test_keys_pick_the_dataset_they_name_not_the_newest(tmp_path):
    doc = "0000000000000002"
    named = write_dataset(tmp_path, doc, "aaaaaaaaaaaa.bbbbbbbbbbbb", 1.0, mtime=1_000)
    newest = write_dataset(tmp_path, doc, "cccccccccccc.dddddddddddd", 2.0, mtime=2_000)

    assert score.find_dataset(tmp_path, doc, "aaaaaaaaaaaa.bbbbbbbbbbbb") == named
    assert score.find_dataset(tmp_path, doc, None) == newest
    with pytest.raises(FileNotFoundError, match=r"no dataset eeeeeeeeeeee\.ffffffffffff\.json for"):
        score.find_dataset(tmp_path, doc, "eeeeeeeeeeee.ffffffffffff")


def test_the_keys_reach_the_scored_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(FIXTURE)
    root = tmp_path / "data"
    named = write_dataset(root, "0000000000000002", "aaaaaaaaaaaa.bbbbbbbbbbbb", 1.0, mtime=1_000)
    write_dataset(root, "0000000000000002", "cccccccccccc.dddddddddddd", 2.0, mtime=2_000)

    cells, text = run(
        tmp_path, "--data-root", str(root), "--keys", "aaaaaaaaaaaa.bbbbbbbbbbbb", "--only", "0000000000000002"
    )

    assert f"`0000000000000002`: `{named}`" in text
    assert [cell["value"] for cell in cells if cell["field"] == "thickness"] == [1.0]


def _spec(kind: str, *, rel_tol: float = 0.0) -> FieldSpec:
    return FieldSpec(name="f", group="g", kind=kind, description="d", keywords=(), rel_tol=rel_tol)  # type: ignore[arg-type]


def test_a_boolean_a_date_and_an_interval_are_scored_by_their_own_rule():
    boolean, date, interval = _spec("boolean"), _spec("date"), _spec("interval", rel_tol=0.01)

    assert score.value_matches(boolean, False, {"value": False})
    assert not score.value_matches(boolean, "false", {"value": False})
    assert score.value_matches(date, "2021-03", {"value": "2021-03"})
    assert not score.value_matches(date, "2021-03-12", {"value": "2021-03"})
    assert score.value_matches(interval, [451.0, None], {"value": [450, None]})
    assert not score.value_matches(interval, [451.0, 500.0], {"value": [450, None]})
    assert not score.value_matches(interval, 450.0, {"value": [450, None]})
