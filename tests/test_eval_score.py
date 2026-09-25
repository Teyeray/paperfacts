"""``eval/score.py``: how a dataset cell is matched against a gold cell.

The script lives outside the package, so it is loaded from its path. Only the matching rules are pinned
here; the alignment and report are exercised by running it against the gold set.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "eval" / "score.py"
_spec = importlib.util.spec_from_file_location("paperfacts_eval_score", SCRIPT)
assert _spec is not None and _spec.loader is not None
score = importlib.util.module_from_spec(_spec)
# dataclasses look their module up in sys.modules while the class body is built.
sys.modules[_spec.name] = score
_spec.loader.exec_module(score)

MODE = score.Spec("mode", "process", "text", 0.0, 0.0, ("DC", "RF", "pulsed DC", "DC+RF", "HiPIMS"))


@pytest.mark.parametrize(
    ("got", "gold"),
    [
        ("HiPIMS", "HiPIMS"),
        ("high-power impulse magnetron sputtering (HiPIMS)", "HiPIMS"),
        ("RF magnetron sputtering", "RF"),
        ("DC pulsed", "pulsed DC"),
        ("DC and RF co-sputtering", "DC+RF"),
    ],
)
def test_a_category_value_matches_its_gold_category(got, gold):
    # The package's own category rule: before, HiPIMS matched nothing, so every HiPIMS cell scored wrong.
    assert score.value_matches(MODE, got, {"value": gold})


@pytest.mark.parametrize(("got", "gold"), [("DC", "RF"), ("pulsed DC", "DC"), ("RF", "DC+RF")])
def test_a_different_category_does_not_match(got, gold):
    assert not score.value_matches(MODE, got, {"value": gold})


def test_the_specs_take_categories_and_default_the_tolerances_like_the_package(tmp_path: Path):
    config = tmp_path / "profile.json"
    config.write_text(
        json.dumps(
            {
                "groups": [{"name": "process", "level": "sample"}, {"name": "film", "level": "sample"}],
                "fields": [
                    {"name": "mode", "group": "process", "kind": "text", "categories": ["DC", "RF"]},
                    {"name": "thickness", "group": "film", "kind": "numeric", "rel_tol": 0.05},
                ],
            }
        ),
        encoding="utf-8",
    )

    specs = score.load_specs(config)

    assert specs["mode"].categories == ("DC", "RF")
    assert (specs["thickness"].rel_tol, specs["thickness"].abs_tol) == (0.05, 0.0)
