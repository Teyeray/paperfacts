"""Files production holds at B1 still load, with their paper-level record (round 2, spec §1.3).

Round 2 renames the persisted ``target`` / ``no_tco_film`` to ``paper`` / ``no_samples`` and the report's
``"target"`` scope to ``"paper"``. Every persisted model ignores unknown keys, so a renamed attribute without a
read alias would load a B1 file with its paper record silently gone. ``tests/fixtures/b1_formats/`` holds the
files as B1's own code writes them (``generate.py``); these tests read them through the current models and must
pass on both sides of the rename. That is why an attribute is looked up under either name: the one the model
declares is the one that must hold the value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from paperfacts.compare import ComparisonReport
from paperfacts.dataset import DatasetPayload, DocumentDataset
from paperfacts.readings import StoredReadings
from paperfacts.records import LaneExtraction

FIXTURES = Path(__file__).parent / "fixtures" / "b1_formats"
PAPER_COMPONENT = "Sn/Ta target 95:5 wt.%"


def declared(model: BaseModel, *names: str):
    """The attribute the model declares under the first of ``names`` it has: ``(new, legacy)`` spellings."""
    name = next(name for name in names if name in type(model).model_fields)
    return getattr(model, name)


def raw(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ---- the fixtures are the old format (so a re-generation after the rename cannot pass unnoticed) ----------


def test_the_fixtures_are_b1_shaped():
    for name in ("lane.json", "lane_paddleocr_vl.json", "lane_no_samples.json"):
        lane = raw(name)
        assert "target" in lane and "no_tco_film" in lane, name
        assert "paper" not in lane and "no_samples" not in lane, name
    assert raw("lane_no_samples.json")["no_tco_film"] is True
    assert "target" in {comparison["scope"] for comparison in raw("report.json")["comparisons"]}
    assert "target" in {row["sample_id"] for row in raw("dataset.json")["quality_rows"]}


# ---- lanes ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["lane.json", "lane_paddleocr_vl.json"])
def test_a_b1_lane_loads_with_its_paper_record(name: str):
    lane = LaneExtraction.read(FIXTURES / name)

    paper = declared(lane, "paper", "target")
    assert paper is not None
    assert [(value.field, value.value_raw, value.grounded) for value in paper.fields] == [
        ("component", PAPER_COMPONENT, True)
    ]
    assert declared(lane, "no_samples", "no_tco_film") is False
    assert lane.samples and lane.samples[0].fields


def test_a_b1_lane_that_found_no_samples_still_says_so():
    lane = LaneExtraction.read(FIXTURES / "lane_no_samples.json")

    assert declared(lane, "no_samples", "no_tco_film") is True
    assert lane.samples == ()
    assert declared(lane, "paper", "target") is not None


# ---- comparison report ------------------------------------------------------------------------------------


def test_a_b1_report_loads_with_its_paper_comparison_and_unmatched_sample():
    report = ComparisonReport.read(FIXTURES / "report.json")

    paper_scopes = [c for c in report.comparisons if not c.scope.startswith("sample:")]
    assert [(c.scope in {"paper", "target"}, c.field, c.status) for c in paper_scopes] == [(True, "component", "agree")]
    assert report.counts.samples_matched == 2
    assert report.counts.samples_unmatched == 1
    assert report.matching.unmatched_b == ("SnO2:Ta reference",)


# ---- dataset ----------------------------------------------------------------------------------------------


def test_a_b1_dataset_loads_with_its_paper_level_cell():
    payload = DatasetPayload.model_validate_json((FIXTURES / "dataset.json").read_text(encoding="utf-8"))
    dataset = DocumentDataset.from_payload(payload)

    assert dataset.paper_row["component"] == PAPER_COMPONENT
    assert len(dataset.sample_rows) == 3
    paper_quality = [row for row in dataset.quality_rows if row["field"] == "component"]
    assert [(row["sample_id"] in {"paper", "target"}, row["decision"]) for row in paper_quality] == [(True, "agree")]


# ---- figure readings --------------------------------------------------------------------------------------


def test_b1_figure_readings_load():
    readings = StoredReadings.read(FIXTURES / "figures.json")

    assert readings.profile == "tco"
    assert readings.complete
    assert [(reading.field, reading.y) for reading in readings.readings] == [
        ("sheet_resistance", 25.0),
        ("sheet_resistance", 40.0),
    ]
