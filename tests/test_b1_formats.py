"""Files production holds at B1 still load, with their paper-level record (round 2, spec §1.3).

Round 2 renames the persisted ``target`` / ``no_tco_film`` to ``paper`` / ``no_samples`` and the report's
``"target"`` scope to ``"paper"``. Every persisted model ignores unknown keys, so a renamed attribute without a
read alias would load a B1 file with its paper record silently gone. ``tests/fixtures/b1_formats/`` holds the
files as B1's own code writes them (``generate.py``); these tests read them through the current models, which
must put every renamed value under its new name: ``paper``, ``no_samples``, the ``"paper"`` scope, and the one
``matching`` as the implicit entity's entry of ``matchings``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from paperfacts.compare import IMPLICIT_ENTITY, ComparisonReport
from paperfacts.dataset import DatasetPayload, DocumentDataset
from paperfacts.readings import StoredReadings
from paperfacts.records import LaneExtraction

FIXTURES = Path(__file__).parent / "fixtures" / "b1_formats"
PAPER_COMPONENT = "Sn/Ta target 95:5 wt.%"


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

    paper = lane.paper
    assert paper is not None
    assert [(value.field, value.value_raw, value.grounded) for value in paper.fields] == [
        ("component", PAPER_COMPONENT, True)
    ]
    assert lane.no_samples is False
    assert lane.samples and lane.samples[0].fields


def test_a_b1_lane_that_found_no_samples_still_says_so():
    lane = LaneExtraction.read(FIXTURES / "lane_no_samples.json")

    assert lane.no_samples is True
    assert lane.samples == ()
    assert lane.paper is not None


# ---- comparison report ------------------------------------------------------------------------------------


def test_a_b1_report_loads_with_its_paper_comparison_and_unmatched_sample():
    report = ComparisonReport.read(FIXTURES / "report.json")

    paper_scopes = [c for c in report.comparisons if not c.scope.startswith("sample:")]
    assert [(c.scope, c.field, c.status) for c in paper_scopes] == [("paper", "component", "agree")]
    assert report.counts.samples_matched == 2
    assert report.counts.samples_unmatched == 1
    assert list(report.matchings) == [IMPLICIT_ENTITY]
    assert len(report.sample_matching().pairs) == 2
    assert report.sample_matching().unmatched_b == ("SnO2:Ta reference",)


def test_a_report_without_the_implicit_entity_s_matching_is_refused_at_load():
    data = json.loads((FIXTURES / "report.json").read_text(encoding="utf-8"))
    data["matchings"] = {"catalyst": data.pop("matching")}

    with pytest.raises(ValidationError, match="matchings has no entry for the implicit entity 'sample'"):
        ComparisonReport.model_validate(data)


def test_a_b1_report_is_written_back_under_the_new_names():
    written = ComparisonReport.read(FIXTURES / "report.json").model_dump(mode="json")

    assert "matching" not in written and set(written["matchings"]) == {IMPLICIT_ENTITY}
    assert "target" not in {comparison["scope"] for comparison in written["comparisons"]}


def test_a_b1_lane_is_written_back_under_the_new_names():
    written = LaneExtraction.read(FIXTURES / "lane_no_samples.json").model_dump(mode="json")

    assert written["no_samples"] is True and written["paper"] is not None
    assert "target" not in written and "no_tco_film" not in written


# ---- dataset ----------------------------------------------------------------------------------------------


def test_a_b1_dataset_loads_with_its_paper_level_cell():
    payload = DatasetPayload.model_validate_json((FIXTURES / "dataset.json").read_text(encoding="utf-8"))
    dataset = DocumentDataset.from_payload(payload)

    assert dataset.paper_row["component"] == PAPER_COMPONENT
    assert len(dataset.sample_rows) == 3
    paper_quality = [row for row in dataset.quality_rows if row["field"] == "component"]
    # The paper-level quality rows' "target" id is read as "paper", as the report's scope is.
    assert [(row["sample_id"], row["decision"]) for row in paper_quality] == [("paper", "agree")]
    assert "target" not in {row["sample_id"] for row in dataset.quality_rows}


def test_a_sample_named_target_keeps_its_quality_rows():
    # Only the paper's rows were ever renamed: a sample really called "target" has a sample row, and keeps its id.
    data = json.loads((FIXTURES / "dataset.json").read_text(encoding="utf-8"))
    data["sample_rows"] = [*data["sample_rows"], {"sample_id": "target"}]

    payload = DatasetPayload.model_validate(data)

    assert "target" in {row["sample_id"] for row in payload.quality_rows}


# ---- figure readings --------------------------------------------------------------------------------------


def test_b1_figure_readings_load():
    readings = StoredReadings.read(FIXTURES / "figures.json")

    assert readings.profile == "tco"
    assert readings.complete
    assert [(reading.field, reading.y) for reading in readings.readings] == [
        ("sheet_resistance", 25.0),
        ("sheet_resistance", 40.0),
    ]
