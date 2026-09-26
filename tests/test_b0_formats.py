"""Files written before profiles still load (AC-5).

The migration adds ``profile_fingerprint`` to the stored lane, comparison report and dataset as a new optional
field -- no renames, no aliases -- so a B0 file reads as it did, with the fingerprint unknown. The fixtures in
``tests/fixtures/b0_formats/`` are the stored shapes without that field. Loading is all they are promised:
combining one with results of a profile is refused (``test_profile_mismatch.py``), since nothing says which
profile it came from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperfacts.compare import ComparisonReport
from paperfacts.dataset import DatasetPayload, DocumentDataset
from paperfacts.records import LaneExtraction

FIXTURES = Path(__file__).parent / "fixtures" / "b0_formats"


@pytest.mark.parametrize("name", ["lane", "report", "dataset"])
def test_the_fixtures_are_b0_shaped(name):
    assert "profile_fingerprint" not in json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def test_a_b0_lane_loads():
    lane = LaneExtraction.read(FIXTURES / "lane.json")

    assert lane.profile_fingerprint is None
    assert [sample.sample_id for sample in lane.samples] == ["A"]
    assert lane.samples[0].fields[0].value_raw == "100"


def test_a_b0_lane_with_a_paper_level_record_loads_it():
    # The recorded B0 lane has none ("target": null), so one is set on its bytes here. Round 2 renamed the
    # attribute to ``paper`` behind a read alias; without the alias it would load None.
    data = json.loads((FIXTURES / "lane.json").read_text(encoding="utf-8"))
    data["target"] = {"source_ids": ["mineru_p0_b1"], "fields": [data["samples"][0]["fields"][1] | {"field": "inch"}]}

    lane = LaneExtraction.model_validate_json(json.dumps(data))

    paper = lane.paper
    assert paper is not None
    assert [(value.field, value.value_raw) for value in paper.fields] == [("inch", "RF")]


def test_a_b0_report_loads():
    report = ComparisonReport.read(FIXTURES / "report.json")

    assert report.profile_fingerprint is None
    assert report.counts.agree == len(report.comparisons) > 0
    assert list(report.matchings) == ["sample"]


def test_a_b0_dataset_loads_and_exports_again():
    payload = DatasetPayload.model_validate_json((FIXTURES / "dataset.json").read_text(encoding="utf-8"))

    dataset = DocumentDataset.from_payload(payload)

    assert payload.profile_fingerprint is None
    assert dataset.paper_row["thickness"] == 100
    # The paper-level quality rows (keyed "target" at B0) all come through.
    assert [row["field"] for row in dataset.quality_rows if row["sample_id"] in {"paper", "target"}] == [
        "component",
        "resistance",
        "density",
        "inch",
    ]
    # Everything but the field list, which is display text built from the profile and no longer carried.
    assert payload.fields
    assert dataset.to_payload() == payload.model_copy(update={"fields": ()})
