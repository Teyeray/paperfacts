"""Results of two profiles are never combined (AC-13).

A lane records the extraction fingerprint of its profile, a comparison report and a dataset the comparison
fingerprint. Two profiles may name the same field with other units or verdict rules, so a comparison across
them would report agreement that means nothing: every such combination is refused with
:class:`ProfileMismatchError`, and so is a file from before profiles, which only an explicitly old key reaches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paperfacts.compare import compare_lanes
from paperfacts.config import Settings
from paperfacts.dataset import consolidate_document
from paperfacts.errors import ProfileMismatchError
from paperfacts.keys import ComparisonOptions, profile_comparison_fingerprint, profile_extraction_fingerprint
from paperfacts.matching import SampleMatch, SampleMatching
from paperfacts.models import DocumentInput
from support.extraction import comparison_options, make_field, make_lane, make_sample
from support.factories import DOC_ID
from support.profiles import make_profile

MATCHING = SampleMatching(
    pairs=(SampleMatch(a_id="A", b_id="A", confidence=1.0, justification="identical", method="exact"),)
)
DOCUMENT = DocumentInput(document_id=DOC_ID, sha256=DOC_ID, pdf_path=Path("paper.pdf"))


def lane(backend="mineru", **update):
    thickness = make_field("thickness", "100", unit_raw="nm", source_ids=[f"{backend}_p0_b1"])
    built = make_lane(backend=backend, samples=[make_sample("A", [thickness])])
    return built.model_copy(update=update)


def other_profile_fingerprint() -> str:
    return profile_extraction_fingerprint(make_profile())


# ---- compare_lanes ------------------------------------------------------------------------------------------


def test_lanes_of_the_options_profile_are_compared_and_the_report_records_it(tco_profile):
    report = compare_lanes(lane(), lane("paddleocr_vl"), MATCHING, comparison_options())

    assert report.profile_fingerprint == profile_comparison_fingerprint(tco_profile)
    assert [c.status for c in report.comparisons] == ["agree"]


def test_two_lanes_of_different_profiles_are_refused():
    b = lane("paddleocr_vl", profile_fingerprint=other_profile_fingerprint())

    with pytest.raises(ProfileMismatchError, match="different profiles"):
        compare_lanes(lane(), b, MATCHING, comparison_options())


def test_lanes_that_agree_with_each_other_but_not_with_the_options_are_refused(tco_profile):
    # Both lanes are the shipped profile's; the options carry another one.
    options = ComparisonOptions.from_settings(Settings(), make_profile())

    with pytest.raises(ProfileMismatchError, match="mineru lane"):
        compare_lanes(lane(), lane("paddleocr_vl"), MATCHING, options)


@pytest.mark.parametrize("backend", ["mineru", "paddleocr_vl"])
def test_a_lane_from_before_profiles_is_refused(backend):
    lanes = {name: lane(name) for name in ("mineru", "paddleocr_vl")}
    lanes[backend] = lanes[backend].model_copy(update={"profile_fingerprint": None})

    with pytest.raises(ProfileMismatchError, match="before profiles"):
        compare_lanes(lanes["mineru"], lanes["paddleocr_vl"], MATCHING, comparison_options())


# ---- consolidate_document -----------------------------------------------------------------------------------


def consolidate(report_update=None, options=None):
    a, b = lane(), lane("paddleocr_vl")
    report = compare_lanes(a, b, MATCHING, comparison_options())
    if report_update is not None:
        report = report.model_copy(update=report_update)
    return consolidate_document(DOCUMENT, {a.backend: a, b.backend: b}, report, options or comparison_options())


def test_a_report_of_the_options_profile_is_consolidated_and_the_dataset_records_it(tco_profile):
    dataset = consolidate()

    assert dataset.profile_fingerprint == profile_comparison_fingerprint(tco_profile)
    assert dataset.to_payload().profile_fingerprint == dataset.profile_fingerprint
    assert dataset.paper_row["thickness"] == 100


def test_a_report_of_another_profile_is_refused():
    with pytest.raises(ProfileMismatchError, match="comparison report"):
        consolidate({"profile_fingerprint": profile_comparison_fingerprint(make_profile())})


def test_a_report_consolidated_under_other_options_is_refused():
    with pytest.raises(ProfileMismatchError, match="comparison report"):
        consolidate(options=ComparisonOptions.from_settings(Settings(), make_profile()))


def test_a_report_from_before_profiles_is_refused():
    with pytest.raises(ProfileMismatchError, match="before profiles"):
        consolidate({"profile_fingerprint": None})
