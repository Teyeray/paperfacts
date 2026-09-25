"""The comparison path reads its field table from the profile its options carry, never from config.json.

``comparison_key`` hashes the profile on the options, so a verdict read from anywhere else could be served
from a file whose key did not move when the rule that produced it did.
"""

from __future__ import annotations

from pathlib import Path

from paperfacts.compare import compare_lanes
from paperfacts.config import Settings
from paperfacts.dataset import consolidate_document, field_columns
from paperfacts.keys import ComparisonOptions, comparison_key, profile_extraction_fingerprint
from paperfacts.matching import SampleMatch, SampleMatching
from paperfacts.models import DocumentInput
from paperfacts.normalize import normalize_lane
from paperfacts.workbook import data_columns
from support.extraction import make_lane, make_sample
from support.factories import DOC_ID
from support.profiles import make_profile
from test_dataset import value

MATCHING = SampleMatching(
    pairs=(SampleMatch(a_id="A", b_id="A", confidence=1.0, justification="identical", method="exact"),)
)
DOCUMENT = DocumentInput(document_id=DOC_ID, sha256=DOC_ID, pdf_path=Path("paper.pdf"))
# coating_thickness is the demo profile's sample-level numeric field; 100 against 104 nm is 4 % apart.
STRICT = make_profile()
LOOSE = make_profile({"fields.1.rel_tol": 0.05})


def lanes():
    fingerprint = profile_extraction_fingerprint(STRICT)
    a = make_lane(samples=[make_sample("A", [value("coating_thickness", "100", "nm")])])
    b = make_lane(
        backend="paddleocr_vl",
        samples=[make_sample("A", [value("coating_thickness", "104", "nm", backend="paddleocr_vl")])],
    )
    return a.model_copy(update={"profile_fingerprint": fingerprint}), b.model_copy(
        update={"profile_fingerprint": fingerprint}
    )


def verdicts(profile):
    a, b = lanes()
    options = ComparisonOptions.from_settings(Settings(), profile)
    report = compare_lanes(a, b, MATCHING, options)
    dataset = consolidate_document(DOCUMENT, {a.backend: a, b.backend: b}, report, options)
    return [c.status for c in report.comparisons], dataset.paper_row["coating_thickness"]


def test_a_tolerance_is_read_from_the_options_profile_and_moves_the_comparison_key():
    # A tolerance is verdict-only, so the lanes extracted under one profile compare under the other.
    assert profile_extraction_fingerprint(LOOSE) == profile_extraction_fingerprint(STRICT)

    assert verdicts(STRICT) == (["conflict"], None)
    assert verdicts(LOOSE) == (["agree"], 100)
    assert comparison_key(ComparisonOptions.from_settings(Settings(), LOOSE)) != comparison_key(
        ComparisonOptions.from_settings(Settings(), STRICT)
    )


def test_normalisation_and_the_columns_follow_the_profile():
    a, _ = lanes()

    normalized = normalize_lane(a, STRICT)

    assert [(f.value, f.unit) for f in normalized.samples[0].fields] == [(100.0, "nm")]
    assert [column.name for column in field_columns(STRICT)] == ["precursor_purity", "coating_thickness", "solvent"]
    assert [key for key, _ in data_columns(STRICT)][-3:] == ["precursor_purity", "coating_thickness", "solvent"]
