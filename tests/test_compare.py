"""Field-level comparison (the rule layer): the deterministic parts happen here, the fuzzy judgment calls
are left to the model layer.

This layer produces the AGREE / CONFLICT / AMBIGUOUS / MISSING counts the PRD asks for directly, so two
things must be pinned down: **status is always the actual comparison outcome** (pairing confidence is
recorded separately, never used to rewrite the observation), and **when uncertain, lean toward the
lower-risk outcome** (when matching fails, unmatched = ambiguous for review, rather than missing/silently
accepted).
"""

from __future__ import annotations

import pytest

from paperfacts.compare import ComparisonReport, compare_lanes, compare_values
from paperfacts.fields import AMBIGUOUS_MATCH_CONFIDENCE, FIELD_BY_NAME
from paperfacts.keys import FINGERPRINT_LENGTH, comparison_key
from paperfacts.matching import SampleMatch, SampleMatching
from paperfacts.normalize import normalize_field
from paperfacts.records import FieldValue, TargetRecord
from support.extraction import make_field, make_lane, make_sample


def normalized(field: FieldValue) -> FieldValue:
    """The comparison layer always receives normalized values; unit tests for compare_values add this step by hand."""
    return normalize_field(field, FIELD_BY_NAME[field.field])


def exact_match(a_id: str = "A", b_id: str = "A") -> SampleMatching:
    return SampleMatching(
        pairs=(SampleMatch(a_id=a_id, b_id=b_id, confidence=1.0, justification="identical", method="exact"),)
    )


def statuses(report: ComparisonReport) -> list[tuple[str, str]]:
    return [(c.field, c.status) for c in report.comparisons]


# ---- compare_values: numeric ------------------------------------------------------------


def test_two_numbers_within_the_field_tolerance_agree():
    a, b = (
        normalized(make_field("thickness", "300", unit_raw="nm")),
        normalized(make_field("thickness", "305", unit_raw="nm")),
    )

    status, detail = compare_values(a, b, FIELD_BY_NAME["thickness"])

    assert status == "agree"
    assert "rel_tol=0.05" in detail


def test_two_numbers_outside_the_tolerance_conflict():
    a, b = (
        normalized(make_field("thickness", "300", unit_raw="nm")),
        normalized(make_field("thickness", "400", unit_raw="nm")),
    )

    status, detail = compare_values(a, b, FIELD_BY_NAME["thickness"])

    assert status == "conflict"
    assert "300 vs 400" in detail


def test_the_same_length_written_in_different_units_agrees_after_conversion():
    # Conversion is the whole reason this layer exists: 300 nm and 0.3 μm are the same fact.
    a, b = (
        normalized(make_field("thickness", "300", unit_raw="nm")),
        normalized(make_field("thickness", "0.3", unit_raw="μm")),
    )

    assert compare_values(a, b, FIELD_BY_NAME["thickness"])[0] == "agree"


def test_a_value_that_could_not_be_parsed_makes_the_pair_ambiguous():
    # If one side fails to parse as a number there's no way to judge agreement; calling it conflict would
    # manufacture a false conflict, and calling it agree would paper over the problem.
    a, b = (
        normalized(make_field("thickness", "300", unit_raw="nm")),
        normalized(make_field("thickness", "n.a.", unit_raw="nm")),
    )

    status, detail = compare_values(a, b, FIELD_BY_NAME["thickness"])

    assert status == "ambiguous"
    assert "unparsed" in detail


def test_two_unparseable_values_with_identical_raw_text_still_agree():
    # If both lanes transcribed identical raw text (including the unit), they really are describing the
    # same thing.
    a = normalized(make_field("thickness", "n.a.", unit_raw="nm"))
    b = normalized(make_field("thickness", "n.a.", unit_raw="nm"))

    assert compare_values(a, b, FIELD_BY_NAME["thickness"]) == (
        "agree",
        "identical raw text (not parsed as a number)",
    )


def test_identical_unparseable_text_with_different_units_is_ambiguous():
    a = normalized(make_field("resistance", "n.a.", unit_raw="mΩ·cm"))
    b = normalized(make_field("resistance", "n.a.", unit_raw="Ω·cm"))

    assert compare_values(a, b, FIELD_BY_NAME["resistance"])[0] == "ambiguous"


def test_the_unit_comparison_is_case_sensitive_so_milli_never_matches_mega():
    """Units are compared with ``clean_unit`` (case preserved), never ``normalize_key`` (which would
    lowercase both mΩ and MΩ down to mΩ).

    Two units that are 10⁹ apart being judged "identical text" would be one of the worst kinds of silent
    error.
    """
    a = normalized(make_field("resistance", "n.a.", unit_raw="mΩ·cm"))
    b = normalized(make_field("resistance", "n.a.", unit_raw="MΩ·cm"))

    assert compare_values(a, b, FIELD_BY_NAME["resistance"])[0] == "ambiguous"


def test_units_that_differ_after_normalization_are_ambiguous():
    # Defensive branch: once units differ after normalization, the values simply can't be compared.
    a = FieldValue(field="thickness", value_raw="1", value=1.0, unit="nm")
    b = FieldValue(field="thickness", value_raw="1", value=1.0, unit="μm")

    status, detail = compare_values(a, b, FIELD_BY_NAME["thickness"])

    assert status == "ambiguous"
    assert "units differ" in detail


# ---- compare_values: text ------------------------------------------------------------


def test_text_values_agree_when_their_normalized_keys_match():
    a, b = make_field("component", "SnO2:Ta"), make_field("component", "sno2 : ta")

    assert compare_values(a, b, FIELD_BY_NAME["component"]) == ("agree", "identical after text normalization")


def test_text_values_that_differ_conflict_and_quote_both_sides():
    a, b = make_field("component", "SnO2:Ta"), make_field("component", "ITO")

    status, detail = compare_values(a, b, FIELD_BY_NAME["component"])

    assert status == "conflict"
    assert "SnO2:Ta" in detail and "ITO" in detail


# ---- compare_lanes: scope levels -------------------------------------------------------------


def test_target_fields_are_compared_at_the_paper_level():
    lane_a = make_lane(backend="mineru", target=TargetRecord(fields=(make_field("density", "98.5", unit_raw="%"),)))
    lane_b = make_lane(
        backend="paddleocr_vl", target=TargetRecord(fields=(make_field("density", "98.6", unit_raw="%"),))
    )

    report = compare_lanes(lane_a, lane_b, SampleMatching())

    assert [(c.scope, c.field, c.status) for c in report.comparisons] == [("target", "density", "agree")]


def test_a_target_field_that_shows_up_inside_a_sample_is_ignored():
    # The target belongs to the paper, not to a sample; comparing it at the sample level would conjure up
    # duplicate facts out of nowhere.
    fields = (make_field("density", "99", unit_raw="%"),)
    lane_a = make_lane(backend="mineru", samples=[make_sample("A", fields)])
    lane_b = make_lane(backend="paddleocr_vl", samples=[make_sample("A", fields)])

    report = compare_lanes(lane_a, lane_b, exact_match())

    assert report.comparisons == ()


def test_matched_samples_are_compared_field_by_field():
    lane_a = make_lane(
        backend="mineru",
        samples=[
            make_sample(
                "A", [make_field("thickness", "300", unit_raw="nm"), make_field("transmittance", "85", unit_raw="%")]
            )
        ],
    )
    lane_b = make_lane(
        backend="paddleocr_vl",
        samples=[
            make_sample(
                "A", [make_field("thickness", "900", unit_raw="nm"), make_field("transmittance", "85.5", unit_raw="%")]
            )
        ],
    )

    report = compare_lanes(lane_a, lane_b, exact_match())

    assert statuses(report) == [("thickness", "conflict"), ("transmittance", "agree")]


def test_the_scope_names_both_sides_of_a_matched_pair():
    lane_a = make_lane(backend="mineru", samples=[make_sample("A1", [make_field("thickness", "300", unit_raw="nm")])])
    lane_b = make_lane(
        backend="paddleocr_vl", samples=[make_sample("B1", [make_field("thickness", "300", unit_raw="nm")])]
    )

    report = compare_lanes(lane_a, lane_b, exact_match("A1", "B1"))

    assert report.comparisons[0].scope == "sample:A1|B1"


def test_lanes_are_normalized_inside_compare_lanes():
    """Callers never have to remember to normalize first — otherwise "forgot to normalize" would silently
    degrade every numeric comparison into a raw-text comparison."""
    lane_a = make_lane(backend="mineru", samples=[make_sample("A", [make_field("thickness", "0.3", unit_raw="μm")])])
    lane_b = make_lane(
        backend="paddleocr_vl", samples=[make_sample("A", [make_field("thickness", "300", unit_raw="nm")])]
    )

    report = compare_lanes(lane_a, lane_b, exact_match())

    assert report.comparisons[0].status == "agree"
    assert report.comparisons[0].a.value == 300.0


# ---- compare_lanes: conditions -------------------------------------------------------------


def test_the_same_field_under_two_conditions_is_two_separate_facts():
    # Transmittance at 550 nm and the full-spectrum average are two different things; merging them into
    # one record would let them contaminate each other.
    fields_a = [
        make_field("transmittance", "85", unit_raw="%", condition="550 nm"),
        make_field("transmittance", "80", unit_raw="%", condition="average 400-800 nm"),
    ]
    fields_b = [
        make_field("transmittance", "85.5", unit_raw="%", condition="550 nm"),
        make_field("transmittance", "80.2", unit_raw="%", condition="average 400-800 nm"),
    ]
    lane_a = make_lane(backend="mineru", samples=[make_sample("A", fields_a)])
    lane_b = make_lane(backend="paddleocr_vl", samples=[make_sample("A", fields_b)])

    report = compare_lanes(lane_a, lane_b, exact_match())

    assert [(c.condition, c.status) for c in report.comparisons] == [
        ("550 nm", "agree"),
        ("average 400-800 nm", "agree"),
    ]


def test_leftover_numeric_values_pair_by_numeric_proximity_when_the_condition_wording_differs():
    """When both lanes phrase the same fact's condition differently, it shouldn't be split into two
    records that each look like they're missing the other's value."""
    lane_a = make_lane(
        backend="mineru",
        samples=[make_sample("A", [make_field("thickness", "300", unit_raw="nm", condition="ellipsometric")])],
    )
    lane_b = make_lane(
        backend="paddleocr_vl",
        samples=[make_sample("A", [make_field("thickness", "300", unit_raw="nm", condition="from ellipsometric")])],
    )

    report = compare_lanes(lane_a, lane_b, exact_match())
    only = report.comparisons[0]

    assert only.status == "agree"
    assert "condition texts differ" in only.detail
    assert only.a is not None and only.b is not None


def test_a_proximity_paired_value_that_disagrees_is_ambiguous_not_conflict():
    # Different condition wording plus different values: there's no way to tell "two facts under
    # different conditions" from "a real conflict over the same fact," so it's left for review.
    lane_a = make_lane(
        backend="mineru",
        samples=[make_sample("A", [make_field("thickness", "300", unit_raw="nm", condition="ellipsometric")])],
    )
    lane_b = make_lane(
        backend="paddleocr_vl",
        samples=[make_sample("A", [make_field("thickness", "900", unit_raw="nm", condition="from SEM")])],
    )

    report = compare_lanes(lane_a, lane_b, exact_match())

    assert report.comparisons[0].status == "ambiguous"
    assert "condition texts differ" in report.comparisons[0].detail


def test_unparsed_leftovers_cannot_be_paired_by_proximity_and_stay_one_sided():
    """Pairing by numeric proximity requires a value on both sides; when neither side parses, they cannot
    be paired blindly and each counts as one-sided instead.

    Forcing a pairing here would claim that two values which failed to parse under two different
    conditions are somehow the same fact.
    """
    lane_a = make_lane(
        backend="mineru",
        samples=[make_sample("A", [make_field("thickness", "n.a.", unit_raw="nm", condition="SEM")])],
    )
    lane_b = make_lane(
        backend="paddleocr_vl",
        samples=[make_sample("A", [make_field("thickness", "not measured", unit_raw="nm", condition="ellipsometry")])],
    )

    report = compare_lanes(lane_a, lane_b, exact_match())

    assert [(c.condition, c.status, c.missing_in) for c in report.comparisons] == [
        ("SEM", "missing", "paddleocr_vl"),
        ("ellipsometry", "missing", "mineru"),
    ]


# ---- compare_lanes: one-sided / missing ---------------------------------------------------------


def test_a_field_only_one_lane_reported_is_missing_in_the_other():
    lane_a = make_lane(backend="mineru", samples=[make_sample("A", [make_field("thickness", "300", unit_raw="nm")])])
    lane_b = make_lane(backend="paddleocr_vl", samples=[make_sample("A", [])])

    report = compare_lanes(lane_a, lane_b, exact_match())
    only = report.comparisons[0]

    assert only.status == "missing"
    assert only.missing_in == "paddleocr_vl"
    assert only.detail == "only in mineru"


def test_an_unmatched_sample_makes_every_one_of_its_fields_missing_in_the_other_lane():
    lane_a = make_lane(backend="mineru", samples=[make_sample("S1", [make_field("thickness", "300", unit_raw="nm")])])
    lane_b = make_lane(
        backend="paddleocr_vl", samples=[make_sample("T1", [make_field("thickness", "400", unit_raw="nm")])]
    )

    report = compare_lanes(lane_a, lane_b, SampleMatching(unmatched_a=("S1",), unmatched_b=("T1",)))

    assert [(c.scope, c.status, c.missing_in) for c in report.comparisons] == [
        ("sample:S1", "missing", "paddleocr_vl"),
        ("sample:T1", "missing", "mineru"),
    ]


def test_a_matching_failure_downgrades_unmatched_samples_to_ambiguous():
    """When the matching model fails, "unmatched" does not mean "the other lane genuinely doesn't have
    it"; mark it ambiguous for review, rather than accepting it outright."""
    lane_a = make_lane(backend="mineru", samples=[make_sample("S1", [make_field("thickness", "300", unit_raw="nm")])])
    lane_b = make_lane(
        backend="paddleocr_vl", samples=[make_sample("T1", [make_field("thickness", "400", unit_raw="nm")])]
    )
    matching = SampleMatching(unmatched_a=("S1",), unmatched_b=("T1",), failed=True, failure="two bad answers")

    report = compare_lanes(lane_a, lane_b, matching)

    assert {c.status for c in report.comparisons} == {"ambiguous"}
    assert all(c.missing_in is None for c in report.comparisons)
    assert all("two bad answers" in c.detail for c in report.comparisons)
    assert report.counts.matching_failed is True


def test_an_unmatched_id_with_no_matching_sample_is_skipped():
    lane_a = make_lane(backend="mineru", samples=[])
    lane_b = make_lane(backend="paddleocr_vl", samples=[])

    report = compare_lanes(lane_a, lane_b, SampleMatching(unmatched_a=("ghost",), unmatched_b=("ghost",)))

    assert report.comparisons == ()


def test_a_pair_naming_a_sample_that_is_not_in_the_lane_is_skipped():
    lane_a = make_lane(backend="mineru", samples=[make_sample("A", [make_field("thickness", "300", unit_raw="nm")])])
    lane_b = make_lane(backend="paddleocr_vl", samples=[])

    report = compare_lanes(lane_a, lane_b, exact_match())

    assert report.comparisons == ()


# ---- Pairing confidence ---------------------------------------------------------------------


def test_a_low_confidence_pair_keeps_the_real_status_and_records_the_confidence():
    """The PRD requires low-confidence pairings to be downgraded to AMBIGUOUS, but that downgrade is a
    downstream consumer's policy decision.

    The comparison layer records the **observation** (status + match_confidence) and never rewrites it —
    otherwise "did this actually agree or not" could never be recovered.
    """
    fields = [make_field("thickness", "300", unit_raw="nm")]
    lane_a = make_lane(backend="mineru", samples=[make_sample("A", fields)])
    lane_b = make_lane(backend="paddleocr_vl", samples=[make_sample("A", fields)])
    weak = SampleMatching(pairs=(SampleMatch(a_id="A", b_id="A", confidence=0.3, justification="weak", method="llm"),))

    report = compare_lanes(lane_a, lane_b, weak)

    assert report.comparisons[0].status == "agree"
    assert report.comparisons[0].match_confidence == 0.3
    assert report.counts.low_confidence_matches == 1


def test_an_exact_pair_carries_no_confidence():
    # An exact pairing is deterministic; a "confidence of 1.0" would be mistaken for a judgment the model
    # actually made.
    fields = [make_field("thickness", "300", unit_raw="nm")]
    lane_a = make_lane(backend="mineru", samples=[make_sample("A", fields)])
    lane_b = make_lane(backend="paddleocr_vl", samples=[make_sample("A", fields)])

    report = compare_lanes(lane_a, lane_b, exact_match())

    assert report.comparisons[0].match_confidence is None


def test_a_confident_llm_pair_is_not_counted_as_low_confidence():
    fields = [make_field("thickness", "300", unit_raw="nm")]
    lane_a = make_lane(backend="mineru", samples=[make_sample("A", fields)])
    lane_b = make_lane(backend="paddleocr_vl", samples=[make_sample("A", fields)])
    strong = SampleMatching(
        pairs=(SampleMatch(a_id="A", b_id="A", confidence=AMBIGUOUS_MATCH_CONFIDENCE, justification="", method="llm"),)
    )

    report = compare_lanes(lane_a, lane_b, strong)

    assert report.counts.low_confidence_matches == 0


# ---- counts -------------------------------------------------------------------------


def test_the_counts_cover_every_status_and_the_sample_bookkeeping():
    lane_a = make_lane(
        backend="mineru",
        samples=[
            make_sample(
                "A", [make_field("thickness", "300", unit_raw="nm"), make_field("transmittance", "85", unit_raw="%")]
            ),
            make_sample("S1", [make_field("thickness", "10", unit_raw="nm")]),
        ],
    )
    lane_b = make_lane(
        backend="paddleocr_vl",
        samples=[
            make_sample(
                "A", [make_field("thickness", "900", unit_raw="nm"), make_field("transmittance", "85.5", unit_raw="%")]
            ),
            make_sample("T1", [make_field("thickness", "20", unit_raw="nm")]),
        ],
    )
    matching = SampleMatching(pairs=exact_match().pairs, unmatched_a=("S1",), unmatched_b=("T1",))

    counts = compare_lanes(lane_a, lane_b, matching).counts

    assert counts.agree == 1 and counts.conflict == 1 and counts.missing == 2 and counts.ambiguous == 0
    assert counts.total == 4
    assert counts.missing_by_backend == {"mineru": 1, "paddleocr_vl": 1}
    assert counts.samples_matched == 1 and counts.samples_unmatched == 2


def test_the_counts_have_every_key_even_when_zero():
    # Every key is present even at zero, so downstream consumers never need to guard against KeyError.
    counts = compare_lanes(make_lane(), make_lane(backend="paddleocr_vl"), SampleMatching()).counts.model_dump()

    assert set(counts) == {
        "agree",
        "conflict",
        "ambiguous",
        "missing",
        "total",
        "missing_by_backend",
        "samples_matched",
        "samples_unmatched",
        "low_confidence_matches",
        "matching_failed",
        "unattributed_by_backend",
    }


# ---- Preconditions and cache key ---------------------------------------------------------------


def test_two_lanes_from_different_extractors_cannot_be_compared():
    # Comparing results extracted with different prompts/models/schemas would no longer say anything
    # about actual parser differences.
    lane_a = make_lane(backend="mineru", extractor_key="aaaaaaaaaaaa")
    lane_b = make_lane(backend="paddleocr_vl", extractor_key="bbbbbbbbbbbb")

    with pytest.raises(ValueError, match="different extractor_key"):
        compare_lanes(lane_a, lane_b, SampleMatching())


def test_the_comparison_key_is_stable_and_short():
    assert comparison_key() == comparison_key()
    assert len(comparison_key()) == FINGERPRINT_LENGTH


def test_the_report_records_both_keys():
    report = compare_lanes(make_lane(), make_lane(backend="paddleocr_vl"), SampleMatching())

    assert report.extractor_key == make_lane().extractor_key
    assert report.comparison_key == comparison_key()


# ---- Persisting to disk --------------------------------------------------------------------


def test_the_report_round_trips_through_disk(tmp_path):
    lane_a = make_lane(backend="mineru", samples=[make_sample("A", [make_field("thickness", "300", unit_raw="nm")])])
    lane_b = make_lane(
        backend="paddleocr_vl", samples=[make_sample("A", [make_field("thickness", "305", unit_raw="nm")])]
    )
    report = compare_lanes(lane_a, lane_b, exact_match())
    path = tmp_path / "comparisons" / "key.json"

    report.write(path)

    assert ComparisonReport.read(path) == report
