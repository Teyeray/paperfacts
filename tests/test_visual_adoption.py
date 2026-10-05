"""Counterfactual adoption only: all facts, identities and pixel receipts are synthetic."""

import hashlib
import json

import pytest

from paperfacts.compare import compare_lanes
from paperfacts.crops import RegionCrop
from paperfacts.dataset import consolidate_document
from paperfacts.keys import ComparisonOptions, profile_extraction_fingerprint
from paperfacts.matching import SampleMatch, SampleMatching
from paperfacts.models import NormalizedBBox
from paperfacts.records import LaneExtraction, SampleRecord
from paperfacts.visual_adoption import AdoptionPolicy, replay_document
from paperfacts.visual_candidates import CandidateSelection, VisualCandidate
from paperfacts.visual_evidence import (
    BoundObservation,
    CandidateReading,
    VisualAttempt,
    VisualEvidenceReport,
    VisualObservation,
)
from support.extraction import make_field
from support.profiles import make_profile


def make_case(document, profile=None):
    profile = profile or make_profile()
    lanes = tuple(
        LaneExtraction(
            document_id=document.document_id,
            backend=backend,
            extractor_key="ex",
            model="fake",
            profile_fingerprint=profile_extraction_fingerprint(profile),
            artifact_sha256=backend[0] * 64,
            samples=(SampleRecord(sample_id="S1"),),
        )
        for backend in ("mineru", "paddleocr_vl")
    )
    matching = SampleMatching(
        pairs=(SampleMatch(a_id="S1", b_id="S1", confidence=1, justification="exact", method="exact"),)
    )
    options = ComparisonOptions(profile=profile, ambiguous_match_confidence=0.6)
    comparison = compare_lanes(*lanes, matching, options)
    dataset = consolidate_document(document, {lane.backend: lane for lane in lanes}, comparison, options).to_payload()
    box = NormalizedBBox(x1=0, y1=0, x2=1, y2=1)
    candidate = VisualCandidate(page=0, bbox=box, reasons=("native_keyword",), sources=("independent",))
    raw = VisualObservation(
        scope="sample",
        entity="sample",
        sample_raw="S1",
        field="coating_thickness",
        value_raw="10",
        unit_raw="nm",
        condition=None,
        condition_status="not_stated",
        evidence_type="printed",
        basis="S1 row and nm column",
    )
    reading = reading_for(candidate, (raw,))
    evidence = VisualEvidenceReport(
        document_id=document.document_id,
        pdf_sha256=document.sha256,
        profile_hash=profile.content_hash,
        model="fake",
        strategy="balanced",
        prompt_sha256="a" * 64,
        baseline_sha256={lane.backend: hashlib.sha256(lane.model_dump_json().encode()).hexdigest() for lane in lanes},
        selection=CandidateSelection(strategy="balanced", candidates=(candidate,), issues=(), unselected_pages=()),
        readings=(reading,),
        logical_requests=1,
        usage={},
        seconds=0,
    )
    return dict(
        dataset=dataset,
        lanes=lanes,
        comparison=comparison,
        evidence=evidence,
        profile=profile,
        policy=AdoptionPolicy(revision="synthetic-v1", fields=("coating_thickness",)),
        pdf_sha256=document.sha256,
    )


@pytest.fixture
def case(document):
    return make_case(document)


def reading_for(candidate, raw):
    response = json.dumps(dict(outcome="observed", reason="", observations=[o.model_dump() for o in raw]))
    crop = RegionCrop(
        page=candidate.page,
        bbox=candidate.bbox,
        dpi=72,
        width_px=100,
        height_px=100,
        image_sha256="b" * 64,
        path="synthetic.png",
    )
    return CandidateReading(
        candidate=candidate,
        status="observed",
        facts=tuple(
            BoundObservation(raw=o, normalized=make_field(o.field, "999", value=999), mapping_status="new") for o in raw
        ),
        attempts=(VisualAttempt(crop=crop, requested=True, raw_response=response),),
    )


def with_raw(case, **changes):
    old = case["evidence"].readings[0]
    raw = old.facts[0].raw.model_copy(update=changes)
    reading = reading_for(old.candidate, (raw,))
    return case | {"evidence": case["evidence"].model_copy(update={"readings": (reading,)})}


def cell(case):
    return replay_document(**case).cells[0]


def test_missing_adopts_recomputed_raw_value_and_keeps_baseline(case):
    before = case["dataset"].model_dump_json()
    result = cell(case)
    assert result.adopted and result.after == 10
    assert result.before is None and result.before_status == "missing"
    assert result.reason == "adopted_from_visual"
    assert result.evidence == ((0, 0),)
    assert case["dataset"].model_dump_json() == before


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"value_raw": "~10"}, "non_exact_value"),
        ({"value_raw": "10–20"}, "non_exact_value"),
        ({"value_raw": "above 10"}, "non_exact_value"),
        ({"value_raw": "NaN"}, "invalid_value"),
        ({"unit_raw": "kg"}, "invalid_value"),
        ({"condition_status": "unclear"}, "unclear_condition"),
        ({"evidence_type": "curve_estimate"}, "not_printed"),
    ],
)
def test_unsafe_measurements_abstain(case, changes, reason):
    result = cell(with_raw(case, **changes))
    assert not result.adopted and result.after is None and result.reason == reason


def test_equivalent_unit_conversion(case):
    assert cell(with_raw(case, value_raw="0.01", unit_raw="μm")).after == 10


def test_new_sample_is_retained_unassigned(case):
    result = replay_document(**with_raw(case, sample_raw="NEW"))
    assert not result.cells[0].adopted
    assert result.unassigned[0].reason == "unknown_or_ambiguous_sample"


def test_populated_baseline_cannot_be_overwritten_even_when_mislabelled_conflict(case):
    dataset = case["dataset"].model_copy(deep=True)
    row = next(r for r in dataset.quality_rows if r["field"] == "coating_thickness")
    row.update(value=9, decision="conflict")
    dataset.sample_rows[0]["coating_thickness"] = 9
    result = cell(case | {"dataset": dataset})
    assert not result.adopted and result.after == 9 and result.reason == "existing_value"


def test_multiple_visual_values_cannot_be_cherry_picked(case):
    old = case["evidence"].readings[0]
    raw = old.facts[0].raw
    reading = reading_for(old.candidate, (raw, raw.model_copy(update={"value_raw": "11"})))
    result = cell(case | {"evidence": case["evidence"].model_copy(update={"readings": (reading,)})})
    assert result.reason == "visual_disagreement" and not result.adopted


def test_duplicate_equivalent_observations_form_one_cell(case):
    old = case["evidence"].readings[0]
    raw = old.facts[0].raw
    reading = reading_for(old.candidate, (raw, raw.model_copy(update={"value_raw": "0.01", "unit_raw": "μm"})))
    result = cell(case | {"evidence": case["evidence"].model_copy(update={"readings": (reading,)})})
    assert result.adopted and result.after == 10 and len(result.evidence) == 2


@pytest.mark.parametrize("reason", ["multipage_source", "incomplete_source"])
def test_incomplete_context_refused(case, reason):
    old = case["evidence"].readings[0]
    candidate = old.candidate.model_copy(update={"reasons": (reason,)})
    reading = reading_for(candidate, (old.facts[0].raw,))
    result = cell(case | {"evidence": case["evidence"].model_copy(update={"readings": (reading,)})})
    assert result.reason == "incomplete_context" and not result.adopted


def test_low_confidence_identity_blocks_even_missing_cells(case):
    comparison = case["comparison"].model_copy(deep=True)
    matching = comparison.matchings["sample"]
    comparison.matchings["sample"] = matching.model_copy(
        update={"pairs": (matching.pairs[0].model_copy(update={"confidence": 0.1}),)}
    )
    assert cell(case | {"comparison": comparison}).reason == "sample_identity_blocked"


def test_raw_response_must_support_bound_fact(case):
    old = case["evidence"].readings[0]
    reading = old.model_copy(update={"attempts": (old.attempts[0].model_copy(update={"raw_response": "{}"}),)})
    result = cell(case | {"evidence": case["evidence"].model_copy(update={"readings": (reading,)})})
    assert result.reason == "invalid_reading_receipt" and not result.adopted


def test_stale_report_is_refused(case):
    evidence = case["evidence"].model_copy(update={"pdf_sha256": "f" * 64})
    with pytest.raises(ValueError, match="identity"):
        replay_document(**(case | {"evidence": evidence}))


def with_baseline(case, document, values, *, condition=None):
    lanes = tuple(
        lane.model_copy(
            update={
                "samples": (
                    SampleRecord(
                        sample_id="S1",
                        fields=()
                        if value is None
                        else (
                            make_field(
                                "coating_thickness",
                                str(value),
                                unit_raw="nm",
                                condition=condition,
                                source_ids=(f"{lane.backend}:p0:b0",),
                            ),
                        ),
                    ),
                )
            }
        )
        for lane, value in zip(case["lanes"], values, strict=True)
    )
    options = ComparisonOptions(
        profile=case["profile"], ambiguous_match_confidence=case["policy"].ambiguous_match_confidence
    )
    comparison = compare_lanes(*lanes, case["comparison"].matchings, options)
    dataset = consolidate_document(document, {lane.backend: lane for lane in lanes}, comparison, options).to_payload()
    evidence = case["evidence"].model_copy(
        update={
            "baseline_sha256": {
                lane.backend: hashlib.sha256(lane.model_dump_json().encode()).hexdigest() for lane in lanes
            }
        }
    )
    return case | dict(lanes=lanes, comparison=comparison, dataset=dataset, evidence=evidence)


def test_real_conflict_decision_can_be_corrected(case, document):
    result = cell(with_baseline(case, document, (9, 20)))
    assert result.before_status == "conflict" and result.adopted and result.after == 10


def test_condition_conflict_cannot_be_disguised_as_numeric_conflict(case, document):
    setup = with_baseline(case, document, (9, 20), condition="300 K")
    result = cell(with_raw(setup, condition="300 °C", condition_status="stated"))
    assert not result.adopted and result.reason == "condition_mismatch"


def test_missing_unit_cannot_assume_canonical_unit(case):
    result = cell(with_raw(case, unit_raw=None))
    assert not result.adopted


@pytest.mark.parametrize(
    "status", ["unanswered", "ambiguous", "unreviewed", "non_scalar", "multiple_conditions", "multiple_values"]
)
def test_other_empty_statuses_do_not_become_exact_values(case, status):
    dataset = case["dataset"].model_copy(deep=True)
    next(row for row in dataset.quality_rows if row["field"] == "coating_thickness")["decision"] = status
    assert not cell(case | {"dataset": dataset}).adopted


def test_genuine_existing_value_is_preserved(case, document):
    result = cell(with_baseline(case, document, (9, 9)))
    assert result.before == result.after == 9 and not result.adopted


def test_condition_embedded_in_number_must_not_be_dropped(case):
    result = cell(with_raw(case, value_raw="10 nm at 300 K", unit_raw=None))
    assert not result.adopted


def test_profile_range_is_enforced_after_unit_conversion(document):
    setup = make_case(document, make_profile({"fields.1.valid_range": {"min": 0, "max": 5}}))
    result = cell(with_raw(setup, value_raw="0.01", unit_raw="μm"))
    assert not result.adopted


def test_required_condition_cannot_be_not_stated(document):
    setup = make_case(
        document,
        make_profile(
            {
                "fields.1.condition_rule": "the substrate",
                "fields.1.condition_hint": "the substrate",
                "fields.1.missing_condition_note_zh": "未注明基底",
            }
        ),
    )
    assert cell(setup).reason == "unclear_condition"


def test_failed_question_blocks_missing_even_with_complete_visual_fact(case):
    from paperfacts.records import FailedQuestion

    lanes = (
        case["lanes"][0].model_copy(
            update={"failed_questions": (FailedQuestion(field="coating_thickness", detail="synthetic failure"),)}
        ),
        case["lanes"][1],
    )
    evidence = case["evidence"].model_copy(
        update={
            "baseline_sha256": {
                lane.backend: hashlib.sha256(lane.model_dump_json().encode()).hexdigest() for lane in lanes
            }
        }
    )
    assert cell(case | dict(lanes=lanes, evidence=evidence)).reason == "unanswered"


def test_policy_rejects_paper_and_nonnumeric_fields(case):
    for name in ("precursor_purity", "solvent", "unknown"):
        with pytest.raises(ValueError, match="scalar numeric sample field"):
            replay_document(**(case | {"policy": AdoptionPolicy(revision="test", fields=(name,))}))


@pytest.mark.parametrize("raw", ["10 +/- 2", "10 +- 2", r"10 \pm 2", "10 nm (+/- 2 nm)", "10 nm (+- 2 nm)"])
def test_every_supported_uncertainty_spelling_is_not_exact(case, raw):
    result = cell(with_raw(case, value_raw=raw))
    assert not result.adopted and result.reason == "non_exact_value"


def test_after_clause_is_not_silently_removed_before_adoption(document):
    setup = make_case(document, make_profile({"fields.1.after_clause": "condition"}))
    result = cell(with_raw(setup, value_raw="10 nm after annealing"))
    assert not result.adopted and result.reason == "embedded_condition"
