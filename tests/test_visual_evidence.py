"""Independent visual facts, tested with synthetic pages and a recording fake client."""

import json

import pytest
from pydantic import ValidationError

from paperfacts.compare import ComparisonReport
from paperfacts.crops import CropStore
from paperfacts.errors import LlmError, LlmOfflineMiss
from paperfacts.llm import LlmResult
from paperfacts.matching import SampleMatch, SampleMatching
from paperfacts.models import NormalizedBBox, sha256_of_file
from paperfacts.records import LaneExtraction, SampleRecord
from paperfacts.storage import DataLayout
from paperfacts.visual_candidates import CandidateSelection, VisualCandidate
from paperfacts.visual_evidence import parse_response, run_evidence, visual_prompt
from support.profiles import make_profile, make_reference_profile
from support.vision import FakeVisionClient


def answer(**changes):
    observation = dict(
        scope="sample",
        entity="sample",
        sample_raw="S1",
        field="coating_thickness",
        value_raw="10",
        unit_raw="nm",
        condition=None,
        condition_status="not_stated",
        evidence_type="printed",
        basis="Row S1, thickness column with unit nm",
    )
    observation.update(changes)
    return dict(outcome="observed", reason="", observations=[observation])


@pytest.fixture
def setup(document, tmp_path):
    profile = make_profile()
    comparison = ComparisonReport(
        document_id=document.document_id,
        extractor_key="ex",
        comparison_key="cp",
        backend_a="mineru",
        backend_b="paddleocr_vl",
        matchings={"sample": SampleMatching()},
    )
    lanes = tuple(
        LaneExtraction(
            document_id=document.document_id,
            backend=backend,
            extractor_key="ex",
            model="m",
            samples=(SampleRecord(sample_id="S1"),),
        )
        for backend in ("mineru", "paddleocr_vl")
    )
    box = NormalizedBBox(x1=0, y1=0, x2=1, y2=1)
    candidate = VisualCandidate(
        page=0, bbox=box, reasons=("native_keyword",), sources=("independent",), selected_by="independent"
    )
    selection = CandidateSelection(strategy="balanced", candidates=(candidate,), issues=(), unselected_pages=(1,))
    crops = CropStore(
        DataLayout(tmp_path / "data"),
        document.document_id,
        document.pdf_path,
        dpi=72,
        max_pixels=100000,
        source_pdf_sha256=sha256_of_file(document.pdf_path),
    )
    return dict(
        document=document, selection=selection, profile=profile, lanes=lanes, comparison=comparison, crops=crops
    )


def test_prompt_is_independent_and_full_fact_is_bound_to_actual_pixels(setup):
    client = FakeVisionClient(answer())
    report = run_evidence(**setup, client=client)
    fact = report.readings[0].facts[0]
    assert fact.raw.value_raw == "10"
    assert fact.normalized.value == 10
    assert not fact.normalized.grounded
    assert fact.mapping_status == "matched"
    assert report.pdf_sha256 == setup["document"].sha256
    assert report.readings[0].attempts[0].crop.image_sha256
    prompt = client.calls[0].system + client.calls[0].user
    assert setup["profile"].prompt.sample_definition in prompt
    assert setup["profile"].prompt.field_scope in prompt
    assert "S1" not in prompt
    assert "Row S1" not in prompt
    assert "source_ids" not in prompt


@pytest.mark.parametrize(
    "change",
    [
        {"field": "invented"},
        {"scope": "paper"},
        {"entity": "invented"},
        {"condition_status": "stated"},
        {"value_raw": 10},
        {"page": 999},
        {"basis": ""},
    ],
)
def test_invalid_facts_cannot_enter_report(change):
    with pytest.raises((ValueError, ValidationError)):
        parse_response(json.dumps(answer(**change)), make_profile())


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n{}\n```",
        '{"outcome":"no_facts","observations":[]}',
        '{"outcome":"observed","reason":"","observations":[]}',
    ],
)
def test_malformed_or_incomplete_answer_is_not_silently_repaired(raw):
    with pytest.raises((ValueError, ValidationError)):
        parse_response(raw, make_profile())


@pytest.mark.parametrize("raw", ["10–20", "~10", ">10", "approximately 10"])
def test_qualifiers_and_ranges_never_become_exact_scalar(setup, raw):
    report = run_evidence(**setup, client=FakeVisionClient(answer(value_raw=raw)))
    fact = report.readings[0].facts[0]
    assert fact.raw.value_raw == raw
    assert fact.normalized.value is None


def test_unseen_and_ambiguous_samples_stay_unmapped(setup):
    report = run_evidence(**setup, client=FakeVisionClient(answer(sample_raw="NEW")))
    assert report.readings[0].facts[0].mapping_status == "new"
    lane = setup["lanes"][0]
    setup["lanes"] = (
        lane.model_copy(update={"samples": (SampleRecord(sample_id="S1"), SampleRecord(sample_id="S 1"))}),
    )
    report = run_evidence(**setup, client=FakeVisionClient(answer()))
    assert report.readings[0].facts[0].mapping_status == "ambiguous"


def test_one_failed_candidate_does_not_hide_the_next(setup):
    candidate = setup["selection"].candidates[0]
    setup["selection"] = setup["selection"].model_copy(
        update={"candidates": (candidate, candidate.model_copy(update={"page": 1}))}
    )
    replies = iter([LlmError("fake error"), answer()])
    client = FakeVisionClient(lambda *_: next(replies))
    report = run_evidence(**setup, client=client)
    assert [reading.status for reading in report.readings] == ["error", "observed"]
    assert report.logical_requests == 2


def test_offline_cache_miss_is_not_a_model_abstention(setup):
    with pytest.raises(LlmOfflineMiss):
        run_evidence(**setup, client=FakeVisionClient(LlmOfflineMiss("no cached response")))


def test_cached_answer_is_validated_and_tokens_are_not_new_usage(setup, monkeypatch):
    client = FakeVisionClient(answer())
    monkeypatch.setattr(
        client,
        "complete_vision",
        lambda **kwargs: LlmResult(text=json.dumps(answer()), usage={"total_tokens": 10}, cached=True),
    )
    report = run_evidence(**setup, client=client)
    assert report.cached_requests == 1
    assert report.usage == {"total_tokens": 10}
    assert report.uncached_usage == {}
    monkeypatch.setattr(
        client,
        "complete_vision",
        lambda **kwargs: LlmResult(text=json.dumps(answer(field="invented")), usage={"total_tokens": 10}, cached=True),
    )
    report = run_evidence(**setup, client=client)
    assert report.readings[0].status == "error"
    assert report.readings[0].facts == ()


def test_only_one_known_zoom_and_no_model_generated_coordinates(setup):
    candidate = setup["selection"].candidates[0]
    zoom = NormalizedBBox(x1=0.1, y1=0.1, x2=0.8, y2=0.5)
    setup["selection"] = setup["selection"].model_copy(
        update={"candidates": (candidate.model_copy(update={"zoom_boxes": (zoom,)}),)}
    )
    client = FakeVisionClient(dict(outcome="needs_zoom", reason="unreadable small print", observations=[]))
    report = run_evidence(**setup, client=client)
    assert len(client.calls) == 2
    assert report.readings[0].status == "needs_zoom"
    assert len(report.readings[0].attempts) == 2
    assert report.readings[0].attempts[1].crop.bbox == zoom


def test_zoom_without_complete_known_region_abstains(setup):
    client = FakeVisionClient(dict(outcome="needs_zoom", reason="small print", observations=[]))
    report = run_evidence(**setup, client=client)
    assert len(client.calls) == 1
    assert report.readings[0].status == "needs_zoom"


def test_si_identity_remains_separate_from_pdf_digest(setup):
    document = setup["document"].model_copy(update={"document_id": "b" * 64, "sha256": "b" * 64})
    setup["document"] = document
    setup["comparison"] = setup["comparison"].model_copy(update={"document_id": document.document_id})
    setup["lanes"] = tuple(lane.model_copy(update={"document_id": document.document_id}) for lane in setup["lanes"])
    setup["crops"] = CropStore(
        setup["crops"].layout,
        document.document_id,
        document.pdf_path,
        dpi=72,
        source_pdf_sha256=sha256_of_file(document.pdf_path),
    )
    report = run_evidence(**setup, client=FakeVisionClient(answer()))
    assert report.document_id == "b" * 64
    assert report.pdf_sha256 != report.document_id


def test_repeated_facts_are_not_silently_lost_across_evidence(setup):
    candidate = setup["selection"].candidates[0]
    setup["selection"] = setup["selection"].model_copy(
        update={"candidates": (candidate, candidate.model_copy(update={"page": 1}))}
    )
    report = run_evidence(**setup, client=FakeVisionClient(answer()))
    assert len(report.readings) == 2
    assert all(len(reading.facts) == 1 for reading in report.readings)


def test_wrong_document_fails_before_any_model_call(setup):
    setup["comparison"] = setup["comparison"].model_copy(update={"document_id": "c" * 64})
    client = FakeVisionClient(answer())
    with pytest.raises(ValueError, match="document"):
        run_evidence(**setup, client=client)
    assert client.calls == []


def test_stale_pdf_crop_identity_is_refused_before_reading(setup):
    setup["crops"].source_pdf_sha256 = "f" * 64
    client = FakeVisionClient(answer())
    with pytest.raises(ValueError, match="PDF"):
        run_evidence(**setup, client=client)
    assert client.calls == []


def test_reference_profile_never_asks_for_an_absent_inventory():
    prompt = visual_prompt(make_reference_profile(), "review")
    assert "from the list" not in prompt
    assert "visible object label" in prompt


@pytest.mark.parametrize("raw,expected", [("ten", 10), ("above 10", None), ("one monolayer", 1)])
def test_normalization_keeps_exact_named_values_but_not_bounds(setup, raw, expected):
    setup["profile"] = make_profile({"fields.1.named_values": {"one monolayer": 1}})
    result = run_evidence(**setup, client=FakeVisionClient(answer(value_raw=raw)))
    assert result.readings[0].facts[0].normalized.value == expected


def test_partial_matching_failure_keeps_existing_exact_pairs(setup):
    pair = SampleMatch(a_id="S1", b_id="S1", method="exact", confidence=1, justification="exact identity")
    setup["comparison"] = setup["comparison"].model_copy(
        update={"matchings": {"sample": SampleMatching(pairs=(pair,), failed=True, failure="unmatched remainder")}}
    )
    report = run_evidence(**setup, client=FakeVisionClient(answer()))
    assert report.readings[0].facts[0].scope == "sample:S1|S1"


def test_pdfium_page_error_is_isolated(setup, monkeypatch):
    from pypdfium2 import PdfiumError

    candidate = setup["selection"].candidates[0]
    setup["selection"] = setup["selection"].model_copy(
        update={"candidates": (candidate, candidate.model_copy(update={"page": 1}))}
    )
    original = setup["crops"].crop

    def crop(region):
        if region.page == 0:
            raise PdfiumError("bad page")
        return original(region)

    monkeypatch.setattr(setup["crops"], "crop", crop)
    report = run_evidence(**setup, client=FakeVisionClient(answer()))
    assert [r.status for r in report.readings] == ["error", "observed"]
    assert report.logical_requests == 1
