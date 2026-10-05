"""Synthetic experiment copies: never alter A/B, never retain stale C values."""

import hashlib

import pytest
from openpyxl import load_workbook

from paperfacts.dataset import DocumentDataset
from paperfacts.visual_snapshot import ExperimentSnapshot, build_snapshot
from paperfacts.workbook import write_dataset
from test_visual_adoption import make_case, with_raw


@pytest.fixture
def case(document):
    return make_case(document)


def build(case, document, **changes):
    args = {key: case[key] for key in ("lanes", "comparison", "evidence", "profile", "policy", "pdf_sha256")}
    args.update(
        document=document,
        expected_model="fake",
        expected_prompt_sha256=case["evidence"].prompt_sha256,
        expected_strategy="balanced",
    )
    args.update(changes)
    return build_snapshot(**args)


def test_copy_rebuilds_derived_rows_and_roundtrips_into_workbook(case, document, tmp_path):
    before = [lane.model_dump_json() for lane in case["lanes"]]
    snapshot = build(case, document)
    assert snapshot.dataset.sample_rows[0]["coating_thickness"] == 10
    assert snapshot.dataset.sample_rows[0]["available_fields"] == 1
    assert snapshot.dataset.sample_rows[0]["agree_fields"] == 0
    assert snapshot.dataset.paper_row == snapshot.dataset.sample_rows[0]
    quality = next(row for row in snapshot.dataset.quality_rows if row["field"] == "coating_thickness")
    assert quality["decision"] == "adopted_from_visual" and "visual:" in quality["source_ids"]
    assert [lane.model_dump_json() for lane in case["lanes"]] == before
    path = tmp_path / "snapshot.json"
    snapshot.write(path)
    loaded = ExperimentSnapshot.model_validate_json(path.read_bytes())
    assert loaded == snapshot
    first, second = tmp_path / "first.xlsx", tmp_path / "second.xlsx"
    for snap, target in ((snapshot, first), (loaded, second)):
        write_dataset([DocumentDataset.from_payload(snap.dataset)], target, case["profile"])

    def rows(path):
        book = load_workbook(path, data_only=True)
        try:
            return {s.title: list(s.values) for s in book}
        finally:
            book.close()

    assert rows(first) == rows(second)
    assert any("adopted_from_visual" in row for row in rows(second)["数据质量"])


@pytest.mark.parametrize(
    "change",
    [
        {"expected_model": "new-model"},
        {"expected_prompt_sha256": "f" * 64},
        {"expected_strategy": "risk_only"},
        {"pdf_sha256": "f" * 64},
        {"evidence": None},
    ],
)
def test_stale_or_absent_evidence_rebuilds_original_baseline(case, document, change):
    assert build(case, document).dataset.paper_row["coating_thickness"] == 10
    stale = build(case, document, **change)
    assert stale.dataset == case["dataset"]
    assert stale.audit is None and stale.report_status in ("stale", "missing")


def test_rules_changed_recompute_and_do_not_reuse_previous_adoption(case, document):
    good = build(case, document)
    refused = build(with_raw(case, value_raw="~10"), document)
    assert good.dataset.paper_row["coating_thickness"] == 10
    assert refused.dataset.paper_row["coating_thickness"] is None
    quality = next(row for row in refused.dataset.quality_rows if row["field"] == "coating_thickness")
    assert "non_exact_value" in quality["detail"]
    assert refused.audit.cells[0].reason == "non_exact_value"
    assert good.fingerprint != refused.fingerprint


def test_experiment_dataset_cannot_be_passed_back_as_baseline(case, document):
    from paperfacts.visual_adoption import replay_document

    derived = build(case, document)
    with pytest.raises(ValueError, match="identity"):
        replay_document(**(case | {"dataset": derived.dataset}))


def test_snapshot_writes_never_overwrite_existing_result(case, document, tmp_path):
    path = tmp_path / "snapshot.json"
    build(case, document).write(path)
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        build(case, document).write(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original


def test_adoption_reselects_whole_sample_and_updates_conditions(case, document):
    from paperfacts.compare import compare_lanes
    from paperfacts.keys import ComparisonOptions
    from paperfacts.matching import SampleMatch
    from paperfacts.records import SampleRecord
    from test_visual_adoption import reading_for

    # Baseline ties choose A; only C discovers a value for S1, so S1 must become the paper row.
    lanes = tuple(
        lane.model_copy(update={"samples": (SampleRecord(sample_id="A"), *lane.samples)}) for lane in case["lanes"]
    )
    matching = case["comparison"].matchings["sample"]
    matching = matching.model_copy(
        update={
            "pairs": (
                SampleMatch(a_id="A", b_id="A", confidence=1, method="exact", justification="synthetic"),
                *matching.pairs,
            )
        }
    )
    comparison = compare_lanes(
        *lanes, matching, ComparisonOptions(profile=case["profile"], ambiguous_match_confidence=0.6)
    )
    old = case["evidence"].readings[0]
    raw = old.facts[0].raw.model_copy(update={"condition": "as deposited", "condition_status": "stated"})
    evidence = case["evidence"].model_copy(
        update={
            "readings": (reading_for(old.candidate, (raw,)),),
            "baseline_sha256": {
                lane.backend: hashlib.sha256(lane.model_dump_json().encode()).hexdigest() for lane in lanes
            },
        }
    )
    result = build(case, document, lanes=lanes, comparison=comparison, evidence=evidence)
    assert result.dataset.paper_row["sample_id"] == "S1"
    assert result.dataset.paper_row["conditions"] == "coating_thickness: as deposited"
    selection = next(r for r in result.dataset.quality_rows if r["field"] == "__selection__")
    assert selection["sample_id"] == "S1"
    assert next(r for r in result.dataset.sample_rows if r["sample_id"] == "A")["coating_thickness"] is None


def test_same_named_entities_and_shared_paper_fields_keep_separate_rows(case, document):
    from paperfacts.compare import compare_lanes
    from paperfacts.dataset import consolidate_document
    from paperfacts.keys import ComparisonOptions, profile_extraction_fingerprint
    from paperfacts.records import PaperRecord, SampleRecord
    from support.extraction import make_field
    from support.profiles import make_entity_profile
    from test_visual_adoption import reading_for

    profile = make_entity_profile()
    lanes = tuple(
        lane.model_copy(
            update={
                "profile_fingerprint": profile_extraction_fingerprint(profile),
                "paper": PaperRecord(
                    fields=(make_field("precursor_purity", "99", unit_raw="%", source_ids=(f"{lane.backend}_p0_b0",)),)
                ),
                "samples": (
                    SampleRecord(sample_id="S1", entity="coating"),
                    SampleRecord(sample_id="S1", entity="wear_test"),
                ),
            }
        )
        for lane in case["lanes"]
    )
    matching = case["comparison"].matchings["sample"]
    options = ComparisonOptions(profile=profile, ambiguous_match_confidence=0.6)
    comparison = compare_lanes(*lanes, {name: matching for name in ("coating", "wear_test")}, options)
    old = case["evidence"].readings[0]
    raw = old.facts[0].raw.model_copy(update={"entity": "coating"})
    evidence = case["evidence"].model_copy(
        update={
            "profile_hash": profile.content_hash,
            "readings": (reading_for(old.candidate, (raw,)),),
            "baseline_sha256": {
                lane.backend: hashlib.sha256(lane.model_dump_json().encode()).hexdigest() for lane in lanes
            },
        }
    )
    baseline = consolidate_document(document, {lane.backend: lane for lane in lanes}, comparison, options).to_payload()
    result = build(case, document, lanes=lanes, comparison=comparison, evidence=evidence, profile=profile)
    secondary = next(row for row in result.dataset.sample_rows if row["entity"] == "wear_test")
    assert secondary == next(row for row in baseline.sample_rows if row["entity"] == "wear_test")
    assert result.dataset.paper_row["entity"] == "coating"
    assert result.dataset.paper_row["available_fields"] == 2
    assert result.dataset.paper_row["agree_fields"] == 1
    assert result.dataset.paper_row["precursor_purity"] == 99


def test_invalid_policy_is_configuration_error_even_without_report(case, document):
    from paperfacts.visual_adoption import AdoptionPolicy

    with pytest.raises(ValueError, match="scalar numeric sample field"):
        build(case, document, evidence=None, policy=AdoptionPolicy(revision="invalid", fields=("unknown",)))
