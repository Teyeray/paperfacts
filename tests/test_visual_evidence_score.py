"""Reviewed-scope scoring contracts; every PDF, lane, report and label here is synthetic."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from paperfacts.models import NormalizedBBox
from paperfacts.records import LaneExtraction
from paperfacts.visual_candidates import CandidateSelection, VisualCandidate
from paperfacts.visual_evidence import BoundObservation, CandidateReading, VisualEvidenceReport, VisualObservation
from support.extraction import make_field, make_lane
from support.factories import make_blank_pdf

SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "visual_evidence_score.py"


@pytest.fixture
def scorer():
    spec = importlib.util.spec_from_file_location("paperfacts_visual_score", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def frozen(path: Path):
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def observation(key, **changes):
    value = dict(
        scope="sample",
        entity="sample",
        sample_raw=f"S-{key}",
        field="coating_thickness",
        value_raw="10",
        unit_raw="nm",
        condition=None,
        condition_status="not_stated",
        evidence_type="printed",
        basis="Printed row and unit",
    )
    value.update(changes)
    return VisualObservation(**value)


def write_report(path, document_id, pdf_sha256, observations, strategy="balanced"):
    candidate = VisualCandidate(
        page=0, bbox=NormalizedBBox(x1=0, y1=0, x2=1, y2=1), reasons=("native_keyword",), sources=("independent",)
    )
    selection = CandidateSelection(candidates=(candidate,), issues=(), unselected_pages=(), strategy=strategy)
    facts = tuple(
        BoundObservation(
            raw=raw, normalized=make_field(raw.field, raw.value_raw, unit_raw=raw.unit_raw), mapping_status="new"
        )
        for raw in observations
    )
    report = VisualEvidenceReport(
        document_id=document_id,
        pdf_sha256=pdf_sha256,
        profile_hash="synthetic",
        model="synthetic",
        strategy=strategy,
        prompt_sha256="a" * 64,
        selection=selection,
        readings=(CandidateReading(candidate=candidate, status="observed" if facts else "no_facts", facts=facts),),
        logical_requests=1,
        usage={},
        seconds=0,
        baseline_sha256={
            backend: hashlib.sha256(
                LaneExtraction.model_validate_json((path.parent / f"{backend}.json").read_bytes())
                .model_dump_json()
                .encode()
            ).hexdigest()
            for backend in ("mineru", "paddleocr_vl")
        },
    )
    report.write(path)
    return frozen(path)


@pytest.fixture
def case(tmp_path):
    pdf_path = make_blank_pdf(tmp_path / "synthetic.pdf", ((100, 100),))
    pdf_ref = frozen(pdf_path)
    document_id = pdf_ref["sha256"]
    baselines = []
    for backend in ("mineru", "paddleocr_vl"):
        path = tmp_path / f"{backend}.json"
        make_lane(backend=backend, document_id=document_id).write(path)
        baselines.append({"document_id": document_id, "backend": backend, **frozen(path)})
    facts = []
    for key, baseline in (("fix", "correction_opportunity"), ("miss", "shared_missing")):
        facts.append(
            dict(
                key=key,
                document_id=document_id,
                scope="sample",
                entity="sample",
                sample=f"S-{key}",
                field="coating_thickness",
                condition=None,
                condition_status="not_stated",
                measurement_state="as_prepared",
                value_raw="10",
                unit_raw="nm",
                baseline=baseline,
                source={"pdf": pdf_ref, "page": 0, "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}},
            )
        )
    report_path = tmp_path / "balanced.json"
    report_ref = write_report(report_path, document_id, pdf_ref["sha256"], (observation("fix"), observation("miss")))
    return dict(
        format=1,
        review_version="review-v1",
        gold_revision="gold-synthetic-v1",
        baseline_sources=baselines,
        reports=[report_ref],
        facts=facts,
        observations=[
            dict(
                report_path=report_path.name,
                reading_index=0,
                fact_index=index,
                fact_key=key,
                verdict="correct",
                reason="Human synthetic full-tuple check",
            )
            for index, key in enumerate(("fix", "miss"))
        ],
    )


def score(scorer, tmp_path, case):
    path = tmp_path / "review.json"
    path.write_text(json.dumps(case))
    return scorer.score_review(path)


def test_complete_review_passes_both_goals_and_records_review_provenance(scorer, tmp_path, case):
    result = score(scorer, tmp_path, case)
    strategy = result["strategies"]["balanced"]

    assert result["scope"] == "reviewed-scope"
    assert result["scientific_acceptance"] == "not_established_by_scorer"
    assert result["evaluation_set"] == "synthetic"
    assert result["review"]["sha256"] == frozen(tmp_path / "review.json")["sha256"]
    assert result["review_version"] == "review-v1"
    assert result["gold_revision"] == "gold-synthetic-v1"
    assert strategy["correction"] == {"opportunities": 1, "correct": 1, "wrong": 0, "net": 1}
    assert strategy["shared_missing"] == {"opportunities": 1, "correct": 1, "wrong": 0, "net": 1}
    assert strategy["metric_result"] == "pass"
    assert result["g2"] == "not_evaluated"
    assert result["observations"][0]["observation"]["raw"]["unit_raw"] == "nm"


def test_repeated_correct_observations_do_not_increase_gain(scorer, tmp_path, case):
    case["reports"][0] = write_report(
        tmp_path / "balanced.json",
        case["facts"][0]["document_id"],
        case["facts"][0]["source"]["pdf"]["sha256"],
        (observation("fix"), observation("miss"), observation("fix")),
    )
    case["observations"].append({**case["observations"][0], "fact_index": 2})

    result = score(scorer, tmp_path, case)

    assert result["strategies"]["balanced"]["correction"]["correct"] == 1
    assert len(result["observations"]) == 3


def test_unselected_shared_missing_fact_remains_in_opportunity_denominator(scorer, tmp_path, case):
    case["reports"][0] = write_report(
        tmp_path / "balanced.json",
        case["facts"][0]["document_id"],
        case["facts"][0]["source"]["pdf"]["sha256"],
        (observation("fix"),),
    )
    case["observations"] = case["observations"][:1]

    result = score(scorer, tmp_path, case)["strategies"]["balanced"]

    assert result["shared_missing"] == {"opportunities": 1, "correct": 0, "wrong": 0, "net": 0}
    assert result["metric_result"] == "fail"


def test_unreviewed_observation_is_pending_not_an_extra_or_a_gain(scorer, tmp_path, case):
    case["observations"] = case["observations"][:1]

    result = score(scorer, tmp_path, case)

    assert result["strategies"]["balanced"]["pending_review"] == 1
    assert result["strategies"]["balanced"]["shared_missing"]["net"] == 0
    assert result["strategies"]["balanced"]["metric_result"] == "needs_review"
    assert result["observations"][1]["verdict"] == "pending_review"
    assert "extra" not in result["strategies"]["balanced"]


@pytest.mark.parametrize("changes", [{"unit_raw": "mm"}, {"sample_raw": "Other sample"}])
def test_human_wrong_unit_or_attribution_is_penalized_without_numeric_matching(scorer, tmp_path, case, changes):
    case["reports"][0] = write_report(
        tmp_path / "balanced.json",
        case["facts"][0]["document_id"],
        case["facts"][0]["source"]["pdf"]["sha256"],
        (observation("fix", **changes), observation("miss")),
    )
    case["observations"][0]["verdict"] = "wrong"

    result = score(scorer, tmp_path, case)["strategies"]["balanced"]

    assert result["correction"] == {"opportunities": 1, "correct": 0, "wrong": 1, "net": -1}
    assert result["shared_missing"]["net"] == 1
    assert result["metric_result"] == "fail"


def test_same_fact_correct_and_wrong_keeps_the_error_penalty(scorer, tmp_path, case):
    case["reports"][0] = write_report(
        tmp_path / "balanced.json",
        case["facts"][0]["document_id"],
        case["facts"][0]["source"]["pdf"]["sha256"],
        (observation("fix"), observation("miss"), observation("fix", unit_raw="mm")),
    )
    case["observations"].append({**case["observations"][0], "fact_index": 2, "verdict": "wrong"})

    result = score(scorer, tmp_path, case)["strategies"]["balanced"]

    assert result["correction"] == {"opportunities": 1, "correct": 1, "wrong": 1, "net": 0}
    assert result["metric_result"] == "fail"


def test_wrong_on_already_correct_fact_penalizes_correction_goal(scorer, tmp_path, case):
    kept = copy.deepcopy(case["facts"][0])
    kept.update(key="kept", sample="S-kept", baseline="already_correct")
    case["facts"].append(kept)
    case["reports"][0] = write_report(
        tmp_path / "balanced.json",
        kept["document_id"],
        kept["source"]["pdf"]["sha256"],
        (observation("fix"), observation("miss"), observation("kept", unit_raw="mm")),
    )
    case["observations"].append({**case["observations"][0], "fact_index": 2, "fact_key": "kept", "verdict": "wrong"})

    result = score(scorer, tmp_path, case)["strategies"]["balanced"]

    assert result["correction"] == {"opportunities": 1, "correct": 1, "wrong": 1, "net": 0}
    assert result["metric_result"] == "fail"


@pytest.mark.parametrize("where", ["baseline", "observation"])
def test_uncertain_review_prevents_pass(scorer, tmp_path, case, where):
    if where == "baseline":
        case["facts"][1]["baseline"] = "uncertain"
    else:
        case["observations"][1]["verdict"] = "uncertain"

    result = score(scorer, tmp_path, case)["strategies"]["balanced"]

    assert result["uncertain"] == 1
    assert result["metric_result"] == "needs_review"


def test_zero_opportunities_cannot_pass(scorer, tmp_path, case):
    case["facts"][1]["baseline"] = "already_correct"

    assert score(scorer, tmp_path, case)["strategies"]["balanced"]["metric_result"] == "insufficient"


@pytest.mark.parametrize("reference", ["report", "baseline", "pdf"])
def test_file_drift_is_refused_before_scoring(scorer, tmp_path, case, reference):
    ref = {
        "report": case["reports"][0],
        "baseline": case["baseline_sources"][0],
        "pdf": case["facts"][0]["source"]["pdf"],
    }[reference]
    with (tmp_path / ref["path"]).open("ab") as handle:
        handle.write(b"\nchanged")

    with pytest.raises(ValueError, match="SHA"):
        score(scorer, tmp_path, case)


def test_same_observation_cannot_be_reviewed_twice(scorer, tmp_path, case):
    case["observations"].append(dict(case["observations"][0]))

    with pytest.raises(ValueError, match="duplicate observation"):
        score(scorer, tmp_path, case)


@pytest.mark.parametrize("change", [{"unit_raw": "mm"}, {"key": "alias"}])
def test_duplicate_or_conflicting_canonical_fact_definitions_are_refused(scorer, tmp_path, case, change):
    case["facts"].append({**case["facts"][0], **change})

    with pytest.raises(ValueError, match="fact"):
        score(scorer, tmp_path, case)


def test_both_complete_baseline_lanes_are_required(scorer, tmp_path, case):
    case["baseline_sources"].pop()

    with pytest.raises(ValueError, match="baseline"):
        score(scorer, tmp_path, case)


def test_each_strategy_scores_the_same_reviewed_scope_independently(scorer, tmp_path, case):
    ref = write_report(
        tmp_path / "risk_only.json",
        case["facts"][0]["document_id"],
        case["facts"][0]["source"]["pdf"]["sha256"],
        (observation("fix"),),
        strategy="risk_only",
    )
    case["reports"].append(ref)
    case["observations"].append({**case["observations"][0], "report_path": "risk_only.json"})

    strategies = score(scorer, tmp_path, case)["strategies"]

    assert strategies["balanced"]["metric_result"] == "pass"
    assert strategies["risk_only"]["metric_result"] == "fail"
    assert strategies["risk_only"]["shared_missing"]["opportunities"] == 1


def test_cli_outputs_only_a_json_score_and_identifies_the_review_version(scorer, tmp_path, case):
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(case))

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--review", str(review_path)], capture_output=True, text=True, check=True
    )

    assert json.loads(completed.stdout)["review_version"] == "review-v1"
    assert completed.stderr == ""


@pytest.mark.parametrize("goal", ["correction", "shared_missing"])
def test_explicit_excluded_error_penalizes_its_reviewed_goal_without_creating_an_opportunity(
    scorer, tmp_path, case, goal
):
    excluded = copy.deepcopy(case["facts"][0])
    excluded.update(
        key="excluded",
        sample="Not this paper",
        baseline="excluded",
        value_raw=None,
        reason="Cited literature sample is outside this paper's scope",
        penalty_goal=goal,
    )
    case["facts"].append(excluded)
    case["reports"][0] = write_report(
        tmp_path / "balanced.json",
        excluded["document_id"],
        excluded["source"]["pdf"]["sha256"],
        (observation("fix"), observation("miss"), observation("excluded")),
    )
    case["observations"].append(
        {**case["observations"][0], "fact_index": 2, "fact_key": "excluded", "verdict": "wrong"}
    )

    result = score(scorer, tmp_path, case)["strategies"]["balanced"]

    assert result[goal] == {"opportunities": 1, "correct": 1, "wrong": 1, "net": 0}
    assert result["metric_result"] == "fail"


def test_different_measurement_conditions_are_distinct_facts_even_with_equal_values(scorer, tmp_path, case):
    second = copy.deepcopy(case["facts"][1])
    second.update(key="miss_other_state", condition="after treatment", condition_status="stated")
    case["facts"].append(second)

    result = score(scorer, tmp_path, case)["strategies"]["balanced"]

    assert result["shared_missing"]["opportunities"] == 2
    assert result["shared_missing"]["correct"] == 1


@pytest.mark.parametrize("field,value", [("fact_index", 99), ("reading_index", 99), ("report_path", "not_frozen.json")])
def test_observation_pointer_must_resolve_to_the_frozen_complete_tuple(scorer, tmp_path, case, field, value):
    case["observations"][0][field] = value

    with pytest.raises(ValueError, match="pointer"):
        score(scorer, tmp_path, case)


def test_report_with_no_observation_cannot_pass_and_keeps_its_frozen_report_identity(scorer, tmp_path, case):
    fact = case["facts"][0]
    ref = write_report(tmp_path / "balanced.json", fact["document_id"], fact["source"]["pdf"]["sha256"], ())
    case["reports"] = [ref]
    case["observations"] = []

    result = score(scorer, tmp_path, case)

    assert result["strategies"]["balanced"]["metric_result"] == "fail"
    assert result["strategies"]["balanced"]["shared_missing"]["opportunities"] == 1
    assert result["reports"] == [
        {
            "path": str(tmp_path / "balanced.json"),
            "sha256": ref["sha256"],
            "document_id": fact["document_id"],
            "strategy": "balanced",
        }
    ]


def test_review_cannot_replace_a_report_baseline_even_with_an_updated_file_sha(scorer, tmp_path, case):
    reference = case["baseline_sources"][0]
    path = tmp_path / reference["path"]
    lane = LaneExtraction.model_validate_json(path.read_bytes())
    lane.model_copy(update={"no_samples": True}).write(path)
    reference["sha256"] = frozen(path)["sha256"]

    with pytest.raises(ValueError, match=r"baseline.*report"):
        score(scorer, tmp_path, case)


def test_report_without_frozen_baseline_digests_is_not_scoreable(scorer, tmp_path, case):
    path = tmp_path / case["reports"][0]["path"]
    data = json.loads(path.read_text())
    data.pop("baseline_sha256")
    path.write_text(json.dumps(data))
    case["reports"][0] = frozen(path)

    with pytest.raises(ValueError, match=r"baseline.*report"):
        score(scorer, tmp_path, case)


def test_incomplete_baseline_is_rejected_even_when_the_report_digest_matches(scorer, tmp_path, case):
    reference = case["baseline_sources"][0]
    path = tmp_path / reference["path"]
    data = json.loads(path.read_text())
    data["failed_questions"] = [{"field": "coating_thickness", "detail": "synthetic failed question"}]
    path.write_text(json.dumps(data))
    reference["sha256"] = frozen(path)["sha256"]
    fact = case["facts"][0]
    case["reports"][0] = write_report(
        tmp_path / "balanced.json",
        fact["document_id"],
        fact["source"]["pdf"]["sha256"],
        (observation("fix"), observation("miss")),
    )

    with pytest.raises(ValueError, match="incomplete baseline"):
        score(scorer, tmp_path, case)


def test_reviewed_source_page_must_exist_in_the_frozen_pdf(scorer, tmp_path, case):
    case["facts"][0]["source"]["page"] = 9999

    with pytest.raises(ValueError, match="source page"):
        score(scorer, tmp_path, case)


def test_frozen_source_must_be_a_readable_pdf_not_just_matching_bytes(scorer, tmp_path, case):
    pdf = tmp_path / case["facts"][0]["source"]["pdf"]["path"]
    pdf.write_bytes(b"not a pdf")
    for fact in case["facts"]:
        fact["source"]["pdf"] = frozen(pdf)
    report_path = tmp_path / case["reports"][0]["path"]
    report = json.loads(report_path.read_text())
    report["pdf_sha256"] = frozen(pdf)["sha256"]
    report_path.write_text(json.dumps(report))
    case["reports"][0] = frozen(report_path)

    with pytest.raises(ValueError, match="source PDF"):
        score(scorer, tmp_path, case)


def test_failed_reading_cannot_contribute_facts_even_with_human_correct_labels(scorer, tmp_path, case):
    path = tmp_path / case["reports"][0]["path"]
    data = json.loads(path.read_text())
    data["readings"][0]["status"] = "error"
    path.write_text(json.dumps(data))
    case["reports"][0] = frozen(path)

    with pytest.raises(ValueError, match="reading status"):
        score(scorer, tmp_path, case)
