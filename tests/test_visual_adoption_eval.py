"""Synthetic frozen G2 replay: gold influences scores, never adoption decisions."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from paperfacts.keys import profile_comparison_fingerprint, profile_extraction_fingerprint
from paperfacts.profile_loader import load_profile
from support.profiles import profile_data
from test_visual_adoption import make_case

SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "visual_adoption.py"


@pytest.fixture
def case(document):
    return make_case(document)


@pytest.fixture
def evaluator():
    spec = importlib.util.spec_from_file_location("paperfacts_adoption_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def frozen(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def replay(tmp_path, case, document):
    path = tmp_path / "demo.json"
    path.write_text(json.dumps(profile_data()))
    profile = load_profile(path)
    lanes = tuple(
        lane.model_copy(update={"profile_fingerprint": profile_extraction_fingerprint(profile)})
        for lane in case["lanes"]
    )
    evidence = case["evidence"].model_copy(
        update={
            "profile_hash": profile.content_hash,
            "baseline_sha256": {
                lane.backend: hashlib.sha256(lane.model_dump_json().encode()).hexdigest() for lane in lanes
            },
        }
    )
    comparison = case["comparison"].model_copy(update={"profile_fingerprint": profile_comparison_fingerprint(profile)})
    dataset = case["dataset"].model_copy(update={"profile_fingerprint": profile_comparison_fingerprint(profile)})
    refs = {}
    for name, model in dict(
        dataset=dataset, comparison=comparison, evidence=evidence, lane_a=lanes[0], lane_b=lanes[1]
    ).items():
        target = tmp_path / f"{name}.json"
        target.write_text(model.model_dump_json())
        refs[name] = frozen(target)
    manifest = dict(
        format=1,
        evaluation_set="synthetic",
        gold_revision="synthetic-v1",
        profile=frozen(path),
        policy=case["policy"].model_dump(),
        cases=[dict(pdf=frozen(document.pdf_path), **refs)],
        gold=[
            dict(
                document_id=document.document_id,
                entity="sample",
                sample_id="S1",
                field="coating_thickness",
                value=10,
                unit="nm",
                condition=None,
                condition_status="not_stated",
                measurement_state="as_prepared",
                before_correct=False,
                observations=[dict(reading_index=0, fact_index=0, verdict="correct")],
            )
        ],
    )
    return manifest


def run(evaluator, tmp_path, replay):
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(replay))
    return evaluator.replay(path)


def test_replay_records_benefit_and_frozen_provenance(evaluator, tmp_path, replay):
    result = run(evaluator, tmp_path, replay)
    assert result["scientific_acceptance"] == "not_established"
    assert result["goals"]["shared_missing"]["improved"] == 1
    assert result["goals"]["shared_missing"]["adopted"] == 1
    assert result["goals"]["correction"]["metric_result"] == "insufficient"
    assert result["metric_result"] == "insufficient"
    assert result["audits"][0]["cells"][0]["after"] == 10
    assert result["frozen_files"] and result["code_sha256"]


def test_changing_gold_cannot_change_decision(evaluator, tmp_path, replay):
    first = run(evaluator, tmp_path, replay)
    replay["gold"][0]["value"] = 20
    replay["gold"][0]["observations"][0]["verdict"] = "wrong"
    second = run(evaluator, tmp_path, replay)
    assert first["audits"] == second["audits"]
    assert second["goals"]["shared_missing"]["incorrect_adoptions"] == 1
    assert second["metric_result"] == "fail"


@pytest.mark.parametrize("change", ["no_gold", "no_review", "uncertain"])
def test_no_complete_review_cannot_pass(evaluator, tmp_path, replay, change):
    if change == "no_gold":
        replay["gold"] = []
    elif change == "no_review":
        replay["gold"][0]["observations"] = []
    else:
        replay["gold"][0]["observations"][0]["verdict"] = "uncertain"
    result = run(evaluator, tmp_path, replay)
    assert result["goals"]["shared_missing"]["pending_review"] == 1
    assert result["metric_result"] == "needs_review"


def test_identical_number_with_wrong_full_tuple_is_error(evaluator, tmp_path, replay):
    replay["gold"][0]["observations"][0]["verdict"] = "wrong"
    result = run(evaluator, tmp_path, replay)
    assert result["goals"]["shared_missing"]["incorrect_adoptions"] == 1


def test_file_hash_drift_stops_before_scoring(evaluator, tmp_path, replay):
    replay["cases"][0]["dataset"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="SHA mismatch"):
        run(evaluator, tmp_path, replay)


def test_duplicate_document_cannot_inflate_counts(evaluator, tmp_path, replay):
    replay["cases"].append(copy.deepcopy(replay["cases"][0]))
    with pytest.raises(ValueError, match="duplicate document"):
        run(evaluator, tmp_path, replay)


def test_gold_wrong_unit_is_not_correct(evaluator, tmp_path, replay):
    replay["gold"][0]["unit"] = "kg"
    result = run(evaluator, tmp_path, replay)
    assert result["goals"]["shared_missing"]["incorrect_adoptions"] == 1


def test_schema_cli_needs_no_model_credentials():
    result = subprocess.run([sys.executable, str(SCRIPT), "--schema"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "cases" in json.loads(result.stdout)["properties"]


def test_known_wrong_observation_is_not_hidden_by_unreviewed_duplicate(evaluator, tmp_path, replay):
    from paperfacts.visual_evidence import VisualEvidenceReport
    from test_visual_adoption import reading_for

    ref = replay["cases"][0]["evidence"]
    path = Path(ref["path"])
    report = VisualEvidenceReport.model_validate_json(path.read_bytes())
    old = report.readings[0]
    reading = reading_for(old.candidate, (old.facts[0].raw, old.facts[0].raw))
    path.write_text(report.model_copy(update={"readings": (reading,)}).model_dump_json())
    replay["cases"][0]["evidence"] = frozen(path)
    replay["gold"][0]["observations"][0]["verdict"] = "wrong"
    result = run(evaluator, tmp_path, replay)
    assert result["goals"]["shared_missing"]["incorrect_adoptions"] == 1
    assert result["goals"]["shared_missing"]["pending_review"] == 1
    assert result["metric_result"] == "fail"


def test_both_goals_need_real_counterfactual_improvement(evaluator, tmp_path, replay, case, document):
    from paperfacts.compare import compare_lanes
    from paperfacts.dataset import consolidate_document
    from paperfacts.keys import ComparisonOptions
    from paperfacts.matching import SampleMatch
    from paperfacts.records import SampleRecord
    from test_visual_adoption import reading_for, with_baseline

    updated = with_baseline(case, document, (9, 20))
    lanes = tuple(
        lane.model_copy(update={"samples": (*lane.samples, SampleRecord(sample_id="S2"))}) for lane in updated["lanes"]
    )
    matching = updated["comparison"].matchings["sample"]
    matching = matching.model_copy(
        update={
            "pairs": (
                *matching.pairs,
                SampleMatch(a_id="S2", b_id="S2", confidence=1, method="exact", justification="exact"),
            )
        }
    )
    options = ComparisonOptions(profile=case["profile"], ambiguous_match_confidence=0.6)
    comparison = compare_lanes(*lanes, matching, options)
    dataset = consolidate_document(document, {lane.backend: lane for lane in lanes}, comparison, options).to_payload()
    old = updated["evidence"].readings[0]
    raw = old.facts[0].raw
    reading = reading_for(old.candidate, (raw, raw.model_copy(update={"sample_raw": "S2"})))
    evidence = updated["evidence"].model_copy(
        update={
            "readings": (reading,),
            "baseline_sha256": {
                lane.backend: hashlib.sha256(lane.model_dump_json().encode()).hexdigest() for lane in lanes
            },
        }
    )
    for name, model in dict(
        dataset=dataset, comparison=comparison, evidence=evidence, lane_a=lanes[0], lane_b=lanes[1]
    ).items():
        target = Path(replay["cases"][0][name]["path"])
        target.write_text(model.model_dump_json())
        replay["cases"][0][name] = frozen(target)
    replay["gold"].append(
        {
            **replay["gold"][0],
            "sample_id": "S2",
            "observations": [dict(reading_index=0, fact_index=1, verdict="correct")],
        }
    )
    result = run(evaluator, tmp_path, replay)
    assert result["metric_result"] == "pass"
    assert result["scientific_acceptance"] == "not_established"
    assert result["goals"]["correction"]["improved"] == result["goals"]["shared_missing"]["improved"] == 1
    assert all(rule["documents"] == 1 and rule["adopted"] == 1 for rule in result["rules"])


def test_zero_adoption_is_insufficient(evaluator, tmp_path, replay):
    from paperfacts.visual_evidence import VisualEvidenceReport

    ref = replay["cases"][0]["evidence"]
    path = Path(ref["path"])
    report = VisualEvidenceReport.model_validate_json(path.read_bytes())
    path.write_text(report.model_copy(update={"readings": ()}).model_dump_json())
    replay["cases"][0]["evidence"] = frozen(path)
    replay["gold"][0]["observations"] = []
    result = run(evaluator, tmp_path, replay)
    assert result["metric_result"] == "insufficient"
    assert result["goals"]["shared_missing"]["opportunities"] == 1
    assert result["goals"]["shared_missing"]["adopted"] == 0
