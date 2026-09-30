"""The supervisor: which comparisons it looks at, how a score becomes a verdict, and what a verdict does to a cell.

Every model call goes through :class:`support.llm.FakeLlmClient`; no request leaves the process.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from paperfacts.compare import FieldComparison, Supervision, compare_lanes
from paperfacts.config import ConfigError, Settings
from paperfacts.dataset import consolidate_document, incomplete_reason
from paperfacts.keys import ComparisonOptions, SupervisorOptions, comparison_key
from paperfacts.matching import SampleMatch, SampleMatching
from paperfacts.models import DocumentInput
from paperfacts.records import FieldValue
from paperfacts.supervisor import (
    MAX_PASSAGE_CHARS,
    carry_supervision,
    is_borderline,
    passage_for,
    sample_of,
    score_value,
    selection_reason,
    supervise_report,
    supervision_summary,
)
from paperfacts.workflow import build_supervisor_client, supervise_document
from support.extraction import comparison_options, make_artifact, make_lane, make_sample
from support.factories import DOC_ID, make_block
from support.llm import FakeLlmClient
from support.profiles import shipped_profile

OPTIONS = SupervisorOptions(model="judge", min_confidence=0.6, vote_threshold=0.6)
# A wider band, so a score can be neither trusted nor doubted.
BANDED = SupervisorOptions(model="judge", min_confidence=0.5, vote_threshold=0.8)
BLOCKS = {
    "mineru": {"mineru_p0_b1": "The film showed a transmittance of 83.5 % at 550 nm."},
    "paddleocr_vl": {"paddleocr_vl_p0_b1": "The film showed a transmittance of 83.5 % at 550 nm."},
}


def reply(score: float, flag: str = "correct", critique: str = "") -> str:
    return json.dumps({"score": score, "flag": flag, "critique": critique})


def value(name: str, raw: str, unit: str | None = None, *, backend: str = "mineru", **kwargs) -> FieldValue:
    return FieldValue(field=name, value_raw=raw, unit_raw=unit, source_ids=(f"{backend}_p0_b1",), **kwargs)


def spec(name: str):
    return shipped_profile().by_name[name]


def comparison(status: str, a: FieldValue | None, b: FieldValue | None, **kwargs) -> FieldComparison:
    return FieldComparison(scope="sample:A|A", field=(a or b).field, status=status, a=a, b=b, **kwargs)


def paired_report(a: list[FieldValue], b: list[FieldValue], options: ComparisonOptions | None = None):
    lane_a = make_lane(samples=[make_sample("A", a)])
    lane_b = make_lane(backend="paddleocr_vl", samples=[make_sample("A", b)])
    matching = SampleMatching(
        pairs=(SampleMatch(a_id="A", b_id="A", confidence=1.0, method="llm", justification="test"),)
    )
    return lane_a, lane_b, compare_lanes(lane_a, lane_b, matching, options or comparison_options())


def supervised_dataset(
    a: list[FieldValue], b: list[FieldValue], responses: list[str], options: SupervisorOptions = OPTIONS
):
    """A consolidated table whose comparison the supervisor scored with the canned ``responses``."""
    comparison = dataclasses.replace(comparison_options(), supervisor=options)
    lane_a, lane_b, report = paired_report(a, b, comparison)
    client = FakeLlmClient(responses)
    report = supervise_report(report, options, client, BLOCKS, comparison.profile)
    document = DocumentInput(document_id=DOC_ID, sha256=DOC_ID, pdf_path=Path("paper.pdf"))
    return consolidate_document(document, {"mineru": lane_a, "paddleocr_vl": lane_b}, report, comparison), client


def decision(result, field: str):
    return next(row for row in result.quality_rows if row["field"] == field)


# ---- Selection ---------------------------------------------------------------------------------------------


def test_a_value_near_an_edge_of_the_plausible_range_is_borderline_and_one_well_inside_is_not():
    transmittance = spec("transmittance")  # at least 60 %: within 6 of it is borderline
    assert is_borderline(transmittance, value("transmittance", "62", "%", value=62.0))
    assert not is_borderline(transmittance, value("transmittance", "85", "%", value=85.0))
    thickness = spec("thickness")  # at most 5000 nm
    assert is_borderline(thickness, value("thickness", "4800", "nm", value=4800.0))
    assert not is_borderline(thickness, value("thickness", "300", "nm", value=300.0))


def test_a_field_without_a_range_or_a_value_that_did_not_normalise_is_never_borderline():
    assert spec("resistivity").valid_range == (None, None)
    assert not is_borderline(spec("resistivity"), value("resistivity", "1e-4", "Ω·cm", value=1e-4))
    assert not is_borderline(spec("transmittance"), value("transmittance", "sixty", "%", value=None))


def test_only_conflicts_and_borderline_agreements_are_selected():
    specs = shipped_profile().by_name
    a, b = value("transmittance", "83.5", "%", value=83.5), value("transmittance", "91", "%", value=91.0)
    assert selection_reason(comparison("conflict", a, b), specs) == "conflict"
    edge = value("transmittance", "62", "%", value=62.0)
    assert selection_reason(comparison("agree", edge, edge), specs) == "borderline"
    assert selection_reason(comparison("agree", a, a), specs) is None
    assert selection_reason(comparison("missing", a, None, missing_in="paddleocr_vl"), specs) is None
    assert selection_reason(comparison("conflict", a, b), {}) == "conflict"  # a conflict needs no spec


def test_the_passage_is_the_cited_blocks_in_citation_order_or_nothing_when_it_cannot_be_shown_whole():
    blocks = {"mineru_p0_b1": "first", "mineru_p0_b2": "second", "mineru_p0_b3": "x" * (MAX_PASSAGE_CHARS + 1)}
    cited = FieldValue(field="thickness", value_raw="300", source_ids=("mineru_p0_b2", "mineru_p0_b1", "missing"))
    assert passage_for(cited, blocks) == "second\n\nfirst"
    assert passage_for(FieldValue(field="thickness", value_raw="300"), blocks) is None
    # A long table cut at the limit could leave out the row the value sits in; the judge is not shown a fragment.
    assert passage_for(FieldValue(field="thickness", value_raw="300", source_ids=("mineru_p0_b3",)), blocks) is None


def test_the_sample_of_a_comparison_is_its_scope_without_the_entity():
    assert sample_of(comparison("agree", value("thickness", "300"), value("thickness", "300"))) == "A|A"
    paper = FieldComparison(scope="paper", field="thickness", status="agree")
    assert sample_of(paper) is None


# ---- Scoring -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "options", "verdict"),
    [(0.95, OPTIONS, "trusted"), (0.6, OPTIONS, "trusted"), (0.2, OPTIONS, "doubted"), (0.6, BANDED, "uncertain")],
)
def test_a_score_is_thresholded_into_a_verdict_where_it_is_produced(score, options, verdict):
    client = FakeLlmClient([reply(score, "plausible", "the passage writes 83.5%")])
    result = score_value(client, spec("transmittance"), value("transmittance", "83.5", "%"), "passage", options)
    assert (result.score, result.flag, result.critique, result.verdict) == (
        score,
        "plausible",
        "the passage writes 83.5%",
        verdict,
    )


def test_the_judge_is_told_the_value_its_unit_and_the_passage():
    client = FakeLlmClient([reply(1.0)])
    score_value(
        client,
        spec("transmittance"),
        value("transmittance", "83.5", "%", condition="550 nm"),
        "The film showed 83.5 % at 550 nm.",
        OPTIONS,
    )
    [call] = client.calls
    assert "Extracted value: 83.5" in call.user and "Extracted unit: %" in call.user
    assert "Sample:" not in call.user
    assert "Stated condition: 550 nm" in call.user and "The film showed 83.5 % at 550 nm." in call.user
    assert "Return ONLY a JSON object" in call.system


def test_a_request_that_fails_twice_is_an_error_score_not_an_exception():
    client = FakeLlmClient(["not json", "still not json"])
    result = score_value(client, spec("transmittance"), value("transmittance", "83.5", "%"), "passage", OPTIONS)
    assert (result.verdict, result.flag, result.score) == ("error", "error", 0.0)
    assert "twice" in result.critique


# ---- The report --------------------------------------------------------------------------------------------


def test_supervise_report_scores_both_sides_of_a_conflict_and_leaves_a_comfortable_agreement_alone():
    a = [value("transmittance", "83.5", "%"), value("thickness", "300", "nm")]
    b = [
        value("transmittance", "91", "%", backend="paddleocr_vl"),
        value("thickness", "300", "nm", backend="paddleocr_vl"),
    ]
    _, _, report = paired_report(a, b)
    client = FakeLlmClient([reply(0.95), reply(0.1, "value_not_in_passage", "the passage says 83.5")])

    scored = supervise_report(report, OPTIONS, client, BLOCKS, shipped_profile())

    by_field = {c.field: c for c in scored.comparisons}
    transmittance = by_field["transmittance"].supervision
    assert transmittance is not None and transmittance.reason == "conflict"
    assert (transmittance.a.verdict, transmittance.b.verdict) == ("trusted", "doubted")
    assert by_field["thickness"].supervision is None
    assert scored.counts == report.counts
    # Each side is judged against its own lane's block.
    assert [BLOCKS["mineru"]["mineru_p0_b1"] in call.user for call in client.calls] == [True, True]


def test_a_borderline_agreement_is_scored_and_a_side_citing_no_block_is_not():
    edge_a, edge_b = value("transmittance", "62", "%"), value("transmittance", "62", "%", backend="paddleocr_vl")
    _, _, report = paired_report([edge_a], [edge_b])
    client = FakeLlmClient([reply(0.3, "unit_mismatch", "the passage gives 62 a.u.")])

    scored = supervise_report(report, OPTIONS, client, {"mineru": BLOCKS["mineru"]}, shipped_profile())

    [c] = scored.comparisons
    assert c.status == "agree" and c.supervision is not None and c.supervision.reason == "borderline"
    assert c.supervision.a is not None and c.supervision.a.verdict == "doubted"
    assert c.supervision.b is None
    assert len(client.calls) == 1


def test_the_judge_is_told_which_sample_the_value_was_attributed_to():
    a = [value("transmittance", "83.5", "%")]
    b = [value("transmittance", "91", "%", backend="paddleocr_vl")]
    _, _, report = paired_report(a, b)
    client = FakeLlmClient([reply(1.0), reply(1.0)])
    supervise_report(report, OPTIONS, client, BLOCKS, shipped_profile())
    assert all("Sample: A|A" in call.user for call in client.calls)


def test_requests_in_flight_together_still_land_on_their_own_side():
    a = [value("transmittance", "83.5", "%")]
    b = [value("transmittance", "91", "%", backend="paddleocr_vl")]
    _, _, report = paired_report(a, b)
    # The reply depends on the request, so whichever thread runs first the sides come out right.
    client = FakeLlmClient(lambda system, user: reply(0.95) if "value: 83.5" in user else reply(0.1))
    [c] = supervise_report(report, OPTIONS, client, BLOCKS, shipped_profile(), concurrency=4).comparisons
    assert (c.supervision.a.verdict, c.supervision.b.verdict) == ("trusted", "doubted")


def test_an_export_carries_the_stored_scores_onto_the_rebuilt_report():
    a = [value("transmittance", "83.5", "%"), value("thickness", "300", "nm")]
    b = [
        value("transmittance", "91", "%", backend="paddleocr_vl"),
        value("thickness", "300", "nm", backend="paddleocr_vl"),
    ]
    _, _, fresh = paired_report(a, b)
    stored = supervise_report(fresh, OPTIONS, FakeLlmClient([reply(0.95), reply(0.1)]), BLOCKS, shipped_profile())

    carried = carry_supervision(fresh, stored)

    assert carried == stored
    # A comparison of other values (the lane re-read differently) takes nothing over.
    other = paired_report([value("transmittance", "84", "%")], b)[2]
    assert all(c.supervision is None for c in carry_supervision(other, stored).comparisons)


def test_a_report_with_a_failed_score_is_incomplete_so_it_is_not_stored():
    a = [value("transmittance", "83.5", "%")]
    b = [value("transmittance", "91", "%", backend="paddleocr_vl")]
    lane_a, lane_b, report = paired_report(a, b)
    lanes = {"mineru": lane_a, "paddleocr_vl": lane_b}
    scored = supervise_report(report, OPTIONS, FakeLlmClient([reply(0.95), "bad", "bad"]), BLOCKS, shipped_profile())
    assert incomplete_reason(lanes, scored) == "supervisor gave no score for transmittance"
    settled = supervise_report(report, OPTIONS, FakeLlmClient([reply(0.95), reply(0.1)]), BLOCKS, shipped_profile())
    assert incomplete_reason(lanes, settled) == ""


def test_a_report_without_supervision_keeps_its_bytes():
    c = comparison("agree", value("thickness", "300", "nm"), value("thickness", "300", "nm", backend="paddleocr_vl"))
    assert "supervision" not in c.model_dump_json()
    scored = c.model_copy(update={"supervision": Supervision(reason="borderline")})
    assert FieldComparison.model_validate_json(scored.model_dump_json()).supervision == scored.supervision


def test_the_stage_detail_counts_what_was_looked_at_doubted_and_unscored():
    a = [value("transmittance", "83.5", "%")]
    b = [value("transmittance", "91", "%", backend="paddleocr_vl")]
    _, _, report = paired_report(a, b)
    client = FakeLlmClient([reply(0.1, "value_not_in_passage"), "bad", "bad"])
    scored = supervise_report(report, OPTIONS, client, BLOCKS, shipped_profile())
    assert supervision_summary(scored) == "supervised 1 (1 doubted, 1 unscored)"


# ---- The cell ----------------------------------------------------------------------------------------------


def test_a_conflict_with_one_trusted_and_one_doubted_side_is_settled_as_supervised_not_agree():
    result, _ = supervised_dataset(
        [value("transmittance", "83.5", "%")],
        [value("transmittance", "91", "%", backend="paddleocr_vl")],
        [reply(0.95), reply(0.1, "value_not_in_passage", "the passage says 83.5 %")],
    )
    row = decision(result, "transmittance")
    assert (row["value"], row["decision"], row["lanes"]) == (83.5, "supervised", "mineru")
    assert "监督模型裁定" in row["detail"] and "the passage says 83.5 %" in row["detail"]
    assert result.paper_row["transmittance"] == 83.5


def test_the_trusted_side_may_be_lane_b():
    result, _ = supervised_dataset(
        [value("transmittance", "91", "%")],
        [value("transmittance", "83.5", "%", backend="paddleocr_vl")],
        [reply(0.1, "value_not_in_passage"), reply(0.95)],
    )
    row = decision(result, "transmittance")
    assert (row["value"], row["decision"], row["lanes"]) == (83.5, "supervised", "paddleocr_vl")


@pytest.mark.parametrize(
    ("responses", "options"),
    [
        ([reply(0.1, "value_not_in_passage"), reply(0.2, "value_not_in_passage")], OPTIONS),  # both doubted
        ([reply(0.95), reply(0.9)], OPTIONS),  # both trusted
        ([reply(0.95), reply(0.6)], BANDED),  # one uncertain
        ([reply(0.95), "bad", "bad"], OPTIONS),  # one unscored
    ],
)
def test_a_conflict_the_supervisor_could_not_settle_still_refuses_the_cell_with_the_scores(responses, options):
    result, _ = supervised_dataset(
        [value("transmittance", "83.5", "%")],
        [value("transmittance", "91", "%", backend="paddleocr_vl")],
        responses,
        options,
    )
    row = decision(result, "transmittance")
    assert (row["value"], row["decision"]) == (None, "conflict")
    assert "监督模型未能裁定" in row["detail"] and "83.5" in row["detail"]


def test_a_winner_grounding_rejected_settles_nothing():
    # The judge only needs a value to cite a block; the cell needs it located in the text. A conflict whose
    # trusted side is ungrounded stays a conflict for a person, instead of turning into an empty "ungrounded" cell.
    result, _ = supervised_dataset(
        [value("transmittance", "83.5", "%", grounded=False)],
        [value("transmittance", "91", "%", backend="paddleocr_vl")],
        [reply(0.95), reply(0.1, "value_not_in_passage")],
    )
    row = decision(result, "transmittance")
    assert (row["value"], row["decision"]) == (None, "conflict")


def test_a_settlement_stays_on_record_when_a_later_check_refuses_the_cell():
    # The 550 nm conflict is settled for lane A's 83.5, but lane A also quotes 85 under the same condition, so
    # the cell is refused as multiple_values; the settlement is still in its audit trail.
    a = [
        value("transmittance", "83.5", "%", condition="550 nm"),
        value("transmittance", "85", "%", condition="550 nm"),
    ]
    b = [value("transmittance", "91", "%", condition="550 nm", backend="paddleocr_vl")]
    # The comparison pairs 85 with 91 and leaves 83.5 one-sided; the judge trusts 85 and doubts 91.
    result, client = supervised_dataset(a, b, [reply(0.95), reply(0.1, "value_not_in_passage")])
    assert ["value: 85" in call.user for call in client.calls] == [True, False]
    row = decision(result, "transmittance")
    assert (row["value"], row["decision"]) == (None, "multiple_values")
    assert "监督模型裁定" in row["detail"]


def test_a_doubted_borderline_agreement_is_still_committed_with_the_doubt_recorded():
    result, client = supervised_dataset(
        [value("transmittance", "62", "%")],
        [value("transmittance", "62", "%", backend="paddleocr_vl")],
        [reply(1.0), reply(0.2, "unit_mismatch", "the table gives 62 a.u.")],
    )
    row = decision(result, "transmittance")
    assert (row["value"], row["decision"]) == (62.0, "agree")
    assert "存疑" in row["detail"] and "the table gives 62 a.u." in row["detail"]
    assert len(client.calls) == 2


def test_a_comfortable_agreement_costs_no_call_and_decides_as_before():
    result, client = supervised_dataset(
        [value("thickness", "300", "nm")], [value("thickness", "300", "nm", backend="paddleocr_vl")], []
    )
    assert (decision(result, "thickness")["value"], decision(result, "thickness")["decision"]) == (300.0, "agree")
    assert client.calls == []


def test_one_lane_quoting_a_value_twice_is_still_single_source():
    # Guards the verdict's meaning: "agree" is two lanes, never one lane repeating itself.
    a = [
        FieldValue(field="transmittance", value_raw="83.5", unit_raw="%", source_ids=("mineru_p0_b1",)),
        FieldValue(field="transmittance", value_raw="83.5", unit_raw="%", source_ids=("mineru_p0_b2",)),
    ]
    lane_a = make_lane(samples=[make_sample("A", a)])
    lane_b = make_lane(backend="paddleocr_vl", samples=[make_sample("A", [])])
    matching = SampleMatching(
        pairs=(SampleMatch(a_id="A", b_id="A", confidence=1.0, method="llm", justification="test"),)
    )
    report = compare_lanes(lane_a, lane_b, matching, comparison_options())
    document = DocumentInput(document_id=DOC_ID, sha256=DOC_ID, pdf_path=Path("paper.pdf"))
    result = consolidate_document(document, {"mineru": lane_a, "paddleocr_vl": lane_b}, report, comparison_options())
    row = decision(result, "transmittance")
    assert (row["value"], row["decision"], row["lanes"]) == (83.5, "single_source", "mineru")


# ---- Keys and settings -------------------------------------------------------------------------------------


def test_the_comparison_key_carries_the_supervisor_only_while_it_is_on():
    off = comparison_options()
    assert off.supervisor is None
    on = dataclasses.replace(off, supervisor=OPTIONS)
    other_model = dataclasses.replace(off, supervisor=dataclasses.replace(OPTIONS, model="other"))
    other_threshold = dataclasses.replace(off, supervisor=dataclasses.replace(OPTIONS, vote_threshold=0.9))
    keys = {comparison_key(o) for o in (off, on, other_model, other_threshold)}
    assert len(keys) == 4
    assert comparison_key(dataclasses.replace(on, supervisor=None)) == comparison_key(off)


def test_the_options_carry_the_settings_supervisor_when_enabled():
    profile = shipped_profile()
    assert ComparisonOptions.from_settings(Settings(), profile).supervisor is None
    enabled = Settings(supervisor_enabled=True, supervisor_model="judge", supervisor_vote_threshold=0.8)
    assert ComparisonOptions.from_settings(enabled, profile).supervisor == SupervisorOptions(
        model="judge", min_confidence=0.6, vote_threshold=0.8
    )


def test_the_shipped_config_leaves_the_supervisor_off_at_the_llm_endpoint_and_key():
    settings = Settings.from_env({})
    assert settings.supervisor_enabled is False
    assert settings.supervisor_api_key_env is None and settings.supervisor_api_key is None
    assert settings.supervisor_base_url == settings.llm_base_url
    assert (settings.supervisor_min_confidence, settings.supervisor_vote_threshold) == (0.6, 0.6)


def test_environment_overrides_reach_every_supervisor_setting():
    settings = Settings.from_env(
        {
            "PAPERFACTS_SUPERVISOR_ENABLED": "true",
            "PAPERFACTS_SUPERVISOR_BASE_URL": "http://localhost:8000/v1/",
            "PAPERFACTS_SUPERVISOR_MODEL": "judge",
            "PAPERFACTS_SUPERVISOR_API_KEY_ENV": "JUDGE_KEY",
            "JUDGE_KEY": " sk-judge ",
            "PAPERFACTS_SUPERVISOR_MIN_CONFIDENCE": "0.5",
            "PAPERFACTS_SUPERVISOR_VOTE_THRESHOLD": "0.8",
            "PAPERFACTS_SUPERVISOR_TIMEOUT_S": "12",
        }
    )
    assert settings.supervisor_enabled is True
    assert settings.supervisor_base_url == "http://localhost:8000/v1"
    assert settings.supervisor_model == "judge"
    assert (settings.supervisor_api_key_env, settings.supervisor_api_key) == ("JUDGE_KEY", "sk-judge")
    assert settings.require_supervisor_api_key() == "sk-judge"
    assert (settings.supervisor_min_confidence, settings.supervisor_vote_threshold) == (0.5, 0.8)
    assert settings.supervisor_timeout_s == 12.0


def test_a_null_base_url_follows_the_llm_endpoint_wherever_it_points():
    settings = Settings.from_env({"PAPERFACTS_LLM_BASE_URL": "http://other-provider/v1/"})
    assert settings.supervisor_base_url == "http://other-provider/v1"


def test_a_config_file_without_the_supervisor_section_is_refused_by_name(tmp_path: Path):
    # The same rule as every other section: a missing setting is named, never silently defaulted.
    stripped = json.loads(Path("config.json").read_text(encoding="utf-8"))
    del stripped["supervisor"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(stripped), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"missing setting 'supervisor\."):
        Settings.from_env({"PAPERFACTS_CONFIG": str(path)})


def test_a_named_key_variable_that_is_unset_is_an_error_and_none_means_the_llm_key():
    named = Settings.from_env({"PAPERFACTS_SUPERVISOR_API_KEY_ENV": "JUDGE_KEY"})
    with pytest.raises(ConfigError, match="JUDGE_KEY"):
        named.require_supervisor_api_key()
    assert Settings(llm_api_key="sk-llm").require_supervisor_api_key() == "sk-llm"


def test_inverted_thresholds_are_refused_by_name():
    with pytest.raises(ConfigError, match=r"supervisor\.min_confidence"):
        Settings.from_env(
            {"PAPERFACTS_SUPERVISOR_MIN_CONFIDENCE": "0.9", "PAPERFACTS_SUPERVISOR_VOTE_THRESHOLD": "0.5"}
        )


# ---- The workflow ------------------------------------------------------------------------------------------


def test_the_supervisor_client_is_its_own_at_the_supervisor_endpoint(tmp_path: Path):
    settings = Settings(
        data_root=tmp_path,
        llm_api_key="sk-llm",
        supervisor_base_url="http://judge:8000/v1",
        supervisor_model="judge",
        supervisor_timeout_s=7.0,
    )
    with build_supervisor_client(settings) as client:
        assert (client.base_url, client.api_key, client.model) == ("http://judge:8000/v1", "sk-llm", "judge")
        assert (client.temperature, client.reasoning_effort, client.timeout_s) == (0.0, None, 7.0)


def test_a_missing_supervisor_key_fails_the_compare_before_matching_is_paid_for(monkeypatch, tmp_path: Path):
    from paperfacts.workflow import compare_document

    a = [value("transmittance", "83.5", "%")]
    b = [value("transmittance", "91", "%", backend="paddleocr_vl")]
    lane_a, lane_b, _ = paired_report(a, b)
    comparison = dataclasses.replace(comparison_options(), supervisor=OPTIONS)
    settings = Settings(data_root=tmp_path, supervisor_enabled=True, supervisor_api_key_env="JUDGE_KEY")
    matcher = FakeLlmClient([])

    def no_matching(*args, **kwargs):
        raise AssertionError("matching must not run")

    monkeypatch.setattr("paperfacts.workflow.match_samples", no_matching)
    document = DocumentInput(document_id=DOC_ID, sha256=DOC_ID, pdf_path=Path("paper.pdf"))
    with pytest.raises(ConfigError, match="JUDGE_KEY"):
        compare_document(document, settings, comparison, matcher, lanes={"mineru": lane_a, "paddleocr_vl": lane_b})


def test_supervise_document_judges_each_lane_against_its_own_parse(monkeypatch, tmp_path: Path):
    a = [value("transmittance", "83.5", "%")]
    b = [value("transmittance", "91", "%", backend="paddleocr_vl")]
    comparison = dataclasses.replace(comparison_options(), supervisor=OPTIONS)
    _, _, report = paired_report(a, b, comparison)
    artifacts = {
        backend: make_artifact(
            (make_block(page=0, order=1, backend=backend, content=f"{backend} says 83.5 %"),), backend=backend
        )
        for backend in ("mineru", "paddleocr_vl")
    }
    monkeypatch.setattr("paperfacts.workflow.load_artifact", lambda document, backend, settings: artifacts[backend])
    client = FakeLlmClient([reply(0.95), reply(0.1, "value_not_in_passage")])
    monkeypatch.setattr("paperfacts.workflow.build_supervisor_client", lambda settings: client)
    document = DocumentInput(document_id=DOC_ID, sha256=DOC_ID, pdf_path=Path("paper.pdf"))

    scored = supervise_document(document, Settings(data_root=tmp_path), comparison, report)

    [c] = scored.comparisons
    assert c.supervision is not None and (c.supervision.a.verdict, c.supervision.b.verdict) == ("trusted", "doubted")
    assert "mineru says 83.5 %" in client.calls[0].user and "paddleocr_vl says 83.5 %" in client.calls[1].user
    assert client.closed
