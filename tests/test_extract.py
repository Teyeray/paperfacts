"""One lane's extraction: ParsedArtifact -> prompt -> LLM (with one repair retry) -> cleaning ->
LaneExtraction.

The fine-grained cleaning rules live in :mod:`tests.test_extraction_records` (that is
``response_to_records``'s job); this file guards the **orchestration**: how many times the model was
called, what went into the prompt, how a failure is reported, and whether the cleaning result is recorded
faithfully on the lane. Every case here uses :class:`support.llm.FakeLlmClient`; not one real call ever
happens.
"""

from __future__ import annotations

import json

import pytest

from paperfacts.adapters import render_markdown
from paperfacts.config import INHERIT
from paperfacts.errors import ContextBudgetError, LlmResponseError
from paperfacts.extract import extract_lane
from paperfacts.keys import FINGERPRINT_LENGTH, ExtractionOptions, extractor_key, schema_fingerprint
from paperfacts.llm import LlmResult
from support.extraction import make_artifact
from support.factories import make_block
from support.llm import FakeLlmClient

# ---- extractor_key -------------------------------------------------------------------


def test_extractor_key_is_stable_for_the_same_inputs():
    # An unstable key means re-calling the LLM on every run -- the cache would be worthless.
    assert extractor_key(ExtractionOptions("deepseek-chat", mode="document")) == extractor_key(
        ExtractionOptions("deepseek-chat", mode="document")
    )


def test_extractor_key_changes_with_the_model():
    # A different model is a different extractor; old results must not be reused under it.
    assert extractor_key(ExtractionOptions("deepseek-chat", mode="document")) != extractor_key(
        ExtractionOptions("gpt-4o", mode="document")
    )


def test_extractor_key_is_unaffected_by_the_default_number_of_passes():
    # passes=1 is the default before self-consistency voting existed; keys written back then must still
    # resolve, so the default must produce the exact same key as omitting the argument.
    assert extractor_key(ExtractionOptions("deepseek-chat", mode="document")) == extractor_key(
        ExtractionOptions("deepseek-chat", passes=1, mode="document")
    )


def test_extractor_key_changes_when_the_number_of_passes_is_not_one():
    # Voting across passes changes both the cost and the result, so it must invalidate whatever was
    # cached under a different pass count.
    assert extractor_key(ExtractionOptions("deepseek-chat", mode="document")) != extractor_key(
        ExtractionOptions("deepseek-chat", passes=3, mode="document")
    )
    assert extractor_key(ExtractionOptions("deepseek-chat", passes=2, mode="document")) != extractor_key(
        ExtractionOptions("deepseek-chat", passes=3, mode="document")
    )


def test_extractor_key_changes_when_the_cleaning_fingerprint_changes(monkeypatch):
    # extract.py and records.py decide which of the model's claims survive; changing either must
    # invalidate stored extractions too, even though the prompt, the model and the schema stayed the same.
    # Re-deriving records is free (the model's answer is itself cached by payload), so this cache miss
    # costs nothing but a bit of local computation.
    monkeypatch.setattr("paperfacts.keys.extraction_code_fingerprint", lambda: "aaaaaaaaaaaa")
    before = extractor_key(ExtractionOptions("deepseek-chat", mode="document"))
    monkeypatch.setattr("paperfacts.keys.extraction_code_fingerprint", lambda: "bbbbbbbbbbbb")
    after = extractor_key(ExtractionOptions("deepseek-chat", mode="document"))

    assert before != after


def test_a_document_mode_key_ignores_everything_only_passage_mode_depends_on(monkeypatch):
    # Whole-document mode sends exactly the request it always sent, so its key must not move when the
    # retrieval rules or the passage prompts change -- otherwise tuning a keyword would rename the stored
    # facts of runs that never used retrieval at all.
    before_document = extractor_key(ExtractionOptions("deepseek-chat", mode="document"))
    before_passage = extractor_key(ExtractionOptions("deepseek-chat", mode="passage"))
    monkeypatch.setattr("paperfacts.keys.retrieval_fingerprint", lambda: "ffffffffffff")

    assert extractor_key(ExtractionOptions("deepseek-chat", mode="document")) == before_document
    assert extractor_key(ExtractionOptions("deepseek-chat", mode="passage")) != before_passage


def test_extractor_key_is_unaffected_by_the_baseline_reasoning_effort():
    # None is the baseline: the parameter is left out of the request, exactly as before it existed, so a
    # checkout that never touched the setting keeps the filenames it already has.
    assert extractor_key(ExtractionOptions("deepseek-chat", mode="document")) == extractor_key(
        ExtractionOptions("deepseek-chat", reasoning_effort=None, mode="document")
    )


def test_extractor_key_changes_with_the_reasoning_effort():
    # How much the model thinks before answering changes the answer, so it changes the stored extraction.
    assert extractor_key(ExtractionOptions("deepseek-chat", mode="document")) != extractor_key(
        ExtractionOptions("deepseek-chat", reasoning_effort="none", mode="document")
    )
    assert extractor_key(ExtractionOptions("deepseek-chat", reasoning_effort="low", mode="document")) != extractor_key(
        ExtractionOptions("deepseek-chat", reasoning_effort="high", mode="document")
    )


def test_extractor_key_ignores_the_inventory_effort_at_its_baseline():
    assert extractor_key(ExtractionOptions("deepseek-chat", mode="passage")) == extractor_key(
        ExtractionOptions("deepseek-chat", mode="passage", inventory_reasoning_effort=INHERIT)
    )


def test_extractor_key_changes_when_the_inventory_question_omits_the_parameter():
    # "omit no parameter at all" is a different request from "inherit whatever the client sends", so it
    # cannot quietly reuse the baseline's stored extractions.
    assert extractor_key(ExtractionOptions("deepseek-chat", mode="passage")) != extractor_key(
        ExtractionOptions("deepseek-chat", mode="passage", inventory_reasoning_effort=None)
    )


def test_extractor_key_changes_with_the_inventory_effort_in_passage_mode():
    assert extractor_key(ExtractionOptions("deepseek-chat", mode="passage")) != extractor_key(
        ExtractionOptions("deepseek-chat", mode="passage", inventory_reasoning_effort="none")
    )


def test_extractor_key_ignores_the_inventory_effort_in_document_mode():
    # Document mode never asks an inventory question, so the setting changes nothing it sends and must
    # not rename its stored facts.
    assert extractor_key(ExtractionOptions("deepseek-chat", mode="document")) == extractor_key(
        ExtractionOptions("deepseek-chat", mode="document", inventory_reasoning_effort="none")
    )


def test_extractor_key_is_short_enough_to_live_in_a_filename():
    key = extractor_key(ExtractionOptions("deepseek-chat", mode="document"))

    assert len(key) == FINGERPRINT_LENGTH == 12
    assert key.isalnum()


def test_the_schema_fingerprint_is_a_hash_of_the_field_table_not_a_hand_written_label():
    # A hand-maintained version number eventually gets forgotten; a fingerprint cannot.
    fingerprint = schema_fingerprint()

    assert len(fingerprint) == FINGERPRINT_LENGTH
    assert fingerprint == schema_fingerprint()


# ---- Normal path --------------------------------------------------------------------


def response_json(**overrides) -> str:
    payload = {
        "target": {
            "source_ids": ["mineru_p0_b0"],
            "fields": [{"field": "density", "value_raw": "98.5", "unit_raw": "%"}],
        },
        "samples": [
            {
                "sample_id": "A",
                "label": "O2 100 sccm",
                "conditions": {"O2 flow": "100 sccm"},
                "source_ids": ["mineru_p0_b0"],
                "fields": [
                    {
                        "field": "sheet_resistance",
                        "value_raw": "12.5",
                        "unit_raw": "Ω/sq",
                        "source_ids": ["mineru_p0_b1"],
                    }
                ],
            }
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_a_valid_response_becomes_a_lane_extraction_in_one_call():
    artifact = make_artifact()
    client = FakeLlmClient([response_json()])

    lane = extract_lane(artifact, client, mode="document")

    assert client.call_count == 1
    assert lane.document_id == artifact.document_id
    assert lane.backend == "mineru"
    assert lane.model == "fake-model"
    assert lane.sample("A").get("sheet_resistance").value_raw == "12.5"
    assert lane.target.get("density").value_raw == "98.5"


def test_the_usage_and_the_raw_response_are_kept_as_evidence():
    text = response_json()
    client = FakeLlmClient([text], usage={"total_tokens": 999})

    lane = extract_lane(make_artifact(), client, mode="document")

    assert lane.usage == {"total_tokens": 999}
    assert lane.raw_response == text


def test_the_lane_records_the_extractor_key_and_the_schema_fingerprint():
    client = FakeLlmClient([response_json()], model="some-model")

    lane = extract_lane(make_artifact(), client, mode="document")

    assert lane.extractor_key == extractor_key(ExtractionOptions("some-model", mode="document"))
    assert lane.schema_version == schema_fingerprint()


def test_the_markdown_of_the_artifact_is_what_reaches_the_model():
    artifact = make_artifact()
    client = FakeLlmClient([response_json()])

    extract_lane(artifact, client, mode="document")

    assert render_markdown(artifact.blocks) in client.users[0]


def test_the_values_reach_the_lane_un_normalized():
    # What is stored is the verbatim-level result; value/unit are filled in by normalization at read
    # time, so changing a normalization rule never needs a fresh LLM call.
    lane = extract_lane(make_artifact(), FakeLlmClient([response_json()]), mode="document")
    field = lane.sample("A").get("sheet_resistance")

    assert (field.value, field.unit, field.normalization_note) == (None, None, None)


# ---- Symmetry between the two lanes ---------------------------------------------------


def test_both_lanes_get_a_byte_identical_system_prompt():
    """Both lanes must go through the exact same prompt, or prompt noise would mix into the disagreement
    and the comparison would no longer say anything about the parsers' actual differences."""
    mineru = make_artifact(backend="mineru")
    paddle = make_artifact(
        [make_block(page=0, order=0, backend="paddleocr_vl", content="Sample A")], backend="paddleocr_vl"
    )
    client_a, client_b = FakeLlmClient([response_json()]), FakeLlmClient([json.dumps({"samples": []})])

    extract_lane(mineru, client_a, mode="document")
    extract_lane(paddle, client_b, mode="document")

    assert client_a.systems[0] == client_b.systems[0]


# ---- Repair retry -----------------------------------------------------------------------


def test_an_invalid_first_answer_triggers_one_repair_request():
    client = FakeLlmClient(["not json at all", response_json()])

    lane = extract_lane(make_artifact(), client, mode="document")

    assert client.call_count == 2
    assert lane.sample("A") is not None


def test_the_repair_request_carries_the_validation_error_and_the_previous_answer():
    # Without feeding the error back, the model would just make the same mistake again -- and the retry
    # would have been paid for nothing.
    client = FakeLlmClient(['{"samples": [{"label": "no id"}]}', response_json()])

    extract_lane(make_artifact(), client, mode="document")
    repair = client.users[1]

    assert "not valid JSON" in repair
    assert "sample_id" in repair  # the validation error text itself
    assert '{"samples": [{"label": "no id"}]}' in repair  # the previous answer


def test_the_repair_restates_the_original_request():
    # The repair request is an independent call; the model sees no prior context, so without restating
    # the original request it would answer an isolated fragment.
    artifact = make_artifact()
    client = FakeLlmClient(["oops", response_json()])

    extract_lane(artifact, client, mode="document")

    assert render_markdown(artifact.blocks) in client.users[1]
    assert client.systems[0] == client.systems[1]


def test_the_usage_of_both_attempts_is_summed():
    # The repair costs money too; counting only the second call would understate the actual cost.
    client = FakeLlmClient(
        [
            LlmResult(text="oops", usage={"total_tokens": 10, "prompt_tokens": 8}, cached=False),
            LlmResult(text=response_json(), usage={"total_tokens": 30, "prompt_tokens": 25}, cached=False),
        ]
    )

    lane = extract_lane(make_artifact(), client, mode="document")

    assert lane.usage == {"total_tokens": 40, "prompt_tokens": 33}


def test_two_invalid_answers_in_a_row_raise_an_llm_response_error():
    client = FakeLlmClient(["nope", "still nope"])

    with pytest.raises(LlmResponseError, match="twice failed"):
        extract_lane(make_artifact(), client, mode="document")

    assert client.call_count == 2


def test_a_response_that_is_valid_json_but_the_wrong_shape_also_triggers_a_repair():
    client = FakeLlmClient(['{"samples": "should be a list"}', response_json()])

    lane = extract_lane(make_artifact(), client, mode="document")

    assert client.call_count == 2
    assert len(lane.samples) == 1


# ---- refresh --------------------------------------------------------------------


def test_refresh_is_forwarded_to_the_client_so_force_really_re_asks_the_model():
    client = FakeLlmClient([response_json()])

    extract_lane(make_artifact(), client, refresh=True, mode="document")

    assert client.refreshes == [True]


def test_refresh_defaults_to_false():
    client = FakeLlmClient([response_json()])

    extract_lane(make_artifact(), client, mode="document")

    assert client.refreshes == [False]


def test_refresh_also_applies_to_the_repair_attempt():
    client = FakeLlmClient(["oops", response_json()])

    extract_lane(make_artifact(), client, refresh=True, mode="document")

    assert client.refreshes == [True, True]


# ---- passes: self-consistency voting -----------------------------------------------------


def test_passes_greater_than_one_makes_that_many_model_calls():
    client = FakeLlmClient([response_json(), response_json(), response_json()])

    lane = extract_lane(make_artifact(), client, passes=3, mode="document")

    assert client.call_count == 3
    assert lane.sample("A").get("sheet_resistance").value_raw == "12.5"


def test_each_pass_after_the_first_gets_its_own_cache_salt():
    # All three calls send the identical payload; without a distinct salt per pass the second and third
    # would just replay the first pass's cached answer instead of asking again.
    client = FakeLlmClient([response_json(), response_json(), response_json()])

    extract_lane(make_artifact(), client, passes=3, mode="document")

    assert [call.cache_salt for call in client.calls] == ["", "pass-1", "pass-2"]


def test_every_pass_sends_the_same_system_and_user_prompt():
    client = FakeLlmClient([response_json(), response_json(), response_json()])

    extract_lane(make_artifact(), client, passes=3, mode="document")

    assert len({call.system for call in client.calls}) == 1
    assert len({call.user for call in client.calls}) == 1


def test_passes_must_be_at_least_one():
    with pytest.raises(ValueError, match="at least 1"):
        extract_lane(make_artifact(), FakeLlmClient([]), passes=0, mode="document")


# ---- Context budget -----------------------------------------------------------------------


def test_an_oversized_document_is_rejected_before_any_model_call_is_made():
    # Failing here instead of letting the API silently truncate the prompt is the whole point of the
    # guard; it only holds if the check runs before extract_lane spends any money.
    client = FakeLlmClient([])

    with pytest.raises(ContextBudgetError):
        extract_lane(make_artifact(), client, context_tokens=100, mode="document")

    assert client.call_count == 0


def test_the_context_budget_error_names_the_ways_to_fix_it():
    with pytest.raises(ContextBudgetError, match="PAPERFACTS_LLM_CONTEXT_TOKENS"):
        extract_lane(make_artifact(), FakeLlmClient([]), context_tokens=100, mode="document")


# ---- Grounding is flagged, not enforced ----------------------------------------------------


def test_an_ungrounded_value_is_flagged_but_not_dropped():
    # A value the cleaning rules cannot dismiss (it has digits, its field is in the schema) but which
    # cannot be found in the block it cites is still evidence of what the model claimed; hiding it would
    # make an untraceable claim indistinguishable from one that was never made.
    payload = json.loads(response_json())
    payload["samples"][0]["fields"][0]["value_raw"] = "999.9"  # absent from every block in the artifact
    client = FakeLlmClient([json.dumps(payload)])

    lane = extract_lane(make_artifact(), client, mode="document")
    field = lane.sample("A").get("sheet_resistance")

    assert field is not None
    assert field.value_raw == "999.9"
    assert field.grounded is False
    assert field in lane.ungrounded()


def test_a_grounded_value_does_not_appear_in_ungrounded():
    lane = extract_lane(make_artifact(), FakeLlmClient([response_json()]), mode="document")

    assert lane.sample("A").get("sheet_resistance") not in lane.ungrounded()


# ---- Cleaning results are recorded faithfully on the lane -----------------------------


def test_invented_source_ids_are_removed_from_the_fields_and_listed_on_the_lane():
    payload = json.loads(response_json())
    payload["samples"][0]["fields"][0]["source_ids"] = ["mineru_p0_b1", "mineru_p9_b9"]
    client = FakeLlmClient([json.dumps(payload)])

    lane = extract_lane(make_artifact(), client, mode="document")

    assert lane.sample("A").get("sheet_resistance").source_ids == ("mineru_p0_b1",)
    assert lane.invalid_source_ids == ("mineru_p9_b9",)


def test_the_known_ids_come_from_the_artifact_blocks():
    # Switch to the other lane's artifact, and the same answer's mineru_* ids all become "nonexistent".
    paddle = make_artifact(
        [make_block(page=0, order=0, backend="paddleocr_vl", content="Sample A")], backend="paddleocr_vl"
    )
    client = FakeLlmClient([response_json()])

    lane = extract_lane(paddle, client, mode="document")

    assert lane.invalid_source_ids == ("mineru_p0_b0", "mineru_p0_b1")


def test_dropped_values_are_listed_on_the_lane():
    payload = json.loads(response_json())
    payload["samples"][0]["fields"] = [
        {"field": "sheet_resistance", "value_raw": "minimum"},
        {"field": "carrier_concentration", "value_raw": "1e20"},
    ]
    client = FakeLlmClient([json.dumps(payload)])

    lane = extract_lane(make_artifact(), client, mode="document")

    assert lane.sample("A").fields == ()
    assert lane.dropped == (
        "sheet_resistance: non-numeric value 'minimum'",
        "carrier_concentration: not in schema",
    )


def test_a_clean_response_reports_nothing_invalid_or_dropped():
    lane = extract_lane(make_artifact(), FakeLlmClient([response_json()]), mode="document")

    assert lane.invalid_source_ids == ()
    assert lane.dropped == ()


# ---- Empty results --------------------------------------------------------------------


def test_a_response_that_found_nothing_is_a_valid_empty_lane():
    lane = extract_lane(make_artifact(), FakeLlmClient(['{"target": null, "samples": []}']), mode="document")

    assert lane.target is None
    assert lane.samples == ()
    assert lane.invalid_source_ids == ()


def test_the_lane_round_trips_through_disk(tmp_path):
    lane = extract_lane(make_artifact(), FakeLlmClient([response_json()]), mode="document")
    path = tmp_path / "facts" / "mineru.key.json"

    lane.write(path)

    from paperfacts.records import LaneExtraction

    assert LaneExtraction.read(path) == lane


def test_tuning_the_matching_prompt_does_not_invalidate_extractions(monkeypatch):
    """The matching prompt pairs samples across lanes, which is part of the comparison, not the extraction.

    Mixing it into ``extractor_key`` would mean that fixing a sample-pairing edge case discards every
    stored per-lane extraction and re-pays the LLM for the expensive step to redo a cheap one.
    """
    before = extractor_key(ExtractionOptions("deepseek-chat", mode="document"))
    monkeypatch.setattr("paperfacts.keys.matching_system_prompt", lambda: "something else")

    assert extractor_key(ExtractionOptions("deepseek-chat", mode="document")) == before


def test_tuning_the_matching_prompt_does_invalidate_comparisons(monkeypatch):
    from paperfacts.keys import comparison_key

    before = comparison_key()
    monkeypatch.setattr("paperfacts.keys.matching_system_prompt", lambda: "something else")

    assert comparison_key() != before
