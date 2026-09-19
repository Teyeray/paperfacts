"""Passage mode: ask which samples exist, then ask about one field at a time.

Document mode hands the whole paper over once; this mode asks a question per field over the blocks
:mod:`paperfacts.passages` retrieved for it. That buys three things this file guards: the model can only
cite blocks it was actually shown, a value has to name the sample it belongs to, and a field no block
mentions is never asked about at all.

The retrieval rules themselves live in :mod:`tests.test_passages` and the per-value cleaning in
:mod:`tests.test_records`; here it is the orchestration -- how many calls, what each carries, and where
each answer ends up. Every case runs on :class:`support.llm.FakeLlmClient`; no real call is ever made.
"""

from __future__ import annotations

import json
import re

import pytest

from paperfacts.extract import extract_lane
from paperfacts.keys import extractor_key
from paperfacts.prompts import field_system_prompt, inventory_system_prompt
from support.extraction import make_artifact
from support.factories import make_block
from support.llm import FakeLlmClient

# One block naming a sample and its deposition condition, one stating a measurement. Retrieval gives
# `sheet_resistance` exactly one keyword candidate (the second block); the "100 sccm" in the first block
# additionally qualifies the three flow-rate fields by unit alone (a unit is a weak signal by design), so
# a lane built from these makes precisely five calls -- which is what makes the call-count assertions
# below readable.
SAMPLE_BLOCK = "Sample A was grown at 100 sccm."
VALUE_BLOCK = "The sheet resistance of Sample A was 12.5 ohm/sq."
# The fields these blocks put in a prompt: the one named by a keyword, plus the three the unit qualifies.
ASKED_FIELDS = {"ar_flow_rate", "o2_flow_rate", "h2_flow_rate", "sheet_resistance"}
# The one field a keyword names, and the id of the block that mentions it.
ASKED_FIELD = "sheet_resistance"


def make_blocks(backend: str = "mineru") -> tuple:
    return (
        make_block(page=0, order=0, backend=backend, content=SAMPLE_BLOCK),
        make_block(page=0, order=1, backend=backend, content=VALUE_BLOCK),
    )


def inventory_json(samples: list[dict] | None = None) -> str:
    if samples is None:
        samples = [{"sample_id": "A", "label": "O2 100 sccm", "conditions": {"flow": "100 sccm"}}]
    return json.dumps({"samples": samples})


# Two samples, for the cases that need a null sample_id to stay genuinely unplaceable: with a single
# sample in the inventory there is only one possible owner, and `passage_records` attributes it.
TWO_SAMPLES = [
    {"sample_id": "A", "label": "O2 100 sccm", "conditions": {"flow": "100 sccm"}},
    {"sample_id": "B", "label": "O2 200 sccm", "conditions": {"flow": "200 sccm"}},
]


def values_json(*values: dict) -> str:
    return json.dumps({"values": list(values)})


def field_of(user: str) -> str:
    """The field a question is about, read back out of the rendered field table line."""
    match = re.search(r"^- `([a-z0-9_]+)`", user, re.MULTILINE)
    return match.group(1) if match else ""


def responder(inventory: str = "", **by_field: str):
    """Answer the inventory question and each field question by name; unasked fields answer nothing."""
    inventory = inventory or inventory_json()

    def respond(system: str, user: str) -> str:
        if system == inventory_system_prompt():
            return inventory
        return by_field.get(field_of(user), values_json())

    return respond


def passage_responder(inventory: str, per_pass: list[dict[str, str]]):
    """Like :func:`responder`, but answers differently on each pass.

    Every pass sends byte-identical prompts, so the pass can only be told apart by counting: each one opens
    with the inventory question.
    """
    state = {"pass_index": -1}

    def respond(system: str, user: str) -> str:
        if system == inventory_system_prompt():
            state["pass_index"] += 1
            return inventory
        return per_pass[state["pass_index"]].get(field_of(user), values_json())

    return respond


def extract(client: FakeLlmClient, *, backend: str = "mineru", passes: int = 1):
    return extract_lane(make_artifact(make_blocks(backend), backend=backend), client, mode="passage", passes=passes)


# ---- how many questions, and about what -------------------------------------------------------


def test_the_inventory_is_asked_once_and_then_only_about_fields_a_block_mentions():
    # The whole point of the mode: twenty fields in the schema, but only the ones this paper mentions cost
    # a call. Asking about the rest would be paying to be told "not stated".
    client = FakeLlmClient(responder())

    extract(client)

    assert client.call_count == 5
    assert client.systems == [inventory_system_prompt()] + [field_system_prompt()] * 4
    assert {field_of(user) for user in client.users[1:]} == ASKED_FIELDS


def test_a_field_no_block_mentions_is_never_asked_about_and_the_lane_says_why():
    # "The model missed it" and "we never asked" are different failures and must stay distinguishable in
    # the audit, or a retrieval gap would read as a model error.
    client = FakeLlmClient(responder())

    lane = extract(client)

    assert any("thickness: no block in this lane mentions it" in entry for entry in lane.dropped)
    assert not any(entry.startswith(f"{ASKED_FIELD}: no block") for entry in lane.dropped)
    # Sixteen of the twenty fields go unasked, which is why only four field calls were made.
    assert sum(1 for entry in lane.dropped if "was not asked about" in entry) == 16


def test_both_lanes_get_a_byte_identical_inventory_question():
    # Any asymmetry between the lanes contaminates the disagreement signal: a difference downstream has to
    # come from the parsers, never from how each lane was asked.
    mineru, paddle = FakeLlmClient(responder()), FakeLlmClient(responder())

    extract(mineru, backend="mineru")
    extract(paddle, backend="paddleocr_vl")

    assert mineru.systems[0] == paddle.systems[0]


def test_both_lanes_get_a_byte_identical_field_question_system_prompt():
    # The field half that varies is the question (the user half); the instructions must not vary at all.
    mineru, paddle = FakeLlmClient(responder()), FakeLlmClient(responder())

    extract(mineru, backend="mineru")
    extract(paddle, backend="paddleocr_vl")

    assert mineru.systems[1] == paddle.systems[1] == field_system_prompt()


def test_the_field_question_carries_that_fields_own_description():
    # The model is told which field it is being asked about in the question, not the system prompt, so the
    # instructions can stay one constant across all twenty fields.
    client = FakeLlmClient(responder())

    extract(client)

    question = next(user for user in client.users[1:] if field_of(user) == ASKED_FIELD)
    assert "`sheet_resistance`" in question
    assert "Sheet resistance of the film" in question


def test_the_field_question_carries_the_sample_list_with_labels_and_conditions():
    # A value can only name a sample the question listed, and the model needs the conditions to tell the
    # samples apart when the paper's own ids are unhelpful.
    client = FakeLlmClient(responder())

    extract(client)

    assert "- id: A | label: O2 100 sccm | conditions: flow=100 sccm" in client.users[1]


def test_the_inventory_question_never_carries_the_field_schema():
    # The inventory question is about identity only. Sending the field table with it would pay for the
    # schema twice and invite the model to answer both questions at once.
    client = FakeLlmClient(responder())

    extract(client)

    assert ASKED_FIELD not in client.systems[0]
    assert ASKED_FIELD not in client.users[0]


def test_the_inventory_question_carries_the_blocks_that_name_samples():
    client = FakeLlmClient(responder())

    extract(client)

    assert SAMPLE_BLOCK in client.users[0]


# ---- citations: a field may only cite what its own question showed ------------------------------


def test_a_citation_the_field_question_never_showed_is_stripped_and_audited():
    # Each question is shown its own retrieved blocks, so an id from elsewhere in the paper is exactly as
    # invented as one the model made up: it cannot be evidence for a value the model never read.
    unseen = make_blocks()[0].source_id
    client = FakeLlmClient(
        responder(sheet_resistance=values_json({"sample_id": "A", "value_raw": "12.5", "source_ids": [unseen]}))
    )

    lane = extract(client)

    assert lane.invalid_source_ids == (unseen,)
    assert lane.samples[0].fields[0].source_ids == ()


def test_a_citation_the_field_question_did_show_survives():
    shown = make_blocks()[1].source_id
    client = FakeLlmClient(
        responder(sheet_resistance=values_json({"sample_id": "A", "value_raw": "12.5", "source_ids": [shown]}))
    )

    lane = extract(client)

    assert lane.invalid_source_ids == ()
    assert lane.samples[0].fields[0].source_ids == (shown,)


# ---- attribution -------------------------------------------------------------------------------


def test_a_value_naming_a_listed_sample_is_attached_to_it():
    client = FakeLlmClient(
        responder(sheet_resistance=values_json({"sample_id": "A", "value_raw": "12.5", "unit_raw": "Ω/sq"}))
    )

    lane = extract(client)

    assert [(sample.sample_id, [field.value_raw for field in sample.fields]) for sample in lane.samples] == [
        ("A", ["12.5"])
    ]
    assert lane.unattributed == ()


def test_attribution_survives_a_difference_in_case_and_spacing():
    # The model is asked to repeat an id it was given, but it reformats them; the same normalised key that
    # pairs samples across lanes decides here, so "S 1" and "s1" are the same sample.
    client = FakeLlmClient(
        responder(
            inventory=inventory_json([{"sample_id": "S 1", "label": "", "conditions": {}}]),
            sheet_resistance=values_json({"sample_id": "s1", "value_raw": "12.5"}),
        )
    )

    lane = extract(client)

    assert lane.samples[0].sample_id == "S 1"
    assert [field.value_raw for field in lane.samples[0].fields] == ["12.5"]
    assert lane.unattributed == ()


def test_a_value_naming_no_sample_is_kept_unattributed_rather_than_guessed_onto_one():
    # Guessing would be invisible in the report; an unplaced value is visible, and a misplaced one is
    # indistinguishable from a real measurement. Two samples, so there is a real choice to refuse.
    client = FakeLlmClient(
        responder(
            inventory=inventory_json(TWO_SAMPLES),
            sheet_resistance=values_json({"sample_id": None, "value_raw": "12.5"}),
        )
    )

    lane = extract(client)

    assert [field.value_raw for field in lane.unattributed] == ["12.5"]
    assert [sample.fields for sample in lane.samples] == [(), ()]


def test_a_value_naming_no_sample_goes_to_the_only_sample_of_a_single_sample_paper():
    # The prompt allows a null sample_id when the excerpts do not say which sample a value belongs to.
    # With one sample in the inventory there is nothing to say: it is not a plausible neighbour, it is the
    # only possible owner, and leaving the value unattributed would compare it with nothing.
    client = FakeLlmClient(responder(sheet_resistance=values_json({"sample_id": None, "value_raw": "12.5"})))

    lane = extract(client)

    assert len(lane.samples) == 1
    assert [field.value_raw for field in lane.samples[0].fields] == ["12.5"]
    assert lane.unattributed == ()


def test_a_value_naming_a_sample_the_inventory_does_not_have_is_kept_unattributed():
    client = FakeLlmClient(responder(sheet_resistance=values_json({"sample_id": "Z", "value_raw": "12.5"})))

    lane = extract(client)

    assert [field.value_raw for field in lane.unattributed] == ["12.5"]
    assert lane.samples[0].fields == ()


def test_an_unattributed_value_still_counts_as_a_value_of_the_lane():
    # Leaving it out of values() would understate what the lane found and hide an ungrounded one.
    client = FakeLlmClient(
        responder(
            inventory=inventory_json(TWO_SAMPLES),
            sheet_resistance=values_json({"sample_id": None, "value_raw": "12.5"}),
        )
    )

    lane = extract(client)

    assert [field.value_raw for field in lane.values()] == ["12.5"]


def test_a_paper_level_value_goes_to_the_target_even_when_it_names_a_sample():
    # The question itself decided the scope, so a stray sample_id on a target field is noise -- not the
    # scope error document mode has to guard against, where the model chose where to file it.
    blocks = (
        make_block(page=0, order=0, content="Films were grown at 100 sccm."),
        make_block(page=0, order=1, content="A sputtering target of 4 inch diameter was used."),
    )
    client = FakeLlmClient(responder(inch=values_json({"sample_id": "A", "value_raw": "4", "unit_raw": "inch"})))

    lane = extract_lane(make_artifact(blocks), client, mode="passage")

    assert lane.target is not None
    assert [(field.field, field.value_raw) for field in lane.target.fields] == [("inch", "4")]
    assert lane.samples[0].fields == ()
    assert lane.unattributed == ()


# ---- cleaning and grounding ---------------------------------------------------------------------


def test_a_numeric_value_carrying_no_digit_is_dropped_with_its_reason():
    # The prompt forbids qualitative answers, but a prompt is a request; the same rule that guards document
    # mode has to guard this one, or the two modes would disagree about what counts as a value.
    client = FakeLlmClient(responder(sheet_resistance=values_json({"sample_id": "A", "value_raw": "minimum"})))

    lane = extract(client)

    assert any("sheet_resistance: non-numeric value 'minimum'" in entry for entry in lane.dropped)
    assert lane.values() == ()


def test_an_unattributed_value_is_grounded_like_any_other():
    # Grounding is what makes a citation mean anything; a value that escaped attribution must not also
    # escape the check that it was really read where it says it was.
    shown = make_blocks()[1].source_id
    client = FakeLlmClient(
        responder(
            inventory=inventory_json(TWO_SAMPLES),
            sheet_resistance=values_json({"sample_id": None, "value_raw": "99.9", "source_ids": [shown]}),
        )
    )

    lane = extract(client)

    assert lane.unattributed[0].grounded is False
    assert lane.ungrounded() == lane.unattributed


def test_an_unattributed_value_found_in_the_block_it_cites_is_grounded():
    shown = make_blocks()[1].source_id
    client = FakeLlmClient(
        responder(
            inventory=inventory_json(TWO_SAMPLES),
            sheet_resistance=values_json({"sample_id": None, "value_raw": "12.5", "source_ids": [shown]}),
        )
    )

    lane = extract(client)

    assert lane.unattributed[0].grounded is True
    assert lane.ungrounded() == ()


# ---- repeated passes -----------------------------------------------------------------------------


def test_three_passes_run_the_whole_flow_three_times():
    client = FakeLlmClient(responder())

    lane = extract(client, passes=3)

    assert client.call_count == 15  # (1 inventory + 4 fields) x 3
    assert lane.passes == 3


def test_each_pass_after_the_first_carries_its_own_cache_salt():
    # The prompts are byte-identical across passes, so without a salt the second pass would replay the
    # first from the request cache and self-consistency would measure nothing.
    client = FakeLlmClient(responder())

    extract(client, passes=3)

    assert [call.cache_salt for call in client.calls] == [""] * 5 + ["pass-1"] * 5 + ["pass-2"] * 5


def test_a_value_only_one_pass_of_three_produced_is_dropped():
    inventory = inventory_json()
    agreed = {"sample_id": "A", "value_raw": "12.5"}
    client = FakeLlmClient(
        passage_responder(
            inventory,
            [
                {ASKED_FIELD: values_json(agreed)},
                {ASKED_FIELD: values_json(agreed)},
                {ASKED_FIELD: values_json(agreed, {"sample_id": "A", "value_raw": "99.9"})},
            ],
        )
    )

    lane = extract(client, passes=3)

    assert [field.value_raw for field in lane.samples[0].fields] == ["12.5"]
    assert any("only 1/3 passes produced '99.9'" in entry for entry in lane.dropped)


def test_an_unattributed_value_a_majority_of_passes_produced_survives_voting():
    # Agreeing three times that a value cannot be placed is still agreement about the value itself, so
    # voting has to reach the unplaced ones too -- otherwise they would vanish whenever passes > 1.
    inventory = inventory_json(TWO_SAMPLES)
    unplaced = {"sample_id": None, "value_raw": "12.5"}
    client = FakeLlmClient(passage_responder(inventory, [{ASKED_FIELD: values_json(unplaced)} for _ in range(3)]))

    lane = extract(client, passes=3)

    assert [field.value_raw for field in lane.unattributed] == ["12.5"]
    assert lane.unattributed[0].agreement == 1.0


# ---- the mode itself ------------------------------------------------------------------------------


def test_passage_mode_stores_a_different_extractor_key_than_document_mode():
    # The two modes read different text and produce different results, so one's cached facts must never be
    # served under the other's name.
    client = FakeLlmClient(responder())

    lane = extract(client)

    assert lane.extractor_key == extractor_key(client.model, mode="passage")
    assert lane.extractor_key != extractor_key(client.model, mode="document")


def test_an_unknown_mode_is_rejected_before_any_call_is_made():
    client = FakeLlmClient(responder())

    with pytest.raises(ValueError, match="unknown extraction mode"):
        extract_lane(make_artifact(make_blocks()), client, mode="passages")  # type: ignore[arg-type]

    assert client.call_count == 0


# ---- Repeats, and what counts as the same value ------------------------------------------------------


def test_the_same_value_quoted_twice_becomes_one_value_carrying_both_citations():
    # A field question routinely gets the same number back from the table and again from the sentence
    # discussing it. That is one fact with two citations, not two facts, and counting it twice would
    # hand the comparison a duplicate to pair against. Both blocks here mention the field, so both are
    # retrieved as candidates and both citations are ones the model was actually shown.
    blocks = (
        *make_blocks(),
        make_block(page=0, order=2, content="Table 1 lists a sheet resistance of 12.5 ohm/sq for Sample A."),
    )
    cited = [blocks[1].source_id, blocks[2].source_id]
    client = FakeLlmClient(
        responder(
            sheet_resistance=values_json(
                {"sample_id": "A", "value_raw": "12.5", "unit_raw": "ohm/sq", "source_ids": [cited[0]]},
                {"sample_id": "A", "value_raw": "12.5", "unit_raw": "ohm/sq", "source_ids": [cited[1]]},
            )
        )
    )

    lane = extract_lane(make_artifact(blocks), client, mode="passage")

    values = lane.samples[0].fields
    assert [value.value_raw for value in values] == ["12.5"]
    assert set(values[0].source_ids) == set(cited)


def test_the_same_number_in_two_units_stays_two_values():
    # "2.1 μm" and "2.1 nm" differ by a factor of a thousand: merging them would delete a parser
    # disagreement, which is the one thing this pipeline exists to surface.
    blocks = make_blocks()
    cited = blocks[1].source_id
    client = FakeLlmClient(
        responder(
            sheet_resistance=values_json(
                {"sample_id": "A", "value_raw": "12.5", "unit_raw": "ohm/sq", "source_ids": [cited]},
                {"sample_id": "A", "value_raw": "12.5", "unit_raw": "kohm/sq", "source_ids": [cited]},
            )
        )
    )

    lane = extract_lane(make_artifact(blocks), client, mode="passage")

    assert [value.unit_raw for value in lane.samples[0].fields] == ["ohm/sq", "kohm/sq"]


# ---- The inventory's own mistakes ---------------------------------------------------------------------


def test_a_sample_listed_twice_under_one_id_is_kept_once_and_audited():
    # The inventory prompt demands unique ids. A repeat means the model conflated two samples, and every
    # value later attributed to that id would land on the first of them, so the collision is recorded.
    blocks = make_blocks()
    client = FakeLlmClient(
        responder(
            inventory=inventory_json(
                [
                    {"sample_id": "A", "label": "first", "conditions": {"flow": "100 sccm"}},
                    {"sample_id": "a ", "label": "second", "conditions": {"flow": "200 sccm"}},
                ]
            )
        )
    )

    lane = extract_lane(make_artifact(blocks), client, mode="passage")

    assert [sample.label for sample in lane.samples] == ["first"]
    assert any("repeats an id" in entry for entry in lane.dropped)


def test_a_sample_with_a_blank_id_is_dropped_with_a_reason():
    # min_length on the response model still admits "  ", and a sample with no id can be neither matched
    # across lanes nor pointed at by a value.
    blocks = make_blocks()
    client = FakeLlmClient(responder(inventory=inventory_json([{"sample_id": "  ", "label": "nameless"}])))

    lane = extract_lane(make_artifact(blocks), client, mode="passage")

    assert lane.samples == ()
    assert any("no usable id" in entry for entry in lane.dropped)
