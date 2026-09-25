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
import threading
import time

import pytest

from paperfacts.config import INHERIT, InventoryReasoningEffort
from paperfacts.extract import extract_lane
from paperfacts.fields import FIELD_SPECS
from paperfacts.keys import ExtractionOptions, extractor_key
from paperfacts.prompts import field_system_prompt, inventory_system_prompt
from support.extraction import lane_options, make_artifact
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

    Every pass sends byte-identical prompts, so the pass can only be told apart by counting. The inventory
    is asked once for the whole lane, so it cannot mark the boundary: a pass ends when a field is asked
    about for the second time.
    """
    state: dict = {"pass_index": 0, "asked": set()}

    def respond(system: str, user: str) -> str:
        if system == inventory_system_prompt():
            return inventory
        field = field_of(user)
        if field in state["asked"]:
            state["pass_index"] += 1
            state["asked"] = set()
        state["asked"].add(field)
        return per_pass[state["pass_index"]].get(field, values_json())

    return respond


def extract(
    client: FakeLlmClient,
    *,
    backend: str = "mineru",
    passes: int = 1,
    concurrency: int = 1,
    inventory_reasoning_effort: InventoryReasoningEffort = INHERIT,
):
    return extract_lane(
        make_artifact(make_blocks(backend), backend=backend),
        client,
        lane_options(client, mode="passage", passes=passes, inventory_reasoning_effort=inventory_reasoning_effort),
        concurrency=concurrency,
    )


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
    # All but the four asked fields go unasked, which is why only four field calls were made.
    unasked = len(FIELD_SPECS) - len(ASKED_FIELDS)
    assert sum(1 for entry in lane.dropped if "was not asked about" in entry) == unasked


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


def test_a_paper_depositing_no_tco_film_is_asked_only_paper_level_fields():
    # A device paper on purchased ITO glass: its sample-level answers could only be some other layer's.
    client = FakeLlmClient(responder(inventory=json.dumps({"samples": [], "no_tco_film": True})))

    lane = extract(client)

    assert client.call_count == 1
    assert lane.samples == () and lane.unattributed == ()
    assert any(entry.startswith(f"{ASKED_FIELD}: the paper deposits no TCO film") for entry in lane.dropped)


def test_an_inventory_empty_for_any_other_reason_still_gets_every_question():
    # The inventory may simply have missed the sample text; that must not cost the lane its values.
    client = FakeLlmClient(
        responder(inventory=inventory_json([]), sheet_resistance=values_json({"sample_id": None, "value_raw": "12.5"}))
    )

    lane = extract(client)

    assert client.call_count == 5
    assert [field.value_raw for field in lane.unattributed] == ["12.5"]


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

    lane = extract_lane(make_artifact(blocks), client, lane_options(client, mode="passage"))

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


def test_three_passes_repeat_the_field_questions_but_ask_the_inventory_once():
    # Which samples exist is a fact about the paper, not a measurement to average: re-asking it let the
    # model rename the samples between passes, and a renamed sample is a scope no value can reach a
    # majority in. Only the field questions, whose noise the vote exists to filter, are repeated.
    client = FakeLlmClient(responder())

    lane = extract(client, passes=3)

    assert client.call_count == 13  # 1 inventory + 4 fields x 3
    assert sum(call.system == inventory_system_prompt() for call in client.calls) == 1
    assert lane.passes == 3


def test_each_pass_after_the_first_carries_its_own_cache_salt():
    # The prompts are byte-identical across passes, so without a salt the second pass would replay the
    # first from the request cache and self-consistency would measure nothing.
    client = FakeLlmClient(responder())

    extract(client, passes=3)

    # The lone inventory call opens the lane on pass 0's empty salt, so a lane re-run with more passes
    # still hits the inventory entry an earlier run cached.
    assert [call.cache_salt for call in client.calls] == [""] * 5 + ["pass-1"] * 4 + ["pass-2"] * 4


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


# ---- the inventory question's own reasoning effort --------------------------------------------------


def test_the_inventory_effort_reaches_the_inventory_question_and_nothing_else():
    # The inventory question reasons an order of magnitude longer than the field questions after it, so it
    # is the one worth turning down. Turning the others down too would change what they answer.
    client = FakeLlmClient(responder())

    extract(client, inventory_reasoning_effort="none")

    inventory = [call for call in client.calls if call.system == inventory_system_prompt()]
    fields = [call for call in client.calls if call.system != inventory_system_prompt()]
    assert [call.reasoning_effort for call in inventory] == ["none"]
    assert fields and all(call.reasoning_effort is INHERIT for call in fields)


def test_the_inventory_question_can_omit_the_parameter_the_client_still_sends():
    # None is the third meaning the sentinel separates out: this one question goes without a
    # reasoning_effort parameter while every field question keeps the client's own.
    client = FakeLlmClient(responder())

    extract(client, inventory_reasoning_effort=None)

    inventory = [call for call in client.calls if call.system == inventory_system_prompt()]
    fields = [call for call in client.calls if call.system != inventory_system_prompt()]
    assert [call.reasoning_effort for call in inventory] == [None]
    assert fields and all(call.reasoning_effort is INHERIT for call in fields)


def test_without_the_setting_every_question_inherits_the_clients_effort():
    client = FakeLlmClient(responder())

    extract(client)

    assert all(call.reasoning_effort is INHERIT for call in client.calls)


def test_the_inventory_effort_is_stored_in_the_extractor_key():
    client = FakeLlmClient(responder())

    lane = extract(client, inventory_reasoning_effort="none")

    assert lane.extractor_key == extractor_key(
        ExtractionOptions(client.model, mode="passage", inventory_reasoning_effort="none")
    )
    assert lane.extractor_key != extractor_key(ExtractionOptions(client.model, mode="passage"))


# ---- the mode itself ------------------------------------------------------------------------------


def test_passage_mode_stores_a_different_extractor_key_than_document_mode():
    # The two modes read different text and produce different results, so one's cached facts must never be
    # served under the other's name.
    client = FakeLlmClient(responder())

    lane = extract(client)

    assert lane.extractor_key == extractor_key(ExtractionOptions(client.model, mode="passage"))
    assert lane.extractor_key != extractor_key(ExtractionOptions(client.model, mode="document"))


def test_an_unknown_mode_is_rejected_before_any_call_is_made():
    client = FakeLlmClient(responder())

    with pytest.raises(ValueError, match="unknown extraction mode"):
        extract_lane(make_artifact(make_blocks()), client, lane_options(client, mode="passages"))  # type: ignore[arg-type]

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

    lane = extract_lane(make_artifact(blocks), client, lane_options(client, mode="passage"))

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

    lane = extract_lane(make_artifact(blocks), client, lane_options(client, mode="passage"))

    assert [value.unit_raw for value in lane.samples[0].fields] == ["ohm/sq", "kohm/sq"]


# ---- The inventory's own mistakes ---------------------------------------------------------------------


def test_a_paper_level_value_flagged_for_every_sample_is_not_stored_as_a_series_value():
    # The flag only means something for a sample-level field; on the target it used to leave series=True.
    client = FakeLlmClient(
        responder(
            inventory=inventory_json(TWO_SAMPLES),
            component=values_json({"sample_id": None, "value_raw": "ITO", "applies_to_all_samples": True}),
        )
    )
    blocks = (*make_blocks(), make_block(page=0, order=2, content="The target composition was ITO."))

    lane = extract_lane(make_artifact(blocks), client, lane_options(client, mode="passage"))

    assert lane.target.get("component").series is False
    assert all(sample.fields == () for sample in lane.samples)


def test_samples_distinguished_by_a_greek_letter_or_a_suffix_case_stay_separate():
    # normalize_key merged "α-ITO" with "β-ITO" (and "ITO-a" with "ITO-A"): the second was dropped as a
    # repeat and every value naming it landed on the first.
    samples = [
        {"sample_id": "α-ITO", "label": "", "conditions": {}},
        {"sample_id": "β-ITO", "label": "", "conditions": {}},
        {"sample_id": "ITO-a", "label": "", "conditions": {}},
        {"sample_id": "ITO-A", "label": "", "conditions": {}},
    ]
    client = FakeLlmClient(
        responder(
            inventory=inventory_json(samples),
            sheet_resistance=values_json(
                {"sample_id": "β-ITO", "value_raw": "20"},
                {"sample_id": "ITO-A", "value_raw": "30"},
            ),
        )
    )

    lane = extract(client)

    assert [sample.sample_id for sample in lane.samples] == ["α-ITO", "β-ITO", "ITO-a", "ITO-A"]
    assert lane.sample("β-ITO").get("sheet_resistance").value_raw == "20"
    assert lane.sample("ITO-A").get("sheet_resistance").value_raw == "30"
    assert lane.sample("α-ITO").fields == lane.sample("ITO-a").fields == ()
    assert not any("repeats an id" in entry for entry in lane.dropped)


def test_a_sample_whose_id_has_no_letter_or_digit_is_dropped_with_a_reason():
    client = FakeLlmClient(
        responder(inventory=inventory_json([{"sample_id": "#", "label": "", "conditions": {}}, *TWO_SAMPLES]))
    )

    lane = extract(client)

    assert [sample.sample_id for sample in lane.samples] == ["A", "B"]
    assert "inventory: a sample was listed with no usable id" in lane.dropped


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

    lane = extract_lane(make_artifact(blocks), client, lane_options(client, mode="passage"))

    assert [sample.label for sample in lane.samples] == ["first"]
    assert any("repeats an id" in entry for entry in lane.dropped)


def test_a_sample_with_a_blank_id_is_dropped_with_a_reason():
    # min_length on the response model still admits "  ", and a sample with no id can be neither matched
    # across lanes nor pointed at by a value.
    blocks = make_blocks()
    client = FakeLlmClient(responder(inventory=inventory_json([{"sample_id": "  ", "label": "nameless"}])))

    lane = extract_lane(make_artifact(blocks), client, lane_options(client, mode="passage"))

    assert lane.samples == ()
    assert any("no usable id" in entry for entry in lane.dropped)


# ---- the field questions overlap ----------------------------------------------------------------
# `concurrency` is a scheduling knob and nothing else: the same questions, the same answers, the same
# records. These cases pin that down, because a shared dict mutated from a worker thread would show up
# here as a usage total that drifts between runs rather than as an exception.


class SlowResponder:
    """A responder that sleeps inside every field question and records how many overlapped."""

    def __init__(self, delay: float = 0.05) -> None:
        self._delay = delay
        self._lock = threading.Lock()
        self._in_flight = 0
        self.peak = 0

    def __call__(self, system: str, user: str) -> str:
        if system == inventory_system_prompt():
            return inventory_json()
        with self._lock:
            self._in_flight += 1
            self.peak = max(self.peak, self._in_flight)
        time.sleep(self._delay)
        with self._lock:
            self._in_flight -= 1
        return values_json({"sample_id": "A", "value_raw": "12.5", "unit_raw": "ohm/sq", "source_ids": []})


def test_several_field_questions_are_in_flight_at_once():
    slow = SlowResponder()

    extract(FakeLlmClient(slow), concurrency=4)

    # This paper asks four field questions; all four should be waiting on the endpoint together.
    assert slow.peak == 4


def test_concurrency_of_one_sends_the_questions_strictly_one_after_another():
    slow = SlowResponder()

    extract(FakeLlmClient(slow), concurrency=1)

    assert slow.peak == 1


def test_a_concurrent_run_produces_the_same_records_and_the_same_usage_as_a_sequential_one():
    answers = responder(
        sheet_resistance=values_json({"sample_id": "A", "value_raw": "12.5", "unit_raw": "ohm/sq"}),
        ar_flow_rate=values_json({"sample_id": "A", "value_raw": "100", "unit_raw": "sccm"}),
    )

    sequential = extract(FakeLlmClient(answers), concurrency=1)
    concurrent = extract(FakeLlmClient(answers), concurrency=4)

    assert concurrent.usage == sequential.usage
    assert concurrent.model_dump(exclude={"created_at"}) == sequential.model_dump(exclude={"created_at"})


def test_the_raw_response_keeps_the_field_order_whatever_the_concurrency():
    """The stored transcript is read by a human, so its sections stay in FIELD_SPECS order."""
    sequential = extract(FakeLlmClient(responder()), concurrency=1)
    concurrent = extract(FakeLlmClient(responder()), concurrency=4)

    assert concurrent.raw_response == sequential.raw_response
    assert re.findall(r"^# (\S+)$", concurrent.raw_response, re.MULTILINE)[0] == "inventory"


def test_a_field_question_that_fails_propagates_instead_of_being_swallowed():
    def explode(system: str, user: str) -> str:
        if system == inventory_system_prompt():
            return inventory_json()
        raise RuntimeError("the endpoint refused the field question")

    with pytest.raises(RuntimeError, match="refused the field question"):
        extract(FakeLlmClient(explode), concurrency=4)


def test_a_concurrency_below_one_is_rejected_before_any_call_is_made():
    client = FakeLlmClient([])

    with pytest.raises(ValueError, match="concurrency must be at least 1"):
        extract(client, concurrency=0)

    assert client.call_count == 0


# ---- series values: stated once for the whole sample list ---------------------------------------


THREE_SAMPLES = [
    {"sample_id": "A", "label": "O2 100 sccm", "conditions": {"flow": "100 sccm"}},
    {"sample_id": "B", "label": "O2 200 sccm", "conditions": {"flow": "200 sccm"}},
    {"sample_id": "C", "label": "O2 300 sccm", "conditions": {"flow": "300 sccm"}},
]


def test_the_field_question_asks_for_the_series_flag_and_says_when_it_is_true():
    system = field_system_prompt()

    assert '"applies_to_all_samples"' in system
    assert "applies_to_all_samples` is true ONLY when the excerpt states the value holds for every sample" in system
    assert "`sample_id` must be null" in system


def test_a_series_value_is_written_onto_every_sample_keeping_its_citation():
    # "Ar flow 3 sccm (deposition of all GZO films)": the paper placed it on every sample at once, so
    # leaving it unattributed would compare real information with nothing.
    shown = make_blocks()[1].source_id
    client = FakeLlmClient(
        responder(
            inventory=inventory_json(THREE_SAMPLES),
            sheet_resistance=values_json(
                {
                    "sample_id": None,
                    "value_raw": "12.5",
                    "source_ids": [shown],
                    "applies_to_all_samples": True,
                }
            ),
        )
    )

    lane = extract(client)

    assert [sample.sample_id for sample in lane.samples] == ["A", "B", "C"]
    for sample in lane.samples:
        assert [(field.value_raw, field.series, field.source_ids) for field in sample.fields] == [
            ("12.5", True, (shown,))
        ]
    assert lane.unattributed == ()


def test_a_value_without_the_series_flag_is_still_left_unattributed():
    # The fan-out is the model's explicit claim about the excerpt; a plain null id keeps meaning "unplaced".
    client = FakeLlmClient(
        responder(
            inventory=inventory_json(THREE_SAMPLES),
            sheet_resistance=values_json({"sample_id": None, "value_raw": "12.5"}),
        )
    )

    lane = extract(client)

    assert [field.value_raw for field in lane.unattributed] == ["12.5"]
    assert [sample.fields for sample in lane.samples] == [(), (), ()]


def test_the_series_flag_is_ignored_when_the_value_names_a_sample():
    # An id and the flag contradict each other; the id is the more specific claim, so it wins.
    client = FakeLlmClient(
        responder(
            inventory=inventory_json(THREE_SAMPLES),
            sheet_resistance=values_json({"sample_id": "B", "value_raw": "12.5", "applies_to_all_samples": True}),
        )
    )

    lane = extract(client)

    assert [(sample.sample_id, [(f.value_raw, f.series) for f in sample.fields]) for sample in lane.samples] == [
        ("A", []),
        ("B", [("12.5", False)]),
        ("C", []),
    ]
    assert lane.unattributed == ()


def test_an_answer_that_omits_the_series_flag_is_accepted_as_not_a_series_value():
    client = FakeLlmClient(responder(sheet_resistance=values_json({"sample_id": "A", "value_raw": "12.5"})))

    lane = extract(client)

    assert lane.samples[0].fields[0].series is False


def test_the_series_flag_survives_a_round_trip_through_the_stored_lane():
    client = FakeLlmClient(
        responder(
            inventory=inventory_json(THREE_SAMPLES),
            sheet_resistance=values_json({"sample_id": None, "value_raw": "12.5", "applies_to_all_samples": True}),
        )
    )

    lane = extract(client)
    reread = type(lane).model_validate_json(lane.model_dump_json())

    assert [field.series for sample in reread.samples for field in sample.fields] == [True, True, True]


@pytest.mark.parametrize("series_first", [True, False])
def test_a_series_value_and_a_quote_naming_the_sample_become_one_sample_specific_value(series_first):
    # The same number arrives twice on sample B: once fanned out from "all films", once quoted for B
    # itself. They are one fact, and the quote naming B is the more precise claim, so it is what survives.
    blocks = (
        *make_blocks(),
        make_block(page=0, order=2, backend="mineru", content="Sheet resistance was 12.5 ohm/sq for every film."),
    )
    series_id, specific_id = blocks[2].source_id, blocks[1].source_id
    series = {
        "sample_id": None,
        "value_raw": "12.5",
        "source_ids": [series_id],
        "applies_to_all_samples": True,
    }
    specific = {"sample_id": "B", "value_raw": "12.5", "source_ids": [specific_id]}
    answer = values_json(*((series, specific) if series_first else (specific, series)))
    client = FakeLlmClient(responder(inventory=inventory_json(TWO_SAMPLES), sheet_resistance=answer))

    lane = extract_lane(make_artifact(blocks, backend="mineru"), client, lane_options(client, mode="passage"))

    sample_b = lane.sample("B")
    assert [(field.value_raw, field.series) for field in sample_b.fields] == [("12.5", False)]
    assert set(sample_b.fields[0].source_ids) == {series_id, specific_id}
    assert [(field.value_raw, field.series) for field in lane.sample("A").fields] == [("12.5", True)]


def test_a_series_value_with_no_samples_to_place_it_on_is_unattributed_and_unflagged():
    # Nothing to fan out to, so the flag would claim a placement the record does not have.
    client = FakeLlmClient(
        responder(
            inventory=inventory_json([]),
            sheet_resistance=values_json({"sample_id": None, "value_raw": "12.5", "applies_to_all_samples": True}),
        )
    )

    lane = extract(client)

    assert lane.samples == ()
    assert [(field.value_raw, field.series) for field in lane.unattributed] == [("12.5", False)]


# ---- One inventory per lane is what makes the sample vote work --------------------------------


def test_every_pass_field_questions_carry_the_same_sample_list():
    # The defect this guards: re-asking the inventory let the model answer "ITO-O2-0.0sccm-480C" once and
    # "ITO-0.0sccm-480C" the next time. A sample id is the scope its values are voted under, so the two
    # spellings shared no scope, no sample reached a majority, and the lane came back empty.
    client = FakeLlmClient(responder())

    extract(client, passes=3)

    field_questions = [call.user for call in client.calls if call.system != inventory_system_prompt()]
    by_field: dict[str, list[str]] = {}
    for question in field_questions:
        by_field.setdefault(field_of(question), []).append(question)

    assert len(field_questions) == 12  # four fields, three passes
    # Byte-identical across passes, so the sample list rendered into them cannot drift either.
    assert all(asked == [asked[0]] * 3 for asked in by_field.values())


def test_a_sample_named_once_survives_every_pass_and_its_agreed_value_is_unanimous():
    inventory = inventory_json([{"sample_id": "ITO-O2-0.0sccm-480C", "label": "0.0 sccm", "conditions": {}}])
    agreed = {"sample_id": "ITO-O2-0.0sccm-480C", "value_raw": "12.5"}
    only_once = {"sample_id": "ITO-O2-0.0sccm-480C", "value_raw": "99.9"}
    client = FakeLlmClient(
        passage_responder(
            inventory,
            [{ASKED_FIELD: values_json(agreed)}, {ASKED_FIELD: values_json(agreed, only_once)}],
        )
    )

    lane = extract(client, passes=2)

    assert [sample.sample_id for sample in lane.samples] == ["ITO-O2-0.0sccm-480C"]
    assert [(field.value_raw, field.agreement) for field in lane.samples[0].fields] == [("12.5", 1.0)]
    assert any("only 1/2 passes produced '99.9'" in entry for entry in lane.dropped)
