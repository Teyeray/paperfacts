"""Sample identity matching: exact-pair by normalized sample_id first, then hand the rest to the model.

The division of labor is fixed by the PRD — **sample identity is judged by the model, rules only do
field-level comparison**. So this file guards two things: the deterministic part never calls the model
(never spending money on it), and whatever the model answers is treated as a mere "suggestion" — a
pairing whose ids don't check out gets discarded rather than trusted.
"""

from __future__ import annotations

import json

from paperfacts.consensus.matching import SampleMatching, match_samples
from support.extraction import make_field, make_lane, make_sample
from support.llm import FakeLlmClient


def lanes(a_ids, b_ids, **kwargs):
    """A minimal LaneExtraction pair where each lane's samples only carry a sample_id."""
    return (
        make_lane(backend="mineru", samples=[make_sample(i, **kwargs) for i in a_ids]),
        make_lane(backend="paddleocr_vl", samples=[make_sample(i, **kwargs) for i in b_ids]),
    )


def matching_json(pairs=(), unmatched_a=(), unmatched_b=()) -> str:
    return json.dumps({"pairs": list(pairs), "unmatched_a": list(unmatched_a), "unmatched_b": list(unmatched_b)})


# ---- Exact pairing: costs nothing ---------------------------------------------------------------


def test_identical_sample_ids_pair_up_without_calling_the_model():
    lane_a, lane_b = lanes(["A", "B"], ["A", "B"])
    client = FakeLlmClient([])

    matching = match_samples(lane_a, lane_b, client)

    assert client.call_count == 0
    assert [(p.a_id, p.b_id, p.method) for p in matching.pairs] == [("A", "A", "exact"), ("B", "B", "exact")]
    assert matching.unmatched_a == () and matching.unmatched_b == ()


def test_exact_pairs_are_full_confidence_and_say_why():
    lane_a, lane_b = lanes(["A"], ["A"])

    matching = match_samples(lane_a, lane_b, FakeLlmClient([]))

    assert matching.pairs[0].confidence == 1.0
    assert matching.pairs[0].justification == "identical sample_id"


def test_ids_that_differ_only_in_case_spacing_or_decoration_still_pair_exactly():
    # Identical after normalization counts as the same id; the two lanes OCR'ing "Film #2" and "film 2"
    # is routine.
    lane_a, lane_b = lanes(["Film #2"], ["film 2"])

    matching = match_samples(lane_a, lane_b, FakeLlmClient([]))

    assert [(p.a_id, p.b_id) for p in matching.pairs] == [("Film #2", "film 2")]


def test_a_hyphen_is_meaningful_and_keeps_two_ids_apart():
    # Documenting current behavior: '-' is in the key whitelist, so "O2-100 sccm" and "O2 100sccm" do not
    # pair exactly and fall through to model judgment.
    lane_a, lane_b = lanes(["O2-100 sccm"], ["O2 100sccm"])
    client = FakeLlmClient([matching_json()])

    matching = match_samples(lane_a, lane_b, client)

    assert matching.pairs == ()
    assert client.call_count == 1


def test_no_model_call_when_the_exact_pass_consumed_one_side_entirely():
    # Every sample in A got paired, leaving only B's leftovers — there's no combination left to judge, so
    # calling the model would be pure waste.
    lane_a, lane_b = lanes(["A"], ["A", "B"])
    client = FakeLlmClient([])

    matching = match_samples(lane_a, lane_b, client)

    assert client.call_count == 0
    assert matching.unmatched_b == ("B",)


# ---- One side empty ----------------------------------------------------------------------


def test_an_empty_lane_means_everything_is_unmatched_and_no_model_call():
    lane_a, lane_b = lanes(["A", "B"], [])
    client = FakeLlmClient([])

    matching = match_samples(lane_a, lane_b, client)

    assert client.call_count == 0
    assert matching.unmatched_a == ("A", "B")
    assert matching.unmatched_b == ()
    assert matching.pairs == ()


def test_two_empty_lanes_produce_an_empty_matching():
    lane_a, lane_b = lanes([], [])

    matching = match_samples(lane_a, lane_b, FakeLlmClient([]))

    assert matching == SampleMatching()
    assert matching.failed is False


# ---- Model-based pairing ----------------------------------------------------------------


def test_the_model_pairs_the_samples_that_the_exact_pass_could_not():
    lane_a, lane_b = lanes(["A1"], ["B1"])
    client = FakeLlmClient([matching_json([{"a": "A1", "b": "B1", "confidence": 0.9, "justification": "same O2"}])])

    matching = match_samples(lane_a, lane_b, client)

    assert client.call_count == 1
    assert [(p.a_id, p.b_id, p.method, p.confidence) for p in matching.pairs] == [("A1", "B1", "llm", 0.9)]
    assert matching.unmatched_a == () and matching.unmatched_b == ()


def test_the_model_usage_and_raw_answer_are_kept():
    lane_a, lane_b = lanes(["A1"], ["B1"])
    client = FakeLlmClient([matching_json()], usage={"total_tokens": 33})

    matching = match_samples(lane_a, lane_b, client)

    assert matching.usage == {"total_tokens": 33}
    assert matching.raw_response == matching_json()


def test_exact_and_model_pairs_are_counted_together():
    lane_a, lane_b = lanes(["A", "X1"], ["A", "Y1"])
    client = FakeLlmClient([matching_json([{"a": "X1", "b": "Y1", "confidence": 0.8, "justification": "same power"}])])

    matching = match_samples(lane_a, lane_b, client)

    assert [p.method for p in matching.pairs] == ["exact", "llm"]
    assert len(matching.pairs) == 2
    assert matching.unmatched_a == () and matching.unmatched_b == ()


def test_only_the_leftovers_are_shown_to_the_model():
    # Samples already paired exactly should never re-enter the prompt: it would waste tokens and give the
    # model a chance to overturn a settled pairing.
    lane_a, lane_b = lanes(["A", "X1"], ["A", "Y1"])
    client = FakeLlmClient([matching_json()])

    match_samples(lane_a, lane_b, client)
    prompt = client.users[0]

    assert "X1" in prompt and "Y1" in prompt
    assert "id: A" not in prompt


# ---- Sanitizing the model's answer ----------------------------------------------------------


def test_a_pair_naming_an_id_that_does_not_exist_is_ignored():
    # The model can hallucinate ids. Trusting it would put a pairing pointing at a nonexistent sample into
    # the report.
    lane_a, lane_b = lanes(["A1"], ["B1"])
    client = FakeLlmClient([matching_json([{"a": "A1", "b": "GHOST", "confidence": 1.0, "justification": "x"}])])

    matching = match_samples(lane_a, lane_b, client)

    assert matching.pairs == ()
    assert matching.unmatched_a == ("A1",) and matching.unmatched_b == ("B1",)


def test_a_second_pair_reusing_an_id_is_ignored():
    # Each sample belongs to at most one pairing; otherwise the same fact would get compared twice.
    lane_a, lane_b = lanes(["A1", "A2"], ["B1"])
    client = FakeLlmClient(
        [
            matching_json(
                [
                    {"a": "A1", "b": "B1", "confidence": 0.9, "justification": "first"},
                    {"a": "A2", "b": "B1", "confidence": 0.9, "justification": "duplicate b"},
                ]
            )
        ]
    )

    matching = match_samples(lane_a, lane_b, client)

    assert [(p.a_id, p.b_id) for p in matching.pairs] == [("A1", "B1")]
    assert matching.unmatched_a == ("A2",)


def test_an_unparseable_answer_gets_one_repair_attempt():
    # Same as extraction: feed the validation error back and ask once more, rather than giving up on the
    # whole document's sample alignment over one formatting hiccup.
    lane_a, lane_b = lanes(["A1"], ["B1"])
    client = FakeLlmClient(
        ["I think A1 and B1 are the same sample.", matching_json([{"a": "A1", "b": "B1", "confidence": 0.9}])]
    )

    matching = match_samples(lane_a, lane_b, client)

    assert client.call_count == 2
    assert "not valid JSON" in client.users[1]
    assert [(p.a_id, p.b_id) for p in matching.pairs] == [("A1", "B1")]
    assert matching.failed is False


def test_two_unparseable_answers_mark_the_matching_as_failed_without_raising():
    """A matching failure shouldn't crash the whole comparison, but it also must **not** pretend "the two
    lanes truly have no match."

    ``failed=True`` lets the comparison layer mark unmatched samples ambiguous (for review) rather than
    missing (silently accepted) — when uncertain, lean toward the lower-risk outcome.
    """
    lane_a, lane_b = lanes(["A1"], ["B1"])
    client = FakeLlmClient(["not json", "still not json"])

    matching = match_samples(lane_a, lane_b, client)

    assert client.call_count == 2
    assert matching.failed is True
    assert matching.failure and "_MatchingResponse" in matching.failure
    assert matching.pairs == ()
    assert matching.unmatched_a == ("A1",) and matching.unmatched_b == ("B1",)


def test_a_failed_matching_still_keeps_the_exact_pairs():
    # Exact pairings are deterministic; a model failure shouldn't discard them too.
    lane_a, lane_b = lanes(["A", "X1"], ["A", "Y1"])
    client = FakeLlmClient(["nope", "nope again"])

    matching = match_samples(lane_a, lane_b, client)

    assert [(p.a_id, p.method) for p in matching.pairs] == [("A", "exact")]
    assert matching.failed is True


def test_an_answer_with_an_out_of_range_confidence_is_treated_as_unparseable():
    lane_a, lane_b = lanes(["A1"], ["B1"])
    bad = matching_json([{"a": "A1", "b": "B1", "confidence": 7, "justification": "x"}])
    client = FakeLlmClient([bad, bad])

    matching = match_samples(lane_a, lane_b, client)

    assert matching.pairs == ()
    assert matching.failed is True


def test_a_pair_without_a_confidence_gets_a_neutral_default():
    lane_a, lane_b = lanes(["A1"], ["B1"])
    client = FakeLlmClient([matching_json([{"a": "A1", "b": "B1"}])])

    matching = match_samples(lane_a, lane_b, client)

    assert matching.pairs[0].confidence == 0.5
    assert matching.pairs[0].justification == ""


def test_the_unmatched_lists_from_the_model_are_not_trusted_verbatim():
    """The unmatched lists are computed by us from "which ids were not consumed by an accepted pairing,"
    never copied from the model's answer.

    The model routinely omits or over-lists unmatched ids; copying it would make ids vanish or appear
    twice out of nowhere.
    """
    lane_a, lane_b = lanes(["A1"], ["B1"])
    client = FakeLlmClient([matching_json(unmatched_a=["A1", "PHANTOM"], unmatched_b=[])])

    matching = match_samples(lane_a, lane_b, client)

    assert matching.unmatched_a == ("A1",)
    assert matching.unmatched_b == ("B1",)


# ---- What goes into the prompt -------------------------------------------------------------


def test_the_prompt_lists_the_id_label_conditions_and_raw_field_values():
    """Matching is justified by preparation conditions and names, so these must actually appear in the prompt."""
    lane_a = make_lane(
        backend="mineru",
        samples=[
            make_sample(
                "S1",
                [make_field("sheet_resistance", "12.5", unit_raw="Ω/sq", condition="RT")],
                label="high flow",
                conditions={"O2 flow": "100 sccm"},
            )
        ],
    )
    lane_b = make_lane(backend="paddleocr_vl", samples=[make_sample("T1")])
    client = FakeLlmClient([matching_json()])

    match_samples(lane_a, lane_b, client)
    prompt = client.users[0]

    assert "id: S1" in prompt
    assert "label: high flow" in prompt
    assert "O2 flow=100 sccm" in prompt
    assert "sheet_resistance=12.5 Ω/sq @RT" in prompt


def test_a_sample_without_label_conditions_or_fields_renders_placeholders():
    lane_a, lane_b = lanes(["S1"], ["T1"])
    client = FakeLlmClient([matching_json()])

    match_samples(lane_a, lane_b, client)

    assert "id: S1 | label: - | conditions: - | fields: -" in client.users[0]


def test_both_backend_names_reach_the_prompt():
    # The report has to spell out "which lane is A"; the prompt needs the same clarity.
    lane_a, lane_b = lanes(["S1"], ["T1"])
    client = FakeLlmClient([matching_json()])

    match_samples(lane_a, lane_b, client)

    assert "mineru" in client.users[0] and "paddleocr_vl" in client.users[0]


# ---- refresh ------------------------------------------------------------------------


def test_refresh_is_forwarded_to_the_client():
    lane_a, lane_b = lanes(["A1"], ["B1"])
    client = FakeLlmClient([matching_json()])

    match_samples(lane_a, lane_b, client, refresh=True)

    assert client.refreshes == [True]


def test_refresh_defaults_to_false():
    lane_a, lane_b = lanes(["A1"], ["B1"])
    client = FakeLlmClient([matching_json()])

    match_samples(lane_a, lane_b, client)

    assert client.refreshes == [False]
