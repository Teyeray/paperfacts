"""The parts of visual validation that are decided by code alone: the reading, the verdict, the region.

No model is called here. The vision stage's own client arrives in a later slice; what is tested is the rule
that the code -- not the model -- decides whether a value survives, and the sliding window that chooses which
pixels the model will be shown.
"""

from __future__ import annotations

import pytest

from paperfacts.crops import Region
from paperfacts.validate import (
    CONTEXT_TYPES,
    VERDICTS,
    Reading,
    adjudicate,
    fill_system_prompt,
    parse_reading,
    region_for,
    region_of_blocks,
    table_transcription_user_prompt,
    transcription_fold,
    validation_system_prompt,
    validation_user_prompt,
    value_key,
)
from support.extraction import make_artifact, make_field
from support.factories import make_block
from support.profiles import make_profile, shipped_profile

# ---- The reading ------------------------------------------------------------------------------------------


def test_a_clean_json_reply_is_read() -> None:
    reading, note = parse_reading('{"transcription": "Rs = 12.5 ohm/sq", "legible": true}')
    assert reading.transcription == "Rs = 12.5 ohm/sq"
    assert reading.legible is True
    assert note == ""


def test_a_fenced_reply_is_read() -> None:
    reading, note = parse_reading('```json\n{"transcription": "40 nm", "legible": true}\n```')
    assert reading.transcription == "40 nm"
    assert note == ""


def test_a_reply_wrapped_in_prose_is_read() -> None:
    reading, note = parse_reading('Sure! Here it is: {"transcription": "40 nm", "legible": true} Hope that helps.')
    assert reading.transcription == "40 nm"
    assert note == ""


def test_a_reply_with_no_json_is_taken_as_the_transcription() -> None:
    """A model that ignored the format but transcribed has still done the useful part of the job."""
    reading, note = parse_reading("Rs = 12.5 ohm/sq")
    assert reading.transcription == "Rs = 12.5 ohm/sq"
    assert reading.legible is True
    assert "not the JSON object" in note


def test_an_empty_reply_is_illegible() -> None:
    reading, note = parse_reading("   ")
    assert reading.legible is False
    assert note


# ---- The verdict ------------------------------------------------------------------------------------------


def test_a_value_in_the_transcription_is_confirmed() -> None:
    verdict, detail = adjudicate(make_field("thickness", "40 nm"), Reading(transcription="The film was 40 nm thick."))
    assert verdict == "confirmed"
    assert "occurs in" in detail


def test_a_value_absent_from_the_transcription_is_contradicted() -> None:
    verdict, _ = adjudicate(make_field("thickness", "40 nm"), Reading(transcription="The film was 65 nm thick."))
    assert verdict == "contradicted"


def test_an_illegible_reading_is_neither() -> None:
    """A region nobody could read must never look like one that was checked and passed."""
    verdict, detail = adjudicate(make_field("thickness", "40 nm"), Reading(transcription="", legible=False))
    assert verdict == "illegible"
    assert "could not read" in detail


def test_a_blank_transcription_is_illegible_even_when_flagged_legible() -> None:
    verdict, _ = adjudicate(make_field("thickness", "40 nm"), Reading(transcription="   ", legible=True))
    assert verdict == "illegible"


def test_the_verdict_uses_groundings_strictness() -> None:
    """A number must not match inside a longer number -- the same rule grounding applies to a parser's text."""
    verdict, _ = adjudicate(make_field("thickness", "5 nm"), Reading(transcription="The film was 235 nm thick."))
    assert verdict == "contradicted"


@pytest.mark.parametrize(
    ("quoted", "transcribed"),
    [
        ("1.2 × 10^-4", "rho = 1.2×10⁻⁴ ohm cm"),  # the model closes the spaces the parser kept
        ("1.2×10^-4", "rho = 1.2 × 10^-4 ohm cm"),  # and the other way round
        ("1.2 x 10^-4", "rho = 1.2×10^-4 ohm cm"),  # ...whichever sign either side used
    ],
)
def test_a_product_is_confirmed_whichever_side_spaces_it(quoted: str, transcribed: str) -> None:
    verdict, _ = adjudicate(make_field("resistivity", quoted), Reading(transcription=transcribed))
    assert verdict == "confirmed"


def test_only_the_value_is_adjudicated_not_its_unit() -> None:
    """The unit lives in ``unit_raw``, and grounding does not fold a symbol to its name (``Ω`` is not ``ohm``).

    So a model spelling the unit its own way cannot contradict the value: what is checked is ``value_raw``.
    Were the unit folded into the check, every symbol the model spells out would read as a wrong value.
    """
    value = make_field("sheet_resistance", "12.5", unit_raw="Ω/sq")
    verdict, _ = adjudicate(value, Reading(transcription="Rs = 12.5 ohm/sq"))
    assert verdict == "confirmed"


def test_the_fold_is_applied_to_both_sides_alike() -> None:
    assert transcription_fold("40 × 10") == transcription_fold("40×10")
    # ...but a space that is a boundary is still a boundary: the fold only closes digit-x-digit.
    assert transcription_fold("5 nm") == "5 nm"


def test_every_verdict_is_a_known_one() -> None:
    assert set(VERDICTS) == {"confirmed", "contradicted", "illegible", "not_checked", "error"}


# ---- The region -------------------------------------------------------------------------------------------


def test_the_region_is_the_cited_block_plus_its_neighbours() -> None:
    blocks = [make_block(page=0, order=i, content=f"block {i}") for i in range(5)]
    artifact = make_artifact(blocks)
    region = region_of_blocks([artifact.block(blocks[2].source_id)], artifact, context=1)

    assert region.source_ids == (blocks[2].source_id,)
    assert set(region.context_ids) == {blocks[1].source_id, blocks[3].source_id}


def test_context_ids_are_in_reading_order() -> None:
    blocks = [make_block(page=0, order=i, content=f"block {i}") for i in range(5)]
    artifact = make_artifact(blocks)
    region = region_of_blocks([artifact.block(blocks[2].source_id)], artifact, context=2)
    assert list(region.context_ids) == [
        blocks[0].source_id,
        blocks[1].source_id,
        blocks[3].source_id,
        blocks[4].source_id,
    ]


def test_a_figure_is_skipped_over_without_being_counted() -> None:
    """Page furniture carries no text worth reading; it would only make the crop larger."""
    blocks = [
        make_block(page=0, order=0, content="wanted before"),
        make_block(page=0, order=1, type="figure", content="/tmp/fig1.png"),
        make_block(page=0, order=2, content="the cited one"),
    ]
    artifact = make_artifact(blocks)
    region = region_of_blocks([artifact.block(blocks[2].source_id)], artifact, context=1)

    assert blocks[1].source_id not in region.context_ids
    assert blocks[0].source_id in region.context_ids
    assert "figure" not in CONTEXT_TYPES


def test_a_neighbour_never_comes_from_another_page() -> None:
    blocks = [
        make_block(page=0, order=0, content="last of page 0"),
        make_block(page=1, order=0, content="the cited one"),
        make_block(page=1, order=1, content="next on page 1"),
    ]
    artifact = make_artifact(blocks)
    region = region_of_blocks([artifact.block(blocks[1].source_id)], artifact, context=2)

    assert region.page == 1
    assert blocks[0].source_id not in region.context_ids


def test_zero_context_is_the_cited_blocks_alone() -> None:
    blocks = [make_block(page=0, order=i, content=f"block {i}") for i in range(3)]
    artifact = make_artifact(blocks)
    region = region_of_blocks([artifact.block(blocks[1].source_id)], artifact, context=0)
    assert region.context_ids == ()


def test_the_box_is_the_union_widened_by_padding() -> None:
    blocks = [make_block(page=0, order=i, content=f"block {i}") for i in range(3)]
    artifact = make_artifact(blocks)
    tight = region_of_blocks([artifact.block(blocks[1].source_id)], artifact, context=0, padding=0.0)
    padded = region_of_blocks([artifact.block(blocks[1].source_id)], artifact, context=0, padding=0.02)
    assert padded.bbox.x1 <= tight.bbox.x1 and padded.bbox.y2 >= tight.bbox.y2


def test_region_for_uses_the_first_cited_page_only() -> None:
    """A box spanning two pages is not a region of any page, so the other page's citation is dropped."""
    blocks = [make_block(page=0, order=0, content="here"), make_block(page=1, order=0, content="and there")]
    artifact = make_artifact(blocks)
    value = make_field("thickness", "40 nm", source_ids=[blocks[0].source_id, blocks[1].source_id])
    region = region_for(value, artifact, context=0)

    assert region is not None
    assert region.page == 0
    assert region.source_ids == (blocks[0].source_id,)


def test_region_for_ignores_an_unknown_citation() -> None:
    artifact = make_artifact()
    value = make_field("thickness", "40 nm", source_ids=["mineru_p0_b999"])
    assert region_for(value, artifact) is None


def test_region_for_without_a_citation_is_none() -> None:
    assert region_for(make_field("thickness", "40 nm"), make_artifact()) is None


def test_a_region_is_a_crops_region() -> None:
    """The stage hands its regions straight to the crop store, so it must be that module's shape."""
    artifact = make_artifact()
    region = region_for(make_field("x", "1", source_ids=[artifact.blocks[0].source_id]), artifact)
    assert isinstance(region, Region)


# ---- A value's identity -----------------------------------------------------------------------------------


def test_the_value_key_separates_lanes_owners_and_conditions() -> None:
    value = make_field("thickness", "40 nm")
    other = make_field("thickness", "40 nm", condition="at 500 °C")
    keys = {
        value_key("mineru", "target:S1", value),
        value_key("paddleocr_vl", "target:S1", value),
        value_key("mineru", "target:S2", value),
        value_key("mineru", "unattributed", value),
        value_key("mineru", "target:S1", other),
    }
    assert len(keys) == 5


# ---- The prompts ------------------------------------------------------------------------------------------


def test_the_validation_prompt_never_names_a_value_or_asks_a_question() -> None:
    """The model transcribes; the code adjudicates. A yes/no question here would break that."""
    system = validation_system_prompt()
    user = validation_user_prompt(shipped_profile().by_name["thickness"])
    for text in (system, user):
        lowered = text.lower()
        # No question about the value's truth. ("correct" does appear, in "do not ... correct": the model is
        # told not to fix what it reads, which is the opposite of being asked to judge it.)
        assert "is this correct" not in lowered
        assert "is this right" not in lowered
        assert "verify" not in lowered
        assert "check whether" not in lowered
        assert "expected value" not in lowered
    assert "do not summarise, interpret, correct" in system.lower()
    assert "transcribe" in system.lower()
    # The field is named so the model knows which small print to take care over; the value never is.
    assert "thickness" in user


def test_the_validation_prompt_is_the_same_for_every_field_and_lane() -> None:
    assert validation_system_prompt() is validation_system_prompt()


def test_the_table_prompt_names_no_field() -> None:
    """One transcription serves every field the table might hold."""
    assert "thickness" not in table_transcription_user_prompt()
    assert "caption" in table_transcription_user_prompt()


def test_the_fill_prompt_is_rendered_from_the_profile() -> None:
    text = fill_system_prompt(shipped_profile())
    assert "{domain_subject}" not in text and "{fields}" not in text
    assert shipped_profile().prompt.domain_subject in text
    assert "verbatim" in text


def test_the_fill_prompt_follows_another_profiles_wording() -> None:
    """Domain words come from the profile, so a different profile asks in its own terms."""
    other = make_profile({"prompt.domain_subject": "electrolyte formulations"})
    text = fill_system_prompt(other)
    assert "electrolyte formulations" in text
    assert shipped_profile().prompt.domain_subject not in text
