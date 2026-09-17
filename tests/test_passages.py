"""Retrieval decides what the model is allowed to see, so its rules are pinned like arithmetic.

Everything here is a pure function of blocks in, blocks out: no model, no network, no fixtures beyond
hand-built :class:`SourceBlock`s. The cases below are the rules the module claims in its own docstring --
whole-token matching, the digit requirement, a unit as a weak signal, and a table that is never cut to fit
a budget -- because each of them exists to stop a specific failure that was seen on a real paper.
"""

from __future__ import annotations

import logging

import pytest

from paperfacts.fields import FIELD_BY_NAME
from paperfacts.models import SourceBlock
from paperfacts.passages import candidate_blocks, fit_budget, inventory_blocks
from support.factories import make_block

COMPONENT = FIELD_BY_NAME["component"]
SHEET_RESISTANCE = FIELD_BY_NAME["sheet_resistance"]
THICKNESS = FIELD_BY_NAME["thickness"]
TRANSMITTANCE = FIELD_BY_NAME["transmittance"]


def ids(blocks: list[SourceBlock]) -> list[str]:
    return [block.source_id for block in blocks]


def text(order: int, content: str, *, page: int = 0) -> SourceBlock:
    return make_block(page=page, order=order, type="text", content=content)


# ---- Keyword matching ------------------------------------------------------------------------------


def test_a_keyword_is_matched_as_a_whole_token_not_inside_a_word():
    # "Rs" is a keyword of sheet_resistance. As a plain substring it also matches "years" and "layers",
    # which would make half of every paper a candidate.
    named = text(0, "The sheet resistance was 12.3 per square.")
    prose = text(1, "Over the years the layers grew by 5 percent.")

    assert ids(candidate_blocks(SHEET_RESISTANCE, [named, prose])) == [named.source_id]


def test_a_standalone_abbreviation_keyword_still_matches():
    # The other half of the same rule: Rs on its own is exactly how papers report the value in prose.
    block = text(0, "The measured Rs was 12.3 per square.")

    assert ids(candidate_blocks(SHEET_RESISTANCE, [block])) == [block.source_id]


def test_matching_ignores_case():
    # Table headers shout, body text does not; both are the same field.
    shouted = text(0, "SHEET RESISTANCE 12.3")
    titled = text(1, "Sheet Resistance of 9.8")

    assert ids(candidate_blocks(SHEET_RESISTANCE, [shouted, titled])) == [shouted.source_id, titled.source_id]


def test_a_keyword_that_ends_in_punctuation_matches_without_a_word_boundary():
    # "%T" is a real keyword of transmittance; demanding \b after "T" would never match "85 %T".
    block = text(0, "The film reached 85 %T in the visible range.")

    assert ids(candidate_blocks(TRANSMITTANCE, [block])) == [block.source_id]


def test_a_numeric_field_needs_a_digit_in_the_block():
    # A paragraph that discusses the trend cannot contain the number, however often it says the word.
    qualitative = text(0, "The sheet resistance decreased sharply with oxygen flow.")
    quantitative = text(1, "The sheet resistance decreased to 12.3 per square.")

    assert ids(candidate_blocks(SHEET_RESISTANCE, [qualitative, quantitative])) == [quantitative.source_id]


def test_a_composition_field_has_no_digit_requirement():
    # A target composition can be entirely words ("tin oxide doped with tantalum"), so the digit rule
    # applies only to numeric fields.
    block = text(0, "A ceramic target of tin oxide doped with tantalum was used.")

    assert ids(candidate_blocks(COMPONENT, [block])) == [block.source_id]


# ---- Units as a weak signal ------------------------------------------------------------------------


def test_a_block_that_only_mentions_the_unit_ranks_below_one_that_names_the_field():
    # Both are plausible, but the one that says "thickness" is the one to spend a place on. With only one
    # place available the wavelength block must lose it.
    named = text(0, "The film thickness was 250 nm.")
    unit_only = text(1, "Transmittance was measured at 550 nm.")

    assert ids(candidate_blocks(THICKNESS, [named, unit_only], limit=1)) == [named.source_id]


def test_a_unit_match_alone_still_qualifies_a_block():
    # The paper that writes "films of 2108 nm" without the word "thickness" would otherwise be lost
    # entirely, so a unit hit qualifies -- it just ranks last.
    unit_only = text(0, "Films of 2108 nm were obtained after the long run.")

    assert ids(candidate_blocks(THICKNESS, [unit_only])) == [unit_only.source_id]


def test_a_block_with_neither_a_name_nor_a_unit_is_not_a_candidate():
    block = text(0, "The crystal structure was examined by X-ray diffraction at 40 kV.")

    assert candidate_blocks(THICKNESS, [block]) == []


# ---- Ranking, the limit, and neighbours -------------------------------------------------------------


def test_the_limit_caps_how_many_scored_blocks_are_chosen():
    blocks = [text(order, f"The sheet resistance was {order + 1}.2 per square.") for order in range(5)]

    assert len(candidate_blocks(SHEET_RESISTANCE, blocks, limit=2)) == 2


def test_a_table_next_to_a_chosen_block_comes_along_beyond_the_limit():
    # One fact, two blocks: the number is in the table and the field name is in the sentence above it.
    # The table arrives outside the ranking, so the result may hold more blocks than the limit allows.
    named = text(0, "The sheet resistance of all 6 films is listed below.")
    table = make_block(page=0, order=1, type="table", content="<table><tr><td>A</td><td>B</td></tr></table>")

    assert ids(candidate_blocks(SHEET_RESISTANCE, [named, table], limit=1)) == [named.source_id, table.source_id]


def test_a_table_on_another_page_is_not_pulled_in():
    # Adjacency in the block list is not adjacency on paper: the last block of one page and the first of
    # the next are neighbours in the list but have nothing to do with each other.
    named = text(0, "The sheet resistance of all 6 films is listed below.")
    table = make_block(page=1, order=0, type="table", content="<table><tr><td>A</td><td>B</td></tr></table>")

    assert ids(candidate_blocks(SHEET_RESISTANCE, [named, table], limit=1)) == [named.source_id]


def test_the_selection_comes_back_in_document_order():
    # The ranking decides which blocks; the document decides their order, so the model reads the paper's
    # own narrative rather than a scoreboard.
    unit_only = text(0, "Measured at 550 nm.")
    named_twice = make_block(
        page=0, order=1, type="caption", content="Film thickness from the SEM cross section: 250 nm"
    )

    selected = candidate_blocks(THICKNESS, [unit_only, named_twice], limit=2)

    assert ids(selected) == [unit_only.source_id, named_twice.source_id]


def test_a_limit_below_one_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        candidate_blocks(SHEET_RESISTANCE, [text(0, "The sheet resistance was 12.3")], limit=0)


# ---- The inventory question -------------------------------------------------------------------------


def test_every_title_table_and_caption_is_in_the_inventory():
    # Samples are enumerated in tables and named in captions; titles are nearly free and tell the model
    # which section it is reading.
    blocks = [
        make_block(page=0, order=0, type="title", content="3. Results"),
        make_block(page=0, order=1, type="table", content="<table><tr><td>S1</td></tr></table>"),
        make_block(page=0, order=2, type="caption", content="Table 1. Deposition conditions."),
    ]

    assert ids(inventory_blocks(blocks)) == ids(blocks)


def test_prose_that_states_a_numeric_condition_is_in_the_inventory():
    flow = text(0, "Films were grown with an oxygen flow of 100 sccm.")
    power = text(1, "The discharge was held at 150 W throughout.")

    assert ids(inventory_blocks([flow, power])) == [flow.source_id, power.source_id]


def test_prose_that_states_a_duration_in_seconds_is_in_the_inventory():
    block = text(0, "The films were deposited for 300 s at room temperature.")

    assert ids(inventory_blocks([block])) == [block.source_id]


def test_the_word_samples_after_a_number_is_not_a_seconds_unit():
    # "\d\s*s\b" must not fire here: after the "s" of "samples" comes the "a" of "amples", both word
    # characters, so \b fails and the prose stays out of the inventory.
    block = text(0, "2 samples were prepared by the same route.")

    assert inventory_blocks([block]) == []


def test_prose_that_states_a_composition_ratio_is_in_the_inventory():
    block = text(0, "The gas mix contained O2 at 3 vol.% of the total flow.")

    assert ids(inventory_blocks([block])) == [block.source_id]


def test_prose_without_a_number_is_left_out_of_the_inventory():
    # "the deposition process" appears in every discussion paragraph; on its own it says nothing about
    # which samples exist.
    block = text(0, "The deposition process was studied in detail by several groups.")

    assert inventory_blocks([block]) == []


def test_a_condition_word_next_to_a_number_is_in_the_inventory():
    # Not every condition carries a recognised unit: "sample 3" is how a paper names what it made.
    block = text(0, "Sample 3 was cleaned before loading.")

    assert ids(inventory_blocks([block])) == [block.source_id]


def test_the_inventory_keeps_document_order():
    title = make_block(page=0, order=0, type="title", content="2. Experimental")
    prose = text(1, "Deposition ran at 300 °C.")
    table = make_block(page=1, order=0, type="table", content="<table><tr><td>S1</td></tr></table>")

    assert ids(inventory_blocks([title, prose, table])) == [title.source_id, prose.source_id, table.source_id]


# ---- The context budget ------------------------------------------------------------------------------


def test_a_selection_within_the_budget_is_returned_unchanged():
    blocks = [text(0, "x" * 50), text(1, "y" * 50)]

    assert ids(fit_budget(blocks, budget_chars=1000)) == ids(blocks)


def test_prose_is_dropped_from_the_end_until_the_budget_fits():
    # Dropping from the end keeps the earliest, most contextual blocks, which is where the methods live.
    first = text(0, "a" * 100)
    second = text(1, "b" * 100)
    third = text(2, "c" * 100)

    assert ids(fit_budget([first, second, third], budget_chars=250)) == [first.source_id, second.source_id]


def test_a_table_is_never_dropped_to_fit_the_budget():
    # A table is the densest evidence in a paper: cutting one to fit a budget throws away the answer.
    prose_before = text(0, "a" * 100)
    table = make_block(page=0, order=1, type="table", content="t" * 100)
    prose_after = text(2, "c" * 100)

    assert ids(fit_budget([prose_before, table, prose_after], budget_chars=150)) == [table.source_id]


def test_a_budget_smaller_than_the_tables_keeps_them_and_says_so(caplog):
    # An impossible budget must not silently return a prompt that is missing its evidence.
    tables = [make_block(page=0, order=index, type="table", content="t" * 100) for index in range(2)]

    with caplog.at_level(logging.WARNING, logger="paperfacts.passages"):
        kept = fit_budget(tables, budget_chars=50)

    assert ids(kept) == ids(tables)
    assert "still exceeds the budget" in caplog.text


def test_what_the_budget_drops_is_logged_by_source_id(caplog):
    prose = [text(index, "x" * 100) for index in range(3)]

    with caplog.at_level(logging.WARNING, logger="paperfacts.passages"):
        fit_budget(prose, budget_chars=150)

    assert prose[2].source_id in caplog.text
