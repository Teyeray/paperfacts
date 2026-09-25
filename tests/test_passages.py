"""Retrieval decides what the model is allowed to see, so its rules are pinned like arithmetic.

Everything here is a pure function of blocks in, blocks out: no model, no network, no fixtures beyond
hand-built :class:`SourceBlock`s. The cases below are the rules the module claims in its own docstring --
whole-token matching, the digit requirement, a unit as a weak signal, and a table that is never cut to fit
a budget -- because each of them exists to stop a specific failure that was seen on a real paper.
"""

from __future__ import annotations

import logging
from functools import partial

import pytest

from paperfacts import passages
from paperfacts.continuation import continuation_pairs
from paperfacts.models import SourceBlock
from paperfacts.passages import fit_budget
from support.factories import make_block
from support.profiles import shipped_profile

# The shipped profile's field table, units and retrieval, at module level because constants and parametrize
# lists need them before any fixture runs.
TCO = shipped_profile()
FIELD_BY_NAME = TCO.by_name
candidate_blocks = partial(passages.candidate_blocks, units=TCO.units)
inventory_blocks = partial(passages.inventory_blocks, retrieval=TCO.retrieval)

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


def test_a_chinese_keyword_matches_inside_running_chinese_text():
    # Chinese writes no spaces between words, so a word boundary beside a Chinese character never matches:
    # "煅烧温度" in "样品的煅烧温度为800" would be missed and the value never asked about.
    assert passages.keyword_hits(["煅烧温度"], "样品的煅烧温度为800 °c") == 1
    assert passages.keyword_hits(["煅烧温度"], "样品的烧结温度为800 °c") == 0
    # An ASCII edge keeps its boundary: "Rs" still does not match inside "years".
    assert passages.keyword_hits(["Rs", "薄层Rs"], "over the years 薄层rsx") == 0


def test_a_profile_pattern_searches_only_the_head_of_a_very_long_block():
    # A profile's own expression is bounded per block; a built-in one searches everything.
    filler = "word " * (passages.PROFILE_PATTERN_SPAN // 5)
    late = text(0, f"{filler} annealed at 500 °C")
    retrieval = TCO.retrieval.__class__(condition_keywords=(), condition_unit_pattern=r"\d\s*°c")

    assert passages.inventory_blocks([late], retrieval) == []
    assert passages.inventory_blocks([text(1, "annealed at 500 °C")], retrieval) != []
    assert passages.candidate_blocks(FIELD_BY_NAME["annealing_temperature"], [late], units=TCO.units) == [late]


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


def test_every_block_naming_the_field_is_chosen_whatever_the_limit():
    # Ranking named blocks against each other cut the later pages, where the results are.
    blocks = [text(order, f"The sheet resistance was {order + 1}.2 per square.") for order in range(5)]

    assert len(candidate_blocks(SHEET_RESISTANCE, blocks, limit=2)) == 5


def test_blocks_matching_only_a_unit_fill_the_places_the_named_ones_left():
    named = text(0, "The sheet resistance was 12 per square.")
    by_unit = [text(order, f"Sample {order} measured {order}0 Ω/sq.") for order in range(1, 5)]

    chosen = candidate_blocks(SHEET_RESISTANCE, [named, *by_unit], limit=3)

    assert ids(chosen) == ids([named, *by_unit[:2]])


def test_a_table_or_caption_carrying_only_the_unit_is_capped_like_any_unit_only_block():
    # A dense block that matches on its unit alone used to score as much as a keyword match and ride along
    # uncapped: with limit=2, 19 of 20 blocks of a paper full of "%" captions went into every question.
    named = text(0, "The average transmittance was 85.2%.")
    captions = [
        make_block(page=0, order=order, type="caption", content=f"Figure {order}: XRD at {order}0% power")
        for order in range(1, 19)
    ]
    table = make_block(page=0, order=19, type="table", content="<table><tr><td>O2</td><td>5%</td></tr></table>")

    chosen = candidate_blocks(TRANSMITTANCE, [named, *captions, table], limit=2)

    # One place left under the limit: the first caption takes it, and its neighbour comes along by the
    # dense-neighbour rule. The other sixteen captions and the table stay out.
    assert ids(chosen) == ids([named, *captions[:2]])


def test_among_unit_only_blocks_a_table_or_caption_outranks_prose():
    prose = text(0, "Sample 1 measured 40 Ω/sq.")
    table = make_block(page=1, order=0, type="table", content="<table><tr><td>S2</td><td>35 Ω/sq</td></tr></table>")

    assert ids(candidate_blocks(SHEET_RESISTANCE, [prose, table], limit=1)) == [table.source_id]


def test_a_keyword_matches_a_spelling_that_lost_one_of_a_doubled_letter():
    # MinerU writes "transmitance" on some papers; the other lane has "transmittance".
    block = text(0, "The average transmitance of the film was 88.6%.")

    assert ids(candidate_blocks(TRANSMITTANCE, [block])) == [block.source_id]


def test_a_unit_written_in_latex_is_recognised():
    block = text(0, r"The value reached $5.1 \times 10^{-4}\ \Omega\cdot\text{cm}$ at 300 C.")

    assert ids(candidate_blocks(FIELD_BY_NAME["resistivity"], [block])) == [block.source_id]


@pytest.mark.parametrize("written", ["300 $^{\\circ}$C", "300 $^\\circ$C", "300 °C"])
def test_a_temperature_written_in_latex_qualifies_a_block(written):
    block = text(0, f"The films were annealed at {written}.")

    assert ids(candidate_blocks(FIELD_BY_NAME["annealing_temperature"], [block])) == [block.source_id]


@pytest.mark.parametrize("written", ["60 W", "0.2 kW", "150W"])
def test_a_power_in_watts_qualifies_a_block_for_sputtering_power(written):
    block = text(0, f"The films were grown at {written} for 30 min.")

    assert ids(candidate_blocks(FIELD_BY_NAME["sputtering_power"], [block])) == [block.source_id]


# ---- Paragraphs cut by a page or column break -----------------------------------------------------------


def test_a_sentence_cut_by_a_page_break_is_linked_across_the_page_furniture():
    before = text(9, "The hydrogen volume was 0.6% and the", page=4)
    furniture = [
        make_block(page=4, order=10, type="caption", content="Fig. 3. XRD patterns."),
        make_block(page=5, order=0, type="figure", content="images/fig3.jpg"),
    ]
    after = text(1, "films showed a transmittance of 92.1%.", page=5)

    assert continuation_pairs([before, *furniture, after]) == [(0, 3)]


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param("The films were annealed.", "then they were cooled.", id="sentence-finished"),
        pytest.param("The films were annealed and", "The next section describes", id="capital-start"),
        pytest.param("as shown in", "(a) the XRD pattern", id="sub-figure-label"),
        pytest.param("the results are", "3 Results and discussion", id="numbered-heading"),
    ],
)
def test_a_block_that_starts_something_new_is_not_a_continuation(before, after):
    assert continuation_pairs([text(0, before, page=0), text(0, after, page=1)]) == []


@pytest.mark.parametrize("start", ["95% SnO2", "$5.1 \\times 10^{-4}$", "(Ar 20 sccm)", "were annealed"])
def test_a_continuation_may_start_with_a_number_formula_or_parenthesis(start):
    assert continuation_pairs([text(0, "composed of", page=0), text(0, start, page=1)]) == [(0, 1)]


def test_a_title_between_two_blocks_breaks_the_link():
    blocks = [
        text(0, "the films were", page=0),
        make_block(page=1, order=0, type="title", content="Results"),
        text(1, "annealed", page=1),
    ]

    assert continuation_pairs(blocks) == []


def test_a_footnote_between_the_halves_is_skipped_and_never_linked_itself():
    before = text(9, "The hydrogen volume was 0.6% and the", page=4)
    footnote = SourceBlock.model_validate(
        make_block(page=4, order=10, content="* corresponding author").model_dump() | {"raw_label": "page_footnote"}
    )
    after = text(0, "films were annealed.", page=5)

    assert continuation_pairs([before, footnote, after]) == [(0, 2)]


def test_the_other_half_of_a_chosen_block_comes_along_beyond_the_limit():
    naming = text(9, "The sample deposited with 0.6% hydrogen", page=4)
    value = text(0, "reached a sheet resistance of 12 Ω/sq.", page=5)
    unrelated = text(1, "Unrelated remarks.", page=5)

    assert ids(candidate_blocks(SHEET_RESISTANCE, [naming, value, unrelated], limit=1)) == ids([naming, value])


def test_the_inventory_brings_the_other_half_of_a_block_it_chose():
    naming = text(9, "The films deposited at 150 W with the", page=4)
    rest = text(0, "hydrogen mixture are named ICO-H.", page=5)

    assert ids(inventory_blocks([naming, rest])) == ids([naming, rest])


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


def test_a_caption_on_the_next_page_follows_its_table_across_the_break():
    # A table typeset at the foot of a page carries its caption to the head of the next one. The condition
    # the numbers were measured under lives in that caption, so the break must not separate them.
    table = make_block(page=0, order=9, type="table", content="<table><tr><td>Rs</td><td>12.3</td></tr></table>")
    caption = make_block(page=1, order=0, type="caption", content="Table 2. Sheet resistance of the films.")

    assert ids(candidate_blocks(SHEET_RESISTANCE, [table, caption], limit=1)) == [
        table.source_id,
        caption.source_id,
    ]


def test_a_table_on_the_next_page_follows_its_caption_across_the_break():
    # The same pair in the other order: the caption scored and the table has to come with it.
    caption = make_block(page=0, order=9, type="caption", content="Table 2. Sheet resistance of the films.")
    table = make_block(page=1, order=0, type="table", content="<table><tr><td>A</td><td>B</td></tr></table>")

    assert ids(candidate_blocks(SHEET_RESISTANCE, [caption, table], limit=1)) == [
        caption.source_id,
        table.source_id,
    ]


def test_two_tables_across_a_page_break_are_still_unrelated():
    # Only a table/caption pair survives the break. Two tables that merely meet at a page boundary are
    # consecutive by accident, and pulling one in would cost the prompt a block for nothing.
    scored = make_block(page=0, order=9, type="table", content="<table><tr><td>Rs</td><td>12.3</td></tr></table>")
    other = make_block(page=1, order=0, type="table", content="<table><tr><td>A</td><td>B</td></tr></table>")

    assert ids(candidate_blocks(SHEET_RESISTANCE, [scored, other], limit=1)) == [scored.source_id]


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


@pytest.mark.parametrize("written", ["0.5 Pa", "3 mTorr", "5×10-3 mbar", "0.4Pa"])
def test_a_pressure_qualifies_a_block_for_the_working_pressure(written):
    block = text(0, f"Deposition proceeded at {written} in argon.")

    assert ids(candidate_blocks(FIELD_BY_NAME["working_pressure"], [block])) == [block.source_id]


def test_a_block_the_inventory_cited_for_the_samples_is_asked_about_every_sample_level_field():
    # "The substrate to target distance is 69 mm" matches no keyword of the field and only a weak unit.
    recipe = text(3, "The substrate to target distance is 69 mm and the films are 240 nm thick.")
    others = [text(order, f"Earlier remark number {order} at 5 mm.") for order in range(3)]

    chosen = candidate_blocks(
        FIELD_BY_NAME["target_substrate_distance"],
        [*others, recipe],
        limit=1,
        sample_blocks=frozenset({recipe.source_id}),
    )

    assert recipe.source_id in ids(chosen)


def test_a_cited_block_without_a_number_is_not_added_to_a_numeric_question():
    recipe = text(0, "The films were sputtered from an ITO target.")

    assert candidate_blocks(THICKNESS, [recipe], sample_blocks=frozenset({recipe.source_id})) == []
