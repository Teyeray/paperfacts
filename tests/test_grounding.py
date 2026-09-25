"""Checking that an extracted value really occurs in the block it cites.

A valid ``source_id`` only proves the model named a real block, not that the value came from it: a model
can quote a plausible number and attach a nearby, real id, and the result still looks perfectly traceable.
Grounding closes that gap by requiring the quoted text to actually be found, after folding away the
formatting differences that come from two different parsers (and, before that, from two different PDFs)
writing the same number very differently.
"""

from __future__ import annotations

import pytest

from paperfacts.grounding import block_adjacency, ground_lane, ground_values, grounding_key, is_grounded
from paperfacts.records import FieldValue, TargetRecord
from support.extraction import make_field, make_lane, make_sample
from support.factories import make_block

# ---- grounding_key: what gets folded away ----------------------------------------------


def test_grounding_key_folds_case_and_collapses_runs_of_decoration_to_one_space():
    # Decoration collapses to a single space rather than vanishing, so that numbers printed next to each
    # other keep a boundary between them. See the containment matrix at the bottom of this module.
    assert grounding_key("  Sheet   Resistance  ") == "sheet resistance"
    assert grounding_key("76.7, 71.3, 68.4") == "76.7 71.3 68.4"


def test_grounding_key_folds_a_multiplication_sign_normalize_text_does_not_already_handle():
    # normalize_text already turns "×" into "x"; "✕" (HEAVY MULTIPLICATION X) is left alone by the general
    # text normalizer, so folding it is grounding_key's own job.
    assert grounding_key("4 ✕ 5") == grounding_key("4 x 5")


def test_grounding_key_strips_latex_wrapper_commands_so_they_do_not_leak_into_the_key():
    # If "\mathrm" survived as literal letters it would sit between "2:" and "ta" and defeat the match
    # outright. What remains is the formula's characters, spaced out the way the typesetting spaced them.
    wrapped = r"$\mathrm { S n O } _ { 2 } : \mathrm { T a }$"

    key = grounding_key(wrapped)

    assert "mathrm" not in key
    assert key.replace(" ", "") == "sno2:ta"


# ---- is_grounded ------------------------------------------------------------------------


def test_a_plain_value_is_grounded_when_the_block_text_contains_it():
    value = make_field("sheet_resistance", "12.5", source_ids=("b1",))
    blocks = {"b1": "The film exhibited a sheet resistance of 12.5 Ω/sq at room temperature."}

    assert is_grounded(value, blocks) is True


def test_a_latex_mangled_number_grounds_a_plain_value():
    # MinerU renders a table cell like "(40 x 10 cm" with a space between every character and wraps it in
    # LaTeX; the model is asked to quote it verbatim, but might normalize the spacing when it copies it.
    value = make_field("inch", "40 × 10", source_ids=("b1",))
    blocks = {"b1": r"$( 4 0 \times 1 0 \mathrm { c m }$"}

    assert is_grounded(value, blocks) is True


def test_a_latex_wrapped_formula_grounds_a_plain_quoted_composition():
    value = make_field("component", "SnO2:Ta", source_ids=("b1",))
    blocks = {"b1": r"the target composition was $\mathrm { S n O } _ { 2 } : \mathrm { T a }$ throughout"}

    assert is_grounded(value, blocks) is True


def test_a_value_with_no_source_ids_is_not_grounded():
    # Even though the text is right there in a block, a value that cites nothing has nothing to be
    # grounded against.
    value = make_field("sheet_resistance", "12.5", source_ids=())
    blocks = {"b1": "sheet resistance of 12.5 Ω/sq"}

    assert is_grounded(value, blocks) is False


def test_a_value_citing_a_block_id_absent_from_the_map_is_not_grounded():
    value = make_field("sheet_resistance", "12.5", source_ids=("ghost",))
    blocks = {"b1": "sheet resistance of 12.5 Ω/sq"}

    assert is_grounded(value, blocks) is False


def test_a_genuinely_absent_number_is_not_grounded():
    value = make_field("sheet_resistance", "99.9", source_ids=("b1",))
    blocks = {"b1": "sheet resistance of 12.5 Ω/sq"}

    assert is_grounded(value, blocks) is False


# ---- ground_values: FieldValue tuples ----------------------------------------------------


def test_ground_values_marks_each_value_grounded_or_not_independently():
    blocks = {"b1": "sheet resistance of 12.5 Ω/sq"}
    grounded = make_field("sheet_resistance", "12.5", source_ids=("b1",))
    ungrounded = make_field("thickness", "300", source_ids=("b1",))

    result = ground_values((grounded, ungrounded), blocks)

    assert [v.grounded for v in result] == [True, False]
    assert [v.field for v in result] == ["sheet_resistance", "thickness"]  # order is preserved


def test_ground_values_does_not_mutate_its_input():
    # FieldValue is frozen, so this should already be structurally impossible; assert it anyway, since it
    # is exactly the invariant grounding depends on to be safe to redo on every read.
    original = make_field("sheet_resistance", "99.9", source_ids=("b1",))

    ground_values((original,), {"b1": "no matching number here"})

    assert original.grounded is True  # the pydantic default, untouched


# ---- ground_lane: a whole LaneExtraction --------------------------------------------------


def test_ground_lane_updates_grounding_on_both_the_target_and_every_sample():
    blocks = {"b1": "target relative density of 98.5 %", "b2": "sheet resistance of 12.5 Ω/sq"}
    lane = make_lane(
        target=TargetRecord(source_ids=("b1",), fields=(make_field("density", "98.5", source_ids=("b1",)),)),
        samples=[
            make_sample(
                "A",
                [
                    make_field("sheet_resistance", "12.5", source_ids=("b2",)),
                    make_field("thickness", "999", source_ids=("b2",)),
                ],
            )
        ],
    )

    grounded = ground_lane(lane, blocks)

    assert grounded.target.get("density").grounded is True
    assert grounded.sample("A").get("sheet_resistance").grounded is True
    assert grounded.sample("A").get("thickness").grounded is False


def test_ground_lane_tolerates_a_lane_with_no_target():
    lane = make_lane(target=None, samples=[make_sample("A", [make_field("thickness", "300", source_ids=("b1",))])])

    grounded = ground_lane(lane, {"b1": "thickness of 300 nm"})

    assert grounded.target is None
    assert grounded.sample("A").get("thickness").grounded is True


def test_ground_lane_leaves_everything_else_about_the_lane_unchanged():
    lane = make_lane(model="some-model", dropped=("x: not in schema",), invalid_source_ids=("ghost",))

    grounded = ground_lane(lane, {})

    assert grounded.model == lane.model
    assert grounded.dropped == lane.dropped
    assert grounded.invalid_source_ids == lane.invalid_source_ids
    assert grounded.extractor_key == lane.extractor_key


# ---- how strictly a value must appear in the block it cites ----------------------------


@pytest.mark.parametrize(
    ("value_raw", "block", "grounded", "why"),
    [
        # Separators matter: without them, adjacent numbers merge into one digit run.
        ("76.7", "the thicknesses were 76.7, 71.3, 68.4 and 57.0 nm", True, "first of a comma-separated list"),
        ("71.3", "the thicknesses were 76.7, 71.3, 68.4 and 57.0 nm", True, "middle of a comma-separated list"),
        # ... but a number must not match inside a longer number.
        ("4", "the films were deposited for 40 minutes", False, "prefix of a longer number"),
        ("2", "a thickness of 2108 nm from ellipsometry", False, "prefix of a longer number"),
        ("10", "sputtered for 110 s", False, "suffix of a longer number"),
        ("0.3", "a resistivity of 10.3 ohm cm", False, "inside a longer decimal"),
        ("5 nm", "a grain size of 235 nm", False, "short value, separators are load-bearing"),
        ("10", "page 10 of 12", True, "a number in its own right"),
        ("2", "a film thickness of 2 um", True, "a number in its own right"),
        ("80.6", "transmittance of 80.6 % in the visible", True, "decimal matches exactly"),
        # A formula LaTeX has shattered into single characters still has to match.
        (
            "SnO2:Sb2O3(95:5)",
            r"<td> $\mathrm { S n O } _ { 2 } : \mathrm { S b } _ { 2 } \mathrm { O } _ { 3 } ( 9 5 { : } 5 )$ </td>",
            True,
            "LaTeX-fragmented chemical formula",
        ),
        ("SnO2:Sb2O3(95:5)", "a paper about ITO films only", False, "a long value still has to be present"),
        ("9999", "nothing like that here", False, "genuinely absent"),
        # A decimal point or a caret continues a number just as a digit does.
        ("5", "a thickness of 0.5 nm", False, "fraction digits of a decimal"),
        ("5", "a thickness of 5.2 nm", False, "integer part of a decimal"),
        ("10", "a resistivity of 10^-4 ohm cm", False, "base of a power of ten"),
        ("10", "a resistivity of 1.2 × 10⁻⁴ Ω·cm", False, "base of a superscript power of ten"),
        ("4", "a resistivity of 10^4 ohm cm", False, "exponent of a power of ten"),
        ("10^-4", "a resistivity of 10⁻⁴ Ω·cm", True, "the whole power matches"),
        ("10^-4", "a resistivity of $1 0 ^ { - 4 }$", True, "LaTeX power with spaced caret"),
        ("5", "the thickness was 5. The films", True, "a full stop is not a decimal point"),
    ],
)
def test_how_strictly_a_value_must_appear_in_the_block_it_cites(value_raw, block, grounded, why):
    r"""Two failure modes pull in opposite directions, and both are silent.

    Discarding separators lets a short value match any digit run containing it -- "4" inside "40 minutes"
    -- and the fields holding short round numbers (target size in inches, sputtering time in minutes) are
    exactly the ones that would be wrongly confirmed. Keeping separators, on the other hand, would reject
    a chemical formula that MinerU emitted as ``$\mathrm { S n O } _ { 2 }$``, where the gaps are an
    artefact of the typesetting rather than part of the value.
    """
    value = FieldValue(field="inch", value_raw=value_raw, source_ids=("b1",))

    assert is_grounded(value, {"b1": block}) is grounded, why


# ---- grounding across a block boundary -----------------------------------------------


def make_blocks(*contents: str, page: int = 0) -> tuple[dict, dict]:
    """A same-page run of blocks plus its two derived inputs: the text map and the adjacency map."""
    seq = [make_block(page=page, order=order, content=content) for order, content in enumerate(contents)]
    return {b.source_id: b.content for b in seq}, block_adjacency(seq)


def test_a_quote_straddling_the_cited_block_and_its_next_neighbour_grounds():
    # The README's observed failure: the model quoted a sentence that continues into the next block,
    # citing only the first. Without adjacency the verdict is a false "ungrounded".
    blocks, adjacency = make_blocks("The film consists of SnO2 and", "Sb2O3 in a 95:5 ratio.")
    value = make_field("component", "and Sb2O3", source_ids=("mineru_p0_b0",))

    assert is_grounded(value, blocks, adjacency=adjacency) is True
    assert is_grounded(value, blocks) is False


def test_a_quote_straddling_a_previous_neighbour_and_the_cited_block_grounds():
    # Same failure, mirrored: only the second half of the sentence was cited.
    blocks, adjacency = make_blocks("The film consists of SnO2 and", "Sb2O3 in a 95:5 ratio.")
    value = make_field("component", "SnO2 and", source_ids=("mineru_p0_b1",))

    assert is_grounded(value, blocks, adjacency=adjacency) is True
    assert is_grounded(value, blocks) is False


def test_a_quote_living_entirely_inside_the_neighbour_does_not_ground():
    # The model cited a block that carries none of the quote; the neighbour merely happens to contain
    # the words. Accepting that would replace grounding with "somewhere near the citation".
    blocks, adjacency = make_blocks("The film consists of SnO2 and", "Sb2O3 in a 95:5 ratio.")
    value = make_field("component", "consists of SnO2", source_ids=("mineru_p0_b1",))

    assert is_grounded(value, blocks, adjacency=adjacency) is False


def test_a_neighbour_on_a_different_page_is_not_tried():
    # A sentence broken by a page break is not one sentence; joining the two blocks would manufacture
    # text that appears nowhere in the PDF.
    first, _ = make_blocks("The film consists of SnO2 and", page=0)
    second, _ = make_blocks("Sb2O3 in a 95:5 ratio.", page=1)
    blocks = {**first, **second}
    # The adjacency built from the actual page-split sequence has no same-page neighbours to offer.
    adjacency = block_adjacency(
        [
            make_block(page=0, order=0, content="The film consists of SnO2 and"),
            make_block(page=1, order=0, content="Sb2O3 in a 95:5 ratio."),
        ]
    )
    value = make_field("component", "and Sb2O3", source_ids=("mineru_p0_b0",))

    assert is_grounded(value, blocks, adjacency=adjacency) is False


def test_a_sentence_cut_by_a_page_break_grounds_across_the_break():
    # The halves of one sentence: the page break is the parser's cut, not a gap in the text.
    seq = [
        make_block(page=4, order=9, content="The film deposited with 0.6% hydrogen had a transmittance of"),
        make_block(page=5, order=0, content="92.1% in the visible range."),
    ]
    blocks = {b.source_id: b.content for b in seq}
    value = make_field("transmittance", "of 92.1", source_ids=("mineru_p5_b0",))

    assert is_grounded(value, blocks, adjacency=block_adjacency(seq)) is True


def test_block_adjacency_reports_same_page_neighbours_only():
    # Finished sentences, so no page-break continuation links the page-1 block in.
    seq = [
        make_block(page=0, order=0, content="One."),
        make_block(page=0, order=1, content="Two."),
        make_block(page=1, order=0, content="three"),
    ]

    adjacency = block_adjacency(seq)

    assert adjacency["mineru_p0_b0"] == (None, "mineru_p0_b1")
    assert adjacency["mineru_p0_b1"] == ("mineru_p0_b0", None)  # the next entry is on page 1
    assert adjacency["mineru_p1_b0"] == (None, None)


def test_a_latex_mangled_formula_split_across_two_blocks_grounds_via_the_squeezed_straddle():
    # MinerU shattered the formula across a block boundary; the squeezed join is the only place the
    # full composition still exists contiguously.
    blocks, adjacency = make_blocks(
        r"target $\mathrm { S n O } _ { 2 } :", r"\mathrm { S b } _ { 2 } \mathrm { O } _ { 3 }$"
    )
    value = make_field("component", "SnO2:Sb2O3", source_ids=("mineru_p0_b0",))

    assert is_grounded(value, blocks, adjacency=adjacency) is True
    assert is_grounded(value, blocks) is False


def test_the_digit_rule_still_applies_across_the_junction():
    # "for 4" + "0 min" is really "for 40 min" split by the parser; a needle "4" ending exactly at the
    # junction must not count as its own number. The check has to run in squeezed coordinates, because
    # the join's space is an artefact we inserted, not spacing the PDF had.
    blocks, adjacency = make_blocks("the films were deposited for 4", "0 min at room temperature")
    value = make_field("sputtering_time", "4", source_ids=("mineru_p0_b1",))

    assert is_grounded(value, blocks, adjacency=adjacency) is False


@pytest.mark.parametrize(
    ("first", "second", "needle"),
    [
        ("a thickness of 0.5", "nm was measured", "5 nm"),
        ("a thickness of 3.5", "nm was measured", "5 nm"),
    ],
)
def test_the_decimal_and_caret_rule_also_applies_across_the_junction(first, second, needle):
    # The quote crosses the junction, so only the boundary rule stands between it and a false confirmation:
    # "5 nm" is the tail of "0.5 nm" or of "3.5 nm", never a thickness of its own.
    blocks, adjacency = make_blocks(first, second)
    value = make_field("thickness", needle, source_ids=("mineru_p0_b1",))

    assert is_grounded(value, blocks, adjacency=adjacency) is False


def test_ground_lane_with_adjacency_flips_a_straddled_value_to_grounded_end_to_end():
    blocks, adjacency = make_blocks("The film is made of SnO2", "and Sb2O3 in a 95:5 ratio.")
    cited = "mineru_p0_b0"
    lane = make_lane(samples=[make_sample("A", [make_field("component", "SnO2 and", source_ids=(cited,))])])

    without = ground_lane(lane, blocks)
    with_adjacency = ground_lane(lane, blocks, adjacency=adjacency)

    assert without.sample("A").get("component").grounded is False
    assert with_adjacency.sample("A").get("component").grounded is True
