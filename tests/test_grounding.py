"""Checking that an extracted value really occurs in the block it cites.

A valid ``source_id`` only proves the model named a real block, not that the value came from it: a model
can quote a plausible number and attach a nearby, real id, and the result still looks perfectly traceable.
Grounding closes that gap by requiring the quoted text to actually be found, after folding away the
formatting differences that come from two different parsers (and, before that, from two different PDFs)
writing the same number very differently.
"""

from __future__ import annotations

import pytest

from paperfacts.grounding import ground_lane, ground_values, grounding_key, is_grounded
from paperfacts.records import FieldValue, TargetRecord
from support.extraction import make_field, make_lane, make_sample

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
