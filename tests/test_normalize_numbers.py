"""Number parsing: the many spellings found in papers -> a float, plus a note explaining the reading.

By design the LLM only transcribes the source text verbatim; all conversion happens here, making this the
one place in the whole system that can offer a determinism guarantee. Two hard rules: **a questionable
reading must leave a note** (so it's traceable in provenance), and **an unreadable value must return None**
(never a guess).
"""

from __future__ import annotations

import pytest

from paperfacts.normalize import parse_number

# ---- Scientific notation ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "1.2 × 10^-4",  # already in ^ form
        "1.2x10⁻⁴",  # superscript, converted to ^-4 by normalize_text
        "1.2 x 10<sup>-4</sup>",  # MinerU's HTML superscript
        "1.2 × 10-4",  # OCR lost the superscript marker, but "x 10" is still there
        "1.2e-4",
        "1.2E-4",
    ],
)
def test_every_scientific_notation_spelling_reaches_the_same_float(raw):
    value, note = parse_number(raw)

    assert value == pytest.approx(1.2e-4)
    assert note is None


def test_a_power_of_ten_without_a_mantissa_is_read_as_one_times_that_power():
    assert parse_number("10^-4") == (1e-4, None)


def test_a_bare_integer_is_never_mistaken_for_scientific_notation():
    # "2108" contains "10"; without requiring an explicit "^" the regex would read it as 2x10^8 — five
    # orders of magnitude off.
    assert parse_number("2108") == (2108.0, None)


def test_a_power_of_ten_without_a_mantissa_needs_the_caret():
    """``"10-4"`` has no mantissa and no "^", so there is no way to tell whether it means 10⁻⁴ or something
    else; rather than guess, the first number is taken and a note is left.

    The range rule requires a < b, so this also can't be misread as the midpoint (7) of a 10~4 range.
    """
    value, note = parse_number("10-4")

    assert value == 10.0
    assert note == "2 numbers found, first used"


def test_a_descending_pair_is_not_a_range():
    assert parse_number("20-10") == (20.0, "2 numbers found, first used")


# ---- Qualifiers ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected", "qualifier"),
    [
        ("~85", 85.0, "~"),
        ("> 80", 80.0, ">"),
        ("≈ 2", 2.0, "≈"),
        ("≥ 90", 90.0, "≥"),
        ("ca. 5", 5.0, "ca."),
        ("about 7", 7.0, "about"),
        (">= 90", 90.0, ">="),
        ("<= 5", 5.0, "<="),
        (">=90", 90.0, ">="),
    ],
)
def test_a_leading_qualifier_is_dropped_but_recorded(raw, expected, qualifier):
    # ">" and "≈" change the strength of a fact; dropping it must leave a trace, or ">80" and "80" would be
    # judged AGREE with no way to trace why.
    value, note = parse_number(raw)

    assert value == expected
    assert note == f"qualifier '{qualifier}' dropped"


# ---- Uncertainty, ranges, parentheses ----------------------------------------------------------------


def test_an_uncertainty_is_dropped_and_the_central_value_kept():
    assert parse_number("3.5 ± 0.2") == (3.5, "uncertainty dropped")


def test_a_range_collapses_to_its_midpoint_with_a_note():
    value, note = parse_number("10–20")

    assert value == 15.0
    assert "range" in note


@pytest.mark.parametrize(
    ("raw", "expected", "token"),
    [
        ("15.6 to 16.3 nm", 15.95, "nm"),
        ("15.6-16.3nm", 15.95, "nm"),
        ("5-10 percent", 7.5, "percent"),
        ("1.2-1.5%", 1.35, "%"),
    ],
)
def test_a_range_with_a_trailing_unit_still_collapses_to_its_midpoint(raw, expected, token):
    # Papers write "15.6 to 16.3 nm" with the unit inside the value; the README promises the midpoint, not
    # the first bound with a "2 numbers" note.
    value, note = parse_number(raw)

    assert value == pytest.approx(expected)
    assert f"trailing unit {token!r} in value ignored" in note
    assert "range" in note


def test_a_qualified_range_with_a_trailing_unit_records_both_readings():
    value, note = parse_number("~15.6-16.3 nm")

    assert value == pytest.approx(15.95)
    assert "qualifier '~' dropped" in note
    assert "trailing unit 'nm' in value ignored" in note


def test_a_multi_number_value_with_x_keeps_the_first_number_path():
    # "40 x 10 cm" ends in a unit, but the "x" in front of it disqualifies stripping: the multiplier
    # reading must survive untouched.
    assert parse_number("40 x 10 cm") == (40.0, "2 numbers found, first used")


@pytest.mark.parametrize("raw", ["10-20", "10 to 20", "10~20"])
def test_every_range_separator_is_recognised(raw):
    value, note = parse_number(raw)

    assert value == 15.0
    assert "range" in note


def test_a_parenthesized_alternative_is_ignored_in_favour_of_the_outer_value():
    # Papers often write "12 (60)" (an alternative value under different conditions); the primary value is
    # the one outside the parens.
    value, note = parse_number("12 (60)")

    assert value == 12.0
    assert "parenthesized" in note


# ---- Plain numbers ----------------------------------------------------------------------


@pytest.mark.parametrize(("raw", "expected"), [("1,200", 1200.0), ("0.3", 0.3), ("-5", -5.0), ("+7", 7.0)])
def test_plain_numbers_parse_without_a_note(raw, expected):
    assert parse_number(raw) == (expected, None)


def test_a_thousands_separator_does_not_split_the_number():
    # If "1,200" were split into 1 and 200, the parsed value would be three orders of magnitude too small.
    assert parse_number("1,200") == (1200.0, None)


def test_the_first_number_wins_when_several_are_present_and_the_count_is_recorded():
    value, note = parse_number("550 nm at 80%")

    assert value == 550.0
    assert note == "2 numbers found, first used"


def test_text_without_any_number_yields_none_and_says_so():
    # The key point: never guess. None lets the comparison layer mark this fact AMBIGUOUS instead of
    # inventing a number.
    assert parse_number("n.a.") == (None, "no number found")


@pytest.mark.parametrize("raw", ["", "not measured", "—"])
def test_other_unparseable_values_also_yield_none(raw):
    value, note = parse_number(raw)

    assert value is None
    assert note == "no number found"


def test_notes_accumulate_when_several_rules_fire():
    # When a qualifier and parentheses both fire, both notes must be kept, or only half the reasoning
    # would be traceable later.
    value, note = parse_number("~12 (60)")

    assert value == 12.0
    assert note == "qualifier '~' dropped; parenthesized alternative ignored"


# ---- MinerU's LaTeX spacing ------------------------------------------------------------------
# A table cell rendered through LaTeX arrives with a space between every character, and the prompt's
# "verbatim" rule keeps it that way. Some cells lose the "$" and the backslash on the way, so the run has
# to be recognised by its own signature: every token a single digit, or a lone "." between digits.


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4 0 0 °C", 400.0),
        ("4 5 0 °C", 450.0),
        ("8 . 4", 8.4),
        ("1 0 ^ { - 4 }", 1e-4),
        ("9 . 1 \\times 1 0 ^ { - 4 }", 9.1e-4),
    ],
)
def test_digits_spaced_out_by_latex_are_read_as_one_number(raw, expected):
    value, _ = parse_number(raw)

    assert value == pytest.approx(expected)


@pytest.mark.parametrize(
    ("raw", "expected", "note"),
    [
        ("8 . 4 \\pm 0 . 1", 8.4, "uncertainty dropped"),
        ("1 0 . 1 \\pm 0 . 5", 10.1, "uncertainty dropped"),
    ],
)
def test_a_spaced_out_measurement_keeps_its_centre_value(raw, expected, note):
    assert parse_number(raw) == (pytest.approx(expected), note)


@pytest.mark.parametrize(
    ("raw", "expected", "note"),
    [
        # Two multi-digit numbers never look like the LaTeX signature, so neither of these may be merged.
        ("300 500", 300.0, "2 numbers found, first used"),
        ("40 x 10 cm", 40.0, "2 numbers found, first used"),
        ("10 20", 10.0, "2 numbers found, first used"),
    ],
)
def test_two_separate_numbers_are_never_joined(raw, expected, note):
    assert parse_number(raw) == (expected, note)


# ---- Approximation markers -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected", "marker"),
    [
        ("around 100 nm", 100.0, "around"),
        ("roughly 100", 100.0, "roughly"),
        ("circa 100", 100.0, "circa"),
        ("approximately 100", 100.0, "approximately"),
        ("∼83.6 %", 83.6, "~"),  # tilde operator U+223C
        ("~86.0 %", 86.0, "~"),
        ("≈ 100", 100.0, "≈"),  # U+2248
    ],
)
def test_approximation_markers_are_dropped_and_recorded(raw, expected, marker):
    value, note = parse_number(raw)

    assert value == pytest.approx(expected)
    assert note.startswith(f"qualifier '{marker}' dropped")


def test_two_single_digit_numbers_are_read_as_one_two_digit_number():
    # The accepted trade-off of the LaTeX-spacing collapse: "2 5" is indistinguishable from a spaced-out
    # "25", and in a table cell 25 is the reading that is almost always right.
    assert parse_number("2 5") == (25.0, None)


@pytest.mark.parametrize("raw", ["2 3 \\mathbf{0}", "2 3 \\mathbf { 0 }", "$2 3 0$"])
def test_a_digit_wrapped_in_a_formatting_command_stays_part_of_its_number(raw):
    # MinerU bolds a table cell's last digit; "2 3 \\mathbf{0}" is 230, not 23 and a stray 0.
    assert parse_number(raw) == (230.0, None)
