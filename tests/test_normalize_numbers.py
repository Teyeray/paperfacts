"""Number parsing: the many spellings found in papers -> a float, plus a note explaining the reading.

By design the LLM only transcribes the source text verbatim; all conversion happens here, making this the
one place in the whole system that can offer a determinism guarantee. Two hard rules: **a questionable
reading must leave a note** (so it's traceable in provenance), and **an unreadable value must return None**
(never a guess).
"""

from __future__ import annotations

import dataclasses

import pytest

from paperfacts.kinds import NO_CONTEXT
from paperfacts.normalize import normalize_field, parse_number, read_range
from support.extraction import make_field
from support.profiles import shipped_profile

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


@pytest.mark.parametrize("raw", ["10-4", "20-10", "300-200", "10-4 Ω·cm", "~10-4"])
def test_a_pair_that_does_not_ascend_is_refused_rather_than_read_as_its_first_number(raw):
    """``"10-4"`` has no mantissa and no "^": it is 10⁻⁴ that lost its caret as often as it is anything else,
    and reading it as 10 makes a resistivity five orders of magnitude off. ``"300-200"`` is no range anybody
    writes. Neither is a midpoint, and neither has a first number worth trusting, so both are refused."""
    value, note = parse_number(raw)

    assert value is None
    assert "ambiguous" in note


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


@pytest.mark.parametrize("raw", ["2.8–4.3 V", "10-20", "3.2 x 10^-4 to 4.1 x 10^-4", "~15.6-16.3 nm"])
def test_a_range_under_reject_gives_no_scalar_and_says_why(raw):
    value, note = parse_number(raw, range_policy="reject")

    assert value is None
    assert "refused (range_policy 'reject')" in note
    assert "midpoint" not in note


@pytest.mark.parametrize("policy", ["reject", "midpoint"])
def test_number_words_under_either_range_policy(policy):
    # "two to three" is words containing numbers, never read as a range (records.spell_number_word), so no
    # policy gives it a value; a single number word is one value, which reject has no reason to refuse.
    spec = dataclasses.replace(shipped_profile().by_name["thickness"], range_policy=policy)

    words_range = normalize_field(
        make_field("thickness", "two to three", unit_raw="nm"), spec, shipped_profile().units, NO_CONTEXT
    )
    one_word = normalize_field(make_field("thickness", "two", unit_raw="nm"), spec, shipped_profile().units, NO_CONTEXT)

    assert words_range.value is None
    assert "midpoint" not in (words_range.normalization_note or "")
    assert one_word.value == 2.0


def test_a_range_behind_a_parenthesised_alternative_is_refused_under_reject():
    value, note = parse_number("10-20 (30)", range_policy="reject")

    assert value is None and "refused (range_policy 'reject')" in note


def test_reject_leaves_everything_that_is_not_a_range_alone():
    for raw in ("12", "3.5 ± 0.2", "~12 (60)", "1.2 × 10^-4 at 300 K", "10–20 nm, 30 nm", "1:4"):
        assert parse_number(raw, range_policy="reject") == parse_number(raw)


def test_midpoint_is_the_default_policy():
    assert parse_number("2.8–4.3 V", range_policy="midpoint") == parse_number("2.8–4.3 V")
    assert parse_number("2.8–4.3 V") == (
        pytest.approx(3.55),
        "trailing unit 'V' in value ignored; range 2.8-4.3 → midpoint",
    )


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


@pytest.mark.parametrize(
    ("raw", "ends"),
    [
        # "to" is the separator, not the first bound's unit: once read as (-60, 20) with unit "to".
        ("-60 to -20", (-60.0, -20.0, "")),
        ("between 450 and 500 °C", (450.0, 500.0, "°C")),
        ("Between 450 °C and 500 °C", (450.0, 500.0, "°C")),
    ],
)
def test_a_negative_range_in_words_and_a_between_range_are_read_as_ranges(raw, ends):
    assert read_range(raw, lambda unit: True) == ends


def test_and_without_between_still_joins_two_values():
    assert parse_number("450 and 500")[0] is None


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
    value, note = parse_number("300 500")

    assert value == 300.0
    assert note == "2 numbers found, first used"


def test_a_condition_after_the_value_is_set_aside_with_a_note():
    # "at ..." states when the value was measured; its numbers are not the value's. The same rule holds for
    # scientific notation, so "1.2 × 10^-4 at 300 K" is not refused for its "300".
    assert parse_number("550 nm at 80%") == (550.0, "condition 'at 80%' ignored")
    assert parse_number("1.2 × 10^-4 at 300 K") == (pytest.approx(1.2e-4), "condition 'at 300 K' ignored")


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


# ---- Real OCR spellings, pinned as one table -------------------------------------------------
# Each row is a spelling seen in a paper or produced by one of the parsers, with the reading the parser must
# give: a value, or None when no reading is safe. The rows marked "refuse" used to return a confident wrong
# number (1e-4 for a 4.5e-4 mantissa, -0.0015 for a range, 10 for 10^-4, 1 -> 100 % for a 1:4 gas ratio).

SPELLINGS = [
    # value_raw, expected value (None = refuse), a fragment the note must contain (None = no note)
    ("(4.5 ± 0.2) × 10^-4", 4.5e-4, "uncertainty dropped"),
    ("(4.5±0.2)×10−4", 4.5e-4, "uncertainty dropped"),
    ("(4.5) × 10⁻⁴", 4.5e-4, None),
    ("$( 4 . 5 \\pm 0 . 2 ) \\times 1 0 ^ { - 4 }$", 4.5e-4, "uncertainty dropped"),
    ("1.2 × 10^(-4)", 1.2e-4, None),
    ("3.2 x 10^-4 to 4.1 x 10^-4", 3.65e-4, "midpoint"),
    ("1.2 x 10^-4 ± 0.1 x 10^-4", 1.2e-4, "uncertainty dropped"),
    ("-1.5 × 10^-3", -1.5e-3, None),
    ("1.2-1.5 × 10^-3", None, "ambiguous"),  # does the exponent apply to 1.2? refuse
    ("4.5 ± 0.2 × 10^-4", None, "ambiguous"),  # the exponent may scale only the uncertainty
    ("4.1 x 10^-4 - 3.2 x 10^-4", None, "descending"),
    ("1.2 × 10^-4 at 550 nm", 1.2e-4, "condition 'at 550 nm' ignored"),
    ("10-4", None, "ambiguous"),
    ("300-200", None, "ambiguous"),
    ("1:4", None, "ratio"),
    ("O2/Ar = 1:4", None, "ratio"),
    ("Ar:O2 = 9:1", None, "ratio"),
    ("(12)", None, "parenthesis"),
    ("x (5)", None, "parenthesis"),
    ("12 (60", None, "parenthes"),
    ("10–20 at 550 nm", 15.0, "midpoint"),
    ("10–20 nm, 30 nm", None, "range among other numbers"),
    # Still accepted, unchanged:
    ("12 (60)", 12.0, "parenthesized alternative ignored"),
    ("1.2 × 10⁻⁴ Ω·cm", 1.2e-4, None),
    ("550 nm at 80%", 550.0, "condition"),
    # Ranges with a unit on each bound, as for "3.2e-4 to 4.1e-4": the midpoint, never the first bound.
    ("80%–85%", 82.5, "midpoint"),
    ("20 W–100 W", 60.0, "midpoint"),
    ("500 °C to 530 °C", 515.0, "midpoint"),
    ("500 ℃ to 530 ℃", 515.0, "midpoint"),
    ("5 nm - 10 μm", None, "different units"),
    # Slash ratios are refused like colon ratios.
    ("10/10", None, "ratio"),
    ("12/10/3", None, "ratio"),
    ("Ar/O2 = 20/1", None, "ratio"),
    # The digits of a formula or a unit exponent are not the value.
    ("O2/(Ar+O2) = 5%", 5.0, "before '='"),
    ("5% H2", 5.0, "formula"),
    ("4.5 × 10^20 cm^-3", 4.5e20, "formula"),
    ("4.5 × 10^20 cm-3", None, "ambiguous"),  # without a caret "-3" may be a second number
    ("x = 0.1", 0.1, "before '='"),
    ("\\sim82", 82.0, "qualifier '~' dropped"),  # MinerU's \sim without the $ markers
    ("$ \\sim $25 and 70", None, "joined by 'and'"),
    ("25 nm or 70 nm", None, "joined by 'and'"),
    ("1.2e-4", 1.2e-4, None),
    ("1.2x10^-4", 1.2e-4, None),
    ("15.6 to 16.3 nm", 15.95, "midpoint"),
    # A list is several values, and a number carrying its own unit before another number is a value beside
    # a second quantity: "550 nm: 85%" is a transmittance of 85 at 550 nm, never 550.
    ("30, 40", None, "separated by"),
    ("550 nm, 80%", None, "separated by"),
    ("550 nm: 85%", None, "separated by"),
    ("20; 30", None, "separated by"),
    ("140 nm ATO/25 nm ITO", None, "its own unit"),
    ("∅32 mm × 40 mm", None, "its own unit"),
    # parse_number sees no field, so a compound duration is refused here; normalize_field reads it.
    ("3 h 30 min", None, "its own unit"),
    ("2 in x 3 in", None, "its own unit"),
    # A condition introduced by "for", "during", "under" or "after" is set aside like one after "at". Not
    # "in": that is also the inch.
    ("400 °C for 2 h", 400.0, "condition 'for 2 h' ignored"),
    ("400 °C in air for 1 h", 400.0, "condition 'for 1 h' ignored"),
    ("500 °C under N2 for 1 h", 500.0, "condition 'under N2 for 1 h' ignored"),
    ("90% for 550 nm", 90.0, "condition 'for 550 nm' ignored"),
    # "after" introduces another state of the sample (after bending, after annealing), not a condition.
    ("100 nm after annealing at 400 °C", None, "after"),
    ("85% after 10 cycles", None, "after"),
    ("15 after 1000 bending cycles", None, "after"),
    # A tail is set aside only when the value keeps a number of its own: a quote opening with the verb reads.
    ("deposited for 10 min", 10.0, None),
    ("12 Ω/sq during 30 min", 12.0, "condition"),
]


@pytest.mark.parametrize(("raw", "expected", "fragment"), SPELLINGS)
def test_real_spellings_read_as_the_value_or_are_refused(raw, expected, fragment):
    value, note = parse_number(raw)

    if expected is None:
        assert value is None
    else:
        assert value == pytest.approx(expected)
    if fragment is None:
        assert note is None
    else:
        assert fragment in note


@pytest.mark.parametrize("raw", ["four", "four-inch", "one of the samples", "five to ten", "ten-fold"])
def test_parse_number_has_no_unit_context_and_so_reads_no_number_word(raw):
    # Whether "four" is a value depends on its unit, which parse_number does not see; normalize_field decides.
    assert parse_number(raw) == (None, "no number found")


# ---- A quote too long to be one number --------------------------------------------------------------------------


def test_a_quote_longer_than_the_cap_is_refused_before_it_is_parsed():
    import time

    from paperfacts.fields import MAX_NUMBER_QUOTE
    from paperfacts.normalize import read_interval

    started = time.perf_counter()
    value, note = parse_number("1" * 20_000)
    clean, why = read_interval("1" * 20_000, lambda unit: True)
    assert time.perf_counter() - started < 0.5
    assert value is None and "20000 characters is too long" in (note or "")
    assert clean is None and "too long" in (why or "")
    # The cap is a guard, not a reading rule: a quote at it is still read.
    assert parse_number(" " * (MAX_NUMBER_QUOTE - 2) + "12")[0] == 12.0


def test_the_cap_is_above_every_corpus_value():
    import json
    from pathlib import Path

    from paperfacts.fields import MAX_NUMBER_QUOTE

    corpus = Path(__file__).parent / "fixtures" / "corpus" / "values.json"
    longest = max(len(entry["value_raw"]) for entry in json.loads(corpus.read_text(encoding="utf-8")))
    assert longest < MAX_NUMBER_QUOTE / 2
