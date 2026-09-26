"""``range_policy`` lower / upper: a range quoted as one value read as the end the field asks for.

The lanes and the comparison read the chosen end (``parse_number``); the dataset cell takes it too, because an
end is a number the paper printed. ``midpoint`` and ``reject`` behave exactly as before, and a bound is never
mistaken for a range.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from paperfacts import keys
from paperfacts.compare import FieldComparison
from paperfacts.decide import decide
from paperfacts.errors import ConfigError
from paperfacts.fields import FieldRole
from paperfacts.kinds import RULES
from paperfacts.normalize import normalize_field, parse_number, read_range, read_value
from support.extraction import make_field
from support.profiles import make_profile, shipped_profile

TCO = shipped_profile()
UNITS = TCO.units
CORPUS_VALUES = json.loads((Path(__file__).parent / "fixtures" / "corpus" / "values.json").read_text(encoding="utf-8"))
ENDS = ("lower", "upper")


def _spec(policy: str):
    return dataclasses.replace(TCO.by_name["annealing_temperature"], range_policy=policy)


def _field(value_raw: str, *, unit_raw: str | None = "°C", bound: str | None = None, condition: str | None = None):
    field = make_field("annealing_temperature", value_raw, unit_raw=unit_raw, condition=condition, source_ids=("b",))
    return field.model_copy(update={"bound": bound})


# ---- parse_number ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "low", "high", "shown"),
    [
        ("10-20", 10.0, 20.0, "range 10-20"),
        ("15.6 to 16.3 nm", 15.6, 16.3, "range 15.6-16.3"),
        ("1.2 × 10^-3 to 1.5 × 10^-3", 1.2e-3, 1.5e-3, "range 0.0012-0.0015"),
        ("3.2 x 10^-4 - 4.1 x 10^-4", 3.2e-4, 4.1e-4, "range 0.00032-0.00041"),
    ],
)
def test_lower_and_upper_read_the_chosen_end_with_a_note(raw, low, high, shown):
    lower_value, lower_note = parse_number(raw, range_policy="lower")
    upper_value, upper_note = parse_number(raw, range_policy="upper")

    assert lower_value == pytest.approx(low)
    assert upper_value == pytest.approx(high)
    assert lower_note.endswith(f"{shown} → lower bound")
    assert upper_note.endswith(f"{shown} → upper bound")
    assert "midpoint" not in lower_note + upper_note


@pytest.mark.parametrize("policy", ["midpoint", "reject", "lower", "upper"])
@pytest.mark.parametrize("raw", ["20-10", "300-200 °C", "1.5 × 10^-3 to 1.2 × 10^-3", "1.2-1.5 × 10^-3"])
def test_a_descending_range_or_one_exponent_for_two_numbers_is_refused_under_every_policy(policy, raw):
    value, note = parse_number(raw, range_policy=policy)

    assert value is None
    assert "ambiguous" in note


@pytest.mark.parametrize("policy", ENDS)
@pytest.mark.parametrize("raw", ["above 80", ">80", "≥ 80 %", "12", "3.5 ± 0.2", "1.2 × 10^-4 at 300 K", "1:4"])
def test_lower_and_upper_leave_everything_that_is_not_a_range_alone(policy, raw):
    assert parse_number(raw, range_policy=policy) == parse_number(raw)


@pytest.mark.parametrize("policy", ENDS)
def test_the_end_policies_move_only_the_corpus_strings_read_as_a_midpoint(policy):
    # Over every numeric string of the real corpus: a string that is no range reads exactly as under the default,
    # and one read as a midpoint reads as one of its ends with the same notes but the last.
    moved = 0
    for row in CORPUS_VALUES:
        raw = row["value_raw"]
        midpoint = parse_number(raw)
        chosen = parse_number(raw, range_policy=policy)
        if midpoint[1] is None or not midpoint[1].endswith(" → midpoint"):
            assert chosen == midpoint, raw
            continue
        moved += 1
        assert chosen[1] == midpoint[1].removesuffix(" → midpoint") + f" → {policy} bound"
        assert chosen[0] != midpoint[0]

    assert moved > 0


# ---- The comparison's reading ---------------------------------------------------------------------------


def test_normalize_field_takes_the_upper_end_in_the_canonical_unit():
    value = normalize_field(_field("450-500"), _spec("upper"), UNITS)

    assert value.value == pytest.approx(500.0)
    assert value.unit == "℃"
    assert value.normalization_note == "range 450-500 → upper bound"


@pytest.mark.parametrize("policy", ENDS)
def test_a_bound_grounding_found_reads_as_under_midpoint(policy):
    # "80" quoted out of "above 80": the reading is "above 80", which no policy reads as a range.
    field = _field("80", bound="above")

    assert normalize_field(field, _spec(policy), UNITS) == normalize_field(field, _spec("midpoint"), UNITS)


def test_read_range_goes_through_read_value():
    assert read_range(read_value(_field("~450-500 °C"), _spec("upper"), UNITS)) == (
        450.0,
        500.0,
        "qualifier '~' dropped; trailing unit '°C' in value ignored; range 450-500",
    )
    # Number words are never a range (records.spell_number_word), here as in the comparison.
    assert read_range(read_value(_field("two to three", unit_raw="h"), TCO.by_name["annealing_time"], UNITS)) is None

    assert read_range(read_value(_field("450-500", bound="above"), _spec("upper"), UNITS)) is None
    assert read_range(read_value(_field("> 450-500"), _spec("upper"), UNITS)) is None
    assert read_range(read_value(_field("450"), _spec("upper"), UNITS)) is None


# ---- The dataset cell -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy", "expected", "word"),
    [("upper", 500.0, "上限"), ("lower", 450.0, "下限")],
)
def test_an_end_fills_the_cell_with_a_note(policy, expected, word):
    value, note = RULES["numeric"].cell(_field("450-500"), _spec(policy), UNITS)

    assert value == pytest.approx(expected)
    assert f"原文为区间 450–500，按字段配置取{word}" in note


def test_an_approximate_range_keeps_its_end_not_a_centre():
    value, note = RULES["numeric"].cell(_field("~450-500"), _spec("lower"), UNITS)

    assert value == pytest.approx(450.0)
    assert "原文为近似值" in note
    assert "中心值" not in note


def test_a_scientific_range_fills_the_cell_with_its_end():
    spec = dataclasses.replace(TCO.by_name["resistivity"], range_policy="upper")
    value, note = RULES["numeric"].cell(
        make_field("resistivity", "1.2 × 10^-3 to 1.5 × 10^-3", unit_raw="Ω·cm", source_ids=("b",)), spec, UNITS
    )

    assert value == pytest.approx(1.5e-3)
    assert "取上限" in note


@pytest.mark.parametrize("policy", ["midpoint", "reject"])
def test_midpoint_and_reject_still_keep_a_range_out_of_the_cell(policy):
    value, note = RULES["numeric"].cell(_field("450-500"), _spec(policy), UNITS)

    assert value is None
    assert note == "含多个数值、范围、上下界或附加条件，不能取中点或第一个数"


@pytest.mark.parametrize("policy", ENDS)
@pytest.mark.parametrize(
    "field",
    [
        _field("above 450-500"),
        _field("> 450-500"),
        _field("450-500", bound="above"),
        _field("80", bound="above"),
        _field(">80"),
        _field("500-450"),
        _field("450-500 °C for 2 h"),
        _field("450-500 (600)"),
        _field("1.2-1.5 × 10^-3"),
    ],
    ids=["above", "qualifier", "bound", "bound-scalar", "gt", "descending", "condition", "alternative", "exponent"],
)
def test_a_bound_or_anything_but_one_clean_range_stays_out_of_the_cell(policy, field):
    value, _ = RULES["numeric"].cell(field, _spec(policy), UNITS)

    assert value is None


def _decision(policy: str):
    field = _field("450-500")
    comparison = FieldComparison(scope="sample:A", field="annealing_temperature", status="missing", a=field)
    return decide(_spec(policy), [("mineru", field)], [comparison], units=UNITS)


def test_an_upper_cell_is_committed_and_a_midpoint_one_is_non_scalar():
    upper = _decision("upper")
    midpoint = _decision("midpoint")

    assert (upper.status, upper.value) == ("single_source", pytest.approx(500.0))
    assert midpoint.status == "non_scalar"
    assert midpoint.value is None


# ---- Profile and keys -----------------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ENDS)
def test_the_loader_accepts_an_end_policy_on_a_numeric_field_only(policy):
    assert make_profile({"fields.1.range_policy": policy}).by_name["coating_thickness"].range_policy == policy
    with pytest.raises(ConfigError, match="range_policy is only meaningful for a numeric field"):
        make_profile({"fields.2.range_policy": policy})


@pytest.mark.parametrize("policy", ENDS)
def test_an_end_policy_moves_the_cleaning_and_verdict_material(policy):
    material = keys._field_material(_spec(policy), FieldRole.CLEANING, FieldRole.VERDICT)

    assert material["range_policy"] == policy
    assert "range_policy" not in keys._field_material(_spec("midpoint"), FieldRole.CLEANING, FieldRole.VERDICT)
