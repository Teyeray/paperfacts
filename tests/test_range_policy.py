"""``range_policy`` lower / upper: a range quoted as one value read as the end the field asks for.

The lanes and the comparison read the chosen end (``parse_number``); the dataset cell takes it too, because an
end is a number the paper printed. Both take it only from what ``read_range`` calls a clean range, so they never
disagree about which quotes have an end. ``midpoint`` and ``reject`` behave exactly as before, and a bound is
never mistaken for a range.
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
from paperfacts.fields import RANGE_ENDS, FieldRole
from paperfacts.kinds import RULES
from paperfacts.normalize import normalize_field, parse_number, read_value, unit_of_value
from paperfacts.readers import read_range
from paperfacts.records import NO_CONTEXT
from support.extraction import make_field
from support.profiles import make_profile, shipped_profile

TCO = shipped_profile()
UNITS = TCO.units
CORPUS_VALUES = json.loads((Path(__file__).parent / "fixtures" / "corpus" / "values.json").read_text(encoding="utf-8"))
ENDS = RANGE_ENDS


def _spec(policy: str):
    return dataclasses.replace(TCO.by_name["annealing_temperature"], range_policy=policy)


def _field(
    value_raw: str,
    *,
    unit_raw: str | None = "°C",
    bound: str | None = None,
    condition: str | None = None,
    name: str = "annealing_temperature",
):
    field = make_field(name, value_raw, unit_raw=unit_raw, condition=condition, source_ids=("b",))
    return field.model_copy(update={"bound": bound})


def _named_spec(name: str, policy: str):
    return dataclasses.replace(TCO.by_name[name], range_policy=policy)


def _units_of(*names: str):
    """Whether a unit written in a quote is one of these TCO fields' (in their canonical unit), as the registry
    reads it."""
    specs = [TCO.by_name[name] for name in names]
    return lambda written: any(unit_of_value(spec, spec.canonical_unit, written, UNITS) is not None for spec in specs)


THICKNESS_UNIT = _units_of("thickness")
TEMPERATURE_UNIT = _units_of("annealing_temperature")


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
    lower_value, lower_note = parse_number(raw, range_policy="lower", range_unit=THICKNESS_UNIT)
    upper_value, upper_note = parse_number(raw, range_policy="upper", range_unit=THICKNESS_UNIT)

    assert lower_value == pytest.approx(low)
    assert upper_value == pytest.approx(high)
    assert lower_note.endswith(f"{shown} → lower end")
    assert upper_note.endswith(f"{shown} → upper end")
    assert "midpoint" not in lower_note + upper_note


@pytest.mark.parametrize("policy", ["midpoint", "reject", "lower", "upper"])
@pytest.mark.parametrize("raw", ["20-10", "300-200 °C", "1.5 × 10^-3 to 1.2 × 10^-3", "1.2-1.5 × 10^-3"])
def test_a_descending_range_or_one_exponent_for_two_numbers_is_refused_under_every_policy(policy, raw):
    value, note = parse_number(raw, range_policy=policy, range_unit=TEMPERATURE_UNIT)

    assert value is None
    assert "ambiguous" in note


@pytest.mark.parametrize("policy", ENDS)
@pytest.mark.parametrize("raw", ["above 80", ">80", "≥ 80 %", "12", "3.5 ± 0.2", "1.2 × 10^-4 at 300 K", "1:4"])
def test_lower_and_upper_leave_everything_that_is_not_a_range_alone(policy, raw):
    assert parse_number(raw, range_policy=policy, range_unit=TEMPERATURE_UNIT) == parse_number(raw)


@pytest.mark.parametrize("policy", ENDS)
@pytest.mark.parametrize("field_units", [True, False], ids=["field-units", "no-unit-accepted"])
def test_the_end_policies_move_only_the_corpus_strings_read_as_a_midpoint(policy, field_units):
    # Over every numeric string of the real corpus: a string that is no range reads exactly as under the default;
    # one read as a midpoint reads as the end of a clean range with the same notes but the last, and any other
    # range is refused. Every corpus range is clean in its field's unit; accepting no unit at all refuses the one
    # that writes one ("500 °C to 530 °C").
    moved = refused = 0
    for row in CORPUS_VALUES:
        raw = row["value_raw"]
        accepts = _units_of(row["field"]) if field_units else (lambda written: False)
        midpoint = parse_number(raw)
        chosen = parse_number(raw, range_policy=policy, range_unit=accepts)
        if midpoint[1] is None or not midpoint[1].endswith(" → midpoint"):
            assert chosen == midpoint, raw
            continue
        if read_range(raw, accepts) is None:
            refused += 1
            assert chosen[0] is None, raw
            assert chosen[1].endswith(f"has no {policy} end: not one clean range in the field's unit; ambiguous")
            continue
        moved += 1
        assert chosen[1] == midpoint[1].removesuffix(" → midpoint") + f" → {policy} end"
        assert chosen[0] != midpoint[0]

    assert moved > 0
    assert (refused > 0) != field_units


# ---- The comparison's reading ---------------------------------------------------------------------------


def test_normalize_field_takes_the_upper_end_in_the_canonical_unit():
    value = normalize_field(_field("450-500"), _spec("upper"), UNITS, NO_CONTEXT)

    assert value.value == pytest.approx(500.0)
    assert value.unit == "℃"
    assert value.normalization_note == "range 450-500 → upper end"


@pytest.mark.parametrize("policy", ENDS)
def test_a_bound_grounding_found_reads_as_under_midpoint(policy):
    # "80" quoted out of "above 80": the reading is "above 80", which no policy reads as a range.
    field = _field("80", bound="above")

    assert normalize_field(field, _spec(policy), UNITS, NO_CONTEXT) == normalize_field(
        field, _spec("midpoint"), UNITS, NO_CONTEXT
    )


def test_read_range_reads_the_text_read_value_built():
    reading = read_value(_field("~450-500 °C"), _spec("upper"), UNITS)
    assert read_range(reading.text, TEMPERATURE_UNIT) == (450.0, 500.0, "°C")
    # Number words are never a range (records.spell_number_word), here as in the comparison.
    spec = TCO.by_name["annealing_time"]
    assert (
        read_range(read_value(_field("two to three", unit_raw="h"), spec, UNITS).text, _units_of("annealing_time"))
        is None
    )
    # A bound grounding found before the quote is put in front of the text.
    assert (
        read_range(read_value(_field("450-500", bound="above"), _spec("upper"), UNITS).text, TEMPERATURE_UNIT) is None
    )
    assert read_range("450", TEMPERATURE_UNIT) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10-20", (10.0, 20.0, "")),
        ("80%–85%", (80.0, 85.0, "%")),
        ("500 °C to 530 °C", (500.0, 530.0, "°C")),
        ("T = 450-500 °C", (450.0, 500.0, "°C")),
        ("about 450-500", (450.0, 500.0, "")),
        ("1.2 × 10^-3 to 1.5 × 10^-3", (1.2e-3, 1.5e-3, "")),
        ("1.2e-4-1.5e-4 Ω·cm", (1.2e-4, 1.5e-4, "Ω.cm")),
        # Anything but one clean range in the field's unit.
        ("450-500 K", None),
        ("450-500 nm", None),
        ("1.2e-4-1.5e-4 nm", None),
        ("> 450-500", None),
        ("≥450-500", None),
        ("above 450-500", None),
        ("below 1.2 x 10^-4 - 1.5 x 10^-4", None),
        ("above 1.2e-4 to 1.5e-4", None),
        ("less than 1.2e-4-1.5e-4", None),
        ("1.2e-4-1.5e-4 Ω cm (sample A)", None),
        ("450-500 (600)", None),
        ("450-500 °C for 2 h", None),
        ("85-90% after annealing", None),
        ("1.2-1.5 × 10^-3", None),
        ("500-450", None),
        ("450 °C-500 K", None),
    ],
)
def test_read_range_is_the_one_definition_of_a_clean_range(raw, expected):
    found = read_range(raw, _units_of("annealing_temperature", "o2_ratio", "resistivity"))

    if expected is None:
        assert found is None
    else:
        assert found is not None
        assert found[:2] == pytest.approx(expected[:2])
        assert found[2] == expected[2]


# ---- The dataset cell -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy", "expected", "word"),
    [("upper", 500.0, "上限"), ("lower", 450.0, "下限")],
)
def test_an_end_fills_the_cell_with_a_note(policy, expected, word):
    value, note = RULES["numeric"].cell(_field("450-500"), _spec(policy), UNITS, NO_CONTEXT)

    assert value == pytest.approx(expected)
    assert f"原文为区间 450–500，按字段配置取{word}" in note


def test_an_approximate_range_keeps_its_end_not_a_centre():
    value, note = RULES["numeric"].cell(_field("~450-500"), _spec("lower"), UNITS, NO_CONTEXT)

    assert value == pytest.approx(450.0)
    assert "原文为近似值" in note
    assert "中心值" not in note


def test_a_scientific_range_fills_the_cell_with_its_end():
    spec = dataclasses.replace(TCO.by_name["resistivity"], range_policy="upper")
    value, note = RULES["numeric"].cell(
        make_field("resistivity", "1.2 × 10^-3 to 1.5 × 10^-3", unit_raw="Ω·cm", source_ids=("b",)),
        spec,
        UNITS,
        NO_CONTEXT,
    )

    assert value == pytest.approx(1.5e-3)
    assert "取上限" in note


@pytest.mark.parametrize("policy", ["midpoint", "reject"])
def test_midpoint_and_reject_still_keep_a_range_out_of_the_cell(policy):
    value, note = RULES["numeric"].cell(_field("450-500"), _spec(policy), UNITS, NO_CONTEXT)

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
    value, _ = RULES["numeric"].cell(field, _spec(policy), UNITS, NO_CONTEXT)

    assert value is None


# The review's exact strings: a range whose own unit is not the field's, and a bound before a scientific range,
# fill no cell and give no lane value; and the lanes and the cell agree on every range.
UNCLEAN = [
    ("annealing_temperature", "450-500 K", "°C"),
    ("annealing_temperature", "450-500 nm", "°C"),
    ("resistivity", "1.2e-4-1.5e-4 nm", "Ω·cm"),
    ("resistivity", "1.2 × 10^-4 - 1.5 × 10^-4 K", "Ω·cm"),
    ("resistivity", "below 1.2 x 10^-4 - 1.5 x 10^-4", "Ω·cm"),
    ("resistivity", "above 1.2e-4 to 1.5e-4", "Ω·cm"),
    ("resistivity", "less than 1.2e-4-1.5e-4", "Ω·cm"),
    ("resistivity", "1.2e-4-1.5e-4 Ω cm (sample A)", "Ω·cm"),
    ("annealing_temperature", "> 450-500", "°C"),
    ("annealing_temperature", "450-500 (600)", "°C"),
    ("annealing_temperature", "450-500 °C for 2 h", "°C"),
]


@pytest.mark.parametrize("policy", ENDS)
@pytest.mark.parametrize(("name", "raw", "unit_raw"), UNCLEAN)
def test_an_unclean_range_has_no_end_in_the_lanes_or_the_cell(policy, name, raw, unit_raw):
    field, spec = _field(raw, unit_raw=unit_raw, name=name), _named_spec(name, policy)

    lane = normalize_field(field, spec, UNITS, NO_CONTEXT)
    cell, _ = RULES["numeric"].cell(field, spec, UNITS, NO_CONTEXT)

    assert lane.value is None
    assert cell is None


@pytest.mark.parametrize("policy", ENDS)
@pytest.mark.parametrize(
    ("name", "raw", "unit_raw"),
    [
        *UNCLEAN,
        ("annealing_temperature", "450-500", "°C"),
        ("annealing_temperature", "450-500 °C", "°C"),
        ("annealing_temperature", "~450-500 °C", "°C"),
        ("annealing_temperature", "450 °C to 500 °C", "°C"),
        ("annealing_temperature", "450-500 ℃", "°C"),
        ("resistivity", "1.2e-4-1.5e-4 Ω·cm", "Ω·cm"),
        ("resistivity", "1.2 × 10^-3 to 1.5 × 10^-3", "Ω·cm"),
        ("o2_ratio", "0.8-1.2", None),
        ("o2_ratio", "0.2-0.5", None),
        ("o2_ratio", "5-10 %", "%"),
    ],
)
def test_the_lanes_and_the_cell_take_the_same_end(policy, name, raw, unit_raw):
    field, spec = _field(raw, unit_raw=unit_raw, name=name), _named_spec(name, policy)

    lane = normalize_field(field, spec, UNITS, NO_CONTEXT)
    cell, _ = RULES["numeric"].cell(field, spec, UNITS, NO_CONTEXT)

    assert lane.value == (None if cell is None else pytest.approx(cell))


@pytest.mark.parametrize(
    ("raw", "lower", "upper"),
    [("0.8-1.2", 0.8, 1.2), ("0.2-0.5", 20.0, 50.0)],
)
def test_a_bare_range_is_a_fraction_or_a_percent_as_a_whole(raw, lower, upper):
    # "0.8-1.2" is not 80 % at one end and 1.2 % at the other: the range decides once, from all of it.
    for policy, expected in (("lower", lower), ("upper", upper)):
        field, spec = _field(raw, unit_raw=None, name="o2_ratio"), _named_spec("o2_ratio", policy)

        assert normalize_field(field, spec, UNITS, NO_CONTEXT).value == pytest.approx(expected)
        assert RULES["numeric"].cell(field, spec, UNITS, NO_CONTEXT)[0] == pytest.approx(expected)


def test_a_bare_midpoint_still_decides_from_the_midpoint():
    field = _field("0.8-1.2", unit_raw=None, name="o2_ratio")

    value = normalize_field(field, _named_spec("o2_ratio", "midpoint"), UNITS, NO_CONTEXT)

    assert value.value == pytest.approx(1.0)
    assert value.normalization_note == "range 0.8-1.2 → midpoint; no unit; read as percent"


def _decision(policy: str):
    field = _field("450-500")
    comparison = FieldComparison(scope="sample:A", field="annealing_temperature", status="missing", a=field)
    return decide(_spec(policy), [("mineru", field)], [comparison], units=UNITS, ctx=NO_CONTEXT)


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
