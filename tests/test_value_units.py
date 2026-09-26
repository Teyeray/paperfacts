"""A unit written inside the quoted value ("1.5e-4 Ω·cm", "450-500 °C", ">80 mW") against the field's unit.

One helper decides it for every reader -- the lanes, the dataset cell's scalar and range paths, and an interval's
range and bound (``normalize.unit_of_value``): units are compared as the registry converts them, never as
spellings; a written unit that converts as ``unit_raw`` does changes nothing; one that converts otherwise is the
more specific statement and is converted from; one the registry cannot read for the field is refused.
"""

from __future__ import annotations

import dataclasses

import pytest

from paperfacts.kinds import RULES, interval_text
from paperfacts.normalize import normalize_field, parse_number, read_number, read_range, unit_of_value
from paperfacts.records import FieldValue
from support.profiles import make_profile, profile_data, shipped_profile

TCO = shipped_profile()
UNITS = TCO.units


def _spec(name: str, policy: str = "midpoint"):
    return dataclasses.replace(TCO.by_name[name], range_policy=policy)


def _field(name: str, value_raw: str, unit_raw: str | None, **update: object) -> FieldValue:
    return FieldValue(field=name, value_raw=value_raw, unit_raw=unit_raw, source_ids=("b",), **update)  # type: ignore[arg-type]


def _lane_and_cell(name: str, raw: str, unit_raw: str | None, policy: str = "midpoint"):
    field, spec = _field(name, raw, unit_raw), _spec(name, policy)
    return normalize_field(field, spec, UNITS), RULES["numeric"].cell(field, spec, UNITS)


# ---- The helper ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit_raw", "written", "expected"),
    [
        ("Ω·cm", "Ω cm", ("Ω·cm", False)),
        ("Ω·cm", "Ω-cm", ("Ω·cm", False)),
        ("Ω·cm", "ohm cm", ("Ω·cm", False)),
        ("×10^-3 Ω·cm", "mΩ·cm", ("×10^-3 Ω·cm", False)),  # the header's power of ten counts
        ("mΩ·cm", "Ω·cm", ("Ω·cm", True)),
        ("×10^-4 Ω·cm", "Ω·cm", ("Ω·cm", True)),
        (None, "Ω·cm", ("Ω·cm", True)),
        ("Ω·cm", "", ("Ω·cm", False)),
        ("Ω·cm", "nm", None),
        ("Ω·cm", "Ω cm (sample A)", None),
    ],
)
def test_a_written_unit_is_compared_by_how_it_converts(unit_raw, written, expected):
    assert unit_of_value(TCO.by_name["resistivity"], unit_raw, written, UNITS) == expected


@pytest.mark.parametrize(("written", "known"), [("C", True), ("°C", True), ("℃", True), ("K", False), ("oC", False)])
def test_a_temperature_is_what_the_registry_reads(written, known):
    # "oC" and "deg C" are no built-in spelling of ℃; a profile that meets them declares them (units.DeclaredUnit).
    assert (unit_of_value(TCO.by_name["annealing_temperature"], "°C", written, UNITS) is not None) == known


# ---- Item 1: a written unit that converts otherwise is converted from ----------------------------------------


def test_a_scalar_is_read_in_its_own_unit_not_unit_raw():
    lane, (cell, note) = _lane_and_cell("resistivity", "1.5e-4 Ω·cm", "mΩ·cm")

    assert lane.value == pytest.approx(1.5e-4)
    assert "own unit 'Ω.cm'" in lane.normalization_note
    assert cell == pytest.approx(1.5e-4)
    assert "自带单位" in note


@pytest.mark.parametrize("policy", ["lower", "upper"])
def test_a_range_is_read_in_its_own_unit_not_unit_raw(policy):
    lane, (cell, _) = _lane_and_cell("resistivity", "1.2e-4-1.5e-4 Ω·cm", "mΩ·cm", policy)
    expected = 1.2e-4 if policy == "lower" else 1.5e-4

    assert lane.value == pytest.approx(expected)
    assert cell == pytest.approx(expected)


def test_a_header_power_of_ten_is_not_applied_to_a_range_that_writes_its_own_unit():
    lane, (cell, note) = _lane_and_cell("resistivity", "1.2-1.5 Ω·cm", "×10^-4 Ω·cm", "upper")

    assert lane.value == pytest.approx(1.5)
    assert cell == pytest.approx(1.5)
    assert "scale factor" not in note


def test_a_midpoint_is_read_in_the_ranges_own_unit():
    lane, _ = _lane_and_cell("resistivity", "1.2e-4-1.5e-4 Ω·cm", "mΩ·cm")

    assert lane.value == pytest.approx(1.35e-4)


def test_a_written_unit_that_converts_as_unit_raw_leaves_the_reading_byte_identical():
    same = _lane_and_cell("resistivity", "1.5 mΩ·cm", "×10^-3 Ω·cm")
    bare = _lane_and_cell("resistivity", "1.5", "×10^-3 Ω·cm")

    assert same[0].value == bare[0].value
    assert same[0].normalization_note == bare[0].normalization_note
    assert same[1] == bare[1]


def test_a_value_without_unit_raw_is_read_in_the_unit_it_writes():
    # Bare, "0.6" would be a fraction, 60 %; the quote says 0.6 %.
    lane, (cell, _) = _lane_and_cell("h2_ratio", "0.6%", None)

    assert lane.value == pytest.approx(0.6)
    assert cell == pytest.approx(0.6)


def test_an_unknown_written_unit_keeps_the_cell_out():
    _, (cell, _) = _lane_and_cell("annealing_temperature", "450 K", "°C")

    assert cell is None


# ---- Item 2: spellings of one unit are one unit ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "raw", "unit_raw", "expected"),
    [
        ("resistivity", "1.2e-4-1.5e-4 Ω cm", "Ω·cm", 1.5e-4),
        ("resistivity", "1.2e-4-1.5e-4 Ω-cm", "Ω·cm", 1.5e-4),
        ("resistivity", "1.2e-4-1.5e-4 ohm cm", "Ω·cm", 1.5e-4),
        ("annealing_temperature", "450 - 500 C", "°C", 500.0),
        ("annealing_temperature", "450-500 ℃", "°C", 500.0),
    ],
)
def test_a_range_in_another_spelling_of_the_fields_unit_has_its_end(name, raw, unit_raw, expected):
    lane, (cell, _) = _lane_and_cell(name, raw, unit_raw, "upper")

    assert lane.value == pytest.approx(expected)
    assert cell == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["450-500 oC", "450–500 deg C", "450-500 K"])
def test_a_range_in_a_unit_the_registry_does_not_read_has_no_end(raw):
    lane, (cell, _) = _lane_and_cell("annealing_temperature", raw, "°C", "upper")

    assert lane.value is None
    assert cell is None


def test_a_scalar_in_another_spelling_of_the_fields_unit_fills_the_cell():
    _, (cell, note) = _lane_and_cell("resistivity", "1.5e-4 Ω cm", "Ω·cm")

    assert cell == pytest.approx(1.5e-4)
    assert note is None


# ---- Item 3: one range grammar, parsed once ------------------------------------------------------------------


def test_a_range_is_what_the_general_reader_reads_as_one():
    # "450to500" is no range to the general reader (its "500" is glued to a letter), so it has no end either.
    assert read_range("450to500", lambda written: True) is None
    assert parse_number("450to500", range_policy="upper", range_unit=lambda written: True) == parse_number("450to500")
    _, (cell, _) = _lane_and_cell("annealing_temperature", "450to500", "°C", "upper")
    assert cell is None


def test_read_number_carries_the_clean_ends_and_the_written_unit():
    reading = read_number("~450-500 °C")

    assert reading.ends == (450.0, 500.0)
    assert reading.unit == "°C"
    assert read_number("450-500 (600)").ends is None
    assert read_number("1.5e-4 Ω·cm").unit == "Ω.cm"
    assert read_number("1.5e-4 Ω·cm at 300 K").unit is None


@pytest.mark.parametrize("policy", ["lower", "upper"])
def test_an_end_policy_needs_the_fields_unit_check(policy):
    with pytest.raises(ValueError, match="range_unit"):
        parse_number("450-500", range_policy=policy)


# ---- Intervals ------------------------------------------------------------------------------------------------

_INTERVALS = [
    {"name": "power_window", "group": "coating", "kind": "interval", "description": "Power.", "canonical_unit": "W"},
    {"name": "pressure_window", "group": "coating", "kind": "interval", "description": "P.", "canonical_unit": "Pa"},
    {"name": "coverage", "group": "coating", "kind": "interval", "description": "Coverage.", "canonical_unit": "%"},
    {"name": "made_on", "group": "precursor", "kind": "date", "description": "Date made."},
]
PROFILE = make_profile({"fields": [*profile_data()["fields"], *_INTERVALS]})


def _interval(name: str, raw: str, unit_raw: str | None = None, **update: object) -> FieldValue:
    return normalize_field(_field(name, raw, unit_raw, **update), PROFILE.by_name[name], PROFILE.units)


@pytest.mark.parametrize(
    ("name", "raw", "bounds"),
    [
        ("power_window", ">80 mW", (0.08, None)),
        ("power_window", ">80 kW", (80000.0, None)),
        ("power_window", ">80 W", (80.0, None)),
        ("pressure_window", "> 5 kPa", (5000.0, None)),
        ("pressure_window", "< 5 mTorr", (None, 5 * 0.133322)),
    ],
)
def test_a_bounds_unit_is_the_unit_it_writes_not_a_suffix_of_it(name, raw, bounds):
    assert _interval(name, raw).bounds == pytest.approx(bounds)


@pytest.mark.parametrize(
    ("name", "raw", "unit_raw"),
    [
        ("coverage", "> 45 (60)", "%"),
        ("coverage", "< 50 % for 2 h", "%"),
        ("coverage", ">80 % at 550 nm", "%"),
        ("power_window", ">80 V", None),
    ],
)
def test_a_bound_is_held_to_a_ranges_contract(name, raw, unit_raw):
    value = _interval(name, raw, unit_raw)

    assert value.bounds is None
    assert value.normalization_note


def test_a_date_quoted_after_a_bound_is_no_date():
    spec = PROFILE.by_name["made_on"]
    field = _field("made_on", "2021", None, bound="up to")

    assert normalize_field(field, spec, PROFILE.units).iso_date is None
    assert RULES["date"].cell(field, spec, PROFILE.units)[0] is None


@pytest.mark.parametrize(
    ("bounds", "text"),
    [
        ((80.0, None), "≥ 80"),
        ((None, 5.0), "≤ 5"),
        ((2.8, 4.3), "2.8–4.3"),
        ((0.123456789, None), "≥ 0.123456789"),  # never cut to six significant digits
        ((1.5e-7, 2e-7), "1.5e-7–2e-7"),  # as JavaScript's String() writes it, like tsv.js
    ],
)
def test_one_formatter_writes_an_interval(bounds, text):
    assert interval_text(bounds) == text
