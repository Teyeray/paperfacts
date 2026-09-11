"""Unit conversion: the spellings found in papers -> a field's canonical unit.

A wrong conversion means "the number lines up but is three orders of magnitude off," which is far more
dangerous than an outright parse failure, so every field's conversion table is pinned down rule by rule.
An equally important rule: **an unrecognized unit is never guessed at** — it returns None and lets the
comparison layer judge AMBIGUOUS.
"""

from __future__ import annotations

import pytest

from paperfacts.extraction.fields import FIELD_BY_NAME, FieldSpec
from paperfacts.normalization.units import CONVERTERS, clean_unit, convert_to_canonical


def convert(field: str, value: float, unit_raw: str | None):
    return convert_to_canonical(FIELD_BY_NAME[field], value, unit_raw)


# ---- Sheet resistance Ω/sq ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit",
    ["Ω/sq", "ohm/sq", "ohms/sq", "OHM/SQ", "Ω/□", "Ω sq−1", "Ω/square", "Ω/SQ", "Ω/sq.", "ohms per sq"],
)
def test_every_spelling_of_ohms_per_square_is_the_canonical_unit_itself(unit):
    # "Ω/□", "Ω sq−1", "ohm/sq" are all the same thing; failing to recognize them would make the two lanes
    # judge each other AMBIGUOUS.
    assert convert("sheet_resistance", 12.5, unit) == (12.5, "Ω/sq", None)


@pytest.mark.parametrize(
    ("unit", "expected"),
    [("kΩ/sq", 1200.0), ("KΩ/sq", 1200.0), ("mΩ/sq", 0.0012), ("MΩ/sq", 1_200_000.0), ("nΩ/sq", 1.2e-9)],
)
def test_si_prefixes_on_ohms_per_square_scale_the_value(unit, expected):
    value, unit_out, note = convert("sheet_resistance", 1.2, unit)

    assert value == pytest.approx(expected)
    assert (unit_out, note) == ("Ω/sq", None)


def test_prefix_case_distinguishes_milli_from_mega():
    # "m" and "M" differ by 10⁹; case carries meaning here and must not be casually lower()'d away.
    milli, _, _ = convert("sheet_resistance", 1.0, "mΩ/sq")
    mega, _, _ = convert("sheet_resistance", 1.0, "MΩ/sq")

    assert mega / milli == pytest.approx(1e9)


# ---- Resistivity Ω·cm -------------------------------------------------------------------


@pytest.mark.parametrize("unit", ["Ω·cm", "Ω cm", "ohm cm", "Ω.cm", "Ωcm", "Ωxcm"])
def test_every_spelling_of_ohm_centimetre_is_the_canonical_unit_itself(unit):
    assert convert("resistance", 0.3, unit) == (0.3, "Ω·cm", None)


@pytest.mark.parametrize(("unit", "expected"), [("mΩ·cm", 2e-3), ("μΩ cm", 2e-6), ("µΩ·cm", 2e-6), ("kΩ·cm", 2e3)])
def test_si_prefixes_on_resistivity_scale_the_value(unit, expected):
    value, unit_out, note = convert("resistance", 2.0, unit)

    assert value == pytest.approx(expected)
    assert (unit_out, note) == ("Ω·cm", None)


# ---- Length nm -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "expected"),
    [("nm", 2.0), ("μm", 2e3), ("um", 2e3), ("µm", 2e3), ("mm", 2e6), ("cm", 2e7), ("Å", 0.2), ("angstrom", 0.2)],
)
def test_thickness_units_convert_to_nanometres(unit, expected):
    value, unit_out, note = convert("thickness", 2.0, unit)

    assert value == pytest.approx(expected)
    assert (unit_out, note) == ("nm", None)


@pytest.mark.parametrize("unit", ["A", "a"])
def test_a_bare_letter_a_is_not_silently_read_as_angstrom(unit):
    # "Å" is recognized, but a bare "a" is not: treating an ampere symbol or a stray letter as angstrom
    # would silently throw thickness off by a factor of ten.
    value, unit_out, note = convert("thickness", 2.0, unit)

    assert (value, unit_out) == (None, None)
    assert "unknown unit" in note


# ---- Time min ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "expected"),
    [("min", 30.0), ("mins", 30.0), ("minutes", 30.0), ("h", 1800.0), ("hr", 1800.0), ("hours", 1800.0)],
)
def test_deposition_time_converts_to_minutes(unit, expected):
    assert convert("sputtering_time", 30.0, unit) == (pytest.approx(expected), "min", None)


@pytest.mark.parametrize("unit", ["s", "sec", "seconds"])
def test_seconds_become_a_fraction_of_a_minute(unit):
    value, unit_out, _ = convert("sputtering_time", 90.0, unit)

    assert value == pytest.approx(1.5)
    assert unit_out == "min"


# ---- Target size inch -----------------------------------------------------------------


@pytest.mark.parametrize("unit", ["inch", "inches", "in", '"'])
def test_target_size_units_that_already_are_inches(unit):
    assert convert("inch", 4.0, unit) == (4.0, "inch", None)


@pytest.mark.parametrize(("unit", "value", "expected"), [("mm", 25.4, 1.0), ("cm", 2.54, 1.0)])
def test_metric_target_sizes_convert_to_inches(unit, value, expected):
    result, unit_out, note = convert("inch", value, unit)

    assert result == pytest.approx(expected)
    assert (unit_out, note) == ("inch", None)


# ---- Percentage % ----------------------------------------------------------------------


@pytest.mark.parametrize("unit", ["%", "percent"])
def test_percentages_pass_through(unit):
    assert convert("transmittance", 85.0, unit) == (85.0, "%", None)


# ---- When there is no unit --------------------------------------------------------------


def test_a_percentage_field_reads_a_value_at_most_one_as_a_fraction():
    # A paper writing "T = 0.85" means 85%; <=1 is the only range where that reading can be made with
    # confidence.
    value, unit, note = convert("transmittance", 0.85, None)

    assert (value, unit) == (85.0, "%")
    assert "fraction" in note


def test_a_percentage_field_above_one_is_read_as_a_percentage():
    value, unit, note = convert("transmittance", 85.0, None)

    assert (value, unit) == (85.0, "%")
    assert note == "no unit; read as percent"


def test_relative_density_follows_the_same_percentage_policy():
    assert convert("density", 0.9, None) == (90.0, "%", "no unit; value ≤ 1 read as a fraction")
    assert convert("density", 98.5, None) == (98.5, "%", "no unit; read as percent")


@pytest.mark.parametrize("field", ["thickness", "sheet_resistance", "sputtering_time", "inch", "resistance"])
def test_a_field_whose_policy_is_reject_refuses_a_bare_number(field):
    # Guessing the wrong unit (reading 500 μm as 500 nm) is a silent three-orders-of-magnitude error;
    # better to withhold the value and let the comparison layer judge ambiguous.
    value, unit, note = convert(field, 500.0, None)

    assert (value, unit) == (None, None)
    assert note == f"no unit; {field} requires one"


def test_the_assume_canonical_policy_takes_the_number_at_face_value():
    """The third policy: assume the author used the canonical unit, but always leave a note for
    traceability.

    No field in today's field table uses this policy (they're all reject or percent_or_fraction), so a
    FieldSpec is constructed directly here — the policy itself must still work, so that a future field can
    safely opt into it.
    """
    spec = FieldSpec(
        name="demo",
        group="film",
        kind="numeric",
        description="demo",
        keywords=(),
        canonical_unit="nm",
        bare_number="assume_canonical",
    )

    assert convert_to_canonical(spec, 500.0, None) == (500.0, "nm", "no unit; assumed nm")


def test_the_bare_number_policy_comes_from_the_field_table_not_from_the_field_name():
    # Only the two fields with canonical_unit == "%" use percent_or_fraction; the policy lives on
    # FieldSpec, not on the field name.
    percent_fields = {spec.name for spec in FIELD_BY_NAME.values() if spec.bare_number == "percent_or_fraction"}

    assert percent_fields == {"transmittance", "density"}


# ---- Unrecognized units ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "unit"), [("sheet_resistance", "Ω/cm"), ("thickness", "m"), ("sputtering_time", "ms"), ("inch", "m")]
)
def test_an_unknown_unit_refuses_to_guess(field, unit):
    value, unit_out, note = convert(field, 1.0, unit)

    assert (value, unit_out) == (None, None)
    assert "unknown unit" in note and unit in note


def test_a_field_without_a_canonical_unit_returns_the_value_untouched():
    # component is a chemical-composition text field with no convertible unit; that must not cause the
    # value to be discarded.
    assert convert_to_canonical(FIELD_BY_NAME["component"], 2.0, "wt%") == (2.0, None, None)


# ---- clean_unit --------------------------------------------------------------------


def test_clean_unit_strips_whitespace_and_trailing_dots_without_interpreting():
    assert clean_unit(" Ω / sq. ") == "Ω/sq"
    assert clean_unit("µm") == "μm"


def test_clean_unit_preserves_case_so_milli_and_mega_stay_distinct():
    # The comparison layer uses clean_unit (not normalize_key) to compare units precisely because case
    # must be distinguished here.
    assert clean_unit("mΩ·cm") != clean_unit("MΩ·cm")


def test_every_canonical_unit_in_the_field_table_has_a_converter():
    # A missing converter would surface as a KeyError deep inside normalization; units.py checks this at
    # import time, and this test pins the contract down.
    canonical = {spec.canonical_unit for spec in FIELD_BY_NAME.values() if spec.canonical_unit}

    assert canonical <= CONVERTERS.keys()
