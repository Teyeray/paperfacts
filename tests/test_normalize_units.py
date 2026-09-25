"""Unit conversion: the spellings found in papers -> a field's canonical unit.

A wrong conversion means "the number lines up but is three orders of magnitude off," which is far more
dangerous than an outright parse failure, so every field's conversion table is pinned down rule by rule.
An equally important rule: **an unrecognized unit is never guessed at** — it returns None and lets the
comparison layer judge AMBIGUOUS.
"""

from __future__ import annotations

import pytest

from paperfacts.fields import FIELD_BY_NAME, FieldSpec
from paperfacts.normalize import CONVERTERS, clean_unit, convert_to_canonical, parse_number, split_scale_factor


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


@pytest.mark.parametrize(
    "unit",
    [
        "Ω·cm",
        "Ω cm",
        "ohm cm",
        "Ω.cm",
        "Ωcm",
        "Ωxcm",
        "Ω-cm",
        "ohm-cm",
        "Ω•cm",
        "Ω∙cm",
        # MinerU's LaTeX, formatting commands and all.
        "\\Omega { \\cdot } \\mathrm { c m }",
    ],
)
def test_every_spelling_of_ohm_centimetre_is_the_canonical_unit_itself(unit):
    assert convert("resistance", 0.3, unit) == (0.3, "Ω·cm", None)


@pytest.mark.parametrize(
    ("unit", "expected"),
    [("mΩ·cm", 2e-3), ("μΩ cm", 2e-6), ("µΩ·cm", 2e-6), ("kΩ·cm", 2e3), ("mΩ-cm", 2e-3)],
)
def test_si_prefixes_on_resistivity_scale_the_value(unit, expected):
    value, unit_out, note = convert("resistance", 2.0, unit)

    assert value == pytest.approx(expected)
    assert (unit_out, note) == ("Ω·cm", None)


def test_a_hyphenated_resistivity_with_a_trailing_period_is_recognised():
    # Sentence-final "Ω-cm." keeps its period; clean_unit strips it before the recogniser sees it.
    assert convert("resistance", 0.3, "Ω-cm.") == (0.3, "Ω·cm", None)


def test_a_resistivity_per_centimetre_is_not_a_resistivity():
    # "Ω/cm" is a different quantity; recognising it would silently corrupt every comparison.
    value, unit_out, note = convert("resistance", 0.3, "Ω/cm")

    assert value is None
    assert unit_out is None
    assert "unknown unit" in note


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
    assert convert("density", 0.9, None) == (90.0, "%", "no unit; value < 1 read as a fraction")
    assert convert("density", 98.5, None) == (98.5, "%", "no unit; read as percent")


@pytest.mark.parametrize("field", ["o2_ratio", "h2_ratio", "transmittance", "density"])
def test_a_bare_one_is_one_percent_not_the_whole(field):
    # "1" is far more often 1 % (1 % O2 in Ar) than a fraction of exactly one; 100 % would turn a trace
    # admixture into the whole gas.
    assert convert(field, 1.0, None) == (1.0, "%", "no unit; read as percent")


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
    # Only the fields with canonical_unit == "%" use percent_or_fraction; the policy lives on FieldSpec,
    # not on the field name.
    percent_fields = {spec.name for spec in FIELD_BY_NAME.values() if spec.bare_number == "percent_or_fraction"}

    assert percent_fields == {"transmittance", "density", "o2_ratio", "h2_ratio"}


# ---- Unrecognized units ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "unit"), [("sheet_resistance", "Ω/cm"), ("thickness", "m"), ("sputtering_time", "ms"), ("inch", "m")]
)
def test_an_unknown_unit_refuses_to_guess(field, unit):
    value, unit_out, note = convert(field, 1.0, unit)

    assert (value, unit_out) == (None, None)
    assert "unknown unit" in note and unit in note


@pytest.mark.parametrize(("unit", "expected"), [("Pa Ar", 1.1), ("Pa (Ar)", 1.1), ("mTorr O2", 0.1466542)])
def test_a_gas_named_after_the_unit_is_set_aside(unit, expected):
    value, unit_out, note = convert("working_pressure", 1.1, unit)

    assert value == pytest.approx(expected) and unit_out == "Pa"
    assert "gas name" in note


@pytest.mark.parametrize("unit", ["Ar", "sccm Ar"])
def test_a_gas_name_does_not_make_an_unknown_unit_known(unit):
    assert convert("working_pressure", 1.1, unit)[:2] == (None, None)


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


# ---- The process-field units: ℃, cm, W, sccm, rpm -------------------------------------------------


@pytest.mark.parametrize("unit", ["℃", "°C", "°c"])
def test_celsius_spellings_all_convert_to_the_canonical_degree(unit):
    # NFKC folds ℃ (U+2103) to "°C", so one table key has to catch the sign, the folded form and a
    # lowercase typo.
    assert convert("substrate_temperature", 250.0, unit) == (250.0, "℃", None)


def test_kelvin_is_refused_rather_than_converted():
    # K → ℃ needs an offset, not a factor; the converter interface is a factor, so an honest ambiguous
    # beats a confidently wrong number.
    value, unit_out, note = convert("substrate_temperature", 523.0, "K")

    assert (value, unit_out) == (None, None)
    assert "unknown unit" in note


@pytest.mark.parametrize(
    ("unit", "expected"), [("cm", 5.0), ("mm", 0.5), ("m", 500.0), ("μm", 5e-4), ("inch", 12.7), ('"', 12.7)]
)
def test_chamber_distances_convert_to_centimetres(unit, expected):
    assert convert("target_substrate_distance", 5.0, unit) == (pytest.approx(expected), "cm", None)


def test_nanometres_are_not_a_chamber_distance(unit=None):
    # Admitting nm would misread every film thickness in the paper as a candidate target-substrate gap.
    value, unit_out, _ = convert("target_substrate_distance", 5.0, "nm")

    assert (value, unit_out) == (None, None)


@pytest.mark.parametrize(("unit", "expected"), [("W", 150.0), ("kW", 150000.0), ("mW", 0.15), ("MW", 1.5e8)])
def test_power_prefixes_stay_case_sensitive(unit, expected):
    # mW and MW differ by nine orders of magnitude; a lowercasing table would fold them together.
    assert convert("sputtering_power", 150.0, unit) == (pytest.approx(expected), "W", None)


@pytest.mark.parametrize("unit", ["sccm", "cm3/min"])
def test_flow_rates_convert_to_sccm(unit):
    # cm³/min at standard conditions *is* the definition of sccm.
    assert convert("ar_flow_rate", 30.0, unit) == (30.0, "sccm", None)


def test_slm_is_refused_rather_than_scaled():
    # slm is a thousand sccm only at exactly the standard condition it names; converting it silently
    # would invent precision the paper did not state.
    value, unit_out, _ = convert("ar_flow_rate", 30.0, "slm")

    assert (value, unit_out) == (None, None)


@pytest.mark.parametrize("unit", ["rpm", "r/min", "rev/min"])
def test_rotation_speed_spellings_convert_to_rpm(unit):
    assert convert("rotation_speed", 20.0, unit) == (20.0, "rpm", None)


# ---- A scale factor written into the unit -------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (19.4, "×10^-4 Ω-cm", 19.4e-4),
        (0.88, "×10^-3 Ω·cm", 0.88e-3),
        (1.0, "×10⁻⁴ Ω·cm", 1e-4),
        (1.0, r"\times 10^{-4} \Omega cm", 1e-4),
        (1.0, "x10-4 Ω·cm", 1e-4),
        (1.0, "10^-4 Ω·cm", 1e-4),
    ],
)
def test_a_power_of_ten_in_the_unit_is_applied_to_the_value(value, unit, expected):
    # A table column headed "ρ (×10⁻⁴ Ω·cm)" leaves the model transcribing the factor as part of the unit.
    # Refusing it as an unknown unit loses the fact; the factor belongs to the value.
    number, canonical, note = convert("resistivity", value, unit)

    assert number == pytest.approx(expected)
    assert canonical == "Ω·cm"
    assert "scale factor" in note


def test_a_unit_that_merely_starts_with_digits_is_not_read_as_a_factor():
    # Only an explicit "x10" or a caret makes a factor; "cm3/min" must stay the flow unit it is.
    assert split_scale_factor("cm3/min", CONVERTERS["sccm"]) == (1.0, "cm3/min")
    assert convert("ar_flow_rate", 30.0, "cm3/min") == (30.0, "sccm", None)


def test_a_unit_that_is_only_a_factor_falls_back_to_the_bare_number_rule():
    # Nothing is left to name the unit, so the field's own bare-number rule decides. resistivity rejects a
    # unitless value rather than assuming Ω·cm, and the factor is still recorded in the note.
    number, canonical, note = convert("resistivity", 19.4, "×10^-4")

    assert (number, canonical) == (None, None)
    assert note == "scale factor 0.0001 taken from the header in the unit; no unit; resistivity requires one"

    # A field with a bare-number rule of its own keeps the scaled value.
    assert convert("transmittance", 0.85, "x10^2")[:2] == (pytest.approx(85.0), "%")


# ---- Aliases seen in the corpus -----------------------------------------------------------------


@pytest.mark.parametrize("unit", ["Ω·sq^-1", "Ω·sq⁻¹", "Ω/□", "Ω/sq."])
def test_dotted_and_boxed_sheet_resistance_spellings_are_recognised(unit):
    assert convert("sheet_resistance", 10.0, unit) == (10.0, "Ω/sq", None)


def test_ohms_per_litre_is_ocr_damage_and_stays_unknown():
    # "Ω/L" is a misread "Ω/□"; guessing it would invent a sheet resistance the paper never stated.
    value, unit_out, note = convert("sheet_resistance", 10.0, "Ω/L")

    assert (value, unit_out) == (None, None)
    assert "unknown unit" in note


@pytest.mark.parametrize("unit", ['"', "''", "″", "′′"])
def test_every_double_prime_spelling_means_inches(unit):
    assert convert("inch", 4.0, unit) == (4.0, "inch", None)
    assert convert("target_substrate_distance", 4.0, unit) == (pytest.approx(10.16), "cm", None)


def test_a_power_of_ten_in_both_the_value_and_the_unit_is_refused():
    # "1.2 × 10⁻⁴" under a column headed "(×10⁻⁴ Ω·cm)" is either 1.2e-4 or 1.2e-8, depending on whether the
    # author applied the header to the cell. Multiplying twice would manufacture a value, so neither lane
    # gets one and the comparison layer judges it AMBIGUOUS.
    spec = FIELD_BY_NAME["resistivity"]
    number, _ = parse_number("1.2 × 10^-4")

    assert convert_to_canonical(spec, number, "×10^-4 Ω·cm", value_text="1.2 × 10^-4") == (
        None,
        None,
        "scale factor in both value and unit; ambiguous",
    )
    # Only one of the two carrying a factor stays unambiguous, whichever one it is.
    assert convert_to_canonical(spec, number, "Ω·cm", value_text="1.2 × 10^-4") == (pytest.approx(1.2e-4), "Ω·cm", None)
    assert convert_to_canonical(spec, 19.4, "×10^-4 Ω-cm", value_text="19.4")[0] == pytest.approx(1.94e-3)


# ---- A power of ten in a table header: on the quantity or on the unit ---------------------------------------
# "ρ × 10^4 (Ω cm)" heads a column of ρ multiplied by 10^4: a cell of 6.8 is 6.8 × 10^-4 Ω·cm (Guillén 2006).
# "ρ (10^-4 Ω cm)" heads a column in units of 10^-4 Ω·cm: a cell of 6.8 is again 6.8 × 10^-4 Ω·cm. The same
# exponent sign means opposite things, so which one the header wrote decides the value: the factor belongs to
# the quantity only when the unit follows it in brackets of its own.


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        # On the quantity, the unit bracketed apart: the cell is divided.
        ("ρ × 10^4 (Ω cm)", 6.8e-4),  # Guillén 2006, Table 1
        ("ρ×10⁴ (Ω·cm)", 6.8e-4),
        (r"$\rho \times 10^{4}$ ($\Omega$ cm)", 6.8e-4),
        ("ρ × 10^4 [Ω cm]", 6.8e-4),
        # Leading the unit, inside its bracket or unbracketed: the cell is multiplied.
        ("ρ (10^-4 Ω cm)", 6.8e-4),
        ("ρ (×10^-4 Ω cm)", 6.8e-4),
        ("(10^-4 Ω cm)", 6.8e-4),
        ("(×10^-4 Ω cm)", 6.8e-4),
        ("×10^-4 Ω·cm", 6.8e-4),
        ("ρ × 10^-4 Ω·cm", 6.8e-4),
        ("ρ ×10^-4 Ωcm", 6.8e-4),
        ("ρ×10⁻⁴ Ω·cm", 6.8e-4),
        (r"\rho \times 10^{-4} \Omega cm", 6.8e-4),
    ],
)
def test_a_header_factor_is_applied_by_the_convention_it_was_written_in(unit, expected):
    number, canonical, note = convert("resistivity", 6.8, unit)

    assert number == pytest.approx(expected)
    assert canonical == "Ω·cm"
    assert "scale factor" in note


@pytest.mark.parametrize(
    ("field", "unit", "expected"),
    [
        ("thickness", "d ×10^2 nm", 680.0),
        ("working_pressure", "P ×10^-1 Pa", 0.68),
        ("sheet_resistance", "R_s × 10^2 (Ω/sq)", 0.068),
    ],
)
def test_the_header_convention_holds_for_every_field(field, unit, expected):
    assert convert(field, 6.8, unit)[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    "unit",
    [
        # A unit before the factor, or a factor glued to the symbol with nothing saying how.
        "Ω·cm × 10^-4",
        "ρ 10^4 Ω cm",
        "ρ, 10^4 (Ω cm)",
        # A factor bracketed alone beside the symbol: "the column is ×10^-4" or "ρ is multiplied by it"?
        "ρ (×10^-4) (Ω cm)",
        "Resistivity (×10^-4) (Ω·cm)",
        "ρ (10^-4) Ω cm",
        "ρ (10^4) (Ω cm)",
        # On the quantity with no unit to say so.
        "ρ × 10^4",
        # A negative power on the quantity: formally ρ × 10^-4, but usually meant as the unit's multiplier.
        "Resistivity ×10^-4 (Ω cm)",
        "R_s × 10^-2 (Ω/sq)",
    ],
)
def test_a_header_factor_whose_convention_cannot_be_told_is_refused(unit):
    # Dividing and multiplying are eight orders of magnitude apart, and both lanes would read the header the same
    # way, so a wrong guess would show as agreement. Neither is guessed.
    number, canonical, note = convert("resistivity", 6.8, unit)

    assert (number, canonical) == (None, None)
    assert "ambiguous" in note


def test_a_quantity_factor_and_a_value_with_its_own_power_of_ten_are_still_refused():
    spec = FIELD_BY_NAME["resistivity"]

    assert convert_to_canonical(spec, 6.8e-4, "ρ × 10^4 (Ω cm)", value_text="6.8 × 10^-4")[0] is None


@pytest.mark.parametrize("unit", ["10mm", "10 mm"])
def test_a_leading_number_without_a_caret_or_x10_is_not_a_factor(unit):
    # "10mm" is a length someone wrote into the unit column, not a scale factor; reading it as 10^10 would
    # be catastrophic and silent.
    assert split_scale_factor(unit, CONVERTERS["nm"]) == (1.0, "10mm")


# ---- Working pressure Pa ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "factor"),
    [("Pa", 1.0), ("kPa", 1e3), ("mPa", 1e-3), ("MPa", 1e6), ("mbar", 100.0), ("Torr", 133.322), ("mTorr", 0.133322)],
)
def test_a_working_pressure_converts_to_pascal_with_milli_and_mega_kept_apart(unit, factor):
    value, canonical, note = convert("working_pressure", 2.0, unit)

    assert (canonical, note) == ("Pa", None)
    assert value == pytest.approx(2.0 * factor)
