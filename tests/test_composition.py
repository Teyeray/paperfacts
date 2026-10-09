"""The composition display: formulas, the two quote shapes, each conversion, and what travels with a column.

Expected numbers are worked by hand from the IUPAC atomic weights, never read back from the module.
"""

from __future__ import annotations

import pytest

from paperfacts.columns import CompositionReading, FieldColumn
from paperfacts.composition import (
    atoms_of,
    convert,
    molar_mass,
    parse_composition,
    readings,
    sentence,
    with_readings,
)

CATIONS = "cations"


def amounts(quote: str) -> list[tuple[str, float]]:
    parsed = parse_composition(quote)
    assert parsed is not None, quote
    return [(component.formula, pytest.approx(component.amount)) for component in parsed.components]


def shown(quote: str, unit: str) -> str:
    parsed = parse_composition(quote)
    assert parsed is not None, quote
    parts = convert(parsed, unit, CATIONS)
    assert parts is not None, quote
    return sentence(unit, parts)


# ---- formulas ----------------------------------------------------------------------------------------------


def test_a_formula_is_read_into_its_atoms_brackets_and_hydrates_included():
    assert atoms_of("In2O3") == {"In": 2, "O": 3}
    assert atoms_of("Ca(OH)2") == {"Ca": 1, "O": 2, "H": 2}
    assert atoms_of("Mg3(PO4)2") == {"Mg": 3, "P": 2, "O": 8}
    assert atoms_of("CuSO4·5H2O") == {"Cu": 1, "S": 1, "O": 9, "H": 10}
    assert atoms_of("In1.9Sn0.1O3") == {"In": pytest.approx(1.9), "Sn": pytest.approx(0.1), "O": 3}


NOT_FORMULAS = ["ITO", "AZO", "target", "Ce-doped", "X2O3", "2In2O3", "(In2O3", "In2O3)", "In 2O3", ""]


@pytest.mark.parametrize("text", NOT_FORMULAS)
def test_an_acronym_a_word_or_a_broken_formula_is_not_a_formula(text):
    assert atoms_of(text) is None


def test_the_molar_mass_is_the_sum_of_the_atomic_weights():
    assert molar_mass(atoms_of("In2O3") or {}) == pytest.approx(277.637, abs=0.01)
    assert molar_mass(atoms_of("SnO2") or {}) == pytest.approx(150.708, abs=0.01)


# ---- reading a quote ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quote",
    [
        "In2O3:SnO2 = 90:10 wt%",
        "In2O3:SnO2 90:10 wt.%",
        "In2O3/SnO2 (90:10 wt %)",
        "In2O3:SnO2 target, 90/10 weight%",
        "  In2O3 : SnO2   90 : 10 Wt%  ",
    ],
)
def test_a_ratio_with_the_unit_after_it_is_read_whatever_the_separators_and_spelling(quote):
    parsed = parse_composition(quote)
    assert parsed is not None and parsed.unit == "wt%"
    assert amounts(quote) == [("In2O3", 90), ("SnO2", 10)]


def test_a_ratio_that_is_not_out_of_100_is_normalised_to_percentages():
    assert amounts("ZnO:Al2O3 49:1 wt%") == [("ZnO", 98), ("Al2O3", 2)]
    parsed = parse_composition("ZnO:Al2O3 49:1 wt%")
    assert parsed is not None and parsed.components[0].stated is None


def test_terms_with_their_own_amounts_are_read_and_one_amount_less_formula_takes_the_balance():
    assert amounts("90 wt% In2O3 and 10 wt% ZnO") == [("In2O3", 90), ("ZnO", 10)]
    assert amounts("2 wt% Ta2O5-doped SnO2") == [("Ta2O5", 2), ("SnO2", 98)]
    after = parse_composition("SnO2 with 2.5 at% Ta")
    assert after is not None and after.unit == "at%"
    assert amounts("SnO2 with 2.5 at% Ta") == [("Ta", 2.5), ("SnO2", 97.5)]


@pytest.mark.parametrize(
    "quote",
    [
        "ITO 90:10 wt%",  # an acronym, not formulas
        "Ce-doped In2O3",  # no amounts
        "SnO2:Ta (2 wt% Ta2O5)",  # two amount-less formulas for one balance
        "In2O3:SnO2 90:10",  # no unit
        "In2O3:SnO2 90:10 wt% and 5 at% F",  # two units
        "In2O3:SnO2 90:10 wt% at 5 Pa",  # a number that is not part of the mix
        "In2O3:SnO2:ZnO 90:10 wt%",  # counts differ
        "90 wt% In2O3 and 20 wt% ZnO",  # does not add up
        "",  # nothing
    ],
)
def test_a_quote_that_does_not_state_one_mix_in_one_unit_is_not_read(quote):
    assert parse_composition(quote) is None


# ---- converting --------------------------------------------------------------------------------------------


def test_weight_percent_of_oxides_becomes_the_atomic_percent_of_their_metals():
    # In2O3 277.637 g/mol, SnO2 150.708 g/mol: 90/277.637 = 0.32417 mol (0.64834 mol In), 10/150.708 = 0.066354 mol Sn.
    assert shown("In2O3:SnO2 = 90:10 wt%", "at%") == "90.7 at% In and 9.3 at% Sn"
    # ZnO 81.379, Al2O3 101.961: 1.20424 mol Zn, 0.03923 mol Al.
    assert shown("ZnO:Al2O3 98:2 wt%", "at%") == "96.8 at% Zn and 3.2 at% Al"


def test_atomic_percent_of_elements_becomes_weight_percent():
    # Sn 118.71, Ta 180.95: 95 × 118.71 = 11277.45, 5 × 180.95 = 904.75.
    assert shown("Sn/Ta 95:5 at.%", "wt%") == "92.6 wt% Sn and 7.4 wt% Ta"


def test_an_atomic_percent_is_the_metals_share_whether_stated_or_converted_and_converts_back_as_such():
    # Stated for oxides, at% counts their cations, so the labels are the metals as after a conversion.
    assert shown("In2O3:SnO2 = 90:10 at%", "at%") == "90 at% In and 10 at% Sn"
    # 90.7 at% In over In2O3 is 45.35 mol In2O3 against 9.3 mol SnO2: 12590.8 g against 1401.6 g.
    assert shown("In2O3:SnO2 = 90.7:9.3 at%", "wt%") == "90 wt% In2O3 and 10 wt% SnO2"


def test_a_mix_stated_in_the_shown_unit_keeps_the_paper_s_digits():
    assert shown("In2O3:SnO2 = 90.0:10.0 wt%", "wt%") == "90.0 wt% In2O3 and 10.0 wt% SnO2"


def test_mole_percent_converts_either_way():
    # Equal moles of In2O3 (277.637) and SnO2 (150.708): 64.8 wt% In2O3; 2 In per 1 Sn: 66.7 at% In.
    assert shown("In2O3:SnO2 50:50 mol%", "wt%") == "64.8 wt% In2O3 and 35.2 wt% SnO2"
    assert shown("In2O3:SnO2 50:50 mol%", "at%") == "66.7 at% In and 33.3 at% Sn"


def test_three_components_read_as_a_list():
    assert shown("In2O3:Ga2O3:ZnO 1:1:1 mol%", "mol%") == "33.3 mol% In2O3, 33.3 mol% Ga2O3 and 33.3 mol% ZnO"


def test_a_component_with_no_counted_atom_cannot_be_given_an_atomic_percent():
    parsed = parse_composition("H2O:O2 50:50 wt%")
    assert parsed is not None and convert(parsed, "at%", CATIONS) is None
    parsed = parse_composition("In2O3:SnO2 90:10 wt%")
    assert parsed is not None and convert(parsed, "at%", None) is None


# ---- what the page is handed -------------------------------------------------------------------------------


def test_the_readings_of_a_quote_say_which_numbers_the_paper_did_not_print():
    assert readings("In2O3:SnO2 = 90:10 wt%", CATIONS) == {
        "wt%": CompositionReading(text="90 wt% In2O3 and 10 wt% SnO2", computed=False),
        "at%": CompositionReading(text="90.7 at% In and 9.3 at% Sn", computed=True),
    }
    # An inferred balance and a normalised ratio are computed numbers even in the paper's own unit.
    balance = readings("2 wt% Ta2O5-doped SnO2", CATIONS)
    assert balance is not None
    assert balance["wt%"] == CompositionReading(text="2 wt% Ta2O5 and 98 wt% SnO2", computed=True)
    ratio = readings("ZnO:Al2O3 49:1 wt%", CATIONS)
    assert ratio is not None and ratio["wt%"].computed
    assert readings("Ce-doped In2O3 (ICO)", CATIONS) is None


def test_a_switchable_column_carries_the_readings_of_the_quotes_its_rows_hold_and_no_other_column_changes():
    mix = FieldColumn(name="mix", scope="paper", kind="composition", cardinality="many", atomic_basis=CATIONS)
    plain = FieldColumn(name="solvent", scope="sample", kind="text")
    rows = [
        {"mix": ["In2O3:SnO2 = 90:10 wt%", "Ce-doped In2O3"], "solvent": "ethanol"},
        {"mix": "Sn/Ta 95:5 at.%", "solvent": "In2O3:SnO2 = 90:10 wt%"},
        {"mix": None},
    ]

    mix_out, plain_out = with_readings((mix, plain), rows)

    assert plain_out == plain
    assert set(mix_out.compositions) == {"In2O3:SnO2 = 90:10 wt%", "Sn/Ta 95:5 at.%"}  # the unreadable quote has none
    assert mix_out.compositions["Sn/Ta 95:5 at.%"]["wt%"].text == "92.6 wt% Sn and 7.4 wt% Ta"
    assert with_readings((plain,), rows) == (plain,)
