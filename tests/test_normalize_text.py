"""Text normalization: unify Unicode variants, fold sub/superscripts back to ASCII, build the key used for
equality comparisons.

Every bug in this layer is **silent**: an exponent losing its minus sign, Ω's two code points not being
equal, μ's two code points not being equal — none of these raise an exception, they just make two facts
that should AGREE get judged CONFLICT instead. So every rule gets its own pinned test case.
"""

from __future__ import annotations

import unicodedata

import pytest

from paperfacts.normalize import normalize_key, normalize_text

# ---- Superscripts: must be handled before NFKC ------------------------------------------------------


def test_superscript_digits_become_a_caret_exponent():
    # Core invariant: NFKC would fold "10⁻⁴" into "10−4" (read as subtraction), losing the exponent meaning.
    assert normalize_text("10⁻⁴") == "10^-4"
    assert unicodedata.normalize("NFKC", "10⁻⁴") == "10−4"  # document exactly what we're guarding against
    assert "^" not in unicodedata.normalize("NFKC", "10⁻⁴")


def test_scientific_notation_keeps_its_mantissa_and_exponent():
    assert normalize_text("1.2 × 10⁻⁴") == "1.2 x 10^-4"


def test_a_positive_superscript_sign_collapses_to_a_bare_exponent():
    # ⁺ carries no information (an exponent defaults to positive), so only ^ and the digits are kept.
    assert normalize_text("10⁺³") == "10^3"


def test_superscripts_without_a_sign_still_get_a_caret():
    assert normalize_text("cm²") == "cm^2"


def test_html_superscript_tags_are_rewritten_the_same_way_as_unicode():
    # MinerU's Markdown writes sub/superscripts with <sup>/<sub>, and both spellings must converge to the
    # same result.
    assert normalize_text("10<sup>-4</sup>") == normalize_text("10⁻⁴") == "10^-4"


def test_html_subscript_tags_drop_to_plain_digits():
    assert normalize_text("SnO<sub>2</sub>:Ta") == normalize_text("SnO₂:Ta") == "SnO2:Ta"


def test_subscript_digits_become_plain_digits():
    assert normalize_text("H₂O") == "H2O"


# ---- Unifying Unicode variants ---------------------------------------------------------


@pytest.mark.parametrize("dash", ["−", "–", "—", "‐", "‑", "‒", "―"])
def test_every_dash_variant_becomes_an_ascii_hyphen(dash):
    # The minus sign / en dash / em dash all show up in papers; without unifying them, "−5" would fail to
    # parse its negative sign. The hyphen family (U+2010 and friends) is what the two parsers disagree
    # about when they transcribe the same sample label.
    assert normalize_text(f"{dash}5") == "-5"


def test_multiplication_sign_becomes_an_ascii_x():
    assert normalize_text("1.2 × 10") == "1.2 x 10"


def test_the_ohm_sign_and_the_greek_omega_collapse_to_one_character():
    # U+2126 OHM SIGN and U+03A9 GREEK CAPITAL OMEGA look identical; without unifying them, "Ω/sq" and
    # "Ω/sq" would not compare equal.
    ohm_sign, greek_omega = "Ω", "Ω"
    assert ohm_sign != greek_omega
    assert normalize_text(f"{ohm_sign}/sq") == normalize_text(f"{greek_omega}/sq") == "Ω/sq"


def test_the_micro_sign_and_the_greek_mu_collapse_to_one_character():
    micro_sign, greek_mu = "µ", "μ"
    assert micro_sign != greek_mu
    assert normalize_text(f"{micro_sign}m") == normalize_text(f"{greek_mu}m") == "μm"


@pytest.mark.parametrize("dot", ["⋅", "·"])
def test_dot_operators_become_a_period(dot):
    assert normalize_text(f"Ω{dot}cm") == "Ω.cm"


def test_a_typographic_apostrophe_becomes_an_ascii_one():
    assert normalize_text("films’ properties") == "films' properties"


# ---- Whitespace --------------------------------------------------------------------


def test_runs_of_whitespace_collapse_to_a_single_space_and_edges_are_trimmed():
    assert normalize_text("  a   b\t\nc  ") == "a b c"


def test_a_non_breaking_space_counts_as_whitespace():
    assert normalize_text("12 Ω") == "12 Ω"


def test_an_empty_string_stays_empty():
    assert normalize_text("") == ""


# ---- normalize_key ------------------------------------------------------------------


def test_normalize_key_lowercases_and_drops_decorative_punctuation():
    assert normalize_key("Sample A#1") == "samplea1"


def test_normalize_key_keeps_the_characters_that_carry_meaning():
    # Digits, letters, μ, % and ./:+- are the criteria for "is this the same thing"; they can't be
    # stripped as mere decoration.
    assert normalize_key("O2-100 sccm") == "o2-100sccm"
    assert normalize_key("1.2 μm") == "1.2μm"
    assert normalize_key("85%") == "85%"


def test_normalize_key_keeps_the_ohm_through_lowercasing():
    # .lower() turns Ω into ω, but the whitelist only allows uppercase Ω; without converting it back, the
    # ohm sign would be stripped out entirely.
    assert normalize_key("Ω") == "Ω"
    assert normalize_key("Ω·cm") == "Ω.cm"
    assert normalize_key("Ω·cm") != normalize_key("·cm")


def test_normalize_key_must_not_be_used_to_compare_units():
    """Document a design constraint: lowercasing would collide "mΩ" (milli) with "MΩ" (mega, 10⁹ apart),
    so unit comparison uses units.clean_unit instead.

    This invariant is relied on by :func:`compare.compare_values`, which compares units via ``_unit_key``
    (case-sensitive).
    """
    assert normalize_key("mΩ") == normalize_key("MΩ")


def test_normalize_key_runs_the_full_text_normalization_first():
    # The two spellings of a sample name must match, or exact pairing degrades into LLM pairing.
    assert normalize_key("Sample ⁻4") == normalize_key("Sample -4")


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_normalize_key_maps_missing_text_to_an_empty_string(blank):
    # A field with a blank condition must key to the same value as a field with no condition at all.
    assert normalize_key(blank) == ""


@pytest.mark.parametrize("dash", ["‐", "‑", "‒", "–", "—", "―", "−"])
def test_normalize_key_keys_every_hyphen_variant_of_a_sample_id_identically(dash):
    # The two lanes' parsers emit different Unicode for the same label, and the hyphen is where they
    # differ most; keying them apart would split one sample into two that are never compared.
    assert normalize_key(f"Sample{dash}A") == normalize_key("Sample-A") == "sample-a"


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("O\u2082-100", "O2-100"),  # subscript digit
        ("O\u00b2-100", "O2-100"),  # superscript digit
        ("\uff33\uff41\uff4d\uff50\uff4c\uff45\uff11", "Sample1"),  # full-width letters and digit
        ("sample a", "SAMPLE\tA"),  # whitespace and case
    ],
)
def test_normalize_key_folds_the_spellings_the_two_parsers_disagree_about(a, b):
    assert normalize_key(a) == normalize_key(b)


def test_normalize_key_is_idempotent():
    once = normalize_key("550 nm (average)")

    assert normalize_key(once) == once
