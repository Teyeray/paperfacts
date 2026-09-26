"""Three data-quality faults found in production answers (gold set B0), each reproduced with the strings the
lanes actually produced."""

from __future__ import annotations

import pytest

from paperfacts.compare import compare_values
from paperfacts.grounding import ground_lane, quoted_bound
from paperfacts.matching import SampleMatch, SampleMatching
from paperfacts.normalize import canonical_category, normalize_field, read_value, same_text
from paperfacts.records import NO_CONTEXT, FieldValue, PaperRecord
from support.extraction import make_lane, make_sample
from support.profiles import shipped_profile
from test_dataset import FIELD_BY_NAME, dataset, decision, paired, value

MODE = FIELD_BY_NAME["mode"]
COMPONENT = FIELD_BY_NAME["component"]
TRANSMITTANCE = FIELD_BY_NAME["transmittance"]
PROFILE = shipped_profile()


# ---- 1. Guillén (08562126…): MinerU dropped the hyphen of "rf-magnetron" -----------------------------------


def test_a_text_value_that_lost_its_hyphen_is_the_same_mode():
    mineru, paddle = "rfmagnetron sputtering", "rf-magnetron sputtering"

    assert same_text(MODE, mineru, paddle)
    assert compare_values(value("mode", mineru), value("mode", paddle), MODE, NO_CONTEXT) == ("agree", "both name RF")


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("DC", "RF"),
        ("DC magnetron sputtering", "RF magnetron sputtering"),
        ("DC and RF", "DC"),
        ("rfmagnetron", "dcmagnetron"),
    ],
)
def test_distinct_modes_still_conflict(a, b):
    assert not same_text(MODE, a, b)
    assert compare_values(value("mode", a), value("mode", b), MODE, NO_CONTEXT)[0] == "conflict"


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("3 wt.% Ga2O3", "3 wt% Ga2O3", True),
        ("ZnO-Ga2O3", "ZnO Ga2O3", True),
        ("10-20", "1020", False),  # a hyphen before a digit is a range or a sign
        ("1.5 wt%", "15 wt%", False),  # a period before a digit is a decimal point
        ("-5", "5", False),
    ],
)
def test_loose_text_equality_keeps_every_number_intact(a, b, same):
    assert same_text(COMPONENT, a, b) is same


def test_the_guillen_mode_cell_agrees_and_states_the_spelling_that_names_the_category():
    result = paired(
        [value("mode", "rfmagnetron sputtering")],
        [value("mode", "rf-magnetron sputtering", backend="paddleocr_vl")],
    )

    row = decision(result, "mode")
    assert row["decision"] == "agree"
    # The eval scores a category field through canonical_category, which the hyphen-less twin does not name.
    assert row["value"] == "rf-magnetron sputtering"
    assert canonical_category(MODE.categories, row["value"]) == "RF"


# ---- 2. GZO (534040e6…): "90" quoted out of "above 90 %" -----------------------------------------------------

# Synthetic stand-in for paddleocr_vl_p6_b2 (the real block is not in the fixtures): the sentence shape the
# lanes quoted from.
GZO_BLOCK = (
    "The average transmittance of all the films in the visible region (400 nm to 800 nm) is above 90 %, which "
    "indicates that the hydrogen annealing does not degrade the optical properties."
)


def _cited(raw: str, unit: str | None = "%") -> FieldValue:
    return FieldValue(field="transmittance", value_raw=raw, unit_raw=unit, source_ids=("b",))


@pytest.mark.parametrize(
    ("raw", "block", "bound"),
    [
        ("90", GZO_BLOCK, "above"),
        ("95%", "HN450 achieved the best transmittance (over 95%) in the visible spectrum", "over"),
        ("80", "a transmittance of more than 80 % for all films", "more than"),
        ("80", "transmittance greater than 80%", "greater than"),
        ("1", "a resistivity lower than 1 × 10^-3 Ω cm", "lower than"),
        ("90", "T >90 % in the visible", ">"),
        ("90", "T $\\geq 90 \\%$ in the visible", "≥"),
        ("90", "T ≥ 90 %", "≥"),
        ("10", "roughness &lt; 10 nm", "<"),
        ("5", "a sheet resistance of at most 5 Ω/sq", "at most"),
        ("100", "films up to 100 nm thick", "up to"),
        # No bound:
        ("above 90 %", GZO_BLOCK, None),  # the quote carries its own bound
        ("90", "T ~90 % in the visible", None),  # approximate is still the value
        ("0.5", "films sputtered under 0.5 Pa", None),  # "under" is a condition
        ("90", "moreover 90 % of the light", None),  # "over" inside a word
        ("90", "a transmittance above 90.5 %; 90 % for the thinnest film", None),  # 90.5 is no occurrence of 90
        ("90", "\\left( 90 \\right)", None),  # \left is not \le
        ("90", "transmittance of 90 %", None),
    ],
)
def test_the_bound_right_before_a_quote_in_its_block(raw, block, bound):
    assert quoted_bound(_cited(raw), {"b": block}) == bound


def test_a_long_block_is_searched_for_a_bound_in_linear_time():
    import time

    # Every occurrence used to rescan the block from its start: 20 000 occurrences took seconds.
    block = "90 % " * 20_000 + "and above      90 % at last"
    started = time.perf_counter()
    assert quoted_bound(_cited("90"), {"b": block}) == "above"
    assert quoted_bound(_cited("90"), {"b": "90 % " * 20_000}) is None
    assert time.perf_counter() - started < 1.0


def test_the_bound_window_finds_what_the_whole_block_search_found():
    import random

    from paperfacts.grounding import _BOUND_BEFORE, _bound_before

    pieces = ["above", "at most", "up to", ">", "≤", " ", "  ", "x", "moreover", "90", "\n"]
    rng = random.Random(7)
    for _ in range(3000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 12)))
        for start in range(len(text) + 1):
            whole = _BOUND_BEFORE.search(text, 0, start)
            window = _bound_before(text, start)
            assert (whole and whole.group(1)) == (window and window.group(1)), (text, start)


def test_any_bounded_occurrence_makes_the_quote_a_bound():
    # Which occurrence the model copied is unknown; a bound read as a scalar is a wrong number, a refused one a blank.
    block = "transmittance above 90 % (400-800 nm), and 90 % at 550 nm"
    assert quoted_bound(_cited("90"), {"b": block}) == "above"


def test_a_bound_before_the_quote_is_read_exactly_like_a_quoted_bound(tco_profile):
    field = _cited("90").model_copy(update={"bound": "above"})

    reading = read_value(field, TRANSMITTANCE, tco_profile.units)
    normalized = normalize_field(field, TRANSMITTANCE, tco_profile.units, NO_CONTEXT)
    quoted = normalize_field(_cited("above 90"), TRANSMITTANCE, tco_profile.units, NO_CONTEXT)

    assert reading.text == "above 90"
    assert (normalized.value, normalized.unit) == (quoted.value, quoted.unit) == (90.0, "%")
    assert "bound 'above' stands before the quote in its cited block" in normalized.normalization_note


def test_the_gzo_transmittance_fills_no_cell_when_one_lane_quoted_the_number_out_of_a_bound():
    mineru = value("transmittance", "above 90 %", condition="400 nm to 800 nm")
    paddle = value("transmittance", "90", "%", condition="400 nm to 800 nm", backend="paddleocr_vl")
    lanes = [
        ground_lane(make_lane(samples=[make_sample("A", [mineru])]), {"mineru_p0_b1": GZO_BLOCK}, profile=PROFILE),
        ground_lane(
            make_lane(backend="paddleocr_vl", samples=[make_sample("A", [paddle])]),
            {"paddleocr_vl_p0_b1": GZO_BLOCK},
            profile=PROFILE,
        ),
    ]
    assert lanes[1].samples[0].fields[0].bound == "above"
    assert lanes[0].samples[0].fields[0].bound is None

    matching = SampleMatching(pairs=(SampleMatch(a_id="A", b_id="A", confidence=1.0, method="llm", justification="t"),))
    row = decision(dataset(*lanes, matching), "transmittance")

    assert (row["value"], row["decision"]) == (None, "non_scalar")


def test_without_the_bound_the_same_quote_still_fills_the_cell():
    # Control: the paddle quote alone, cited in a block that states 90 % plainly, is a scalar.
    block = "The average transmittance in the visible region (400 nm to 800 nm) is 90 %."
    paddle = value("transmittance", "90", "%", condition="400 nm to 800 nm", backend="paddleocr_vl")
    lane = ground_lane(
        make_lane(backend="paddleocr_vl", samples=[make_sample("A", [paddle])]),
        {"paddleocr_vl_p0_b1": block},
        profile=PROFILE,
    )
    row = decision(dataset(make_lane(), lane), "transmittance")

    assert (row["value"], row["decision"]) == (90.0, "single_source")


# ---- 3. GZO target component: "3 wt.% wt.%" --------------------------------------------------------------------


def test_the_gzo_component_audit_does_not_repeat_a_unit_the_quote_already_ends_with():
    result = dataset(
        make_lane(
            paper=PaperRecord(fields=(value("component", "3 wt.%", "wt.%"), value("component", "97 wt.%", "wt.%")))
        ),
        make_lane(
            backend="paddleocr_vl",
            paper=PaperRecord(
                fields=(
                    value(
                        "component",
                        "3 wt.% gallium oxide (purity 99.95%) and 97 wt.% zinc oxide (purity 99.95%)",
                        "wt.%",
                        backend="paddleocr_vl",
                    ),
                )
            ),
        ),
    )
    row = decision(result, "component")

    assert "wt.% wt.%" not in row["detail"]
    assert "mineru: 3 wt.%" in row["detail"]
    # Still refused: MinerU quoted two bare percentages with no compound named, the other lane one sentence
    # naming both oxides. They are not the same text, and no rule may make them so.
    assert row["value"] is None
    assert row["decision"] == "conflict"
