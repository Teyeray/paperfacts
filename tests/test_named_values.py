"""A value a paper states in words ("at RT and 200 °C"), read as the number the profile declares for those words.

coatings-13-01719: the model would not quote "RT" for a numeric field and the cleaning would not keep it, so the
room-temperature films had no substrate temperature. With ``named_values`` the model quotes the words and the
code reads them; the cell says where its number came from.
"""

from __future__ import annotations

import dataclasses

from paperfacts.compare import FieldComparison, compare_values
from paperfacts.decide import decide
from paperfacts.kinds import rules_for
from paperfacts.normalize import normalize_field
from paperfacts.records import NO_CONTEXT, FieldValue
from support.profiles import shipped_profile

TCO = shipped_profile()
SPEC = dataclasses.replace(
    TCO.by_name["substrate_temperature"], named_values=(("room temperature", 25.0), ("RT", 25.0))
)


def quoted(raw, *, unit=None, backend="mineru", bound=None):
    field = FieldValue(field=SPEC.name, value_raw=raw, unit_raw=unit, source_ids=(f"{backend}_p0_b1",), bound=bound)
    return normalize_field(field, SPEC, TCO.units, NO_CONTEXT)


def test_the_cell_reads_the_declared_number_and_says_the_paper_wrote_words():
    value, note = rules_for(SPEC).cell(quoted("RT"), SPEC, TCO.units, NO_CONTEXT)

    assert value == 25.0
    assert note == "原文为文字表述 'RT'，按字段配置读作 25 ℃"


def test_one_lane_quoting_the_words_agrees_with_the_other_quoting_the_number():
    words, number = quoted("RT"), quoted("25", unit="°C", backend="paddleocr_vl")

    status, _ = compare_values(words, number, SPEC, NO_CONTEXT)
    comparison = FieldComparison(scope="sample:A|A", field=SPEC.name, status=status, a=words, b=number)
    result = decide(SPEC, [("mineru", words), ("paddleocr_vl", number)], [comparison], units=TCO.units, ctx=NO_CONTEXT)

    assert status == "agree"
    assert (result.status, result.value) == ("agree", 25.0)


def test_words_under_a_bound_are_no_scalar():
    # "above RT": the bound grounding found before the quote makes it a bound, and a bound fills no cell.
    a = quoted("RT", bound="above")
    b = quoted("RT", bound="above", backend="paddleocr_vl")
    status, _ = compare_values(a, b, SPEC, NO_CONTEXT)
    comparison = FieldComparison(scope="sample:A|A", field=SPEC.name, status=status, a=a, b=b)

    result = decide(SPEC, [("mineru", a), ("paddleocr_vl", b)], [comparison], units=TCO.units, ctx=NO_CONTEXT)

    assert a.value is None
    assert result.status == "non_scalar" and result.value is None
