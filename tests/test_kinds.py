"""The kind rows: every kind has one, and a cell is formatted by its column's kind and cardinality."""

from __future__ import annotations

import typing

import pytest

from paperfacts.columns import FieldColumn, field_columns
from paperfacts.fields import DIGIT_KINDS, FieldKind
from paperfacts.kinds import RULES, NumericRules, TextRules, rules_for
from paperfacts.profile import DomainProfile
from paperfacts.workbook import format_cell


def test_every_kind_has_a_row():
    assert set(RULES) == set(typing.get_args(FieldKind))


def test_every_digit_kind_is_read_as_a_number():
    # records.py and passages.py sit below kinds.py and read DIGIT_KINDS; the rows must agree with it.
    assert {kind for kind, rules in RULES.items() if isinstance(rules, NumericRules)} == DIGIT_KINDS


def test_a_field_is_dispatched_by_its_kind(tco_profile: DomainProfile):
    for spec in tco_profile.fields:
        assert isinstance(rules_for(spec), NumericRules if spec.kind == "numeric" else TextRules)


def test_no_kind_adds_a_note_to_a_field_line_yet(tco_profile: DomainProfile):
    # The field line's {note} renders "" for every kind so far, which keeps the TCO prompts byte-identical.
    assert {rules_for(spec).note(spec) for spec in tco_profile.fields} == {""}


def test_every_column_carries_its_field_kind(tco_profile: DomainProfile):
    columns = field_columns(tco_profile)

    assert [(column.kind, column.cardinality) for column in columns] == [
        (spec.kind, "one") for spec in tco_profile.fields
    ]


def _column(kind: FieldKind, cardinality: typing.Literal["one", "many"] = "one") -> FieldColumn:
    return FieldColumn(name="f", scope="sample", kind=kind, cardinality=cardinality)


@pytest.mark.parametrize(
    ("column", "value", "written"),
    [
        (_column("numeric"), 12.5, 12.5),
        (_column("numeric"), None, None),
        (_column("text"), "DC+RF", "DC+RF"),
        (_column("composition"), "In2O3:Sn", "In2O3:Sn"),
        (_column("text", "many"), ["XRD", "XPS"], "XRD; XPS"),
        (_column("composition", "many"), ["LiOH", "NiSO4"], "LiOH; NiSO4"),
        (_column("text", "many"), None, None),
    ],
)
def test_a_cell_is_written_as_its_column_says(column: FieldColumn, value, written):
    assert format_cell(value, column) == written
