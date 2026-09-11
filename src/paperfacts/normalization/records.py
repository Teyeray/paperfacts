"""Fill in the canonical value for every FieldValue in a LaneExtraction from its raw text.

Pure and idempotent: always returns a new object.
"""

from __future__ import annotations

from paperfacts.extraction.fields import FIELD_BY_NAME, FieldSpec
from paperfacts.extraction.records import FieldValue, LaneExtraction, TargetRecord
from paperfacts.normalization.numbers import parse_number
from paperfacts.normalization.units import convert_to_canonical


def normalize_field(field: FieldValue, spec: FieldSpec) -> FieldValue:
    if spec.kind != "numeric":
        # Text/composition fields are compared on the fly via normalize_key; no canonical text is cached here
        return field

    number, parse_note = parse_number(field.value_raw)
    if number is None:
        return field.model_copy(update={"value": None, "unit": None, "normalization_note": parse_note})
    value, unit, unit_note = convert_to_canonical(spec, number, field.unit_raw)
    note = "; ".join(n for n in (parse_note, unit_note) if n) or None
    return field.model_copy(update={"value": value, "unit": unit, "normalization_note": note})


def _normalize_fields(fields: tuple[FieldValue, ...]) -> tuple[FieldValue, ...]:
    # Fields outside the schema were already dropped at extraction time; this is a defensive second check
    return tuple(normalize_field(f, FIELD_BY_NAME[f.field]) if f.field in FIELD_BY_NAME else f for f in fields)


def normalize_lane(lane: LaneExtraction) -> LaneExtraction:
    target: TargetRecord | None = None
    if lane.target is not None:
        target = lane.target.model_copy(update={"fields": _normalize_fields(lane.target.fields)})
    samples = tuple(sample.model_copy(update={"fields": _normalize_fields(sample.fields)}) for sample in lane.samples)
    return lane.model_copy(update={"target": target, "samples": samples})
