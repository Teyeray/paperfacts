"""Data models for extraction results: what the model must return, and what is stored.

The two layers are deliberately separate, which is how "the model only quotes, the code converts" is
enforced structurally rather than by asking nicely:

- :class:`ExtractionResponse` is the shape the model must produce. It has ``value_raw`` / ``unit_raw`` and
  no ``value`` / ``unit``, so the model has nowhere to put a converted number even if it wants to.
  ``extra="ignore"`` keeps it tolerant of extra keys.
- :class:`LaneExtraction` is what is written to disk (frozen). ``value`` / ``unit`` are filled in by
  :mod:`paperfacts.normalize` at read time, by deterministic code.

:func:`response_to_records` does the conversion and the cleaning (schema filter, non-numeric values,
citation validation) in one pass for document mode; passage mode assembles its records in
:mod:`paperfacts.extract` but cleans each value through the same :class:`ResponseCleaning`, so neither mode
can quietly become more permissive than the other.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from paperfacts.fields import FIELD_BY_NAME, FieldSpec
from paperfacts.models import Backend

# ---- Response models: field names are exactly the JSON keys the prompt asks for ----------------


class ResponseField(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field: str
    value_raw: str = Field(min_length=1, description="verbatim as printed in the paper")
    unit_raw: str | None = None
    condition: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    note: str | None = None


class ResponseSample(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sample_id: str = Field(min_length=1)
    label: str = ""
    conditions: dict[str, str] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)
    fields: list[ResponseField] = Field(default_factory=list)


class ResponseTarget(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_ids: list[str] = Field(default_factory=list)
    fields: list[ResponseField] = Field(default_factory=list)


class ExtractionResponse(BaseModel):
    """Top-level JSON the model must return in document mode."""

    model_config = ConfigDict(extra="ignore")

    target: ResponseTarget | None = None
    samples: list[ResponseSample] = Field(default_factory=list)


# ---- Passage mode: one answer about the samples, then one answer per field ----------------------


class InventorySample(BaseModel):
    """A sample as the inventory question describes it: identity and conditions, deliberately no values."""

    model_config = ConfigDict(extra="ignore")

    sample_id: str = Field(min_length=1)
    label: str = ""
    conditions: dict[str, str] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)


class InventoryResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    samples: list[InventorySample] = Field(default_factory=list)
    no_tco_film: bool = Field(
        default=False,
        description="the paper deposits no TCO film of its own; an empty list for any other reason says nothing",
    )


class ResponseValue(BaseModel):
    """One value of the field that was asked about.

    There is no ``field`` key: the question named the field, so unlike document mode the model has no
    opportunity to file a value under a name that is not in the schema.
    """

    model_config = ConfigDict(extra="ignore")

    sample_id: str | None = None
    value_raw: str = Field(min_length=1, description="verbatim as printed in the paper")
    unit_raw: str | None = None
    condition: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    note: str | None = None
    applies_to_all_samples: bool = Field(
        default=False, description="the excerpt states this value holds for every sample in the list"
    )


class FieldResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    values: list[ResponseValue] = Field(default_factory=list)


# ---- Stored records --------------------------------------------------------------------------


class FieldValue(BaseModel):
    """One value of one field: what the paper says, what it means, and where it came from."""

    model_config = ConfigDict(frozen=True)

    field: str
    value_raw: str
    unit_raw: str | None = None
    condition: str | None = None
    source_ids: tuple[str, ...] = ()
    note: str | None = None
    grounded: bool = Field(default=True, description="value_raw was located in the text of one of its cited blocks")
    series: bool = Field(
        default=False,
        description="the paper stated this for the whole sample series; attached to this sample by fan-out",
    )
    agreement: float = Field(
        default=1.0, ge=0.0, le=1.0, description="fraction of extraction passes that produced this value"
    )
    # Filled in by paperfacts.normalize at read time; always null in the stored file.
    value: float | None = None
    unit: str | None = None
    normalization_note: str | None = None


class TargetRecord(BaseModel):
    """The sputtering target, which belongs to the paper rather than to any one sample."""

    model_config = ConfigDict(frozen=True)

    source_ids: tuple[str, ...] = ()
    fields: tuple[FieldValue, ...] = ()

    def get(self, name: str) -> FieldValue | None:
        return next((f for f in self.fields if f.field == name), None)


class SampleRecord(BaseModel):
    """One film sample: the deposition conditions that identify it, and what was measured on it."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    label: str = ""
    conditions: dict[str, str] = Field(default_factory=dict)
    source_ids: tuple[str, ...] = ()
    fields: tuple[FieldValue, ...] = ()

    def get(self, name: str) -> FieldValue | None:
        return next((f for f in self.fields if f.field == name), None)


class LaneExtraction(BaseModel):
    """Everything one parser lane yielded, with its provenance audit and its cost."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    backend: Backend
    extractor_key: str
    model: str
    schema_version: str
    target: TargetRecord | None = None
    samples: tuple[SampleRecord, ...] = ()
    invalid_source_ids: tuple[str, ...] = Field(
        default=(), description="ids the model cited that do not exist in the artifact (removed from fields)"
    )
    dropped: tuple[str, ...] = Field(default=(), description="values removed while cleaning, with the reason")
    unattributed: tuple[FieldValue, ...] = Field(
        default=(),
        description="sample-level values the model could not place on a sample: kept and grounded, never compared",
    )
    passes: int = Field(default=1, ge=1, description="extraction passes that were merged into this result")
    usage: dict[str, int] = Field(default_factory=dict)
    raw_response: str = Field(default="", description="the model's raw JSON, kept as evidence")

    def values(self) -> tuple[FieldValue, ...]:
        """Every field value in the lane, target first, unplaced ones last.

        Unattributed values are included because they were extracted and grounded like any other; leaving
        them out here would understate what the lane found and would hide an ungrounded one.
        """
        target = self.target.fields if self.target else ()
        return (*target, *(value for sample in self.samples for value in sample.fields), *self.unattributed)

    def ungrounded(self) -> tuple[FieldValue, ...]:
        """Values that could not be located in the block they cite."""
        return tuple(value for value in self.values() if not value.grounded)

    def sample(self, sample_id: str) -> SampleRecord | None:
        return next((s for s in self.samples if s.sample_id == sample_id), None)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class ExtractedRecords(BaseModel):
    """What one extraction produced, plus its cleaning log."""

    model_config = ConfigDict(frozen=True)

    target: TargetRecord | None
    samples: tuple[SampleRecord, ...]
    invalid_source_ids: tuple[str, ...]
    dropped: tuple[str, ...]
    unattributed: tuple[FieldValue, ...] = Field(
        default=(), description="sample-level values the model could not place on a sample; kept, never compared"
    )


class ResponseCleaning:
    """The audit kept while a model's answer is turned into records, shared by both extraction modes.

    Two things are recorded rather than silently discarded: ids the model cited that it was never shown, and
    values dropped for being impossible (a numeric field whose value has no digit in it).
    """

    def __init__(self) -> None:
        self.invalid: set[str] = set()
        self.dropped: list[str] = []

    def keep_ids(self, ids: Sequence[str], known: frozenset[str]) -> tuple[str, ...]:
        """Citations the model was actually shown, in order, deduplicated; the rest go to the audit."""
        unique = tuple(dict.fromkeys(ids))
        self.invalid.update(source_id for source_id in unique if source_id not in known)
        return tuple(source_id for source_id in unique if source_id in known)

    def value(
        self,
        spec: FieldSpec,
        *,
        value_raw: str,
        unit_raw: str | None,
        condition: str | None,
        source_ids: Sequence[str],
        note: str | None,
        known_ids: frozenset[str],
        series: bool = False,
    ) -> FieldValue | None:
        """One cleaned value, or None when it cannot be one (the reason lands in ``dropped``)."""
        text = value_raw.strip()
        if spec.kind == "numeric" and not any(character.isdigit() for character in text):
            self.dropped.append(f"{spec.name}: non-numeric value {text!r}")
            return None
        return FieldValue(
            field=spec.name,
            value_raw=text,
            unit_raw=_clean(unit_raw),
            condition=_clean(condition),
            source_ids=self.keep_ids(source_ids, known_ids),
            note=_clean(note),
            series=series,
        )


def response_to_records(response: ExtractionResponse, *, known_ids: frozenset[str]) -> ExtractedRecords:
    """Convert a validated response into records, cleaning as it goes.

    Three things are removed: fields that are not in the schema, numeric fields whose value contains no
    digit at all ("minimum", "n.a."), and fields filed under the wrong scope -- a paper-level field such as
    the target composition attached to an individual sample, or the reverse. The prompt forbids all three,
    but a prompt is a request, not an enforcement mechanism; the observed failure is a film's dopant
    concentration being reported as the sputtering target's composition.

    Citations naming a block the model was not shown are stripped from the value and collected into
    ``invalid_source_ids``, so the audit is never silent.
    """
    cleaning = ResponseCleaning()

    def fields_of(items: list[ResponseField], *, sample_level: bool) -> tuple[FieldValue, ...]:
        kept: list[FieldValue] = []
        for item in items:
            spec = FIELD_BY_NAME.get(item.field)
            if spec is None:
                cleaning.dropped.append(f"{item.field}: not in schema")
                continue
            if spec.is_sample_level != sample_level:
                scope = "a sample" if sample_level else "the target"
                cleaning.dropped.append(f"{item.field}: {spec.group}-level field reported under {scope}")
                continue
            value = cleaning.value(
                spec,
                value_raw=item.value_raw,
                unit_raw=item.unit_raw,
                condition=item.condition,
                source_ids=item.source_ids,
                note=item.note,
                known_ids=known_ids,
            )
            if value is not None:
                kept.append(value)
        return tuple(kept)

    target = None
    if response.target is not None:
        # Validate first: if every field is dropped, invented ids still belong in the audit.
        target_ids = cleaning.keep_ids(response.target.source_ids, known_ids)
        fields = fields_of(response.target.fields, sample_level=False)
        if fields:
            target = TargetRecord(source_ids=target_ids, fields=fields)
    samples = tuple(
        SampleRecord(
            sample_id=sample_id,
            label=sample.label.strip(),
            conditions={str(k).strip(): str(v).strip() for k, v in sample.conditions.items()},
            source_ids=cleaning.keep_ids(sample.source_ids, known_ids),
            fields=fields_of(sample.fields, sample_level=True),
        )
        for sample in response.samples
        # min_length on the raw field still admits "  "; a sample with no id cannot be matched or reported.
        if (sample_id := sample.sample_id.strip())
    )
    return ExtractedRecords(
        target=target,
        samples=samples,
        invalid_source_ids=tuple(sorted(cleaning.invalid)),
        dropped=tuple(cleaning.dropped),
    )


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
