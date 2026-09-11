"""Data models for extraction results: what the model must return, and what is stored.

The two layers are deliberately separate, which is how "the model only quotes, the code converts" is
enforced structurally rather than by asking nicely:

- :class:`ExtractionResponse` is the shape the model must produce. It has ``value_raw`` / ``unit_raw`` and
  no ``value`` / ``unit``, so the model has nowhere to put a converted number even if it wants to.
  ``extra="ignore"`` keeps it tolerant of extra keys.
- :class:`LaneExtraction` is what is written to disk (frozen). ``value`` / ``unit`` are filled in by
  :mod:`paperfacts.normalization` at read time, by deterministic code.

:func:`response_to_records` does the conversion and the cleaning (schema filter, non-numeric values,
citation validation) in one pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from paperfacts.extraction.fields import FieldSpec
from paperfacts.models.artifact import Backend

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
    """Top-level JSON the model must return."""

    model_config = ConfigDict(extra="ignore")

    target: ResponseTarget | None = None
    samples: list[ResponseSample] = Field(default_factory=list)


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
    agreement: float = Field(
        default=1.0, ge=0.0, le=1.0, description="fraction of extraction passes that produced this value"
    )
    # Filled in by paperfacts.normalization at read time; always null in the stored file.
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
    passes: int = Field(default=1, ge=1, description="extraction passes that were merged into this result")
    usage: dict[str, int] = Field(default_factory=dict)
    raw_response: str = Field(default="", description="the model's raw JSON, kept as evidence")

    def values(self) -> tuple[FieldValue, ...]:
        """Every field value in the lane, target first."""
        target = self.target.fields if self.target else ()
        return (*target, *(value for sample in self.samples for value in sample.fields))

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
    """What :func:`response_to_records` produced, plus its cleaning log."""

    model_config = ConfigDict(frozen=True)

    target: TargetRecord | None
    samples: tuple[SampleRecord, ...]
    invalid_source_ids: tuple[str, ...]
    dropped: tuple[str, ...]


def response_to_records(
    response: ExtractionResponse, *, known_ids: frozenset[str], known_fields: Mapping[str, FieldSpec]
) -> ExtractedRecords:
    """Convert a validated response into records, cleaning as it goes.

    Three things are removed: fields that are not in the schema, numeric fields whose value contains no
    digit at all ("minimum", "n.a."), and fields filed under the wrong scope -- a paper-level field such as
    the target composition attached to an individual sample, or the reverse. The prompt forbids all three,
    but a prompt is a request, not an enforcement mechanism; the observed failure is a film's dopant
    concentration being reported as the sputtering target's composition.

    Citations naming a block the model was not shown are stripped from the value and collected into
    ``invalid_source_ids``, so the audit is never silent.
    """
    invalid: set[str] = set()
    dropped: list[str] = []

    def keep_ids(ids: list[str]) -> tuple[str, ...]:
        unique = tuple(dict.fromkeys(ids))
        invalid.update(sid for sid in unique if sid not in known_ids)
        return tuple(sid for sid in unique if sid in known_ids)

    def fields_of(items: list[ResponseField], *, sample_level: bool) -> tuple[FieldValue, ...]:
        kept: list[FieldValue] = []
        for item in items:
            spec = known_fields.get(item.field)
            if spec is None:
                dropped.append(f"{item.field}: not in schema")
                continue
            if spec.is_sample_level != sample_level:
                scope = "a sample" if sample_level else "the target"
                dropped.append(f"{item.field}: {spec.group}-level field reported under {scope}")
                continue
            value_raw = item.value_raw.strip()
            if spec.kind == "numeric" and not any(ch.isdigit() for ch in value_raw):
                dropped.append(f"{item.field}: non-numeric value {value_raw!r}")
                continue
            kept.append(
                FieldValue(
                    field=item.field,
                    value_raw=value_raw,
                    unit_raw=_clean(item.unit_raw),
                    condition=_clean(item.condition),
                    source_ids=keep_ids(item.source_ids),
                    note=_clean(item.note),
                )
            )
        return tuple(kept)

    target = None
    if response.target is not None:
        # Validate first: if every field is dropped, invented ids still belong in the audit.
        target_ids = keep_ids(response.target.source_ids)
        fields = fields_of(response.target.fields, sample_level=False)
        if fields:
            target = TargetRecord(source_ids=target_ids, fields=fields)
    samples = tuple(
        SampleRecord(
            sample_id=sample_id,
            label=sample.label.strip(),
            conditions={str(k).strip(): str(v).strip() for k, v in sample.conditions.items()},
            source_ids=keep_ids(sample.source_ids),
            fields=fields_of(sample.fields, sample_level=True),
        )
        for sample in response.samples
        # min_length on the raw field still admits "  "; a sample with no id cannot be matched or reported.
        if (sample_id := sample.sample_id.strip())
    )
    return ExtractedRecords(
        target=target, samples=samples, invalid_source_ids=tuple(sorted(invalid)), dropped=tuple(dropped)
    )


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
