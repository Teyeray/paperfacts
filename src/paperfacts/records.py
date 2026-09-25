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
:mod:`paperfacts.extract`. Both clean each value through the same :class:`ResponseCleaning`, build their
sample list through :func:`clean_samples` and place a whole-series value through
:func:`place_on_every_sample`, so neither mode can quietly become more permissive than the other.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, create_model

from paperfacts.fields import FieldSpec
from paperfacts.models import Backend
from paperfacts.storage import write_text_atomic

logger = logging.getLogger(__name__)

# ---- Response models: field names are exactly the JSON keys the prompt asks for ----------------


class ResponseField(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field: str
    value_raw: str = Field(min_length=1, description="verbatim as printed in the paper")
    unit_raw: str | None = None
    condition: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    note: str | None = None
    applies_to_all_samples: bool = Field(
        default=False, description="a sample-level value under the target that the paper states for every sample"
    )


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


@dataclass(frozen=True)
class ResponseModels:
    """The two answer shapes whose keys a profile names: document mode's and the inventory's."""

    extraction: type[ExtractionResponse]
    inventory: type[InventoryResponse]


@cache
def response_models(paper_key: str, no_samples_key: str) -> ResponseModels:
    """The answer shapes for a profile that tells the model to emit ``paper_key`` and ``no_samples_key``.

    The keys are validation aliases onto the unchanged attributes ``target`` and ``no_tco_film``, so every
    stored record keeps its shape. The class names are the base classes' own: a rejected answer goes back to
    the model as ``str(ValidationError)``, which names the class, and under keys equal to the attribute names
    the text is byte-identical to the base class's, as is every request that carries it.
    """
    extraction = create_model(
        ExtractionResponse.__name__,
        __base__=ExtractionResponse,
        __doc__=ExtractionResponse.__doc__,
        target=(ResponseTarget | None, Field(default=None, validation_alias=paper_key)),
    )
    inventory = create_model(
        InventoryResponse.__name__,
        __base__=InventoryResponse,
        __doc__=InventoryResponse.__doc__,
        no_tco_film=(
            bool,
            Field(
                default=False,
                validation_alias=no_samples_key,
                description=InventoryResponse.model_fields["no_tco_film"].description,
            ),
        ),
    )
    return ResponseModels(extraction=extraction, inventory=inventory)


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


# ---- Sample identity ----------------------------------------------------------------------------
# A sample id is a name, not a value, and the letters in it carry identity: "α-ITO" and "β-ITO" are two
# films, "ITO-a" and "ITO-A" two samples of a paper that uses both. normalize_key's whitelist deletes the
# Greek letter and lower-cases the suffix, which merged such pairs and put the second one's values on the
# first. So ids have their own rule, and it lives here, beside the records it identifies:
#   - tokens are runs of letters (any script: Greek, CJK), numbers (with a decimal point), a sign that starts
#     the id or follows "=" ("T=-5"), and "%" or "+";
#   - everything else -- spaces, hyphens, underscores, brackets, HTML tags -- is dropped, and the tokens are
#     joined with nothing between them: "ITO-1", "ITO 1", "ITO_1" are one id, and so are "WOx" and MinerU's
#     subscripted "WO_x". Only two numbers keep a boundary ("1-2" is not "12");
#   - letters are case-folded ("Sample", "SAMPLE", "S1" vs "s1"), except a trailing single-letter suffix,
#     which keeps its case: that is where papers distinguish samples by case ("ITO-a" vs "ITO-A");
#   - a LaTeX Greek command is the letter it typesets (\\varepsilon too): MinerU writes "$\\alpha$-ITO" where
#     PaddleOCR-VL reads "α-ITO", and the two lanes must key the one sample alike.
_SAMPLE_TOKEN = re.compile(r"[^\W\d_]+|\d+(?:\.\d+)?|(?:(?<==)|^)\s*[-−](?=\d)|[%+]")
_LATEX_LETTER = re.compile(r"\\(?:var)?([A-Za-z]+)")
_HTML_TAG = re.compile(r"<[^>]*>")


def _greek(match: re.Match[str]) -> str:
    name = match.group(1)
    case = "CAPITAL" if name[0].isupper() else "SMALL"
    try:
        return unicodedata.lookup(f"GREEK {case} LETTER {name.upper()}")
    except KeyError:
        return " "


def sample_key(sample_id: str | None) -> str:
    """The key two spellings of one sample id share, and two different samples never do. Used wherever
    samples are keyed -- inventory de-duplication, value attribution, the vote across passes and the exact
    pairing across lanes -- so both lanes, both modes and every pass apply the same rule."""
    if not sample_id:
        return ""
    text = unicodedata.normalize("NFKC", _LATEX_LETTER.sub(_greek, _HTML_TAG.sub(" ", sample_id)))
    tokens = [token.strip() for token in _SAMPLE_TOKEN.findall(text)]
    suffix = len(tokens) > 1 and len(tokens[-1]) == 1 and tokens[-1].isalpha()
    last = len(tokens) - 1
    folded = [token if suffix and index == last else token.lower() for index, token in enumerate(tokens)]
    key = ""
    for index, token in enumerate(folded):
        if index and token[0].isdigit() and folded[index - 1][-1].isdigit():
            key += " "
        key += token.replace("−", "-")
    return key


class FailedQuestion(BaseModel):
    """A field question the model gave no valid answer to (invalid twice, or cut off at max_tokens).

    It costs that field in that lane, not the lane. Its invalid answers were never cached, so extracting the
    lane again re-asks exactly this question while every other one replays from the LLM cache.
    """

    model_config = ConfigDict(frozen=True)

    field: str
    detail: str


class LaneExtraction(BaseModel):
    """Everything one parser lane yielded, with its provenance audit and its cost."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    backend: Backend
    extractor_key: str
    model: str
    # Older files also carry ``schema_version``, which held this same fingerprint. It is no longer written, and an
    # unknown key is ignored on reading, so such a file still loads.
    profile_fingerprint: str | None = Field(
        default=None,
        description="keys.profile_extraction_fingerprint of the profile the lane was extracted under; None in "
        "files written before profiles existed",
    )
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
    no_tco_film: bool = Field(
        default=False,
        description="the inventory found no sample because the paper deposits no TCO film of its own, so no "
        "sample-level question was asked; an empty lane for any other reason leaves it False",
    )
    passes: int = Field(default=1, ge=1, description="extraction passes that were merged into this result")
    failed_questions: tuple[FailedQuestion, ...] = Field(
        default=(),
        description="field questions with no valid answer in some pass; while any is here the lane is incomplete: "
        "it is extracted again on the next run and neither its comparison nor its table is stored",
    )
    usage: dict[str, int] = Field(default_factory=dict)
    raw_response: str = Field(default="", description="the model's raw JSON, kept as evidence")
    artifact_sha256: str | None = Field(
        default=None,
        description="ParsedArtifact.content_hash of the parse this lane was extracted from; None in files "
        "written before it was recorded, which count as unknown rather than as a mismatch",
    )

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
        write_text_atomic(path, self.model_dump_json(indent=2))

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


# English number words a value may be written in: "a four-inch target", "two targets". One to twelve only:
# beyond that papers write digits, and "a dozen" is a round figure, not a count. Defined here rather than in
# normalize.py because the cleaning below must let such a value through, and normalize imports this module.
NUMBER_WORDS = {
    word: index
    for index, word in enumerate(
        ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"), 1
    )
}
# The word, then optionally its unit after a hyphen or a space ("four-inch", "four inch"); nothing else.
_NUMBER_WORD = re.compile(rf"^(?P<word>{'|'.join(NUMBER_WORDS)})(?:(?:-|\s+)(?P<rest>\S.*))?$", re.IGNORECASE)


def _unit_fold(unit: str) -> str:
    return unicodedata.normalize("NFKC", unit).replace(" ", "").rstrip(".").casefold()


def spell_number_word(value_raw: str, unit_raw: str | None) -> str:
    """``value_raw`` as digits when it is an English number word standing for the value: "four" or
    "four-inch" with ``unit_raw`` "inch" -> "4". Anything else comes back unchanged.

    The word must be the whole value, or be followed only by the unit the value was quoted with, and the value
    must carry a unit. "one of the samples", "five to ten", "one-third", "ten-fold", "two-step" and "one order
    of magnitude" are words that contain a number, not values; reading them would put a made-up number into the
    comparison, which is worse than the drop they get.
    """
    stripped = value_raw.strip()
    match = _NUMBER_WORD.match(stripped)
    if match is None or not unit_raw or not unit_raw.strip():
        return value_raw
    rest = match.group("rest")
    if rest is not None and _unit_fold(rest) != _unit_fold(unit_raw):
        return value_raw
    return str(NUMBER_WORDS[match.group("word").lower()])


class ResponseCleaning:
    """The audit kept while a model's answer is turned into records, shared by both extraction modes.

    Two things are recorded rather than silently discarded: ids the model cited that it was never shown, and
    values dropped for being impossible (a numeric field whose value has no digit and no number word in it).
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
    ) -> FieldValue | None:
        """One cleaned value, or None when it cannot be one (the reason lands in ``dropped``)."""
        text = value_raw.strip()
        if spec.kind == "numeric" and not any(character.isdigit() for character in spell_number_word(text, unit_raw)):
            self.dropped.append(f"{spec.name}: non-numeric value {text!r}")
            return None
        return FieldValue(
            field=spec.name,
            value_raw=text,
            unit_raw=_clean(unit_raw),
            condition=_clean(condition),
            source_ids=self.keep_ids(source_ids, known_ids),
            note=_clean(note),
        )


class ListedSample(Protocol):
    """A sample as either mode's answer lists it: the inventory's entry, or document mode's sample."""

    sample_id: str
    label: str
    conditions: dict[str, str]
    source_ids: list[str]


def clean_samples(
    listed: Sequence[ListedSample], cleaning: ResponseCleaning, known_ids: frozenset[str]
) -> tuple[list[SampleRecord], list[int | None]]:
    """The sample list both modes build their records on, and where each listed entry went.

    Returns the records (without fields) and, per listed entry, the index of the record it became: a repeat
    of an id already listed points at the first one, and an entry with no usable id points nowhere (None).
    Both cases are audited in ``cleaning.dropped``. Ids are compared by :func:`sample_key`, the key that
    also pairs the lanes, so the two modes and the two lanes agree on what "the same sample" is.
    """
    records: list[SampleRecord] = []
    placement: list[int | None] = []
    index_by_key: dict[str, int] = {}
    for item in listed:
        sample_id = item.sample_id.strip()
        key = sample_key(sample_id)
        # min_length on the response model still admits "  " or "#": such a sample can be neither matched
        # nor reported.
        if not key:
            cleaning.dropped.append("inventory: a sample was listed with no usable id")
            placement.append(None)
            continue
        if key in index_by_key:
            # One sample listed twice under one id: everything filed under that id belongs to the first.
            cleaning.dropped.append(f"inventory: sample {sample_id!r} repeats an id already listed")
            placement.append(index_by_key[key])
            continue
        index_by_key[key] = len(records)
        placement.append(len(records))
        records.append(
            SampleRecord(
                sample_id=sample_id,
                label=item.label.strip(),
                conditions={str(name).strip(): str(value).strip() for name, value in item.conditions.items()},
                source_ids=cleaning.keep_ids(item.source_ids, known_ids),
            )
        )
    return records, placement


def place_on_every_sample(
    value: FieldValue, sample_fields: Sequence[list[FieldValue]], unattributed: list[FieldValue]
) -> bool:
    """Place a value the paper states for the whole series ("all films were RF sputtered").

    It is written onto every sample with ``series=True`` -- a value the paper placed on all of them at once,
    explicitly flagged by the model, never inferred by code. With no sample to carry it the flag would claim
    a placement the record does not have, so it is kept unattributed instead. Returns whether it fanned out.
    """
    if not sample_fields:
        unattributed.append(value.model_copy(update={"series": False}))
        return False
    series = value.model_copy(update={"series": True})
    for fields in sample_fields:
        fields.append(series)
    return True


def response_to_records(
    response: ExtractionResponse, *, fields: Mapping[str, FieldSpec], known_ids: frozenset[str]
) -> ExtractedRecords:
    """Convert a validated response into records, cleaning as it goes.

    Three things are removed: fields that are not in the schema, numeric fields whose value contains no
    digit at all ("minimum", "n.a."), and fields filed under the wrong scope -- a paper-level field such as
    the target composition attached to an individual sample, or the reverse. The prompt forbids all three,
    but a prompt is a request, not an enforcement mechanism; the observed failure is a film's dopant
    concentration being reported as the sputtering target's composition.

    The one sample-level value allowed under the target is one flagged ``applies_to_all_samples``: the paper
    states it once for the whole series, and it is written onto every sample exactly as passage mode does.
    A sample listed without a usable id keeps its values, unattributed; a sample listed twice has its
    values filed under the first listing. Both are audited, as in passage mode.

    Citations naming a block the model was not shown are stripped from the value and collected into
    ``invalid_source_ids``, so the audit is never silent.
    """
    cleaning = ResponseCleaning()
    samples, placement = clean_samples(response.samples, cleaning, known_ids)
    sample_fields: list[list[FieldValue]] = [[] for _ in samples]
    unattributed: list[FieldValue] = []

    def cleaned(item: ResponseField, spec: FieldSpec) -> FieldValue | None:
        return cleaning.value(
            spec,
            value_raw=item.value_raw,
            unit_raw=item.unit_raw,
            condition=item.condition,
            source_ids=item.source_ids,
            note=item.note,
            known_ids=known_ids,
        )

    def in_schema(item: ResponseField) -> FieldSpec | None:
        spec = fields.get(item.field)
        if spec is None:
            cleaning.dropped.append(f"{item.field}: not in schema")
        return spec

    target = None
    if response.target is not None:
        # Validate first: if every field is dropped, invented ids still belong in the audit.
        target_ids = cleaning.keep_ids(response.target.source_ids, known_ids)
        target_fields: list[FieldValue] = []
        for item in response.target.fields:
            spec = in_schema(item)
            if spec is None:
                continue
            if spec.is_sample_level and not item.applies_to_all_samples:
                cleaning.dropped.append(f"{item.field}: {spec.group}-level field reported under the target")
                continue
            value = cleaned(item, spec)
            if value is None:
                continue
            if spec.is_sample_level:
                place_on_every_sample(value, sample_fields, unattributed)
            else:
                target_fields.append(value)
        if target_fields:
            target = TargetRecord(source_ids=target_ids, fields=tuple(target_fields))

    for listed, index in zip(response.samples, placement, strict=True):
        for item in listed.fields:
            spec = in_schema(item)
            if spec is None:
                continue
            if not spec.is_sample_level:
                cleaning.dropped.append(f"{item.field}: {spec.group}-level field reported under a sample")
                continue
            value = cleaned(item, spec)
            if value is None:
                continue
            # A value filed under a sample the list could not keep has no owner to be compared under.
            (unattributed if index is None else sample_fields[index]).append(value)

    return ExtractedRecords(
        target=target,
        samples=tuple(
            sample.model_copy(update={"fields": tuple(values)})
            for sample, values in zip(samples, sample_fields, strict=True)
        ),
        invalid_source_ids=tuple(sorted(cleaning.invalid)),
        dropped=tuple(cleaning.dropped),
        unattributed=tuple(unattributed),
    )


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
