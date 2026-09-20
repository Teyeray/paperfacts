"""The target field table: what to extract, in what unit, and how close counts as the same.

The table itself lives in ``config.json`` and is read at import. Changing a field is therefore an edit to a
JSON file rather than to Python -- which is the point, since which facts a group wants out of its papers is
the thing that differs between groups. Every entry is validated on the way in: an unknown key or a
misspelled group is a :class:`ConfigError` naming the field, not a surprise three stages later.

Fields fall into three groups: the sputtering target (paper-level -- a paper usually has one), the
deposition process (sample-level), and film characterisation (sample-level).

One table drives five things: the field descriptions given to the model, unit conversion, what to do with
a bare number that has no unit, the numeric tolerance used when comparing the two lanes, and -- through
``keywords`` -- which blocks passage-mode retrieval puts in front of the model. Its contents are hashed
into the cache keys (:mod:`paperfacts.keys`), so changing any cell invalidates exactly the caches that
depended on it.

``keywords`` are the names a paper uses for the field, not its units: units are recognised separately by
:mod:`paperfacts.passages`, which knows which of them are specific enough to identify a field on their own.
Keywords are matched as whole tokens, case-insensitively, after Unicode folding.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Literal, get_args

from paperfacts.config import ConfigDocument, configuration
from paperfacts.errors import ConfigError

FieldGroup = Literal["target", "process", "film"]
# numeric: a number with a unit; composition: a chemical formula; text: anything else
FieldKind = Literal["numeric", "composition", "text"]
# What a bare number with no unit means. Declared per field so normalisation never special-cases a name.
BareNumberPolicy = Literal["reject", "assume_canonical", "percent_or_fraction"]


@dataclass(frozen=True)
class FieldSpec:
    """One extractable field, exactly as ``config.json`` describes it."""

    name: str
    group: FieldGroup
    kind: FieldKind
    description: str
    keywords: tuple[str, ...]
    canonical_unit: str | None = None
    # A short Chinese name for the column header. Display only: it reaches no prompt and no verdict, so it
    # stays out of the cache keys (see keys._SCHEMA_EXCLUDED). Empty means the UI falls back to ``name``.
    label: str = ""
    # Numeric tolerance: |a-b| <= max(rel_tol * max(|a|,|b|), abs_tol)
    rel_tol: float = 0.0
    abs_tol: float = 0.0
    condition_hint: str | None = None
    bare_number: BareNumberPolicy = "reject"
    # A closed set of canonical answers for a text field, e.g. ("DC", "RF", "DC+RF"). When a field has one,
    # comparison goes through paperfacts.normalize.canonical_category instead of raw text equality, so
    # "DC and RF magnetron co-sputtering" and "DC and RF" stop reading as two different modes.
    categories: tuple[str, ...] = ()

    @property
    def is_sample_level(self) -> bool:
        return self.group != "target"


def _field_spec(entry: Any, position: int, source: str) -> FieldSpec:
    """One validated entry of ``config.json``'s ``fields`` list."""
    where = f"{source}: fields[{position}]"
    if not isinstance(entry, Mapping):
        raise ConfigError(f"{where} must be an object, got {type(entry).__name__}")
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{where} needs a non-empty 'name'")
    where = f"{source}: field {name!r}"

    known = {spec.name for spec in dataclass_fields(FieldSpec)}
    unknown = sorted(set(entry) - known)
    if unknown:
        raise ConfigError(f"{where} has unknown key(s) {', '.join(unknown)}; valid keys are {', '.join(sorted(known))}")

    def choice(key: str, allowed: tuple[str, ...]) -> str:
        value = entry.get(key)
        if value not in allowed:
            raise ConfigError(f"{where}: {key} must be one of {', '.join(allowed)}, got {value!r}")
        return str(value)

    def number(key: str, default: float) -> float:
        value = entry.get(key, default)
        if type(value) not in (int, float):
            raise ConfigError(f"{where}: {key} must be a number, got {value!r}")
        return float(value)

    def text_or_none(key: str) -> str | None:
        value = entry.get(key)
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"{where}: {key} must be a string or null, got {value!r}")
        return value

    keywords = entry.get("keywords", [])
    if not isinstance(keywords, list) or not all(isinstance(word, str) and word for word in keywords):
        raise ConfigError(f"{where}: keywords must be a list of non-empty strings")
    description = entry.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ConfigError(f"{where} needs a non-empty 'description'; it is what the model is told to look for")

    label = entry.get("label", "")
    if not isinstance(label, str) or ("label" in entry and not label.strip()):
        raise ConfigError(f"{where}: label must be a non-empty string when present, got {entry.get('label')!r}")

    categories = entry.get("categories", [])
    if not isinstance(categories, list) or not all(isinstance(word, str) and word.strip() for word in categories):
        raise ConfigError(f"{where}: categories must be a list of non-empty strings")
    if categories and entry.get("kind") != "text":
        raise ConfigError(f"{where}: categories is only meaningful for a text field, not a {entry.get('kind')!r} one")

    return FieldSpec(
        name=name,
        group=choice("group", get_args(FieldGroup)),  # type: ignore[arg-type]
        kind=choice("kind", get_args(FieldKind)),  # type: ignore[arg-type]
        description=description,
        keywords=tuple(keywords),
        canonical_unit=text_or_none("canonical_unit"),
        label=label,
        rel_tol=number("rel_tol", 0.0),
        abs_tol=number("abs_tol", 0.0),
        condition_hint=text_or_none("condition_hint"),
        bare_number=choice("bare_number", get_args(BareNumberPolicy)) if "bare_number" in entry else "reject",  # type: ignore[arg-type]
        categories=tuple(categories),
    )


def load_field_specs(document: ConfigDocument) -> tuple[FieldSpec, ...]:
    """The whole field table, validated. An empty table is refused: it would extract nothing, silently."""
    entries = document.entries("fields")
    specs = tuple(_field_spec(entry, index, str(document.path)) for index, entry in enumerate(entries))
    if not specs:
        raise ConfigError(f"{document.path}: fields is empty, so there is nothing to extract")
    duplicates = sorted({spec.name for spec in specs if sum(s.name == spec.name for s in specs) > 1})
    if duplicates:
        raise ConfigError(f"{document.path}: fields has more than one entry named {', '.join(duplicates)}")
    return specs


def load_condition_keywords(document: ConfigDocument) -> tuple[str, ...]:
    words = document.entries("condition_keywords")
    if not all(isinstance(word, str) and word for word in words):
        raise ConfigError(f"{document.path}: condition_keywords must be a list of non-empty strings")
    return tuple(words)


_CONFIG = configuration()

FIELD_SPECS: tuple[FieldSpec, ...] = load_field_specs(_CONFIG)

# Words that mark a block as describing how a sample was made, used to choose what the sample inventory
# question is shown. Deliberately about the process, not about measured results: the inventory question is
# "which samples exist and what distinguishes them", answered in the Methods section and in table headers.
# Numeric conditions ("100 sccm", "150 W") are recognised by pattern in paperfacts.passages.
CONDITION_KEYWORDS: tuple[str, ...] = load_condition_keywords(_CONFIG)

# Sample-pairing confidence below this counts as low confidence: the fact is still compared, but the report
# counts it separately so a reviewer can look at it. It sits here because comparison_key hashes it, and
# keys.py cannot import compare.py without a cycle.
AMBIGUOUS_MATCH_CONFIDENCE: float = _CONFIG.get("comparison.ambiguous_match_confidence", float)

FIELD_BY_NAME: dict[str, FieldSpec] = {spec.name: spec for spec in FIELD_SPECS}
TARGET_FIELDS: tuple[FieldSpec, ...] = tuple(spec for spec in FIELD_SPECS if spec.group == "target")
SAMPLE_FIELDS: tuple[FieldSpec, ...] = tuple(spec for spec in FIELD_SPECS if spec.is_sample_level)
