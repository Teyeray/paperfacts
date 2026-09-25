"""A field: what to extract, in what unit, and how close counts as the same.

The table itself is a domain profile's ``fields`` (``profiles/<name>.json``, read by :mod:`paperfacts.profile`),
and this module holds no table of its own: every stage is handed the profile it runs under. Changing a field is
therefore an edit to a JSON file rather than to Python -- which is the point, since which facts a group wants
out of its papers is the thing that differs between groups. Every entry is validated on the way in by
:func:`field_spec`: an unknown key or a misspelled group is a :class:`ConfigError` naming the field, not a
surprise three stages later. A field's level (paper or sample) is its group's, as the profile declares it.

One table drives five things: the field descriptions given to the model, unit conversion, what to do with
a bare number that has no unit, the numeric tolerance used when comparing the two lanes, and -- through
``keywords`` -- which blocks passage-mode retrieval puts in front of the model. Its contents are hashed
into the cache keys (:mod:`paperfacts.keys`), so changing any cell invalidates exactly the caches that
depended on it.

``keywords`` are the names a paper uses for the field, not its units: units are recognised separately by
:mod:`paperfacts.units`, which knows which of them are specific enough to identify a field on their own.
Keywords are matched as whole tokens, case-insensitively, after Unicode folding.

Every attribute carries its :class:`FieldRole` set in its dataclass metadata: which stages read it (the
prompt, the cleaning of an answer, a verdict, retrieval, figure reading), or that it is display text only.
``tests/test_field_roles.py`` pins the roles against the attribute sets :mod:`paperfacts.keys` hashes, so
adding an attribute is a decision about which keys it belongs to.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from enum import StrEnum
from typing import Any, Literal, get_args

from paperfacts.errors import ConfigError

# Whether a field belongs to the paper as a whole or to each of its samples. A profile declares it per group.
FieldLevel = Literal["paper", "sample"]
# A number as a measurement condition states it: "550", "400" and "800" in "average 400–800 nm".
CONDITION_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# numeric: a number with a unit; composition: a chemical formula; text: anything else
FieldKind = Literal["numeric", "composition", "text"]
# What a bare number with no unit means. Declared per field so normalisation never special-cases a name.
BareNumberPolicy = Literal["reject", "assume_canonical", "percent_or_fraction"]
# How a workbook prints a numeric cell: plainly, or in scientific notation for values spanning decades.
DisplayFormat = Literal["plain", "scientific"]
# What a range quoted as one value ("10-20") becomes: its midpoint, or no value at all.
RangePolicy = Literal["midpoint", "reject"]


class FieldRole(StrEnum):
    """A stage that reads a field attribute. An attribute's roles decide which cache keys hash it."""

    PROMPT = "prompt"  # rendered into what the model is asked
    CLEANING = "cleaning"  # decides which of the model's values survive, or what they convert to
    VERDICT = "verdict"  # decides a comparison or a dataset cell
    RETRIEVAL = "retrieval"  # decides which blocks a passage-mode question is shown
    FIGURE = "figure"  # decides which charts are read, what the vision model is told, or a reading's value
    DISPLAY = "display"  # reaches no model and no decision, so it is never hashed


def _roles(*roles: FieldRole) -> dict[str, frozenset[FieldRole]]:
    return {"roles": frozenset(roles)}


def field_roles(name: str) -> frozenset[FieldRole]:
    """The roles of the :class:`FieldSpec` attribute ``name``."""
    return next(item.metadata["roles"] for item in dataclass_fields(FieldSpec) if item.name == name)


@dataclass(frozen=True)
class FieldSpec:
    """One extractable field, exactly as its profile describes it."""

    name: str = field(metadata=_roles(FieldRole.PROMPT, FieldRole.FIGURE))
    # A group the table declares; the field line shows it to the model ("group: film").
    group: str = field(metadata=_roles(FieldRole.PROMPT))
    kind: FieldKind = field(metadata=_roles(FieldRole.PROMPT))
    description: str = field(metadata=_roles(FieldRole.PROMPT, FieldRole.FIGURE))
    keywords: tuple[str, ...] = field(metadata=_roles(FieldRole.RETRIEVAL, FieldRole.FIGURE))
    canonical_unit: str | None = field(default=None, metadata=_roles(FieldRole.PROMPT, FieldRole.FIGURE))
    # A short Chinese name for the column header. Display only: it reaches no prompt and no verdict, so it
    # stays out of the cache keys (FieldRole.DISPLAY). Empty means the UI falls back to ``name``.
    label: str = field(default="", metadata=_roles(FieldRole.DISPLAY))
    # The Chinese explanation of the field, for the web header tooltip and the Excel field sheet. Display
    # only, like ``label``: no prompt and no verdict reads it, so it stays out of the cache keys.
    description_zh: str = field(default="", metadata=_roles(FieldRole.DISPLAY))
    # Numeric tolerance: |a-b| <= max(rel_tol * max(|a|,|b|), abs_tol)
    rel_tol: float = field(default=0.0, metadata=_roles(FieldRole.VERDICT))
    abs_tol: float = field(default=0.0, metadata=_roles(FieldRole.VERDICT))
    condition_hint: str | None = field(default=None, metadata=_roles(FieldRole.PROMPT))
    bare_number: BareNumberPolicy = field(default="reject", metadata=_roles(FieldRole.CLEANING, FieldRole.FIGURE))
    # A closed set of canonical answers for a text field, e.g. ("DC", "RF", "DC+RF"). When a field has one,
    # comparison goes through paperfacts.normalize.canonical_category instead of raw text equality, so
    # "DC and RF magnetron co-sputtering" and "DC and RF" stop reading as two different modes.
    categories: tuple[str, ...] = field(default=(), metadata=_roles(FieldRole.VERDICT))
    # Plausible (min, max) in canonical_unit, either end open. A value outside it is almost always a
    # different quantity the model mistook for this one -- the spin-coating rpm of an absorber read as the
    # substrate rotation, a perovskite layer's thickness read as the electrode's -- so the model is told the
    # range and a converted value outside it is dropped with an audited reason.
    valid_range: tuple[float | None, float | None] = field(
        default=(None, None), metadata=_roles(FieldRole.PROMPT, FieldRole.CLEANING)
    )
    # Which measurement fills the dataset cell when a sample has several, in order of preference: each entry
    # names the numbers a condition states ("400-800" for an average over 400-800 nm, "550" for one
    # wavelength). A verdict rule only -- the model is never told it -- so it is kept out of the schema
    # fingerprint and hashed into comparison_key alone.
    condition_preference: tuple[str, ...] = field(default=(), metadata=_roles(FieldRole.VERDICT))
    # Resolved from the field's group by the loader, never written in a field entry. The default suits a
    # sample-level group only: a FieldSpec built by hand for a paper-level group must pass it.
    level: FieldLevel = field(
        default="sample", metadata=_roles(FieldRole.PROMPT, FieldRole.CLEANING, FieldRole.VERDICT)
    )
    # What ``condition`` must always hold for this field ("the wavelength or spectral range"): for a quantity
    # whose value means nothing without the condition it was measured under.
    condition_rule: str | None = field(default=None, metadata=_roles(FieldRole.PROMPT))
    # The dataset note when such a field's value arrives without its condition; None gives a generic one.
    missing_condition_note_zh: str | None = field(default=None, metadata=_roles(FieldRole.VERDICT))
    # Whether a chart's y axis may be read for this field: a numeric property of the sample itself.
    figure_readable: bool = field(default=False, metadata=_roles(FieldRole.FIGURE))
    display_format: DisplayFormat = field(default="plain", metadata=_roles(FieldRole.DISPLAY))
    range_policy: RangePolicy = field(default="midpoint", metadata=_roles(FieldRole.CLEANING, FieldRole.VERDICT))

    @property
    def is_sample_level(self) -> bool:
        return self.level == "sample"

    def describe_range(self) -> str | None:
        """The plausible range in words, e.g. "at most 100 rpm", or None when the field declares none. Both ends
        are inclusive, as in :meth:`in_range`."""
        low, high = self.valid_range
        unit = self.canonical_unit or ""
        if low is not None and high is not None:
            return f"between {low:g} and {high:g} {unit}"
        if high is not None:
            return f"at most {high:g} {unit}"
        if low is not None:
            return f"at least {low:g} {unit}"
        return None

    def in_range(self, value: float) -> bool:
        low, high = self.valid_range
        return (low is None or value >= low) and (high is None or value <= high)


# Attributes a field entry never states: the loader derives them.
_DERIVED = {"level"}


def field_spec(entry: Any, position: int, source: str, levels: Mapping[str, FieldLevel]) -> FieldSpec:
    """One validated entry of a ``fields`` list; ``levels`` maps each declared group to its level."""
    where = f"{source}: fields[{position}]"
    if not isinstance(entry, Mapping):
        raise ConfigError(f"{where} must be an object, got {type(entry).__name__}")
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{where} needs a non-empty 'name'")
    where = f"{source}: field {name!r}"

    known = {spec.name for spec in dataclass_fields(FieldSpec)} - _DERIVED
    unknown = sorted(set(entry) - known)
    if unknown:
        raise ConfigError(f"{where} has unknown key(s) {', '.join(unknown)}; valid keys are {', '.join(sorted(known))}")

    def choice(key: str, allowed: tuple[str, ...]) -> str:
        value = entry.get(key)
        if value not in allowed:
            raise ConfigError(f"{where}: {key} must be one of {', '.join(allowed)}, got {value!r}")
        return str(value)

    def tolerance(key: str) -> float:
        # A negative tolerance makes |a - b| <= tol impossible even for a == b: every value would conflict.
        value = entry.get(key, 0.0)
        if type(value) not in (int, float) or value < 0:
            raise ConfigError(f"{where}: {key} must be a number of at least 0, got {value!r}")
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

    description_zh = entry.get("description_zh", "")
    if not isinstance(description_zh, str) or ("description_zh" in entry and not description_zh.strip()):
        raise ConfigError(
            f"{where}: description_zh must be a non-empty string when present, got {entry.get('description_zh')!r}"
        )

    preference = entry.get("condition_preference", [])
    if not isinstance(preference, list) or not all(
        isinstance(word, str) and CONDITION_NUMBER.search(word) for word in preference
    ):
        raise ConfigError(
            f"{where}: condition_preference must be a list of strings each naming a number, like '400-800'"
        )

    categories = entry.get("categories", [])
    if not isinstance(categories, list) or not all(isinstance(word, str) and word.strip() for word in categories):
        raise ConfigError(f"{where}: categories must be a list of non-empty strings")
    if categories and entry.get("kind") != "text":
        raise ConfigError(f"{where}: categories is only meaningful for a text field, not a {entry.get('kind')!r} one")

    bare_number = choice("bare_number", get_args(BareNumberPolicy)) if "bare_number" in entry else "reject"
    if bare_number == "percent_or_fraction" and entry.get("canonical_unit") != "%":
        # The policy reads a bare 0.8 as 80: meaningful for a percentage, an invented number for anything
        # else (0.8 would become 80 nm).
        raise ConfigError(
            f"{where}: bare_number 'percent_or_fraction' needs canonical_unit '%', got {entry.get('canonical_unit')!r}"
        )

    numeric = entry.get("kind") == "numeric"
    condition_rule = text_or_none("condition_rule")
    if condition_rule is not None and (not condition_rule.strip() or entry.get("condition_hint") is None):
        # The rule tells the model to fill a condition that the field line must first say the field has.
        raise ConfigError(f"{where}: condition_rule must be a non-empty string and needs a condition_hint")
    missing_note = text_or_none("missing_condition_note_zh")
    if missing_note is not None and not missing_note.strip():
        raise ConfigError(f"{where}: missing_condition_note_zh must be a non-empty string when present")
    figure_readable = entry.get("figure_readable", False)
    if type(figure_readable) is not bool:
        raise ConfigError(f"{where}: figure_readable must be true or false, got {figure_readable!r}")
    if figure_readable and (not numeric or entry.get("canonical_unit") is None):
        raise ConfigError(f"{where}: figure_readable needs a numeric field with a canonical_unit")
    for key in ("display_format", "range_policy"):
        if key in entry and not numeric:
            raise ConfigError(f"{where}: {key} is only meaningful for a numeric field, not a {entry.get('kind')!r} one")
    display_format = choice("display_format", get_args(DisplayFormat)) if "display_format" in entry else "plain"
    range_policy = choice("range_policy", get_args(RangePolicy)) if "range_policy" in entry else "midpoint"
    group = choice("group", tuple(levels))

    return FieldSpec(
        name=name,
        group=group,
        kind=choice("kind", get_args(FieldKind)),  # type: ignore[arg-type]
        description=description,
        keywords=tuple(keywords),
        canonical_unit=text_or_none("canonical_unit"),
        label=label,
        description_zh=description_zh,
        rel_tol=tolerance("rel_tol"),
        abs_tol=tolerance("abs_tol"),
        condition_hint=text_or_none("condition_hint"),
        bare_number=bare_number,  # type: ignore[arg-type]
        categories=tuple(categories),
        valid_range=_valid_range(entry, where),
        condition_preference=tuple(preference),
        level=levels[group],
        condition_rule=condition_rule,
        missing_condition_note_zh=missing_note,
        figure_readable=figure_readable,
        display_format=display_format,  # type: ignore[arg-type]
        range_policy=range_policy,  # type: ignore[arg-type]
    )


def _valid_range(entry: Mapping[str, Any], where: str) -> tuple[float | None, float | None]:
    if "valid_range" not in entry:
        return (None, None)
    bounds = entry["valid_range"]
    if not isinstance(bounds, Mapping) or not set(bounds) <= {"min", "max"}:
        raise ConfigError(f"{where}: valid_range must be an object with 'min' and/or 'max', got {bounds!r}")
    if entry.get("kind") != "numeric" or entry.get("canonical_unit") is None:
        raise ConfigError(f"{where}: valid_range needs a numeric field with a canonical_unit to be read in")
    low, high = bounds.get("min"), bounds.get("max")
    for key, value in (("min", low), ("max", high)):
        if value is not None and type(value) not in (int, float):
            raise ConfigError(f"{where}: valid_range.{key} must be a number or null, got {value!r}")
    if low is None and high is None:
        raise ConfigError(f"{where}: valid_range needs at least one of 'min' and 'max'")
    if low is not None and high is not None and low >= high:
        raise ConfigError(f"{where}: valid_range.min ({low}) must be below valid_range.max ({high})")
    return (None if low is None else float(low), None if high is None else float(high))
