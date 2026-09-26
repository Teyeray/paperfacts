"""A field: what to extract, in what unit, and how close counts as the same.

The table itself is a domain profile's ``fields`` (``profiles/<name>.json``, read by
:mod:`paperfacts.profile_loader`), and this module holds no table of its own: every stage is handed the profile it
runs under. Changing a field is therefore an edit to a JSON file rather than to Python -- which is the point,
since which facts a group wants out of its papers is the thing that differs between groups. Every entry is
validated on the way in by :func:`paperfacts.profile_loader.field_spec`: an unknown key or a misspelled group is a
:class:`ConfigError` naming the field, not a surprise three stages later. A field's level (paper or sample) is its
group's, as the profile declares it. This module holds the attribute defaults the cache keys omit, which is why
its source is hashed.

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

from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from enum import StrEnum
from typing import Literal

# Whether a field belongs to the paper as a whole or to each of its samples. A profile declares it per group.
FieldLevel = Literal["paper", "sample"]
# numeric: a number with a unit; composition: a chemical formula; text: anything else
FieldKind = Literal["numeric", "composition", "text"]
# The kinds whose value is quoted as a number, so an answer or a block without a digit cannot hold one. Here
# rather than in paperfacts.kinds because cleaning (records.py) and retrieval (passages.py) sit below that module.
DIGIT_KINDS: frozenset[FieldKind] = frozenset({"numeric"})
# How many values of a field one sample (or the paper) holds. Only "one" exists yet; a dataset column carries it
# so that the workbook and the web format a cell by its column.
Cardinality = Literal["one", "many"]
# What a bare number with no unit means. Declared per field so normalisation never special-cases a name.
BareNumberPolicy = Literal["reject", "assume_canonical", "percent_or_fraction"]
# How a workbook prints a numeric cell: plainly, or in scientific notation for values spanning decades.
DisplayFormat = Literal["plain", "scientific"]
# What a range quoted as one value ("10-20") becomes: its midpoint, no value at all, or the end the field asks
# for. Only an end fills a dataset cell: it is a number the paper printed, a midpoint is not.
RangePolicy = Literal["midpoint", "reject", "lower", "upper"]
# What a value quoted with an "after ..." clause ("92.5% after 100 cycles") becomes: refused as the value of
# another state of the sample, or read with the clause moved into its measurement condition.
AfterClause = Literal["refuse", "condition"]


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
    # PROMPT for the field line; VERDICT because a field with a hint is measured along an axis the paper states
    # beside the value, so the numbers in two conditions decide whether two values are one measurement.
    condition_hint: str | None = field(default=None, metadata=_roles(FieldRole.PROMPT, FieldRole.VERDICT))
    bare_number: BareNumberPolicy = field(default="reject", metadata=_roles(FieldRole.CLEANING, FieldRole.FIGURE))
    # A closed set of canonical answers for a text field, e.g. ("DC", "RF", "DC+RF"). When a field has one,
    # comparison goes through paperfacts.normalize.canonical_category instead of raw text equality, so
    # "DC and RF magnetron co-sputtering" and "DC and RF" stop reading as two different modes.
    categories: tuple[str, ...] = field(default=(), metadata=_roles(FieldRole.VERDICT))
    # Plausible (min, max) in canonical_unit, either end open; a field with no unit (a cycle count) is judged on
    # the number as parsed. A value outside it is almost always a different quantity the model mistook for
    # this one -- the spin-coating rpm of an absorber read as the substrate rotation, a perovskite layer's
    # thickness read as the electrode's -- so the model is told the range and a converted value outside it is
    # dropped with an audited reason.
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
    # VERDICT as well: a dataset cell of such a field whose value arrives without its condition carries a note.
    condition_rule: str | None = field(default=None, metadata=_roles(FieldRole.PROMPT, FieldRole.VERDICT))
    # The dataset note when such a field's value arrives without its condition. Required with condition_rule
    # (the loader refuses one without the other), so no generic wording is ever stored in its place.
    missing_condition_note_zh: str | None = field(default=None, metadata=_roles(FieldRole.VERDICT))
    # Whether a chart's y axis may be read for this field: a numeric property of the sample itself.
    figure_readable: bool = field(default=False, metadata=_roles(FieldRole.FIGURE))
    display_format: DisplayFormat = field(default="plain", metadata=_roles(FieldRole.DISPLAY))
    range_policy: RangePolicy = field(default="midpoint", metadata=_roles(FieldRole.CLEANING, FieldRole.VERDICT))
    # "refuse" suits a quantity whose "after annealing" value is a different sample state ("100 nm after
    # annealing" is not the as-deposited thickness); "condition" suits one that is only ever stated after
    # something (a capacity retention after N cycles), where the clause is what the value was measured under.
    after_clause: AfterClause = field(default="refuse", metadata=_roles(FieldRole.CLEANING, FieldRole.VERDICT))

    @property
    def is_sample_level(self) -> bool:
        return self.level == "sample"

    def describe_range(self) -> str | None:
        """The plausible range in words, e.g. "at most 100 rpm", or None when the field declares none. Both ends
        are inclusive, as in :meth:`in_range`."""
        low, high = self.valid_range
        unit = f" {self.canonical_unit}" if self.canonical_unit else ""
        if low is not None and high is not None:
            return f"between {low:g} and {high:g}{unit}"
        if high is not None:
            return f"at most {high:g}{unit}"
        if low is not None:
            return f"at least {low:g}{unit}"
        return None

    def in_range(self, value: float) -> bool:
        low, high = self.valid_range
        return (low is None or value >= low) and (high is None or value <= high)
