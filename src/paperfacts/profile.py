"""Domain profiles: which facts a group wants out of its papers, and how its prompts name them.

A profile is one JSON file, ``profiles/<name>.json``: the groups and fields to extract, the prompt text that
describes the domain (in short slots, so the rules around them stay in code), the chart slots, the words and
unit pattern that find how a sample was made, any units of its own, and the Chinese display copy. It is read
once per process by :mod:`paperfacts.profile_loader` and passed on as a :class:`DomainProfile` value.

This module holds the value types only. Its source is hashed into the cache keys, because the slot defaults
live here; display defaults live in :mod:`paperfacts.ui_copy` and the reading and checking of a file in
:mod:`paperfacts.profile_loader`, neither of which is hashed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

from paperfacts.fields import FieldLevel, FieldSpec
from paperfacts.ui_copy import UiCopy
from paperfacts.units import UnitRegistry

# The markers a template fills in from computed values rather than from a slot, and the one pattern every
# marker matches: prompts.py renders with it, and a slot may not contain a marker it would fill.
COMPUTED_MARKERS = ("sample_groups", "paper_groups", "condition_rules", "subset_scope", "fields")
MARKER = re.compile(r"\{([a-z_]+)\}")
Maturity = Literal["production", "example"]


@dataclass(frozen=True)
class GroupSpec:
    """A group of fields. The name is shown to the model ("group: film"); the level decides the scope rules."""

    name: str
    level: FieldLevel
    label_zh: str = ""


@dataclass(frozen=True)
class PromptSlots:
    """The domain text the prompt templates are filled with. Three slots are required; the rest default to
    generic wording, and ``paper_level_rule`` is generated from the groups when it is None."""

    domain_subject: str
    sample_definition: str
    field_scope: str
    fact_noun: str = "scientific facts"
    sample_plural: str = "samples"
    sample_singular: str = "sample"
    sample_unit: str = "sample / preparation condition set"
    sample_examples: str = "(e.g. one per varied preparation condition)"
    sample_id_example: str = "S1"
    condition_noun: str = "preparation conditions"
    condition_examples: str = "(temperature, time, composition...)"
    unit_examples: str = '"nm", "°C", "%"'
    paper_key: str = "paper"
    paper_level_rule: str | None = None
    no_samples_key: str = "no_samples_in_scope"
    no_samples_clause: str = "the paper reports no such sample"
    no_samples_condition: str = "the paper reports no sample of its own that is in scope"
    samples_present_condition: str = "such a sample is reported"
    subset_examples: str = '"the samples annealed at 500 °C", "the doped samples"'
    whole_series_examples: str = '"all samples", "for all samples"'
    partial_collective_example: str = '"the doped samples", when one listed sample is undoped'
    multi_condition_example: str = "one quantity measured under two conditions"
    matching_condition_examples: str = "(temperature, time, composition...)"
    matching_value_examples: str = "(measured values...)"
    matching_justification_example: str = "both are the sample annealed at 500 °C"
    # Rule 2's examples of a table header that carries a power of ten, and of one that carries only a unit.
    scaled_header_examples: str = '"X × 10^3 (unit)" or "X (×10^-3 unit)"'
    plain_header_example: str = 'a column "Temperature (°C)" gives just "°C"'
    # Where a number outside a field's plausible range usually comes from, in every field line that has one.
    implausible_origin: str = "a different sample, state or quantity"


@dataclass(frozen=True)
class FigureSlots:
    """The domain text of the chart-reading prompt; required when a field is ``figure_readable``."""

    subject: str
    property_noun: str
    chart_definition: str
    axis_example: str
    # An axis whose multiplier sits on the quantity symbol, quoted as the axis and as the unit reported for it.
    symbol_axis_example: str = 'axis "X × 10^3 (unit)" => unit "X × 10^3 (unit)"'
    # Tick labels as they are to be reported for x: numbers and category names.
    x_label_examples: str = '400, 1.5, "As-prepared", "Sample A"'


@dataclass(frozen=True)
class RetrievalSpec:
    """How the inventory question finds the blocks that say how samples were made."""

    condition_keywords: tuple[str, ...]
    # Compiled with re.IGNORECASE and matched on passages.searchable() text.
    condition_unit_pattern: str


@dataclass(frozen=True)
class DomainProfile:
    name: str
    title_zh: str
    maturity: Maturity
    description_zh: str
    groups: tuple[GroupSpec, ...]
    # Declaration order is question order and column order.
    fields: tuple[FieldSpec, ...]
    prompt: PromptSlots
    figures: FigureSlots | None
    retrieval: RetrievalSpec
    units: UnitRegistry
    ui: UiCopy
    # sha256 of everything but the display text, computed once by the loader.
    content_hash: str
    source: Path = field(compare=False)

    def __hash__(self) -> int:
        # Hashing the deep tuples on every lookup of a profile-keyed cache would cost more than the lookup.
        return hash(self.content_hash)

    @cached_property
    def by_name(self) -> dict[str, FieldSpec]:
        return {spec.name: spec for spec in self.fields}

    @cached_property
    def paper_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(spec for spec in self.fields if not spec.is_sample_level)

    @cached_property
    def sample_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(spec for spec in self.fields if spec.is_sample_level)

    @cached_property
    def figure_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(spec for spec in self.fields if spec.figure_readable)

    @cached_property
    def paper_groups(self) -> tuple[GroupSpec, ...]:
        return tuple(group for group in self.groups if group.level == "paper")

    @cached_property
    def sample_groups(self) -> tuple[GroupSpec, ...]:
        return tuple(group for group in self.groups if group.level == "sample")
