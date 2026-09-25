"""Domain profiles: which facts a group wants out of its papers, and how its prompts name them.

A profile is one JSON file, ``profiles/<name>.json``: the groups and fields to extract, the prompt text that
describes the domain (in short slots, so the rules around them stay in code), the chart slots, the words and
unit pattern that find how a sample was made, any units of its own, and the Chinese display copy. It is read
once per process and passed on as a :class:`DomainProfile` value.

Every error names the file and the key, as ``config.json``'s do: a profile is edited by hand, and a typo found
here costs a minute where one found three stages later costs a paid run.

This module's source is hashed into the cache keys, because the slot defaults live here; display defaults
live in :mod:`paperfacts.ui_copy`, which is not hashed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache, cached_property
from pathlib import Path
from typing import Any, Literal, get_args

from paperfacts.config import Settings
from paperfacts.errors import ConfigError
from paperfacts.fields import FieldLevel, FieldRole, FieldSpec, field_roles, field_spec
from paperfacts.ui_copy import UiCopy
from paperfacts.units import UnitRegistry, compile_pattern, load_units

logger = logging.getLogger(__name__)

PROFILE_FORMAT = 1
PROFILES_DIRNAME = "profiles"
# A profile name, a group name, a field name and the two JSON keys the model is told to emit.
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
# Names the dataset, the web API or the model's answer already use for something else.
RESERVED_FIELD_NAMES = frozenset(
    {
        "document_id",
        "filename",
        "sample_id",
        "sample_label",
        "conditions",
        "available_fields",
        "agree_fields",
        "field",
        "target",
        "paper",
        "unattributed",
        "samples",
    }
)
# Every field is one question per lane in passage mode: past this many a run costs noticeably more.
FIELD_WARNING_COUNT = 40
MAX_FIELDS = 100
MAX_SLOT_LENGTH = 2000
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


@dataclass(frozen=True)
class FigureSlots:
    """The domain text of the chart-reading prompt; required when a field is ``figure_readable``."""

    subject: str
    property_noun: str
    chart_definition: str
    axis_example: str


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


def profile_path(settings: Settings) -> Path:
    """The file ``settings.profile`` names: a bare name is looked up in the repository's ``profiles/``, a value
    with a "/" or ending in ".json" is a path."""
    value = settings.profile
    if "/" in value or value.endswith(".json"):
        return Path(value)
    return settings.repo_root / PROFILES_DIRNAME / f"{value}.json"


def load_profile(path: Path) -> DomainProfile:
    """The profile in ``path``, validated. Read once per file for the life of the process, so a running server
    picks up an edit only when it restarts."""
    return _load_resolved(path.resolve())


def default_profile() -> DomainProfile:
    """The profile the built-in settings select, for code the profile is not passed to yet. The field table
    those callers read still comes from ``config.json``'s built-in location too, so the two agree; this goes
    once every caller receives the profile it runs under."""
    return load_profile(profile_path(Settings()))


@cache
def _load_resolved(path: Path) -> DomainProfile:
    if not path.is_file():
        raise ConfigError(f"no profile at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    return parse_profile(data, path)


_TOP_KEYS = (
    "$comment",
    "format",
    "name",
    "title_zh",
    "maturity",
    "description_zh",
    "groups",
    "prompt",
    "figures",
    "retrieval",
    "units",
    "ui",
    "fields",
)


def parse_profile(data: Any, source: Path) -> DomainProfile:
    """A profile from its parsed JSON. ``source`` names it in errors, and its stem must be the profile's name."""
    where = str(source)
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where} must hold a JSON object, got {type(data).__name__}")
    _refuse_unknown(data, _TOP_KEYS, where)
    for key in ("format", "name", "groups", "prompt", "retrieval", "fields"):
        if key not in data:
            raise ConfigError(f"{where}: missing key {key!r}")
    if data["format"] != PROFILE_FORMAT:
        raise ConfigError(f"{where}: format must be {PROFILE_FORMAT}, got {data['format']!r}")
    name = data["name"]
    if not isinstance(name, str) or not IDENTIFIER.match(name):
        raise ConfigError(f"{where}: name must match {IDENTIFIER.pattern}, got {name!r}")
    if name != source.stem:
        raise ConfigError(f"{where}: name {name!r} must equal the file name {source.stem!r}")
    maturity = data.get("maturity", "example")
    if maturity not in get_args(Maturity):
        raise ConfigError(f"{where}: maturity must be one of {', '.join(get_args(Maturity))}, got {maturity!r}")

    groups = _groups(data["groups"], where)
    units = load_units(data.get("units", {}), where)
    fields = _fields(data["fields"], {group.name: group.level for group in groups}, units, where)
    prompt = _text_record(PromptSlots, data["prompt"], f"{where}: prompt", slot=True)
    _check_prompt_keys(prompt, f"{where}: prompt")
    figures = None
    if data.get("figures") is not None:
        figures = _text_record(FigureSlots, data["figures"], f"{where}: figures", slot=True)
    if any(spec.figure_readable for spec in fields) != (figures is not None):
        raise ConfigError(f"{where}: figures must be given exactly when some field is figure_readable")
    retrieval = _retrieval(data["retrieval"], f"{where}: retrieval")
    ui = _text_record(UiCopy, data.get("ui", {}), f"{where}: ui", slot=False)

    material = {
        "groups": [[group.name, group.level] for group in groups],
        "fields": [_hashed_attributes(spec) for spec in fields],
        "prompt": dataclasses.asdict(prompt),
        "figures": None if figures is None else dataclasses.asdict(figures),
        "retrieval": dataclasses.asdict(retrieval),
        "units": units.material(),
    }
    return DomainProfile(
        name=name,
        title_zh=_display_text(data, "title_zh", name, where),
        maturity=maturity,
        description_zh=_display_text(data, "description_zh", "", where),
        groups=groups,
        fields=fields,
        prompt=prompt,
        figures=figures,
        retrieval=retrieval,
        units=units,
        ui=ui,
        content_hash=hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        source=source,
    )


def _refuse_unknown(data: Mapping[str, Any], valid: tuple[str, ...], where: str) -> None:
    unknown = sorted(set(data) - set(valid))
    if unknown:
        raise ConfigError(f"{where} has unknown key(s) {', '.join(unknown)}; valid keys are {', '.join(valid)}")


def _display_text(data: Mapping[str, Any], key: str, default: str, where: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{where}: {key} must be a string, got {value!r}")
    return value


def _groups(entries: Any, where: str) -> tuple[GroupSpec, ...]:
    if not isinstance(entries, list):
        raise ConfigError(f"{where}: groups must be a list, got {type(entries).__name__}")
    groups: list[GroupSpec] = []
    valid = tuple(item.name for item in dataclasses.fields(GroupSpec))
    for index, entry in enumerate(entries):
        at = f"{where}: groups[{index}]"
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{at} must be an object, got {type(entry).__name__}")
        _refuse_unknown(entry, valid, at)
        name, level, label = entry.get("name"), entry.get("level"), entry.get("label_zh", "")
        if not isinstance(name, str) or not IDENTIFIER.match(name):
            raise ConfigError(f"{at}: name must match {IDENTIFIER.pattern}, got {name!r}")
        if level not in get_args(FieldLevel):
            raise ConfigError(f"{at}: level must be one of {', '.join(get_args(FieldLevel))}, got {level!r}")
        if not isinstance(label, str):
            raise ConfigError(f"{at}: label_zh must be a string, got {label!r}")
        if any(group.name == name for group in groups):
            raise ConfigError(f"{where}: groups has more than one group named {name!r}")
        groups.append(GroupSpec(name=name, level=level, label_zh=label))
    if not any(group.level == "sample" for group in groups):
        raise ConfigError(f"{where}: groups needs at least one group with level 'sample'")
    return tuple(groups)


def _fields(entries: Any, levels: Mapping[str, FieldLevel], units: UnitRegistry, where: str) -> tuple[FieldSpec, ...]:
    if not isinstance(entries, list):
        raise ConfigError(f"{where}: fields must be a list, got {type(entries).__name__}")
    if len(entries) > MAX_FIELDS:
        raise ConfigError(f"{where}: fields has {len(entries)} entries; at most {MAX_FIELDS} are allowed")
    if len(entries) > FIELD_WARNING_COUNT:
        logger.warning(
            "%s: %d fields; each is one question per lane in passage mode, so a run costs that many requests",
            where,
            len(entries),
        )
    specs: list[FieldSpec] = []
    for index, entry in enumerate(entries):
        spec = field_spec(entry, index, where, levels)
        at = f"{where}: field {spec.name!r}"
        if not IDENTIFIER.match(spec.name):
            raise ConfigError(f"{at}: name must match {IDENTIFIER.pattern}")
        if spec.name in RESERVED_FIELD_NAMES:
            raise ConfigError(
                f"{at}: the name is reserved; reserved names are {', '.join(sorted(RESERVED_FIELD_NAMES))}"
            )
        if any(other.name == spec.name for other in specs):
            raise ConfigError(f"{where}: fields has more than one entry named {spec.name!r}")
        if spec.kind == "numeric" and spec.canonical_unit is not None:
            units.check(spec.canonical_unit, at)
        specs.append(spec)
    if not any(spec.is_sample_level for spec in specs):
        raise ConfigError(f"{where}: fields needs at least one field in a group with level 'sample'")
    return tuple(specs)


def _text_record[T](cls: type[T], data: Any, where: str, *, slot: bool) -> T:
    """A dataclass of strings from a JSON object: unknown and missing keys refused, every value a string. A
    prompt slot is also bounded and may not carry a marker a template would fill in."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where} must be an object, got {type(data).__name__}")
    attributes = dataclasses.fields(cls)  # type: ignore[arg-type]
    _refuse_unknown(data, tuple(item.name for item in attributes), where)
    missing = [
        item.name
        for item in attributes
        if item.default is dataclasses.MISSING and item.default_factory is dataclasses.MISSING and item.name not in data
    ]
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {', '.join(missing)}")
    markers = {item.name for item in dataclasses.fields(PromptSlots)} | set(COMPUTED_MARKERS)
    for key, value in data.items():
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{where}: {key} must be a non-empty string, got {value!r}")
        if slot and len(value) > MAX_SLOT_LENGTH:
            raise ConfigError(f"{where}: {key} has {len(value)} characters; at most {MAX_SLOT_LENGTH} are allowed")
        found = [marker for marker in MARKER.findall(value) if marker in markers] if slot else []
        if found:
            raise ConfigError(
                f"{where}: {key} contains the template marker {{{found[0]}}}; slots are inserted verbatim"
            )
    return cls(**data)


def _check_prompt_keys(prompt: PromptSlots, where: str) -> None:
    """The two JSON keys the model is told to emit must be plain identifiers, distinct, and not "samples"."""
    for key in ("paper_key", "no_samples_key"):
        value = getattr(prompt, key)
        if not IDENTIFIER.match(value):
            raise ConfigError(f"{where}: {key} must match {IDENTIFIER.pattern}, got {value!r}")
        if value == "samples":
            raise ConfigError(f"{where}: {key} cannot be 'samples', which the answer already uses")
    if prompt.paper_key == prompt.no_samples_key:
        raise ConfigError(f"{where}: paper_key and no_samples_key must differ, both are {prompt.paper_key!r}")


def _retrieval(data: Any, where: str) -> RetrievalSpec:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where} must be an object, got {type(data).__name__}")
    valid = tuple(item.name for item in dataclasses.fields(RetrievalSpec))
    _refuse_unknown(data, valid, where)
    words = data.get("condition_keywords")
    if not isinstance(words, list) or not all(isinstance(word, str) and word for word in words):
        raise ConfigError(f"{where}: condition_keywords must be a list of non-empty strings")
    pattern = data.get("condition_unit_pattern")
    compile_pattern(pattern, f"{where}: condition_unit_pattern", re.IGNORECASE)
    return RetrievalSpec(condition_keywords=tuple(words), condition_unit_pattern=pattern)


def _hashed_attributes(spec: FieldSpec) -> dict[str, Any]:
    """A field without its display text: what :attr:`DomainProfile.content_hash` covers."""
    return {
        name: value
        for name, value in dataclasses.asdict(spec).items()
        if field_roles(name) != frozenset({FieldRole.DISPLAY})
    }
