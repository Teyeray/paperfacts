"""Reading a domain profile: ``profiles/<name>.json`` to a validated :class:`~paperfacts.profile.DomainProfile`.

The profile's value types and the defaults the keys omit live in hashed modules (:mod:`paperfacts.profile`,
:mod:`paperfacts.fields`, :mod:`paperfacts.units`); this module only checks a file and builds those values, and is
in no cache key. That is sound because everything it decides reaches a key as a value: a field attribute, a slot, a
unit declaration or a derived retrieval pattern is hashed as what it is, and one left at its default is omitted
against the default declared in the hashed module. A reworded error message or a stricter check here therefore
renames nothing.

Every error names the file and the key, as ``config.json``'s do: a profile is edited by hand, and a typo found
here costs a minute where one found three stages later costs a paid run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import re
import re._constants as sre_constants
import re._parser as sre_parser
from collections.abc import Callable, Mapping
from dataclasses import fields as dataclass_fields
from functools import cache
from pathlib import Path
from typing import Any, get_args

from paperfacts.config import Settings
from paperfacts.errors import ConfigError
from paperfacts.fields import (
    AfterClause,
    BareNumberPolicy,
    DisplayFormat,
    FieldKind,
    FieldLevel,
    FieldRole,
    FieldSpec,
    RangePolicy,
    field_roles,
)
from paperfacts.profile import (
    COMPUTED_MARKERS,
    MARKER,
    DomainProfile,
    FigureSlots,
    GroupSpec,
    Maturity,
    PromptSlots,
    RetrievalSpec,
)
from paperfacts.text import clean_unit, normalize_text
from paperfacts.ui_copy import UiCopy
from paperfacts.units import (
    BUILTIN_CONVERTERS,
    BUILTIN_RETRIEVAL,
    DeclaredUnit,
    UnitRegistry,
    derive_retrieval,
    fold_spelling,
)

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


# sha256 of the bytes each loaded file was parsed from, by resolved path. Beside the profile rather than in it:
# the bytes include the display text, which content_hash and every key leave out on purpose.
_FILE_SHA256: dict[Path, str] = {}


def loaded_file_sha256(profile: DomainProfile) -> str | None:
    """The sha256 of the file bytes ``profile`` was parsed from; None for a profile not loaded from a file."""
    return _FILE_SHA256.get(profile.source)


@cache
def _load_resolved(path: Path) -> DomainProfile:
    if not path.is_file():
        raise ConfigError(f"no profile at {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        # An unreadable file (permissions, a directory race) is a configuration problem naming the file, like a
        # missing one, not a traceback from the first stage that needed the profile.
        raise ConfigError(f"cannot read the profile at {path}: {exc}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    profile = parse_profile(data, path)
    _FILE_SHA256[path] = hashlib.sha256(raw).hexdigest()
    return profile


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
    "ignored_unit_suffixes",
    "ui",
    "fields",
)
_REQUIRED_KEYS = ("format", "name", "groups", "prompt", "retrieval", "fields")


def parse_profile(data: Any, source: Path) -> DomainProfile:
    """A profile from its parsed JSON. ``source`` names it in errors, and its stem must be the profile's name.

    Every section is checked even after one fails, and each field on its own, so one ConfigError names every
    problem, one per line: an author fixing a file by hand should not learn about them one run at a time."""
    where = str(source)
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where} must hold a JSON object, got {type(data).__name__}")
    errors: list[str] = []

    def checked[T](step: Callable[[], T]) -> T | None:
        try:
            return step()
        except ConfigError as exc:
            errors.append(str(exc))
            return None

    checked(lambda: _refuse_unknown(data, _TOP_KEYS, where))
    errors += [f"{where}: missing key {key!r}" for key in _REQUIRED_KEYS if key not in data]
    if "format" in data and data["format"] != PROFILE_FORMAT:
        errors.append(f"{where}: format must be {PROFILE_FORMAT}, got {data['format']!r}")
    name = data.get("name")
    if "name" in data and (not isinstance(name, str) or not IDENTIFIER.fullmatch(name)):
        errors.append(f"{where}: name must match {IDENTIFIER.pattern}, got {name!r}")
    elif "name" in data and name != source.stem:
        errors.append(f"{where}: name {name!r} must equal the file name {source.stem!r}")
    maturity = data.get("maturity", "example")
    if maturity not in get_args(Maturity):
        errors.append(f"{where}: maturity must be one of {', '.join(get_args(Maturity))}, got {maturity!r}")

    groups = checked(lambda: _groups(data["groups"], where)) if "groups" in data else None
    units = checked(lambda: load_units(data.get("units", {}), where, data.get("ignored_unit_suffixes", [])))
    fields = None
    if groups is not None and "fields" in data:
        levels = {group.name: group.level for group in groups}
        fields = checked(lambda: _fields(data["fields"], levels, units, where, errors))
    prompt = None
    if "prompt" in data:
        prompt = checked(lambda: _text_record(PromptSlots, data["prompt"], f"{where}: prompt", slot=True))
    if prompt is not None:
        checked(lambda: _check_prompt_keys(prompt, f"{where}: prompt"))
    figures = None
    if data.get("figures") is not None:
        figures = checked(lambda: _text_record(FigureSlots, data["figures"], f"{where}: figures", slot=True))
    if fields is not None and any(spec.figure_readable for spec in fields) != (data.get("figures") is not None):
        errors.append(f"{where}: figures must be given exactly when some field is figure_readable")
    retrieval = checked(lambda: _retrieval(data["retrieval"], f"{where}: retrieval")) if "retrieval" in data else None
    ui = checked(lambda: _text_record(UiCopy, data.get("ui", {}), f"{where}: ui", slot=False))
    title_zh = checked(lambda: _display_text(data, "title_zh", name, where))
    description_zh = checked(lambda: _display_text(data, "description_zh", "", where))
    if errors:
        raise ConfigError("\n".join(errors))
    # Every step above succeeded, so none of these is None; the asserts only tell the type checker so.
    assert groups is not None and units is not None and fields is not None and prompt is not None
    assert retrieval is not None and ui is not None and title_zh is not None and description_zh is not None

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
        title_zh=title_zh,
        maturity=maturity,
        description_zh=description_zh,
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
        if not isinstance(name, str) or not IDENTIFIER.fullmatch(name):
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


def _fields(
    entries: Any, levels: Mapping[str, FieldLevel], units: UnitRegistry | None, where: str, errors: list[str]
) -> tuple[FieldSpec, ...] | None:
    """The field table, or None when some entry is invalid; each invalid entry adds its own line to ``errors``.
    Without ``units`` (their section failed) canonical units are not checked against them."""
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
    failed = False
    for index, entry in enumerate(entries):
        try:
            spec = _field(entry, index, levels, units, where)
        except ConfigError as exc:
            errors.append(str(exc))
            failed = True
            continue
        if any(other.name == spec.name for other in specs):
            errors.append(f"{where}: fields has more than one entry named {spec.name!r}")
            failed = True
        specs.append(spec)
    if not any(spec.is_sample_level for spec in specs) and not failed:
        raise ConfigError(f"{where}: fields needs at least one field in a group with level 'sample'")
    return None if failed else tuple(specs)


def _field(
    entry: Any, index: int, levels: Mapping[str, FieldLevel], units: UnitRegistry | None, where: str
) -> FieldSpec:
    spec = field_spec(entry, index, where, levels)
    at = f"{where}: field {spec.name!r}"
    if not IDENTIFIER.fullmatch(spec.name):
        raise ConfigError(f"{at}: name must match {IDENTIFIER.pattern}")
    if spec.name in RESERVED_FIELD_NAMES:
        raise ConfigError(f"{at}: the name is reserved; reserved names are {', '.join(sorted(RESERVED_FIELD_NAMES))}")
    if spec.condition_rule is not None and spec.missing_condition_note_zh is None:
        # The note is what a dataset cell without the condition says. Built from the label instead, it would be
        # display text stored inside a verdict, which no key covers.
        raise ConfigError(f"{at}: condition_rule needs missing_condition_note_zh, the note a cell without it gets")
    if spec.kind == "numeric" and spec.canonical_unit is not None and units is not None:
        units.check(spec.canonical_unit, at)
    return spec


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
        if not IDENTIFIER.fullmatch(value):
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


# ---- Regular expressions ---------------------------------------------------------------------------------------

MAX_PATTERN_LENGTH = 500
_REPEATS = (sre_constants.MAX_REPEAT, sre_constants.MIN_REPEAT, sre_constants.POSSESSIVE_REPEAT)


def compile_pattern(pattern: Any, where: str, flags: int = 0) -> re.Pattern[str]:
    """A regular expression from a profile: a string of at most ``MAX_PATTERN_LENGTH`` characters that compiles
    and repeats no group that itself repeats or alternates."""
    if not isinstance(pattern, str) or not pattern or len(pattern) > MAX_PATTERN_LENGTH:
        raise ConfigError(f"{where} must be a regular expression of 1 to {MAX_PATTERN_LENGTH} characters")
    try:
        compiled = re.compile(pattern, flags)
    except re.error as exc:
        raise ConfigError(f"{where} is not a valid regular expression: {exc}") from exc
    if _backtracks(sre_parser.parse(pattern, flags), repeated=False):
        # Every block of every paper is searched with it, so a pattern whose backtracking grows exponentially
        # with the text would stall a run on one long paragraph.
        raise ConfigError(
            f"{where} repeats a group that itself repeats or alternates, like (x+)+ or (a|b)*: such a pattern can"
            " take exponential time on a long block; write the repetition once"
        )
    return compiled


def _backtracks(items: Any, *, repeated: bool) -> bool:
    """Whether a parsed pattern holds a repeat, or an alternation, inside another repeat. A conservative
    reading of catastrophic backtracking: it also refuses some patterns that would run fast, never one that
    would not."""
    for op, argument in items:
        if op in _REPEATS:
            _, high, body = argument
            if high > 1 and repeated:
                return True
            if _backtracks(body, repeated=repeated or high > 1):
                return True
        elif op is sre_constants.BRANCH:
            if repeated or any(_backtracks(branch, repeated=repeated) for branch in argument[1]):
                return True
        elif op is sre_constants.SUBPATTERN:
            if _backtracks(argument[3], repeated=repeated):
                return True
        elif op in (sre_constants.ASSERT, sre_constants.ASSERT_NOT):
            if _backtracks(argument[1], repeated=repeated):
                return True
        elif op is sre_constants.ATOMIC_GROUP:
            if _backtracks(argument, repeated=repeated):
                return True
        elif op is sre_constants.GROUPREF_EXISTS:
            if any(branch is not None and _backtracks(branch, repeated=repeated) for branch in argument[1:]):
                return True
    return False


# ---- Fields ------------------------------------------------------------------------------------------------

# A number as a measurement condition states it: "550", "400" and "800" in "average 400–800 nm".
CONDITION_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Attributes a field entry never states: the loader derives them.
_DERIVED = {"level"}
# What an entry that leaves an attribute out gets: the dataclass's own default, which is also what the keys omit.
_FIELD_DEFAULTS = {
    item.name: item.default for item in dataclass_fields(FieldSpec) if item.default is not dataclasses.MISSING
}


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
        value = entry.get(key, _FIELD_DEFAULTS.get(key))
        if value not in allowed:
            raise ConfigError(f"{where}: {key} must be one of {', '.join(allowed)}, got {value!r}")
        return str(value)

    def tolerance(key: str) -> float:
        # A negative tolerance makes |a - b| <= tol impossible even for a == b: every value would conflict.
        value = entry.get(key, _FIELD_DEFAULTS[key])
        if type(value) not in (int, float) or value < 0:
            raise ConfigError(f"{where}: {key} must be a number of at least 0, got {value!r}")
        return float(value)

    def text_or_none(key: str) -> str | None:
        value = entry.get(key)
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"{where}: {key} must be a string or null, got {value!r}")
        return value

    def nonempty_text(key: str) -> str:
        value = entry.get(key, _FIELD_DEFAULTS[key])
        if not isinstance(value, str) or (key in entry and not value.strip()):
            raise ConfigError(f"{where}: {key} must be a non-empty string when present, got {entry.get(key)!r}")
        return value

    keywords = entry.get("keywords", [])
    if not isinstance(keywords, list) or not all(isinstance(word, str) and word for word in keywords):
        raise ConfigError(f"{where}: keywords must be a list of non-empty strings")
    description = entry.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ConfigError(f"{where} needs a non-empty 'description'; it is what the model is told to look for")
    label, description_zh = nonempty_text("label"), nonempty_text("description_zh")

    preference = entry.get("condition_preference", list(_FIELD_DEFAULTS["condition_preference"]))
    if not isinstance(preference, list) or not all(
        isinstance(word, str) and CONDITION_NUMBER.search(word) for word in preference
    ):
        raise ConfigError(
            f"{where}: condition_preference must be a list of strings each naming a number, like '400-800'"
        )

    categories = entry.get("categories", list(_FIELD_DEFAULTS["categories"]))
    if not isinstance(categories, list) or not all(isinstance(word, str) and word.strip() for word in categories):
        raise ConfigError(f"{where}: categories must be a list of non-empty strings")
    if categories and entry.get("kind") != "text":
        raise ConfigError(f"{where}: categories is only meaningful for a text field, not a {entry.get('kind')!r} one")

    bare_number = choice("bare_number", get_args(BareNumberPolicy))
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
    figure_readable = entry.get("figure_readable", _FIELD_DEFAULTS["figure_readable"])
    if type(figure_readable) is not bool:
        raise ConfigError(f"{where}: figure_readable must be true or false, got {figure_readable!r}")
    if figure_readable and (not numeric or entry.get("canonical_unit") is None):
        raise ConfigError(f"{where}: figure_readable needs a numeric field with a canonical_unit")
    for key in ("display_format", "range_policy", "after_clause"):
        if key in entry and not numeric:
            raise ConfigError(f"{where}: {key} is only meaningful for a numeric field, not a {entry.get('kind')!r} one")
    display_format = choice("display_format", get_args(DisplayFormat))
    range_policy = choice("range_policy", get_args(RangePolicy))
    after_clause = choice("after_clause", get_args(AfterClause))
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
        after_clause=after_clause,  # type: ignore[arg-type]
    )


def _valid_range(entry: Mapping[str, Any], where: str) -> tuple[float | None, float | None]:
    if "valid_range" not in entry:
        return _FIELD_DEFAULTS["valid_range"]
    bounds = entry["valid_range"]
    if not isinstance(bounds, Mapping) or not set(bounds) <= {"min", "max"}:
        raise ConfigError(f"{where}: valid_range must be an object with 'min' and/or 'max', got {bounds!r}")
    if entry.get("kind") != "numeric":
        raise ConfigError(f"{where}: valid_range is only meaningful for a numeric field, not a {entry.get('kind')!r}")
    low, high = bounds.get("min"), bounds.get("max")
    for key, value in (("min", low), ("max", high)):
        if value is not None and type(value) not in (int, float):
            raise ConfigError(f"{where}: valid_range.{key} must be a number or null, got {value!r}")
    if low is None and high is None:
        raise ConfigError(f"{where}: valid_range needs at least one of 'min' and 'max'")
    if low is not None and high is not None and low >= high:
        raise ConfigError(f"{where}: valid_range.min ({low}) must be below valid_range.max ({high})")
    return (None if low is None else float(low), None if high is None else float(high))


# ---- Units -------------------------------------------------------------------------------------------------

OFFSET_UNITS = frozenset({"℃", "K"})
MAX_ALIASES = 50
MAX_IGNORED_SUFFIXES = 50
_UNIT_KEYS = ("aliases", "case_sensitive", "retrieval", "extends_builtin", "exclude")
_ALIAS_KEYS = ("factor", "offset")


def load_units(data: Any, where: str, ignored_suffixes: Any = ()) -> UnitRegistry:
    """A profile's ``units`` object and its ``ignored_unit_suffixes`` list, validated; ``where`` names the file in
    every error."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: units must be an object, got {type(data).__name__}")
    if (
        not isinstance(ignored_suffixes, list | tuple)
        or len(ignored_suffixes) > MAX_IGNORED_SUFFIXES
        or not all(isinstance(word, str) and word and not any(c.isspace() for c in word) for word in ignored_suffixes)
        or len(set(ignored_suffixes)) != len(ignored_suffixes)
    ):
        # No spaces: a quoted unit is compared with its spaces removed, so a suffix with one would never match.
        raise ConfigError(
            f"{where}: ignored_unit_suffixes must be a list of at most {MAX_IGNORED_SUFFIXES} distinct words"
            " without spaces"
        )
    declared = (_declared_unit(canonical, entry, f"{where}: units[{canonical!r}]") for canonical, entry in data.items())
    return UnitRegistry(tuple(declared), tuple(ignored_suffixes))


def _declared_unit(canonical: str, entry: Any, where: str) -> DeclaredUnit:
    if not canonical.strip():
        raise ConfigError(f"{where}: a unit needs a non-empty name")
    if not isinstance(entry, Mapping):
        raise ConfigError(f"{where} must be an object, got {type(entry).__name__}")
    unknown = sorted(set(entry) - set(_UNIT_KEYS))
    if unknown:
        raise ConfigError(f"{where} has unknown key(s) {', '.join(unknown)}; valid keys are {', '.join(_UNIT_KEYS)}")
    flags = {key: entry.get(key, False) for key in ("case_sensitive", "extends_builtin")}
    for key, value in flags.items():
        if type(value) is not bool:
            raise ConfigError(f"{where}: {key} must be true or false, got {value!r}")
    case_sensitive, extends = flags["case_sensitive"], flags["extends_builtin"]
    builtin = canonical in BUILTIN_CONVERTERS
    if builtin and not extends:
        # Redefining a built-in would silently change what every field in that unit converts to.
        raise ConfigError(f"{where}: {canonical!r} is a built-in unit; set extends_builtin to add spellings to it")
    if extends and not builtin:
        raise ConfigError(
            f"{where}: extends_builtin needs a built-in unit; the built-ins are {', '.join(BUILTIN_CONVERTERS)}"
        )
    exclude = _exclusions(canonical, entry.get("exclude"), extends, where) if "exclude" in entry else ()

    table = entry.get("aliases", {})
    # An extension may only take spellings away; everything else declares at least one.
    fewest = 0 if exclude else 1
    if not isinstance(table, Mapping) or not fewest <= len(table) <= MAX_ALIASES:
        raise ConfigError(f"{where}: aliases must be an object of {fewest} to {MAX_ALIASES} spellings")
    aliases: list[tuple[str, float, float]] = []
    seen: dict[str, str] = {}
    for spelling, value in table.items():
        key = fold_spelling(spelling, case_sensitive)
        if not key:
            raise ConfigError(f"{where}: aliases has an empty spelling {spelling!r}")
        if key in seen:
            raise ConfigError(f"{where}: aliases {seen[key]!r} and {spelling!r} are the same spelling once folded")
        seen[key] = spelling
        factor, offset = _alias_value(value, f"{where}: aliases[{spelling!r}]")
        if offset and canonical not in OFFSET_UNITS:
            raise ConfigError(
                f"{where}: aliases[{spelling!r}] has an offset; only {', '.join(sorted(OFFSET_UNITS))} take one"
            )
        aliases.append((key, factor, offset))
    if not extends and (fold_spelling(canonical, case_sensitive), 1.0, 0.0) not in aliases:
        # The canonical spelling itself must convert, or a value quoted in the very unit asked for is refused.
        raise ConfigError(f"{where}: aliases must list {canonical!r} itself with factor 1 and no offset")

    retrieval = entry.get("retrieval")
    if retrieval is None:
        retrieval = derive_retrieval(table) if table else None
    else:
        compile_pattern(retrieval, f"{where}: retrieval", re.IGNORECASE)
    return DeclaredUnit(
        canonical=canonical,
        aliases=tuple(aliases),
        case_sensitive=case_sensitive,
        retrieval=retrieval,
        extends_builtin=extends,
        exclude=exclude,
    )


def _exclusions(canonical: str, value: Any, extends: bool, where: str) -> tuple[str, ...]:
    """The built-in spellings an extension takes away. Each must be one the built-in reads -- by its converter,
    or by its retrieval pattern after a number -- and must no longer be found once taken away."""
    if not extends:
        raise ConfigError(f"{where}: exclude takes spellings away from a built-in unit; it needs extends_builtin")
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= MAX_ALIASES
        or not all(isinstance(spelling, str) and spelling.strip() for spelling in value)
    ):
        raise ConfigError(f"{where}: exclude must be a list of 1 to {MAX_ALIASES} non-empty spellings")
    convert, pattern = BUILTIN_CONVERTERS[canonical], BUILTIN_RETRIEVAL[canonical]
    remaining = UnitRegistry((DeclaredUnit(canonical, (), extends_builtin=True, exclude=tuple(value)),))
    for spelling in value:
        text = f"1 {normalize_text(spelling).lower()}"
        if convert(clean_unit(spelling)) is None and not pattern.search(text):
            raise ConfigError(f"{where}: exclude {spelling!r} is no spelling the built-in {canonical!r} reads")
        if remaining.retrieval(canonical).search(text):  # type: ignore[union-attr]
            raise ConfigError(
                f"{where}: exclude {spelling!r} is still found by the built-in retrieval pattern of {canonical!r}"
            )
    return tuple(value)


def _alias_value(value: Any, where: str) -> tuple[float, float]:
    """``(factor, offset)`` from a bare factor or a ``{"factor": ..., "offset": ...}`` object."""
    if isinstance(value, Mapping):
        if set(value) - set(_ALIAS_KEYS) or "factor" not in value:
            raise ConfigError(f"{where} must hold a factor and optionally an offset; valid keys are factor, offset")
        factor, offset = value["factor"], value.get("offset", 0)
    else:
        factor, offset = value, 0
    for name, number in (("factor", factor), ("offset", offset)):
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ConfigError(f"{where}: {name} must be a finite number, got {number!r}")
    if factor <= 0:
        raise ConfigError(f"{where}: factor must be greater than 0, got {factor!r}")
    return float(factor), float(offset)
