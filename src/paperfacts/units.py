"""Units: the built-in converters and retrieval patterns, and the units a profile declares.

Two questions are asked of a unit, and each has its own table here. *Conversion* (``BUILTIN_CONVERTERS``) answers
"what do I multiply a value quoted in this spelling by to reach the canonical unit?", with anchored expressions
over a unit string the model already quoted. *Retrieval* (``BUILTIN_RETRIEVAL``) answers "does this block of
running text carry a number in this unit?", with looser expressions over :func:`paperfacts.passages.searchable`
text. A canonical unit a field may use needs both, which :meth:`UnitRegistry.check` enforces.

A profile may declare further units (:class:`DeclaredUnit`) as factor tables with an optional temperature
offset, or extend a built-in one with more spellings. The built-in tables are TCO's conventions and stay
exactly as they were measured; a new domain adds to them and never edits them.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

from paperfacts.errors import ConfigError
from paperfacts.text import clean_unit, normalize_text

# ---- Built-in converters ---------------------------------------------------------------------------------
# One recogniser per canonical unit, each handling only certain conversions. An unrecognised unit is never
# guessed: it returns None with a reason and the comparison layer decides AMBIGUOUS. What a bare number
# means is decided by ``FieldSpec.bare_number``, never by field name here.

# Case is meaningful (m = milli, M = mega): the regexes ignore case for the unit word, never for the prefix.
_PREFIX = {"": 1.0, "k": 1e3, "K": 1e3, "M": 1e6, "m": 1e-3, "μ": 1e-6, "n": 1e-9}
_OHM = r"(?:Ω|(?i:ohms?))"
# The separator may be "/", a dot (normalize_text folds "·" to "."), or the word "per"; "Ω/L" is OCR
# damage rather than a spelling of "per square" and stays unrecognised.
_PER_SQUARE = re.compile(rf"^(?P<p>[kKMmμn]?){_OHM}\s*(?:[./]|per)?\s*(?i:sq|square|□)\.?(?:\^?-1)?$")
# The separator class needs "-" because "Ω-cm" / "ohm-cm" is at least as common in papers as the dotted
# spellings; the hyphen survives where "·" and "⋅" are folded to "." by normalize_text.
_RESISTIVITY = re.compile(rf"^(?P<p>[kKMmμn]?){_OHM}\s*[.x*-]?\s*(?i:cm)$")
_LENGTH = {"nm": 1.0, "μm": 1e3, "um": 1e3, "mm": 1e6, "cm": 1e7, "å": 0.1, "angstrom": 0.1}
_TIME = {
    "min": 1.0,
    "mins": 1.0,
    "minute": 1.0,
    "minutes": 1.0,
    "h": 60.0,
    "hr": 60.0,
    "hrs": 60.0,
    "hour": 60.0,
    "hours": 60.0,
    "s": 1 / 60,
    "sec": 1 / 60,
    "seconds": 1 / 60,
}
# A paper writes inches as a double prime; OCR renders it as one of four characters.
_INCH_MARKS = ('"', "''", "″", "′′")
_SIZE = {"inch": 1.0, "inches": 1.0, "in": 1.0, "mm": 1 / 25.4, "cm": 1 / 2.54} | dict.fromkeys(_INCH_MARKS, 1.0)
_PERCENT = {"%": 1.0, "percent": 1.0}
# NFKC folds ℃ (U+2103) to "°C" before the table's lowercased lookup, so one key catches all three
# spellings. Kelvin is deliberately absent: K → ℃ needs an offset (−273.15), not a factor, and this
# interface is a factor -- an unknown unit is reported as ambiguous rather than converted wrongly.
_TEMPERATURE = {"°c": 1.0, "c": 1.0}
# Distances in a deposition chamber; nm is left out on purpose: no target-holder gap is written in
# nanometres, and admitting it would misread every film thickness as a candidate distance.
_DISTANCE = {"cm": 1.0, "mm": 0.1, "m": 100.0, "μm": 1e-4, "um": 1e-4, "inch": 2.54, "in": 2.54} | dict.fromkeys(
    _INCH_MARKS, 2.54
)
# sccm is defined as cm³/min at standard conditions, so the two spellings are the same unit.
_FLOW = {"sccm": 1.0, "cm3/min": 1.0}
_ROTATION = {"rpm": 1.0, "r/min": 1.0, "rev/min": 1.0}
# Power prefixes are case-sensitive (mW ≠ MW), so the table's lowercasing cannot be used here.
_POWER = re.compile(r"^(?P<p>[kKMmμn]?)[Ww]$")
# Working pressure in Pa. "mPa" and "MPa" differ only in case, so those two are looked up as written and
# everything else case-folded.
_PRESSURE_EXACT = {"mPa": 1e-3, "MPa": 1e6}
_PRESSURE = {"pa": 1.0, "hpa": 100.0, "kpa": 1e3, "mbar": 100.0, "bar": 1e5, "torr": 133.322, "mtorr": 0.133322}


def _pressure(unit: str) -> float | None:
    return _PRESSURE_EXACT.get(unit, _PRESSURE.get(unit.lower()) if unit.lower() != "mpa" else None)


Converter = Callable[[str], float | None]


def _by_table(table: dict[str, float]) -> Converter:
    def convert(unit: str) -> float | None:
        return table.get(unit.lower())

    return convert


def _by_pattern(pattern: re.Pattern[str]) -> Converter:
    def convert(unit: str) -> float | None:
        match = pattern.match(unit)
        return None if match is None else _PREFIX[match.group("p")]

    return convert


# Canonical unit -> "multiply by what to reach it".
BUILTIN_CONVERTERS: dict[str, Converter] = {
    "Ω/sq": _by_pattern(_PER_SQUARE),
    "Ω·cm": _by_pattern(_RESISTIVITY),
    "nm": _by_table(_LENGTH),
    "min": _by_table(_TIME),
    "inch": _by_table(_SIZE),
    "%": _by_table(_PERCENT),
    "℃": _by_table(_TEMPERATURE),
    "cm": _by_table(_DISTANCE),
    "sccm": _by_table(_FLOW),
    "rpm": _by_table(_ROTATION),
    "W": _by_pattern(_POWER),
    "Pa": _pressure,
}


# ---- Built-in retrieval patterns ---------------------------------------------------------------------
# How each canonical unit is recognised **inside running text**. The converters above are anchored: they
# answer "is this whole string the unit?" for a value the model already quoted. Searching prose for a unit is
# a different question and needs looser expressions.
#
# A unit match ranks below a name match rather than being filtered out, because the two failure modes are
# not symmetric. "%" and "nm" appear in every paper, so treating them as proof would drown the prompt;
# refusing them outright loses the paper that writes "films of 2108 nm" without the word "thickness". As a
# weaker class they only fill places no named block wanted.
# Lowercase omega, not the ohm sign: searchable() lowercases, and "Ω".lower() is "ω". normalize_key has to
# undo the same fold for the same reason. Spelling it uppercase here would silently match only "ohm".
_OHM_TEXT = r"(?:ohms?|ω)"
# Every pattern runs on searchable() text, which is lower case: an upper-case letter in one never matches.
# "W" was written that way once, and sputtering_power went unasked in 22 of 54 lanes that said "60 W".
BUILTIN_RETRIEVAL: dict[str, re.Pattern[str]] = {
    "Ω/sq": re.compile(rf"{_OHM_TEXT}\s*(?:/|per)?\s*(?:sq|square|□)"),
    "Ω·cm": re.compile(rf"{_OHM_TEXT}\s*[.x*·-]?\s*cm"),
    # "4 in." and "2 inch" are target sizes; a bare "in" is the English word, so a digit must precede it.
    "inch": re.compile(r"\d\s*(?:inch|inches|in\.|\")"),
    "nm": re.compile(r"\d\s*(?:nm|µm|μm|um)\b"),
    "min": re.compile(r"\d\s*(?:min|mins|minutes?|h|hr|hrs|hours?|s|sec|secs|seconds?)\b"),
    "%": re.compile(r"\d\s*%"),
    # K is admitted as a retrieval signal even though the converter refuses it: a block saying "annealed
    # at 573 K" belongs in the prompt, and the honest ambiguous verdict is the comparison's job, not
    # retrieval's.
    "℃": re.compile(r"\d\s*(?:°\s*[ck]\b|℃|c\b|k\b)"),
    "cm": re.compile(r"\d\s*(?:cm|mm|m|µm|μm|um)\b"),
    "W": re.compile(r"\d\s*[km]?w\b"),
    "sccm": re.compile(r"\d\s*(?:sccm|slm)\b"),
    "rpm": re.compile(r"\d\s*(?:rpm|r/min)\b"),
    "Pa": re.compile(r"\d\s*(?:[mkh]?pa|m?torr|m?bar)\b"),
}


# ---- Declared units ------------------------------------------------------------------------------------------
# A profile's own units, as factor tables: ``value * factor + offset`` reaches the canonical unit. The offset is
# for temperature alone -- K to ℃ is the one conversion a factor cannot do -- and is never applied to a bare
# number, which has no unit to convert from.

OFFSET_UNITS = frozenset({"℃", "K"})
MAX_ALIASES = 50
MAX_PATTERN_LENGTH = 500
_UNIT_KEYS = ("aliases", "case_sensitive", "retrieval", "extends_builtin")
_ALIAS_KEYS = ("factor", "offset")


@dataclass(frozen=True)
class DeclaredUnit:
    """One unit a profile declares, or the spellings it adds to a built-in one (``extends_builtin``)."""

    canonical: str
    # (spelling as it is looked up, factor, offset). A spelling is cleaned the way a quoted unit is
    # (:func:`paperfacts.text.clean_unit`) and case-folded unless the unit is case-sensitive.
    aliases: tuple[tuple[str, float, float], ...]
    case_sensitive: bool = False
    # The pattern that finds this unit in running text: the author's, or the one the loader derives from the
    # spellings as written (:func:`derive_retrieval`), since the cleaned ones above have lost their spaces.
    retrieval: str | None = None
    extends_builtin: bool = False


@dataclass(frozen=True)
class UnitRegistry:
    """The built-in units plus what one profile declares."""

    declared: tuple[DeclaredUnit, ...] = ()
    # Words a paper writes after a unit to say whose quantity it is ("1.1 Pa Ar", "3 mTorr (O2)"): set aside
    # when the unit is otherwise unknown. A profile's own list (``ignored_unit_suffixes``); none by default.
    ignored_suffixes: tuple[str, ...] = ()

    def known(self) -> tuple[str, ...]:
        return (*BUILTIN_CONVERTERS, *(unit.canonical for unit in self.declared if not unit.extends_builtin))

    def knows(self, canonical: str) -> bool:
        return canonical in self.known()

    def has_retrieval(self, canonical: str) -> bool:
        return self.retrieval(canonical) is not None

    def convert(self, canonical: str, unit: str) -> tuple[float, float] | None:
        """``(factor, offset)`` taking a value quoted in ``unit`` to ``canonical`` as ``value * factor + offset``,
        or None when nothing here reads that spelling.

        A built-in unit's own converter is asked first, so an extension can add spellings to it but never change
        how one it already reads converts."""
        builtin = BUILTIN_CONVERTERS.get(canonical)
        factor = None if builtin is None else builtin(unit)
        if factor is not None:
            return factor, 0.0
        for declared in self.declared:
            if declared.canonical == canonical:
                key = fold_spelling(unit, declared.case_sensitive)
                for spelling, alias_factor, offset in declared.aliases:
                    if spelling == key:
                        return alias_factor, offset
        return None

    def retrieval(self, canonical: str) -> re.Pattern[str] | None:
        """The pattern that finds a number in ``canonical`` in :func:`paperfacts.passages.searchable` text.

        A built-in unit nothing extends keeps its own pattern object; an extension is searched beside it."""
        patterns = [unit.retrieval for unit in self.declared if unit.canonical == canonical and unit.retrieval]
        builtin = BUILTIN_RETRIEVAL.get(canonical)
        if not patterns:
            return builtin
        if builtin is not None:
            patterns.insert(0, builtin.pattern)
        if len(patterns) == 1:
            return _compiled(patterns[0])
        return _compiled("|".join(f"(?:{pattern})" for pattern in patterns))

    def without_ignored_suffix(self, unit: str) -> str:
        """``unit`` (already cleaned, so without spaces) with a trailing ignored suffix removed, bracketed or not."""
        if not self.ignored_suffixes:
            return unit
        return _suffix_pattern(self.ignored_suffixes).sub("", unit)

    def check(self, canonical: str, where: str) -> None:
        """Refuse a canonical unit nothing converts into, or one retrieval cannot find in running text."""
        if not self.knows(canonical):
            raise ConfigError(
                f"{where}: canonical_unit {canonical!r} has no converter; known units are {', '.join(self.known())}"
            )
        if not self.has_retrieval(canonical):
            raise ConfigError(f"{where}: canonical_unit {canonical!r} has no retrieval pattern")

    def material(self) -> list[dict[str, Any]]:
        """What the declared units and the ignored suffixes contribute to a fingerprint: nothing when a profile
        declares neither."""
        material = [dataclasses.asdict(unit) for unit in self.declared]
        if self.ignored_suffixes:
            material.append({"ignored_suffixes": list(self.ignored_suffixes)})
        return material


# The suffixes the built-in tables were measured with: the gases of a sputtering chamber. The TCO profile
# declares exactly these; they are here only for a caller that converts without a profile.
BUILTIN_IGNORED_SUFFIXES = ("Ar", "O2", "N2", "H2", "He", "Kr", "Xe", "air")
# What a caller that converts without a profile gets: the built-in units, read as they always were.
BUILTIN_UNITS = UnitRegistry(ignored_suffixes=BUILTIN_IGNORED_SUFFIXES)
MAX_IGNORED_SUFFIXES = 50


@cache
def _suffix_pattern(suffixes: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(rf"\(?(?:{'|'.join(re.escape(suffix) for suffix in suffixes)})\)?$")


@cache
def _compiled(pattern: str) -> re.Pattern[str]:
    # Searched text is lower-cased, so an author's "mAh" would otherwise never match and nothing would say why.
    return re.compile(pattern, re.IGNORECASE)


def derive_retrieval(spellings: Iterable[str]) -> str:
    """A retrieval pattern for spellings as an author wrote them.

    Each spelling must follow a digit, is folded the way :func:`paperfacts.passages.searchable` folds text, has
    its runs of spaces made optional (papers write "mAh g-1" and "mAhg-1" alike), and is closed by a word
    boundary when it ends in a letter or digit, so "V" does not match the start of "Vis"."""
    alternatives = []
    for spelling in spellings:
        folded = normalize_text(spelling).lower()
        boundary = r"\b" if folded[-1:].isalnum() else ""
        alternatives.append(r"\s*".join(re.escape(part) for part in folded.split(" ")) + boundary)
    return rf"\d\s*(?:{'|'.join(alternatives)})"


def compile_pattern(pattern: Any, where: str, flags: int = 0) -> re.Pattern[str]:
    """A regular expression from a profile: a string of at most ``MAX_PATTERN_LENGTH`` characters that compiles."""
    if not isinstance(pattern, str) or not pattern or len(pattern) > MAX_PATTERN_LENGTH:
        raise ConfigError(f"{where} must be a regular expression of 1 to {MAX_PATTERN_LENGTH} characters")
    try:
        return re.compile(pattern, flags)
    except re.error as exc:
        raise ConfigError(f"{where} is not a valid regular expression: {exc}") from exc


def fold_spelling(spelling: str, case_sensitive: bool) -> str:
    """A declared unit's lookup key: how an alias is stored and how a quoted unit must be looked up."""
    cleaned = clean_unit(spelling)
    return cleaned if case_sensitive else cleaned.casefold()


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

    table = entry.get("aliases")
    if not isinstance(table, Mapping) or not 1 <= len(table) <= MAX_ALIASES:
        raise ConfigError(f"{where}: aliases must be an object of 1 to {MAX_ALIASES} spellings")
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
        retrieval = derive_retrieval(table)
    else:
        compile_pattern(retrieval, f"{where}: retrieval", re.IGNORECASE)
    return DeclaredUnit(
        canonical=canonical,
        aliases=tuple(aliases),
        case_sensitive=case_sensitive,
        retrieval=retrieval,
        extends_builtin=extends,
    )


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
