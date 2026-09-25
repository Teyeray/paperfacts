"""Units: the built-in converters and retrieval patterns, and the units a profile declares.

Two questions are asked of a unit, and each has its own table here. *Conversion* (``BUILTIN_CONVERTERS``) answers
"what do I multiply a value quoted in this spelling by to reach the canonical unit?", with anchored expressions
over a unit string the model already quoted. *Retrieval* (``BUILTIN_RETRIEVAL``) answers "does this block of
running text carry a number in this unit?", with looser expressions over :func:`paperfacts.passages.searchable`
text. A canonical unit a field may use needs both, which :meth:`UnitRegistry.check` enforces.

A profile may declare further units (:class:`DeclaredUnit`) as factor tables with an optional temperature
offset, or extend a built-in one with more spellings and take away the built-in spellings its domain reads
otherwise. The built-in tables are TCO's conventions and stay exactly as they were measured; a new domain
changes what it sees of them through its declarations and never edits them. Nothing here is a default: every
conversion is asked of the :class:`UnitRegistry` a profile was loaded with.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from typing import Any

from paperfacts.errors import ConfigError
from paperfacts.text import clean_unit, is_word_edge, normalize_text

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
# number, which has no unit to convert from. They are read and validated by :mod:`paperfacts.profile_loader`.


@dataclass(frozen=True)
class DeclaredUnit:
    """One unit a profile declares, or what it changes about a built-in one (``extends_builtin``)."""

    canonical: str
    # (spelling as it is looked up, factor, offset). A spelling is cleaned the way a quoted unit is
    # (:func:`paperfacts.text.clean_unit`) and case-folded unless the unit is case-sensitive.
    aliases: tuple[tuple[str, float, float], ...]
    case_sensitive: bool = False
    # The pattern that finds this unit in running text: the author's, or the one the loader derives from the
    # spellings as written (:func:`derive_retrieval`), since the cleaned ones above have lost their spaces.
    retrieval: str | None = None
    extends_builtin: bool = False
    # Built-in spellings an extension takes away, as written and matched case-insensitively: the built-in
    # converter no longer reads them, and its retrieval pattern no longer finds them after a number. A built-in
    # table is one domain's conventions -- "1 C" is a temperature to one group and a C-rate to another -- and an
    # extension that could only add would leave the other group's reading wrong.
    exclude: tuple[str, ...] = ()

    def material(self) -> dict[str, Any]:
        """This unit's part of a fingerprint. ``exclude`` only when it removes something, so a declaration that
        excludes nothing keeps the fingerprint it had before exclusions existed."""
        material = dataclasses.asdict(self)
        if not self.exclude:
            del material["exclude"]
        return material


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

    def excluded(self, canonical: str) -> tuple[str, ...]:
        """The built-in spellings of ``canonical`` this profile's extensions take away, as written."""
        return tuple(spelling for unit in self.declared if unit.canonical == canonical for spelling in unit.exclude)

    def convert(self, canonical: str, unit: str) -> tuple[float, float] | None:
        """``(factor, offset)`` taking a value quoted in ``unit`` to ``canonical`` as ``value * factor + offset``,
        or None when nothing here reads that spelling.

        A built-in unit's own converter is asked first, so an extension can add spellings to it but never change
        how one it already reads converts -- unless it excludes that spelling, which the built-in then refuses."""
        builtin = BUILTIN_CONVERTERS.get(canonical)
        excluded = self.excluded(canonical)
        if excluded and fold_spelling(unit, False) in {fold_spelling(spelling, False) for spelling in excluded}:
            builtin = None
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

        A built-in unit nothing extends keeps its own pattern object; an extension is searched beside it, and an
        excluded spelling is refused where the built-in pattern would start matching it."""
        patterns = [unit.retrieval for unit in self.declared if unit.canonical == canonical and unit.retrieval]
        excluded = self.excluded(canonical)
        builtin = BUILTIN_RETRIEVAL.get(canonical)
        if not patterns and not excluded:
            return builtin
        if builtin is not None:
            source = builtin.pattern
            if excluded:
                # The lookahead starts at the number, as the derived pattern of the excluded spellings does, so it
                # refuses exactly those matches of a built-in pattern that also starts there. The loader checks,
                # for each excluded spelling, that it is no longer found.
                source = f"(?!{derive_retrieval(excluded)})(?:{source})"
            patterns.insert(0, source)
        if not patterns:
            return None
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
        material = [unit.material() for unit in self.declared]
        if self.ignored_suffixes:
            material.append({"ignored_suffixes": list(self.ignored_suffixes)})
        return material


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
    boundary when it ends in a letter or digit of a spaced script (:func:`paperfacts.text.is_word_edge`), so
    "V" does not match the start of "Vis"."""
    alternatives = []
    for spelling in spellings:
        folded = normalize_text(spelling).lower()
        boundary = r"\b" if is_word_edge(folded[-1:]) else ""
        alternatives.append(r"\s*".join(re.escape(part) for part in folded.split(" ")) + boundary)
    return rf"\d\s*(?:{'|'.join(alternatives)})"


def fold_spelling(spelling: str, case_sensitive: bool) -> str:
    """A declared unit's lookup key: how an alias is stored and how a quoted unit must be looked up."""
    cleaned = clean_unit(spelling)
    return cleaned if case_sensitive else cleaned.casefold()
