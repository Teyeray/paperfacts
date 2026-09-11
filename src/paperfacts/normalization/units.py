"""Unit recognition and conversion: fold the many unit spellings found in papers down to each field's
canonical unit.

One recognizer per canonical unit, each handling only **certain** conversions (Ω/□ is synonymous with
Ω/sq, mΩ·cm -> Ω·cm, μm -> nm, ...); an unrecognized unit is never guessed at — it returns None with a
reason and lets the comparison layer decide AMBIGUOUS.
What to do when there is no unit at all is decided by :attr:`FieldSpec.bare_number`, not special-cased
here by field name.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from paperfacts.extraction.fields import FIELD_SPECS, FieldSpec
from paperfacts.normalization.text import normalize_text

# SI prefix -> multiplier. Case is meaningful (m = milli, M = mega), so the regexes below ignore case only
# for the unit word itself, never for the prefix.
_PREFIX = {"": 1.0, "k": 1e3, "K": 1e3, "M": 1e6, "m": 1e-3, "μ": 1e-6, "n": 1e-9}
_OHM = r"(?:Ω|(?i:ohms?))"
_PER_SQUARE = re.compile(rf"^(?P<p>[kKMmμn]?){_OHM}\s*(?:/|per)?\s*(?i:sq|square|□)\.?(?:\^?-1)?$")
_RESISTIVITY = re.compile(rf"^(?P<p>[kKMmμn]?){_OHM}\s*[.x*]?\s*(?i:cm)$")
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
_SIZE = {"inch": 1.0, "inches": 1.0, "in": 1.0, '"': 1.0, "mm": 1 / 25.4, "cm": 1 / 2.54}
_PERCENT = {"%": 1.0, "percent": 1.0}

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


# Canonical unit -> recognizer (returns "multiply by what to reach the canonical unit").
# Ω/sq and Ω·cm are prefix x several spellings, so they use regexes; length/time/size/percent are finite
# enumerations, so they use lookup tables.
CONVERTERS: dict[str, Converter] = {
    "Ω/sq": _by_pattern(_PER_SQUARE),
    "Ω·cm": _by_pattern(_RESISTIVITY),
    "nm": _by_table(_LENGTH),
    "min": _by_table(_TIME),
    "inch": _by_table(_SIZE),
    "%": _by_table(_PERCENT),
}

# Every canonical unit in the field table must have a recognizer; fail at import time rather than with a
# KeyError buried deep inside normalization, field by field.
_MISSING_CONVERTERS = {spec.canonical_unit for spec in FIELD_SPECS if spec.canonical_unit} - CONVERTERS.keys()
if _MISSING_CONVERTERS:
    raise RuntimeError(f"no converter registered for canonical unit(s): {sorted(_MISSING_CONVERTERS)}")


def clean_unit(unit_raw: str) -> str:
    """The unit string with whitespace and decorative characters stripped (no interpretation; case preserved)."""
    return normalize_text(unit_raw).replace(" ", "").rstrip(".")


def convert_to_canonical(
    spec: FieldSpec, value: float, unit_raw: str | None
) -> tuple[float | None, str | None, str | None]:
    """Return ``(canonical value, canonical unit, note)``; the canonical value is None when conversion fails."""
    canonical = spec.canonical_unit
    if canonical is None:
        return value, None, None
    if unit_raw is None:
        return _bare_number(spec, value)

    factor = CONVERTERS[canonical](clean_unit(unit_raw))
    if factor is None:
        return None, None, f"unknown unit {unit_raw!r} for {canonical}"
    return value * factor, canonical, None


def _bare_number(spec: FieldSpec, value: float) -> tuple[float | None, str | None, str | None]:
    canonical = spec.canonical_unit
    match spec.bare_number:
        case "percent_or_fraction":
            if 0.0 <= value <= 1.0:
                return value * 100.0, canonical, "no unit; value ≤ 1 read as a fraction"
            return value, canonical, "no unit; read as percent"
        case "assume_canonical":
            return value, canonical, f"no unit; assumed {canonical}"
        case "reject":
            return None, None, f"no unit; {spec.name} requires one"
