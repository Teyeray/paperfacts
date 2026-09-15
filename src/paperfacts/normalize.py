"""Turn the text the model transcribed into comparable canonical values: text folding, number parsing, unit
conversion, and applying all three to a lane.

Pure functions, millisecond-fast, the one layer that offers a determinism guarantee. The model only
transcribes (``value_raw`` / ``unit_raw``); every conversion happens here, because a model's unit conversion
is wrong *silently*, and the two lanes fail differently, which would flood CONFLICT with noise unrelated to
the parsers. This module's source is hashed into ``comparison_key``, so changing a rule invalidates stored
comparisons but never the stored extractions.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

from paperfacts.fields import FIELD_BY_NAME, FIELD_SPECS, FieldSpec
from paperfacts.records import FieldValue, LaneExtraction, TargetRecord

# ---- Text ------------------------------------------------------------------------------------------------
# Superscript digits are folded **before** NFKC, which would collapse "10⁻⁴" to "10-4" and lose the exponent.

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPT_RUN = re.compile(r"[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
# MinerU's Markdown writes sub/superscripts as HTML tags: SnO<sub>2</sub>, 10<sup>-4</sup>
_HTML_SUP = re.compile(r"<sup>\s*([^<]*?)\s*</sup>", re.IGNORECASE)
_HTML_SUB = re.compile(r"<sub>\s*([^<]*?)\s*</sub>", re.IGNORECASE)
# Only variants NFKC does not already fold (OHM SIGN, MICRO SIGN and NBSP are covered by NFKC).
_REPLACEMENTS = {
    "−": "-",  # minus sign U+2212
    "–": "-",  # en dash
    "—": "-",  # em dash
    "×": "x",  # multiplication sign
    "⋅": ".",  # dot operator U+22C5
    "·": ".",  # middle dot U+00B7
    "’": "'",
}
# Characters that carry meaning in a value: digits, letters, units and the punctuation inside numbers.
KEY_CHARACTERS = "0-9a-zΩμ%./:+-"
_NON_KEY = re.compile(f"[^{KEY_CHARACTERS}]+")
_SPACES = re.compile(r"\s+")


def _ascii_superscripts(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        run = match.group(0)
        sign = "-" if run.startswith("⁻") else ""
        digits = run.lstrip("⁺⁻").translate(_SUPERSCRIPTS)
        return f"^{sign}{digits}"

    return _SUPERSCRIPT_RUN.sub(repl, text)


def normalize_text(text: str) -> str:
    """Normalise spelling while preserving meaning: sub/superscripts to ASCII, Unicode variants folded,
    whitespace collapsed."""
    text = _HTML_SUP.sub(lambda m: f"^{m.group(1)}", text)
    text = _HTML_SUB.sub(lambda m: m.group(1), text)
    text = _ascii_superscripts(text).translate(_SUBSCRIPTS)
    text = unicodedata.normalize("NFKC", text)
    for source, target in _REPLACEMENTS.items():
        text = text.replace(source, target)
    return _SPACES.sub(" ", text).strip()


def normalize_key(text: str | None) -> str:
    """A key for "are these the same" comparisons of text, conditions and compositions. Never use it for
    units: lowercasing collides mΩ with MΩ; use :func:`clean_unit` there."""
    if not text:
        return ""
    # .lower() turns Ω into ω; put it back before the whitelist filter or Ω would be stripped.
    return _NON_KEY.sub("", normalize_text(text).lower().replace("ω", "Ω"))


# ---- Numbers ------------------------------------------------------------------------------------------------
# Returns ``(value, note)``: None when parsing fails, with the note saying why, so "why couldn't the two
# lanes be compared" stays traceable.

_NUM = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+\.\d*|\.\d+|\d+)"
# Scientific notation matches three explicit spellings only ("2108" must never be read as 2x10^8):
#   1.2 x 10^-4 / 1.2x10^-4 / 1.2 x 10-4  (an explicit "x 10" survives OCR losing the superscript)
#   10^-4                                  (no mantissa, so the "^" is mandatory)
#   1.2e-4 / 1.2E-4
_SCI = re.compile(
    rf"(?P<m>{_NUM})\s*x\s*10\s*\^?\s*(?P<e>[-+]?\d+)"
    rf"|10\s*\^\s*(?P<e1>[-+]?\d+)"
    rf"|(?P<e2m>{_NUM})[eE](?P<e2>[-+]?\d+)"
)
_RANGE = re.compile(rf"^(?P<a>{_NUM})\s*(?:-|to|~)\s*(?P<b>{_NUM})$")
_PLUS_MINUS = re.compile(rf"^(?P<a>{_NUM})\s*(?:\+/-|±|\+-)\s*{_NUM}")
_NUMBER = re.compile(_NUM)
# Multi-character qualifiers first, or "<=" is swallowed by the lone "<" in the character class.
_QUALIFIERS = re.compile(r"^(?P<q>>=|<=|ca\.?|approx\.?|about|[~≈≃≅≥≤<>])\s*", re.IGNORECASE)

_LATEX_MARKERS = ("$", "\\")
_LATEX_COMMANDS = {r"\times": " x ", r"\cdot": " x ", r"\,": " ", r"\;": " ", "\\ ": " "}
_DIGIT_GAP = re.compile(r"(?<=[0-9.])\s+(?=[0-9.])")
_CARET_GAP = re.compile(r"\^\s*([-+]?)\s*(?=\d)")


def delatex(text: str) -> str:
    """Undo the LaTeX MinerU produces for numbers in tables and formulas.

    ``6.4 × 10⁻³`` arrives as ``$6 . 4 \\times 1 0 ^ { - 3 }$``, a space between every character, and the
    prompt's "verbatim" rule keeps it that way. Spaces between digits are collapsed only when the text
    carries a LaTeX marker, so ordinary "10 20" is left alone.
    """
    text = re.sub(r"\^\s*\{\s*([-+]?\s*\d+)\s*\}", lambda m: "^" + m.group(1).replace(" ", ""), text)
    if not any(marker in text for marker in _LATEX_MARKERS):
        return text
    for command, replacement in _LATEX_COMMANDS.items():
        text = text.replace(command, replacement)
    text = text.replace("$", " ").replace("{", " ").replace("}", " ")
    text = _DIGIT_GAP.sub("", text)
    return _CARET_GAP.sub(r"^\1", text)


def parse_number(raw: str) -> tuple[float | None, str | None]:
    text = delatex(normalize_text(raw))
    notes: list[str] = []

    match = _QUALIFIERS.match(text)
    if match:
        notes.append(f"qualifier '{match.group('q')}' dropped")
        text = text[match.end() :].strip()

    # A parenthesised alternative ("12 (60)"): take the value outside the parentheses.
    if "(" in text:
        notes.append("parenthesized alternative ignored")
        text = re.sub(r"\([^)]*\)", " ", text).strip()

    sci = _SCI.search(text)
    if sci:
        if sci.group("e2") is not None:
            value = float(_plain(sci.group("e2m"))) * 10 ** int(sci.group("e2"))
        elif sci.group("e1") is not None:
            value = 10 ** int(sci.group("e1"))
        else:
            value = float(_plain(sci.group("m"))) * 10 ** int(sci.group("e"))
        return float(value), _join(notes)

    pm = _PLUS_MINUS.match(text)
    if pm:
        notes.append("uncertainty dropped")
        return float(_plain(pm.group("a"))), _join(notes)

    rng = _RANGE.match(text)
    if rng:
        a, b = float(_plain(rng.group("a"))), float(_plain(rng.group("b")))
        # Only a < b is a range; "10-4" is probably 10^-4 that lost its superscript and falls through.
        if a < b:
            notes.append(f"range {a:g}-{b:g} → midpoint")
            return (a + b) / 2, _join(notes)

    numbers = _NUMBER.findall(text)
    if not numbers:
        notes.append("no number found")
        return None, _join(notes)
    if len(numbers) > 1:
        notes.append(f"{len(numbers)} numbers found, first used")
    return float(_plain(numbers[0])), _join(notes)


def _plain(token: str) -> str:
    return token.replace(",", "")


def _join(notes: list[str]) -> str | None:
    return "; ".join(notes) if notes else None


# ---- Units -----------------------------------------------------------------------------------------------
# One recogniser per canonical unit, each handling only certain conversions. An unrecognised unit is never
# guessed: it returns None with a reason and the comparison layer decides AMBIGUOUS. What a bare number
# means is decided by ``FieldSpec.bare_number``, never by field name here.

# Case is meaningful (m = milli, M = mega): the regexes ignore case for the unit word, never for the prefix.
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


# Canonical unit -> "multiply by what to reach it".
CONVERTERS: dict[str, Converter] = {
    "Ω/sq": _by_pattern(_PER_SQUARE),
    "Ω·cm": _by_pattern(_RESISTIVITY),
    "nm": _by_table(_LENGTH),
    "min": _by_table(_TIME),
    "inch": _by_table(_SIZE),
    "%": _by_table(_PERCENT),
}

# Fail at import time rather than with a KeyError buried in normalisation, field by field.
_MISSING_CONVERTERS = {spec.canonical_unit for spec in FIELD_SPECS if spec.canonical_unit} - CONVERTERS.keys()
if _MISSING_CONVERTERS:
    raise RuntimeError(f"no converter registered for canonical unit(s): {sorted(_MISSING_CONVERTERS)}")


def clean_unit(unit_raw: str) -> str:
    """Whitespace and decoration stripped, case preserved; no interpretation."""
    return normalize_text(unit_raw).replace(" ", "").rstrip(".")


def convert_to_canonical(
    spec: FieldSpec, value: float, unit_raw: str | None
) -> tuple[float | None, str | None, str | None]:
    """``(canonical value, canonical unit, note)``; the value is None when conversion fails."""
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


# ---- Applying it to a lane ----------------------------------------------------------------------------------


def normalize_field(field: FieldValue, spec: FieldSpec) -> FieldValue:
    if spec.kind != "numeric":
        # Text and composition fields are compared through normalize_key on the fly.
        return field
    number, parse_note = parse_number(field.value_raw)
    if number is None:
        return field.model_copy(update={"value": None, "unit": None, "normalization_note": parse_note})
    value, unit, unit_note = convert_to_canonical(spec, number, field.unit_raw)
    note = "; ".join(n for n in (parse_note, unit_note) if n) or None
    return field.model_copy(update={"value": value, "unit": unit, "normalization_note": note})


def _normalize_fields(fields: tuple[FieldValue, ...]) -> tuple[FieldValue, ...]:
    # Fields outside the schema were dropped at extraction time; this is a defensive second check.
    return tuple(normalize_field(f, FIELD_BY_NAME[f.field]) if f.field in FIELD_BY_NAME else f for f in fields)


def normalize_lane(lane: LaneExtraction) -> LaneExtraction:
    """Fill in ``value`` / ``unit`` for every field. Pure and idempotent: always returns a new object."""
    target: TargetRecord | None = None
    if lane.target is not None:
        target = lane.target.model_copy(update={"fields": _normalize_fields(lane.target.fields)})
    samples = tuple(sample.model_copy(update={"fields": _normalize_fields(sample.fields)}) for sample in lane.samples)
    return lane.model_copy(update={"target": target, "samples": samples})
