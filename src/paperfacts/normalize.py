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
from functools import cache

from paperfacts.fields import FIELD_BY_NAME, FIELD_SPECS, FieldSpec
from paperfacts.records import ExtractedRecords, FieldValue, LaneExtraction, TargetRecord

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
    "‐": "-",  # hyphen U+2010 (NFKC also folds the non-breaking hyphen U+2011 to it)
    "‒": "-",  # figure dash
    "―": "-",  # horizontal bar
    "×": "x",  # multiplication sign
    "⋅": ".",  # dot operator U+22C5
    "·": ".",  # middle dot U+00B7
    "’": "'",
    "∼": "~",  # tilde operator U+223C, what papers actually print for "approximately"
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


# ---- Closed category sets --------------------------------------------------------------------------------
# A text field may declare a closed set of answers (FieldSpec.categories). Papers write one mode many ways --
# "DC and RF", "DC and RF co-sputtering", "DC and RF magnetron co-sputtering" -- and raw text equality reads
# those as three different modes, so the two lanes CONFLICT over a difference that is only phrasing.
#
# A category is identified by the tokens its own name contains: "pulsed DC" is {pulsed, dc}, "DC+RF" is
# {dc, rf}, "DC" is {dc}. A raw value is reduced to whichever of those tokens it mentions, everything else
# ("magnetron", "co-sputtering", "and") being vocabulary the set does not define, and matches the category
# whose token set it reproduces exactly. So "DC and RF magnetron co-sputtering" is {dc, rf} -> "DC+RF",
# while "DC" stays {dc} -> "DC" and can never equal "RF". A value mentioning no token at all, or a
# combination no category names, resolves to None and is compared as ordinary text -- never guessed into
# the nearest category.

_CATEGORY_TOKEN = re.compile(r"[a-z0-9]+")


@cache
def _category_index(categories: tuple[str, ...]) -> dict[frozenset[str], str]:
    """Token set -> canonical spelling. The first category claiming a token set keeps it."""
    index: dict[frozenset[str], str] = {}
    for category in categories:
        index.setdefault(frozenset(_CATEGORY_TOKEN.findall(category.lower())), category)
    return index


def canonical_category(categories: tuple[str, ...], raw: str | None) -> str | None:
    """The canonical spelling ``raw`` names, or None when it names none of them."""
    if not categories or not raw:
        return None
    index = _category_index(categories)
    vocabulary = frozenset().union(*index.keys())
    present = frozenset(token for token in _CATEGORY_TOKEN.findall(normalize_text(raw).lower()) if token in vocabulary)
    return index.get(present)


def text_key(spec: FieldSpec, raw: str | None) -> str:
    """The key deciding whether two text values are the same fact: the canonical category when the field has
    a closed set and the value names one of them, the folded text otherwise. The NUL prefix keeps a category
    from ever colliding with a value whose folded text happens to spell it."""
    category = canonical_category(spec.categories, raw)
    return f"\0category:{category}" if category is not None else normalize_key(raw)


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
# A unit token at the very end of a value: letters, Ω, μ, % with optional "." or "/" inside ("vol.%").
# A digit, "-" or "x" anywhere disqualifies it, so "1.2 x 10^-4" and "40 x 10 cm" can never be stripped.
_TRAILING_UNIT = re.compile(r"[a-zA-ZΩμ%]+(?:[./][a-zA-ZΩμ%]+)*$")
_PLUS_MINUS = re.compile(rf"^(?P<a>{_NUM})\s*(?:\+/-|±|\+-)\s*{_NUM}")
NUMBER_RE = re.compile(_NUM)
"""Every plain number in a piece of text. Public because the comparison layer reads the numbers out of a
measurement condition ("550 nm") and must use the same notion of "a number" this module parses with."""
# Multi-character qualifiers first, or "<=" is swallowed by the lone "<" in the character class.
_QUALIFIERS = re.compile(
    r"^(?P<q>>=|<=|approximately|approx\.?|roughly|around|about|circa|ca\.?|[~≈≃≅≥≤<>])\s*", re.IGNORECASE
)

_LATEX_MARKERS = ("$", "\\")
# \Omega and \mu are unit symbols rather than spacing: a cell reading "\times 10^{-4} \Omega cm" is a
# resistivity, and without them the unit is unrecognised.
_LATEX_COMMANDS = {
    r"\times": " x ",
    r"\cdot": " x ",
    r"\pm": "±",
    r"\Omega": "Ω",
    r"\omega": "Ω",
    r"\mu": "μ",
    r"\,": " ",
    r"\;": " ",
    "\\ ": " ",
}
_DIGIT_GAP = re.compile(r"(?<=[0-9.])\s+(?=[0-9.])")
# The LaTeX spacing signature: a run of single characters, each a digit or a lone ".", separated by single
# spaces ("4 0 0", "8 . 4", "1 0"). Two multi-digit numbers ("300 500", "40 x 10") never look like this, so
# collapsing the run cannot merge two genuinely separate numbers. Two single-digit numbers ("2 5") do look
# like it and are read as 25: in a table cell that is the right reading, and it is the accepted trade-off.
_SPACED_DIGITS = re.compile(r"(?<![0-9.])[0-9.](?: [0-9.])+(?![0-9.])")
_CARET_GAP = re.compile(r"\^\s*([-+]?)\s*(?=\d)")


def delatex(text: str) -> str:
    """Undo the LaTeX MinerU produces for numbers in tables and formulas.

    ``6.4 × 10⁻³`` arrives as ``$6 . 4 \\times 1 0 ^ { - 3 }$``, a space between every character, and the
    prompt's "verbatim" rule keeps it that way. Spaces between digits are collapsed only when the text
    carries a LaTeX marker, so ordinary "10 20" is left alone.
    """
    text = re.sub(r"\^\s*\{\s*([-+]?\s*\d+)\s*\}", lambda m: "^" + m.group(1).replace(" ", ""), text)
    # MinerU drops the LaTeX markers from some cells ("4 0 0 °C", "1 0 ^ { - 4 }"), so this run has to be
    # collapsed on its own signature rather than on the presence of "$" or a backslash.
    text = _SPACED_DIGITS.sub(lambda m: m.group(0).replace(" ", ""), text)
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

    # "15.6 to 16.3 nm": the trailing unit made the anchored range match fail. Strip exactly one trailing
    # unit token and try again; the token restrictions above keep the scientific-notation and
    # multi-number spellings on their existing paths.
    stripped = _TRAILING_UNIT.sub("", text, count=1).rstrip()
    if stripped != text:
        rng = _RANGE.match(stripped)
        if rng:
            a, b = float(_plain(rng.group("a"))), float(_plain(rng.group("b")))
            if a < b:
                token = text[len(stripped) :].strip()
                notes.append(f"trailing unit {token!r} in value ignored")
                notes.append(f"range {a:g}-{b:g} → midpoint")
                return (a + b) / 2, _join(notes)

    numbers = NUMBER_RE.findall(text)
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
    "℃": _by_table(_TEMPERATURE),
    "cm": _by_table(_DISTANCE),
    "sccm": _by_table(_FLOW),
    "rpm": _by_table(_ROTATION),
    "W": _by_pattern(_POWER),
}

# Fail at import time rather than with a KeyError buried in normalisation, field by field.
_MISSING_CONVERTERS = {spec.canonical_unit for spec in FIELD_SPECS if spec.canonical_unit} - CONVERTERS.keys()
if _MISSING_CONVERTERS:
    raise RuntimeError(f"no converter registered for canonical unit(s): {sorted(_MISSING_CONVERTERS)}")


# A power-of-ten factor written into the unit: "×10^-4 Ω·cm", "x10-4Ω.cm", "10^-4Ω.cm" (normalize_text has
# already folded "×" to "x" and superscript digits to "^-4"). The caret, or an explicit "x10", is required,
# so a unit that merely starts with digits can never be read as a factor.
_SCALE_FACTOR = re.compile(r"^(?:x\s*10\s*\^?|10\s*\^)\s*(?P<e>[-+]?\d+)")


def split_scale_factor(unit_raw: str) -> tuple[float, str]:
    """``(factor, unit)``: a scale factor written into the unit belongs to the value, not to the unit.

    Papers head a table column "ρ (×10⁻⁴ Ω·cm)" and the model transcribes the whole parenthesis as the unit,
    leaving the value a bare "19.4". Without this the unit is unrecognised and the fact is lost.
    """
    unit = clean_unit(delatex(normalize_text(unit_raw)))
    match = _SCALE_FACTOR.match(unit)
    if match is None:
        return 1.0, unit
    return 10.0 ** int(match.group("e")), unit[match.end() :].lstrip(".x*")


def has_scale_factor(text: str) -> bool:
    """Whether a transcribed *value* already carries its own power of ten ("1.2 x 10^-4", "1.2e-4")."""
    return _SCI.search(delatex(normalize_text(text))) is not None


def clean_unit(unit_raw: str) -> str:
    """Whitespace and decoration stripped, case preserved; no interpretation."""
    return normalize_text(unit_raw).replace(" ", "").rstrip(".")


def convert_to_canonical(
    spec: FieldSpec, value: float, unit_raw: str | None, *, value_text: str | None = None
) -> tuple[float | None, str | None, str | None]:
    """``(canonical value, canonical unit, note)``; the value is None when conversion fails.

    ``value_text`` is the raw text ``value`` was parsed from. It is only consulted to detect a power of ten
    written twice, once in the value and once in the unit, which no reading can resolve.
    """
    canonical = spec.canonical_unit
    if canonical is None:
        return value, None, None
    if unit_raw is None:
        return _bare_number(spec, value)
    scale, unit = split_scale_factor(unit_raw)
    if scale != 1.0 and value_text is not None and has_scale_factor(value_text):
        # "1.2 × 10⁻⁴" under a column headed "(×10⁻⁴ Ω·cm)" is either 1.2e-4 or 1.2e-8 depending on whether
        # the author applied the header. Applying the factor twice would manufacture a value; refuse.
        return None, None, "scale factor in both value and unit; ambiguous"
    scale_note = f"scale factor {scale:g} taken from the unit" if scale != 1.0 else None
    value *= scale
    if not unit:
        canonical_value, canonical_unit, note = _bare_number(spec, value)
        return canonical_value, canonical_unit, _join([n for n in (scale_note, note) if n])
    factor = CONVERTERS[canonical](unit)
    if factor is None:
        return None, None, f"unknown unit {unit_raw!r} for {canonical}"
    return value * factor, canonical, scale_note


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
    value, unit, unit_note = convert_to_canonical(spec, number, field.unit_raw, value_text=field.value_raw)
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
    # Unattributed values are compared now, so they need canonical values like every other; leaving them
    # raw would silently turn every such comparison into "unparsed" and bury real agreements.
    unattributed = _normalize_fields(lane.unattributed)
    return lane.model_copy(update={"target": target, "samples": samples, "unattributed": unattributed})


def drop_implausible(records: ExtractedRecords) -> ExtractedRecords:
    """Drop every value whose converted number falls outside its field's ``valid_range``, with the reason.

    The range lives in the canonical unit, so this has to run on the converted value: "2 μm" is outside a
    500 nm ceiling although its digits are not. A value that cannot be converted is kept, since there is no
    number to judge and the comparison already reports it as unparsed.
    """
    dropped: list[str] = []

    def plausible(value: FieldValue) -> bool:
        spec = FIELD_BY_NAME.get(value.field)
        if spec is None or spec.describe_range() is None:
            return True
        number = normalize_field(value, spec).value
        if number is None or spec.in_range(number):
            return True
        unit = f" {value.unit_raw}" if value.unit_raw else ""
        dropped.append(
            f"{spec.name}: {value.value_raw!r}{unit} is {number:g} {spec.canonical_unit}, "
            f"outside the plausible range ({spec.describe_range()})"
        )
        return False

    def kept(values: tuple[FieldValue, ...]) -> tuple[FieldValue, ...]:
        return tuple(value for value in values if plausible(value))

    target = records.target
    if target is not None:
        target = target.model_copy(update={"fields": kept(target.fields)})
    samples = tuple(sample.model_copy(update={"fields": kept(sample.fields)}) for sample in records.samples)
    unattributed = kept(records.unattributed)
    if not dropped:
        return records
    return records.model_copy(
        update={
            "target": target,
            "samples": samples,
            "unattributed": unattributed,
            "dropped": (*records.dropped, *dropped),
        }
    )
