"""Turn the text the model transcribed into comparable canonical values: number parsing, unit conversion, and
applying both to a lane. The text folding they share is :mod:`paperfacts.text`; the unit tables are
:mod:`paperfacts.units`.

Pure functions, millisecond-fast, the one layer that offers a determinism guarantee. The model only
transcribes (``value_raw`` / ``unit_raw``); every conversion happens here, because a model's unit conversion
is wrong *silently*, and the two lanes fail differently, which would flood CONFLICT with noise unrelated to
the parsers. This module's source is hashed into both keys: into ``comparison_key`` because it decides
verdicts, and into ``extractor_key`` because extraction drops implausible values it converts here. The LLM
cache is keyed by request payload, so a rule change re-derives stored extractions from cached answers
without asking the model again.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from functools import cache

from paperfacts.fields import FieldSpec, RangePolicy
from paperfacts.profile import DomainProfile
from paperfacts.records import ExtractedRecords, FieldValue, LaneExtraction, TargetRecord, spell_number_word
from paperfacts.text import LATEX_WRAPPERS, clean_unit, delatex, normalize_key, normalize_text
from paperfacts.units import BUILTIN_CONVERTERS as CONVERTERS  # noqa: F401 -- see below
from paperfacts.units import BUILTIN_UNITS, UnitRegistry

# CONVERTERS is not used here any more: tests/fixtures/units/generate.py imports it from this module, and that
# generator is frozen together with the recording it made.

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

_UNSIGNED = r"(?:\d{1,3}(?:,\d{3})+|\d+\.\d*|\.\d+|\d+)"
_NUM = rf"[-+]?{_UNSIGNED}"
# A mantissa may carry a sign only at the very start of the value: anywhere else a "-" in front of it is
# the dash of a range ("1.2-1.5 × 10^-3"), and reading it as a minus sign turns a range into a negative number.
_MANTISSA_NUM = rf"(?:^[-+])?{_UNSIGNED}"
# Scientific notation matches three explicit spellings only ("2108" must never be read as 2x10^8):
#   1.2 x 10^-4 / 1.2x10^-4 / 1.2 x 10-4  (an explicit "x 10" survives OCR losing the superscript)
#   10^-4                                  (no mantissa, so the "^" is mandatory)
#   1.2e-4 / 1.2E-4
_SCI = re.compile(
    rf"(?P<m>{_MANTISSA_NUM})\s*x\s*10\s*\^?\s*(?P<e>[-+]?\d+)"
    rf"|10\s*\^\s*(?P<e1>[-+]?\d+)"
    rf"|(?P<e2m>{_MANTISSA_NUM})[eE](?P<e2>[-+]?\d+)"
)
# The vocabulary every spelling below shares, spelled once.
_PM = r"(?:\+/-|±|\+-)"
_RANGE_SEP = r"(?:-|to|~)"
# A unit token inside a value: letters, Ω, μ, % with an optional "." or "/" inside ("vol.%"), or a degree.
# A digit, "-" or "x" on its own is no unit, so "1.2 x 10^-4" and "40 x 10 cm" never read as a range.
_UNIT_TOKEN = r"(?:°?[a-zA-ZΩμ%]+(?:[./][a-zA-ZΩμ%]+)*)"
_PLUS_MINUS_SIGN = re.compile(_PM)
_RANGE_SEPARATOR = re.compile(_RANGE_SEP)
# "10-20", "15.6 to 16.3 nm", "80%–85%", "500 °C to 530 °C": two bounds, each with an optional unit.
_RANGE = re.compile(
    rf"^(?P<a>{_NUM})\s*(?P<ua>{_UNIT_TOKEN})?\s*{_RANGE_SEP}\s*(?P<b>{_NUM})\s*(?P<ub>{_UNIT_TOKEN})?$"
)
# "(4.5 ± 0.2) × 10^-4": the parenthesis holds the mantissa and its uncertainty, the exponent applies to both.
_MANTISSA = re.compile(rf"^\(\s*(?P<m>{_NUM})\s*(?:(?P<pm>{_PM})\s*{_UNSIGNED}\s*)?\)\s*x\s*10\s*\^?\s*(?P<e>[-+]?\d+)")
_PARENTHESES = re.compile(r"\([^()]*\)")
_CARET_PARENS = re.compile(r"\^\s*\(\s*([-+]?\d+)\s*\)")
# "1:4", "20/1", "12/10/3": two numbers set against each other.
_RATIO = re.compile(r"\d\s*[:/]\s*\.?\d")
# Two numbers joined by a dash or tilde: a range, wherever it sits in the text.
_JOINED = re.compile(r"\d\s*[-~]\s*[\d.]")
_PLUS_MINUS = re.compile(rf"^(?P<a>{_NUM})\s*{_PM}\s*{_NUM}")
# Digits glued to a letter belong to a chemical formula ("O2", "SnO2", "H2") or a unit exponent ("cm2",
# "cm^-3"), never to the value: "O2/(Ar+O2) = 5%" read its first number as the 2 of "O2". e and x are left
# alone because they carry the exponent of "1.2e-4" and "1.2x10^-4". A sign needs the caret: "W-100" in
# "20 W-100 W" is a range, not an exponent.
_GLUED_DIGITS = re.compile(r"(?<=[A-DF-WYZa-df-wyzΩμ])(?:\^[-+]?)?\d+(?![\d.])")
# "O2/(Ar+O2) = 5%": what stands before "=" names the quantity; the value is what follows it.
_NAMED = re.compile(r"^[^=]*=\s*")
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
# "25 and 70": two values, not a value and a remark.
_CONJOINED = re.compile(r"\d\s*\S*\s+(?:and|or)\s+\d", re.IGNORECASE)
# "30, 40", "550 nm: 85%": numbers listed or labelled one by another, looked for only between two numbers,
# so the comma of a thousands separator ("1,200"), which is part of its number, is never one.
_LIST_SEPARATOR = re.compile(r"[,;:]")
# A number followed by a unit of its own and then another number ("140 nm ATO/25 nm", "3 h 30 min"): a
# second quantity stands beside the first, and which one is the value is not the parser's to guess. A lone
# "x" is the multiplication sign of "40 x 10 cm", not a unit.
_OWN_UNIT = re.compile(rf"^\s*(?!x\b){_UNIT_TOKEN}")
# Where a measurement or process condition stated after the value begins ("550 nm at 80%", "1.2 × 10^-4 at
# 300 K", "400 °C for 2 h", "500 °C under N2"): its numbers describe when the value was measured, not the value.
# Not "in": that is also the inch, and "2 in x 3 in" would read as 2. Not "after": see _AFTER.
_CONDITION = re.compile(r"\s+(?:at|@|for|during|under)\s+(?=.*\d)", re.IGNORECASE)
# "85% after 10 cycles", "100 nm after annealing": another state of the sample, not a condition of this value --
# unless the field says otherwise (FieldSpec.after_clause, read by split_after_clause).
_AFTER = re.compile(r"\s+after\s+\S", re.IGNORECASE)
# The words of a condition tail that may name a unit ("2 h", "550 nm", "°C").
_TAIL_WORD = re.compile(r"[^\d\s,;:()\[\]]+")
NUMBER_RE = re.compile(_NUM)
"""Every plain number in a piece of text. Public because the comparison layer reads the numbers out of a
measurement condition ("550 nm") and must use the same notion of "a number" this module parses with."""
_QUALIFIERS = re.compile(
    r"^(?P<q>>=|<=|approximately|approx\.?|roughly|around|about|circa|ca\.?|[~≈≃≅≥≤<>])\s*", re.IGNORECASE
)


def parse_number(raw: str, *, range_policy: RangePolicy = "midpoint") -> tuple[float | None, str | None]:
    """``(value, note)``: the number ``raw`` spells, or None with the reason it was refused.

    A qualifier ("~", ">", "about") is dropped and recorded first; what is left must then match one of the
    spellings in :data:`_SPELLINGS`, tried in order. Each spelling either claims the text -- with a value, or
    with a refusal -- or passes it on. A refusal is always better than a guess: the comparison turns None
    into AMBIGUOUS, while a wrong number is indistinguishable from a real measurement.

    ``range_policy`` is the field's (``FieldSpec.range_policy``): ``"midpoint"`` reads a range as its midpoint,
    ``"reject"`` refuses it, for a quantity whose range is a window rather than a scatter around one value (a
    cathode's "2.8–4.3 V" is the cycling window; its midpoint was never measured).
    """
    text, notes, _ = set_aside(raw)
    if _AFTER.search(text):
        return None, _join(
            [*notes, "a value stated 'after' a treatment belongs to another state of the sample; ambiguous"]
        )
    # The digits of a formula or a unit exponent are set aside before the value's own numbers are counted.
    unglued = _GLUED_DIGITS.sub(" ", text)
    if unglued != text:
        notes.append("digits of a formula or unit exponent ignored")
        text = unglued.strip()
    value, reading, is_range = _read(text)
    if range_policy == "reject" and is_range:
        refused = [
            f"{n.removesuffix(_MIDPOINT)} refused (range_policy 'reject')" if n.endswith(_MIDPOINT) else n
            for n in reading
        ]
        return None, _join([*notes, *refused])
    return value, _join([*notes, *reading])


def split_after_clause(text: str) -> tuple[str, str]:
    """``(value, clause)``: ``text`` cut where an "after ..." clause follows a number ("92.5% after 100 cycles" ->
    "92.5%", "after 100 cycles"), or ``(text, "")`` when there is none. Only for a field whose ``after_clause``
    is "condition"; every other field refuses such a value in :func:`parse_number`."""
    match = _AFTER.search(text)
    if match is None or not NUMBER_RE.search(text[: match.start()]):
        return text, ""
    return text[: match.start()].strip(), text[match.start() :].strip()


def _with_after_condition(field: FieldValue, clause: str) -> FieldValue:
    """``field`` with ``clause`` in its condition, unless the condition already says it: normalising twice must
    give the same value."""
    condition = field.condition
    if condition and normalize_key(clause) in normalize_key(condition):
        return field
    return field.model_copy(update={"condition": f"{condition}; {clause}" if condition else clause})


def set_aside(raw: str) -> tuple[str, list[str], str]:
    """``(value text, notes, condition)``: ``raw`` without what surrounds the value -- typesetting, a qualifier,
    a name before "=", a condition after it -- a note for each thing set aside, and the condition itself ("" when
    there is none). The same for every spelling, scientific, plain or compound, and for the dataset cell
    (``decide``), so none of them reads a condition's number as the value.

    A condition is set aside only where the value before it keeps a number: "deposited for 10 min" is the
    quote of a value that opens with its verb, not a condition with no value in front of it."""
    # "10^(-4)" is the caret spelling with its exponent bracketed, not a parenthesised alternative.
    text = _CARET_PARENS.sub(r"^\1", delatex(normalize_text(raw)))
    # A command delatex has no reading for is typesetting; left in place, its letters would glue to the
    # digits after it ("\\sim82").
    text = _LATEX_COMMAND.sub(" ", text).strip()
    notes: list[str] = []
    match = _QUALIFIERS.match(text)
    if match:
        notes.append(f"qualifier '{match.group('q')}' dropped")
        text = text[match.end() :].strip()
    named = _NAMED.match(text)
    if named and text[named.end() :]:
        notes.append(f"name {text[: named.end()].rstrip(' =')!r} before '=' ignored")
        text = text[named.end() :]
    for start in _CONDITION.finditer(text):
        if NUMBER_RE.search(text[: start.start()]):
            condition = text[start.start() :].strip()
            notes.append(f"condition {condition!r} ignored")
            return text[: start.start()].strip(), notes, condition
    return text, notes, ""


# (value, notes, is_range): is_range is set by the spellings that read a whole range as its midpoint, which is
# what range_policy 'reject' refuses.
_Reading = tuple[float | None, list[str], bool]
_MIDPOINT = " → midpoint"


def _read(text: str) -> _Reading:
    for spelling in _SPELLINGS:
        reading = spelling(text)
        if reading is not None:
            return reading
    raise AssertionError("the last spelling always answers")  # pragma: no cover


def _refuse(reason: str) -> _Reading:
    return None, [reason], False


def _ratio(text: str) -> _Reading | None:
    """ "1:4", "Ar:O2 = 9:1", "10/10", "12/10/3": a ratio is two numbers, and reading its first as the value is
    exactly the silent wrong answer this parser exists to prevent."""
    return _refuse("ratio notation a:b or a/b is not a single number") if _RATIO.search(text) else None


def _parenthesised_mantissa(text: str) -> _Reading | None:
    """ "(4.5 ± 0.2) × 10^-4": the parenthesis holds the mantissa, so it cannot be discarded as an alternative."""
    match = _MANTISSA.match(text)
    if match is None:
        return None
    if NUMBER_RE.search(text[match.end() :]):
        return _refuse("numbers outside the scientific notation; ambiguous")
    value = float(_plain(match.group("m"))) * 10 ** int(match.group("e"))
    return value, ["uncertainty dropped"] if match.group("pm") else [], False


def _leading_parenthesis(text: str) -> _Reading | None:
    """A parenthesis before any number is no alternative to a value outside it; there is nothing to prefer."""
    opening = text.find("(")
    if opening < 0 or NUMBER_RE.search(text[:opening]):
        return None
    return _refuse("parenthesis before the value; ambiguous")


def _parenthesised_alternative(text: str) -> _Reading | None:
    """ "12 (60)": an alternative value under other conditions; the primary one is outside the parentheses."""
    if "(" not in text and ")" not in text:
        return None
    rest = _PARENTHESES.sub(" ", text).strip()
    if "(" in rest or ")" in rest:
        return _refuse("unbalanced or nested parentheses; ambiguous")
    value, reading, is_range = _read(rest)
    return value, ["parenthesized alternative ignored", *reading], is_range


def _scientific(text: str) -> _Reading | None:
    """Scientific notation, alone or as a range or uncertainty of two. Any other number beside it -- the
    lower bound of "1.2-1.5 × 10^-3", whose exponent may or may not apply to it -- is refused."""
    matches = list(_SCI.finditer(text))
    if not matches:
        return None
    values = [_sci_value(match) for match in matches]
    rest = _cut(text, matches)
    if len(matches) == 2:
        between = text[matches[0].end() : matches[1].start()].strip()
        if not NUMBER_RE.search(rest) and _PLUS_MINUS_SIGN.fullmatch(between):
            return values[0], ["uncertainty dropped"], False
        if not NUMBER_RE.search(rest) and _RANGE_SEPARATOR.fullmatch(between):
            low, high = values
            if low < high:
                return (low + high) / 2, [f"range {low:g}-{high:g}{_MIDPOINT}"], True
            return _refuse("descending range in scientific notation; ambiguous")
    if len(matches) > 1 or NUMBER_RE.search(rest):
        return _refuse("numbers outside the scientific notation; ambiguous")
    return values[0], [], False


def _uncertainty(text: str) -> _Reading | None:
    match = _PLUS_MINUS.match(text)
    return None if match is None else (float(_plain(match.group("a"))), ["uncertainty dropped"], False)


def _range(text: str) -> _Reading | None:
    """ "10-20", "15.6 to 16.3 nm", "80%–85%": the midpoint, as for a range in scientific notation. Each bound
    may carry a unit, but not two different ones. A pair that does not ascend is refused: "10-4" is as likely
    10^-4 that lost its caret, and "300-200" is no range anybody writes."""
    match = _RANGE.match(text)
    if match is None:
        return None
    first, second = match.group("ua"), match.group("ub")
    if first and second and first != second:
        return _refuse(f"range bounds in different units ({first!r}, {second!r}); ambiguous")
    low, high = float(_plain(match.group("a"))), float(_plain(match.group("b")))
    if low >= high:
        return _refuse("descending range, or an exponent without its caret; ambiguous")
    unit = second or first
    notes = [f"trailing unit {unit!r} in value ignored"] if unit else []
    return (low + high) / 2, [*notes, f"range {low:g}-{high:g}{_MIDPOINT}"], True


def _first_number(text: str) -> _Reading:
    """The fallback: one number is the value. Several separated only by spaces or a multiplication sign ("300
    500", "40 x 10 cm") keep the first with a note. A range buried among other numbers, a list ("30, 40"), and a
    number carrying its own unit before another ("550 nm: 85%", "140 nm ATO/25 nm ITO") have no first value
    worth keeping: each is a second quantity beside the first, and they are refused."""
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return _refuse("no number found")
    if len(numbers) == 1:
        return float(_plain(numbers[0])), [], False
    if _JOINED.search(text):
        return _refuse("a range among other numbers; ambiguous")
    if _CONJOINED.search(text):
        return _refuse("two values joined by 'and' or 'or'; ambiguous")
    spans = list(NUMBER_RE.finditer(text))
    gaps = [text[a.end() : b.start()] for a, b in itertools.pairwise(spans)]
    if any(_LIST_SEPARATOR.search(gap) for gap in gaps):
        return _refuse("numbers separated by ',', ';' or ':'; ambiguous")
    if _OWN_UNIT.match(gaps[0]):
        return _refuse("a number with its own unit followed by another number; ambiguous")
    return float(_plain(numbers[0])), [f"{len(numbers)} numbers found, first used"], False


# Tried in order; the first to claim the text decides. The ratio and parenthesis checks come first because
# they change what the rest of the text means.
_SPELLINGS: tuple[Callable[[str], _Reading | None], ...] = (
    _ratio,
    _parenthesised_mantissa,
    _leading_parenthesis,
    _parenthesised_alternative,
    _scientific,
    _uncertainty,
    _range,
    _first_number,
)


def _sci_value(match: re.Match[str]) -> float:
    if match.group("e2") is not None:
        return float(_plain(match.group("e2m"))) * 10 ** int(match.group("e2"))
    if match.group("e1") is not None:
        return float(10 ** int(match.group("e1")))
    return float(_plain(match.group("m"))) * 10 ** int(match.group("e"))


def _cut(text: str, matches: list[re.Match[str]]) -> str:
    """``text`` with every match removed."""
    kept, start = [], 0
    for match in matches:
        kept.append(text[start : match.start()])
        start = match.end()
    kept.append(text[start:])
    return " ".join(kept)


def _plain(token: str) -> str:
    return token.replace(",", "")


def _join(notes: list[str]) -> str | None:
    return "; ".join(notes) if notes else None


# ---- Units -----------------------------------------------------------------------------------------------
# The converters are a paperfacts.units registry's: the built-ins plus what a profile declares. Every canonical
# unit was checked against it when the field table was loaded. What a bare number means is decided by
# ``FieldSpec.bare_number``, never by field name here.


# A power-of-ten factor in a transcribed table header: "×10^-4 Ω·cm", "x10-4Ω.cm", "10^-4Ω.cm", "ρ × 10^4"
# (normalize_text has already folded "×" to "x" and superscript digits to "^-4"). The caret, or an explicit
# "x10", is required, so a unit that merely starts with digits can never be read as a factor.
_SCALE_FACTOR = re.compile(r"(?:(?P<x>x)\s*10\s*\^?|10\s*\^)\s*(?P<e>[-+]?\d+)")
# The symbol or name of the quantity a header names before its factor: "ρ", "R_s", "Resistivity", "\rho".
_QUANTITY_SYMBOL = re.compile(r"\\?[A-Za-z\u0370-\u03ffΩμ□_]+")
_OPENING, _CLOSING = "([", ")]"


def split_scale_factor(unit_raw: str, is_unit: Callable[[str], object]) -> tuple[float | None, str]:
    """``(factor, unit)``: the factor the cell is multiplied by to give the value, and the unit left over; the
    factor is None when the header does not say which way its power of ten goes.

    A table header carries a power of ten in one of two conventions, which the model copies into ``unit_raw``:

    - on the **unit** -- "×10^-4 Ω·cm", "(10^-4 Ω cm)", "ρ (×10^-4 Ω cm)", "ρ × 10^-4 Ω·cm": the column is in
      units of 10^-4 Ω·cm, so a cell of 6.8 is 6.8 × 10^-4 Ω·cm, and the factor multiplies.
    - on the **quantity** -- "ρ × 10^4 (Ω cm)": the column holds ρ multiplied by 10^4, the unit bracketed apart,
      so the same cell is again 6.8 × 10^-4 Ω·cm, and the factor divides.

    The same exponent sign means opposite things, and both lanes read one header the same way, so a wrong guess
    would pass as agreement. What decides is where the brackets are, so they are read before anything is cleaned
    away. A header that fits neither is refused: a unit before the factor (``is_unit`` names the field's units),
    a factor glued to a symbol with nothing joining them, a factor bracketed alone beside the symbol
    ("ρ (×10^-4) (Ω cm)": the column's multiplier, or ρ's?), or a factor on the quantity with no unit after it.
    """
    text = LATEX_WRAPPERS.sub(" ", delatex(normalize_text(unit_raw))).strip()
    match = _SCALE_FACTOR.search(text)
    if match is None:
        return 1.0, clean_unit(text)
    factor = 10.0 ** int(match.group("e"))
    head, tail = text[: match.start()].strip(), text[match.end() :].strip()

    def unit_of(rest: str) -> str:
        return clean_unit(rest.strip(_OPENING + _CLOSING + " ")).lstrip(".x*")

    if not head.rstrip(_OPENING):
        # The factor leads what follows, bracketed or not: "×10^-4 Ω·cm", "(10^-4 Ω cm)".
        return factor, unit_of(tail)
    symbol = head.rstrip(_OPENING).strip()
    if not _QUANTITY_SYMBOL.fullmatch(symbol) or is_unit(symbol) is not None:
        return None, clean_unit(text)
    if head[-1] in _OPENING:
        inside, _, _after = tail.partition(_CLOSING[_OPENING.index(head[-1])])
        if unit_of(inside):
            return factor, unit_of(inside)  # "ρ (10^-4 Ω cm)": the factor leads the bracketed unit
        return None, clean_unit(text)  # "ρ (×10^-4) (Ω cm)": bracketed alone, it says nothing about direction
    if not match.group("x"):
        return None, clean_unit(text)  # "ρ 10^4 Ω cm": nothing joins the symbol and the factor
    if tail[:1] in _OPENING and unit_of(tail):
        if factor < 1:
            # "ρ ×10^-4 (Ω cm)" formally says ρ was multiplied by 10^-4, but authors who write a negative power on
            # the quantity usually mean the unit's multiplier; the two readings differ by 10^8, so neither is taken.
            return None, clean_unit(text)
        return 1 / factor, unit_of(tail)  # "ρ × 10^4 (Ω cm)": ρ multiplied, the unit bracketed apart
    if tail and tail[0] not in _OPENING:
        return factor, unit_of(tail)  # "ρ × 10^-4 Ω·cm": the factor leads the unit written after it
    return None, clean_unit(text)  # "ρ × 10^4": on the quantity, but no unit says so


def has_scale_factor(text: str) -> bool:
    """Whether a transcribed *value* already carries its own power of ten ("1.2 x 10^-4", "1.2e-4")."""
    return _SCI.search(delatex(normalize_text(text))) is not None


def convert_to_canonical(
    spec: FieldSpec,
    value: float,
    unit_raw: str | None,
    units: UnitRegistry = BUILTIN_UNITS,
    *,
    value_text: str | None = None,
) -> tuple[float | None, str | None, str | None]:
    """``(canonical value, canonical unit, note)``; the value is None when conversion fails.

    The value is multiplied by any scale factor in the header first, then converted as ``value * factor +
    offset``: the header's power of ten counts in the unit it was written in, before a temperature is shifted.
    ``value_text`` is the raw text ``value`` was parsed from. It is only consulted to detect a power of ten
    written twice, once in the value and once in the unit, which no reading can resolve.
    """
    canonical = spec.canonical_unit
    if canonical is None:
        return value, None, None
    if unit_raw is None:
        return _bare_number(spec, value)

    def convert(unit: str) -> tuple[float, float] | None:
        return units.convert(canonical, unit)

    scale, unit = split_scale_factor(unit_raw, convert)
    if scale is None:
        return (
            None,
            None,
            f"power of ten in {unit_raw!r}: cannot tell whether it scales the quantity or the unit; ambiguous",
        )
    if scale != 1.0 and value_text is not None and has_scale_factor(value_text):
        # "1.2 × 10⁻⁴" under a column headed "(×10⁻⁴ Ω·cm)" is either 1.2e-4 or 1.2e-8 depending on whether
        # the author applied the header. Applying the factor twice would manufacture a value; refuse.
        return None, None, "scale factor in both value and unit; ambiguous"
    scale_note = f"scale factor {scale:g} taken from the header in the unit" if scale != 1.0 else None
    value *= scale
    if not unit:
        canonical_value, canonical_unit, note = _bare_number(spec, value)
        return canonical_value, canonical_unit, _join([n for n in (scale_note, note) if n])
    conversion = convert(unit)
    if conversion is None:
        # "1.1 Pa Ar", "3 mTorr (O2)": a pressure or flow named with the gas it belongs to. The gas says whose
        # quantity it is, not what unit, so the unit is read without it -- only when the rest is a known unit.
        # Which words are such suffixes is the profile's (ignored_unit_suffixes).
        gasless = units.without_ignored_suffix(unit)
        if gasless != unit and gasless:
            conversion = convert(gasless)
            if conversion is not None:
                factor, offset = conversion
                note = f"gas name in the unit ({unit[len(gasless) :].strip('()')}) set aside"
                return _apply(value, factor, offset), canonical, _join([n for n in (scale_note, note) if n])
        return None, None, f"unknown unit {unit_raw!r} for {canonical}"
    factor, offset = conversion
    return _apply(value, factor, offset), canonical, scale_note


def _apply(value: float, factor: float, offset: float) -> float:
    # A zero offset is not added at all: -0.0 + 0.0 is 0.0, and a factor-only unit keeps the bits it always gave.
    return value * factor + offset if offset else value * factor


def _bare_number(spec: FieldSpec, value: float) -> tuple[float | None, str | None, str | None]:
    canonical = spec.canonical_unit
    match spec.bare_number:
        case "percent_or_fraction":
            # Strictly below 1: a bare "1" is far more often 1 % (1 % O2 in Ar) than a fraction of exactly
            # one, and reading it as 100 % turns a trace admixture into the whole gas.
            if 0.0 <= value < 1.0:
                return value * 100.0, canonical, "no unit; value < 1 read as a fraction"
            return value, canonical, "no unit; read as percent"
        case "assume_canonical":
            return value, canonical, f"no unit; assumed {canonical}"
        case "reject":
            return None, None, f"no unit; {spec.name} requires one"


# ---- Applying it to a lane ----------------------------------------------------------------------------------


# "3 h 30 min", "2 hours and 15 minutes", "1 min 30 s": one duration written in two units, larger first.
_COMPOUND = re.compile(
    rf"^(?P<a>{_UNSIGNED})\s*(?P<ua>[a-zA-Z]+)\s*(?:and\s+)?(?P<b>{_UNSIGNED})\s*(?P<ub>[a-zA-Z]+)$", re.IGNORECASE
)
# The canonical units whose quantity is written as a sum of units. Only a duration is: anywhere else a second
# unit restates the same value ("0.5 Pa 3.75 mTorr", "2 in 50 mm"), and adding the two doubles it.
_SUMMED_UNITS = {"min"}


def compound_value(spec: FieldSpec, text: str, units: UnitRegistry) -> float | None:
    """The canonical value of a duration spelled in two of its units, larger first ("3 h 30 min" -> 210), or
    None for anything else. The larger part must be whole and the smaller one less than one of the larger unit:
    "1 h 90 min" is no way anyone writes 150 minutes, and "0.5 h 30 min" restates 30 minutes. Both units carry
    their own factor, so the model's ``unit_raw`` -- which can name only one of them -- plays no part. ``text``
    is the value alone: callers set aside what surrounds it (:func:`set_aside`).

    The one reader of compound durations, for the comparison (:func:`normalize_field`) and for the dataset
    cell (``decide``) alike, so the two never read one string differently."""
    if spec.canonical_unit not in _SUMMED_UNITS:
        return None
    match = _COMPOUND.match(normalize_text(text).strip())
    if match is None:
        return None
    canonical = spec.canonical_unit
    big, small = units.convert(canonical, match.group("ua")), units.convert(canonical, match.group("ub"))
    # A unit with an offset is no part of a sum: only a duration is summed, and none of its units has one.
    if big is None or small is None or big[1] or small[1]:
        return None
    big, small = big[0], small[0]
    if big <= small or not match.group("a").isdigit():
        # A fractional larger part ("0.5 h 30 min") is a restatement, not a sum: nobody writes 30 min that way.
        return None
    part = float(_plain(match.group("b"))) * small
    if part >= big:
        return None
    return float(_plain(match.group("a"))) * big + part


def _names_unit_of(spec: FieldSpec, text: str, units: UnitRegistry) -> bool:
    """Whether ``text`` names a unit of the field's own quantity ("2 h" for a time, "°C" for a temperature)."""
    canonical = spec.canonical_unit
    if canonical is None:
        return False
    return any(units.convert(canonical, word) is not None for word in _TAIL_WORD.findall(normalize_text(text)))


def normalize_field(field: FieldValue, spec: FieldSpec, units: UnitRegistry) -> FieldValue:
    if spec.kind != "numeric":
        # Text and composition fields are compared through normalize_key on the fly.
        return field
    # A number word is decided here, not in parse_number: only the unit tells "four-inch" from "ten-fold".
    spelled = spell_number_word(field.value_raw, field.unit_raw)
    lead_note = f"number word {field.value_raw.strip()!r} read as {spelled}" if spelled != field.value_raw else None
    if spec.after_clause == "condition":
        # "92.5% after 100 cycles": the number is the value, the clause is what it was measured after. Moved into
        # the condition, it separates "after 50 cycles" from "after 100 cycles" in the comparison and the cell.
        spelled, clause = split_after_clause(spelled)
        if clause:
            field = _with_after_condition(field, clause)
            lead_note = "; ".join(n for n in (lead_note, f"{clause!r} moved into the condition") if n)
    bare, context_notes, condition = set_aside(spelled)
    if condition and _names_unit_of(spec, condition, units) and not _names_unit_of(spec, bare, units):
        # "400 °C for 2 h" on annealing_time: the time is in the tail, and the number kept is a temperature.
        note = f"the condition {condition!r} holds this field's quantity and the value does not; ambiguous"
        return field.model_copy(update={"value": None, "unit": None, "normalization_note": note})
    compound = compound_value(spec, bare, units)
    if compound is not None:
        compound_note = f"compound {bare!r} read as {compound:g} {spec.canonical_unit}"
        note = "; ".join(n for n in (lead_note, *context_notes, compound_note) if n)
        return field.model_copy(update={"value": compound, "unit": spec.canonical_unit, "normalization_note": note})
    number, parse_note = parse_number(spelled, range_policy=spec.range_policy)
    if number is None:
        return field.model_copy(update={"value": None, "unit": None, "normalization_note": parse_note})
    value, unit, unit_note = convert_to_canonical(spec, number, field.unit_raw, units, value_text=spelled)
    note = "; ".join(n for n in (lead_note, parse_note, unit_note) if n) or None
    return field.model_copy(update={"value": value, "unit": unit, "normalization_note": note})


def _normalize_fields(fields: tuple[FieldValue, ...], profile: DomainProfile) -> tuple[FieldValue, ...]:
    # Fields outside the schema were dropped at extraction time; this is a defensive second check.
    specs = profile.by_name
    return tuple(normalize_field(f, specs[f.field], profile.units) if f.field in specs else f for f in fields)


def normalize_lane(lane: LaneExtraction, profile: DomainProfile) -> LaneExtraction:
    """Fill in ``value`` / ``unit`` for every field, in ``profile``'s units. Pure and idempotent: always returns
    a new object."""
    target: TargetRecord | None = None
    if lane.target is not None:
        target = lane.target.model_copy(update={"fields": _normalize_fields(lane.target.fields, profile)})
    samples = tuple(
        sample.model_copy(update={"fields": _normalize_fields(sample.fields, profile)}) for sample in lane.samples
    )
    # Unattributed values are compared now, so they need canonical values like every other; leaving them
    # raw would silently turn every such comparison into "unparsed" and bury real agreements.
    unattributed = _normalize_fields(lane.unattributed, profile)
    return lane.model_copy(update={"target": target, "samples": samples, "unattributed": unattributed})


def drop_implausible(records: ExtractedRecords, profile: DomainProfile) -> ExtractedRecords:
    """Drop every value whose converted number falls outside its field's ``valid_range``, with the reason.

    The range lives in the canonical unit, so this has to run on the converted value, in ``profile``'s units:
    "2 μm" is outside a 500 nm ceiling although its digits are not. A value that cannot be converted is kept,
    since there is no number to judge and the comparison already reports it as unparsed.
    """
    dropped: list[str] = []

    def plausible(value: FieldValue) -> bool:
        spec = profile.by_name.get(value.field)
        if spec is None or spec.describe_range() is None:
            return True
        number = normalize_field(value, spec, profile.units).value
        if number is None or spec.in_range(number):
            return True
        unit = f" {value.unit_raw}" if value.unit_raw else ""
        canonical = f" {spec.canonical_unit}" if spec.canonical_unit else ""
        dropped.append(
            f"{spec.name}: {value.value_raw!r}{unit} is {number:g}{canonical}, "
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
