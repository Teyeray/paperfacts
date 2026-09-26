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

import datetime
import itertools
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

from paperfacts.fields import RANGE_ENDS, FieldSpec, RangePolicy
from paperfacts.grounding import LOWER_BOUND_WORDS, UPPER_BOUND_WORDS
from paperfacts.profile import DomainProfile
from paperfacts.records import ExtractedRecords, FieldValue, LaneExtraction, PaperRecord, spell_number_word
from paperfacts.text import LATEX_WRAPPERS, clean_unit, delatex, normalize_key, normalize_text
from paperfacts.units import UnitRegistry

if TYPE_CHECKING:
    from paperfacts.kinds import KindContext

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


# A hyphen or a period one parser keeps and the other drops: MinerU read "rf-magnetron sputtering" as
# "rfmagnetron sputtering", and "wt.%" is also written "wt%". Before a digit either is part of a number (a sign,
# a range, a decimal point), so there it stays: "10-20" is not "1020", nor "1.5" "15".
LOOSE_PUNCTUATION = re.compile(r"[-.](?!\d)")


def same_text(spec: FieldSpec, a: str | None, b: str | None) -> bool:
    """Whether two text values state the same fact, for the comparison and the dataset cell alike.

    Equal :func:`text_key` decides first. Failing that, the two agree when they differ only by spacing, case and a
    dropped hyphen or period -- unless both name a category, which is then the whole answer ("DC" is never "RF").
    Only one of them naming a category is exactly what a lost hyphen causes: "rf-magnetron" reads as RF, its
    glued twin "rfmagnetron" names nothing."""
    if text_key(spec, a) == text_key(spec, b):
        return True
    if canonical_category(spec.categories, a) is not None and canonical_category(spec.categories, b) is not None:
        return False
    return LOOSE_PUNCTUATION.sub("", normalize_key(a)) == LOOSE_PUNCTUATION.sub("", normalize_key(b))


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
# "10-20", "15.6 to 16.3 nm", "80%–85%", "500 °C to 530 °C", "-60 to -20", "between 450 and 500 °C": two bounds,
# each with an optional unit. "to" is the separator, never the first bound's unit ("-60 to -20" is no -60 to, 20).
_RANGE = re.compile(
    rf"^(?:(?P<between>[Bb]etween)\s+)?(?P<a>{_NUM})\s*(?P<ua>(?!to\b){_UNIT_TOKEN})?"
    rf"\s*(?(between)and|{_RANGE_SEP})\s*(?P<b>{_NUM})\s*(?P<ub>{_UNIT_TOKEN})?$"
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
NUMBER_RE = re.compile(_NUM)
"""Every plain number in a piece of text. Public because the comparison layer reads the numbers out of a
measurement condition ("550 nm") and must use the same notion of "a number" this module parses with."""
_QUALIFIERS = re.compile(
    r"^(?P<q>>=|<=|approximately|approx\.?|roughly|around|about|circa|ca\.?|[~≈≃≅≥≤<>])\s*", re.IGNORECASE
)
# A qualifier that makes what follows a one-sided bound: "> 450-500" is no range with two printed ends.
_BOUND_SIGNS = frozenset({">=", "<=", "≥", "≤", "<", ">"})
# One number as a scalar is written -- plain, "1.2 x 10^-4", "10^-4" or "1.2e-4" -- optionally "± another", then
# whatever follows it (``tail``). The dataset cell demands the tail be a unit of the field; the lanes read the
# unit a quote writes after its number from the same tail.
NUMBER_ATOM = rf"(?:{_NUM}\s*x\s*10\s*\^?\s*[-+]?\d+|10\s*\^\s*[-+]?\d+|{_NUM}(?:[eE][-+]?\d+)?)"
SCALAR = re.compile(
    rf"^(?P<center>{NUMBER_ATOM})(?:\s*(?:±|\+/-|\+-|\\pm)\s*(?P<uncertainty>{NUMBER_ATOM}))?(?P<tail>.*)$"
)


@dataclass(frozen=True)
class NumberReading:
    """What :func:`read_number` reads from a quote: the number, and what the unit check needs, so no reader of
    the same quote parses it a second time."""

    value: float | None
    note: str | None
    # What the quote writes after its number or its clean range, stripped: "" when nothing, None when the quote
    # is neither one scalar nor one clean range, or carries a condition or a parenthesis, so no single text
    # follows "the" number. Whether it is a unit of the field is :func:`unit_of_value`'s to say.
    unit: str | None
    # The (low, high) of a clean range (:func:`read_range`), whatever the policy; None for anything else.
    ends: tuple[float, float] | None


def parse_number(
    raw: str, *, range_policy: RangePolicy = "midpoint", range_unit: Callable[[str], bool] | None = None
) -> tuple[float | None, str | None]:
    """``(value, note)`` of :func:`read_number`."""
    reading = read_number(raw, range_policy=range_policy, range_unit=range_unit)
    return reading.value, reading.note


def read_number(
    raw: str, *, range_policy: RangePolicy = "midpoint", range_unit: Callable[[str], bool] | None = None
) -> NumberReading:
    """The number ``raw`` spells, or None with the reason it was refused.

    A qualifier ("~", ">", "about") is dropped and recorded first; what is left must then match one of the
    spellings in :data:`_SPELLINGS`, tried in order. Each spelling either claims the text -- with a value, or
    with a refusal -- or passes it on. A refusal is always better than a guess: the comparison turns None
    into AMBIGUOUS, while a wrong number is indistinguishable from a real measurement.

    ``range_policy`` is the field's (``FieldSpec.range_policy``): ``"midpoint"`` reads a range as its midpoint,
    ``"reject"`` refuses it, for a quantity whose range is a window rather than a scatter around one value (a
    cathode's "2.8–4.3 V" is the cycling window; its midpoint was never measured). ``"lower"`` / ``"upper"``
    read it as the end the field asks for (a calcination "at 450-500 °C" reported by its upper end): unlike
    the midpoint, an end is a number the paper printed. Only a clean range has an end (:func:`read_range`), and
    only when the unit written after it is one of the field's (``range_unit``, required under these two); any
    other range is refused under them, exactly as the dataset cell refuses it.
    """
    if range_policy in RANGE_ENDS and range_unit is None:
        raise ValueError(f"range_policy {range_policy!r} needs range_unit: an end is read only in the field's unit")
    bare, notes, condition = set_aside(raw)
    if _AFTER.search(bare):
        refusal = "a value stated 'after' a treatment belongs to another state of the sample; ambiguous"
        return NumberReading(None, _join([*notes, refusal]), None, None)
    # The digits of a formula or a unit exponent are set aside before the value's own numbers are counted.
    text = bare
    unglued = _GLUED_DIGITS.sub(" ", text)
    if unglued != text:
        notes.append("digits of a formula or unit exponent ignored")
        text = unglued.strip()
    value, reading, ends = _read(text)
    clean = _clean_range(raw, bare, text, condition, ends)
    if ends is None:
        alone = value is not None and not condition and "(" not in bare and ")" not in bare
        scalar = SCALAR.fullmatch(bare) if alone else None
        unit = scalar.group("tail").strip() if scalar is not None else None
        return NumberReading(value, _join([*notes, *reading]), unit, None)
    low, high, _ = ends
    unit = None if clean is None else clean[2]
    clean_ends = None if clean is None else (low, high)
    if range_policy == "midpoint":
        return NumberReading(value, _join([*notes, *reading, f"range {low:g}-{high:g} → midpoint"]), unit, clean_ends)
    if range_policy == "reject":
        note = _join([*notes, *reading, f"range {low:g}-{high:g} refused (range_policy 'reject')"])
        return NumberReading(None, note, unit, clean_ends)
    if clean is not None and (not clean[2] or (range_unit is not None and range_unit(clean[2]))):
        end = f"range {low:g}-{high:g} → {range_policy} end"
        chosen = low if range_policy == "lower" else high
        return NumberReading(chosen, _join([*notes, *reading, end]), unit, clean_ends)
    unclean = f"range {low:g}-{high:g} has no {range_policy} end: not one clean range in the field's unit; ambiguous"
    return NumberReading(None, _join([*notes, *reading, unclean]), unit, clean_ends)


def _clean_range(
    raw: str, bare: str, text: str, condition: str, ends: tuple[float, float, str | None] | None
) -> tuple[float, float, str] | None:
    """``(low, high, unit)`` when the general reader found a range (``ends``) that is clean: see :func:`read_range`.

    The reader's range spellings (:func:`_range`, :func:`_scientific`) decide what a range is; this only refuses
    what surrounds one. ``text`` is ``bare`` with glued digits taken out: a range that needed that ("450to500")
    was never read as a range by the reader either."""
    if ends is None or ends[2] is None or condition or text != bare or "(" in bare or ")" in bare:
        return None
    qualifier = _QUALIFIERS.match(_typeset(raw))
    if qualifier is not None and qualifier.group("q") in _BOUND_SIGNS:
        return None
    return ends[0], ends[1], ends[2]


def split_after_clause(text: str) -> tuple[str, str]:
    """``(value, clause)``: ``text`` cut where an "after ..." clause follows a number ("92.5% after 100 cycles" ->
    "92.5%", "after 100 cycles"), or ``(text, "")`` when there is none. Only for a field whose ``after_clause``
    is "condition"; every other field refuses such a value in :func:`parse_number`."""
    match = _AFTER.search(text)
    if match is None or not NUMBER_RE.search(text[: match.start()]):
        return text, ""
    return text[: match.start()].strip(), text[match.start() :].strip()


def _typeset(raw: str) -> str:
    """``raw`` with its typesetting folded away, as every reader of a value sees it."""
    # "10^(-4)" is the caret spelling with its exponent bracketed, not a parenthesised alternative.
    text = _CARET_PARENS.sub(r"^\1", delatex(normalize_text(raw)))
    # A command delatex has no reading for is typesetting; left in place, its letters would glue to the
    # digits after it ("\\sim82").
    return _LATEX_COMMAND.sub(" ", text).strip()


def set_aside(raw: str) -> tuple[str, list[str], str]:
    """``(value text, notes, condition)``: ``raw`` without what surrounds the value -- typesetting, a qualifier,
    a name before "=", a condition after it -- a note for each thing set aside, and the condition itself ("" when
    there is none). The same for every spelling, scientific, plain or compound, and for the dataset cell
    (``kinds``), so none of them reads a condition's number as the value.

    A condition is set aside only where the value before it keeps a number: "deposited for 10 min" is the
    quote of a value that opens with its verb, not a condition with no value in front of it."""
    text = _typeset(raw)
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


# (value, notes, ends): ends is the (low, high, unit) of a spelling that reads a whole range as its midpoint, which
# is what range_policy decides on (read_number writes the range's note); None for every other reading. The unit is
# what follows the range, "" when nothing does, None when something stands before it too.
_Reading = tuple[float | None, list[str], tuple[float, float, str | None] | None]


def _read(text: str) -> _Reading:
    for spelling in _SPELLINGS:
        reading = spelling(text)
        if reading is not None:
            return reading
    raise AssertionError("the last spelling always answers")  # pragma: no cover


def _refuse(reason: str) -> _Reading:
    return None, [reason], None


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
    return value, ["uncertainty dropped"] if match.group("pm") else [], None


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
    value, reading, ends = _read(rest)
    return value, ["parenthesized alternative ignored", *reading], ends


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
            return values[0], ["uncertainty dropped"], None
        if not NUMBER_RE.search(rest) and _RANGE_SEPARATOR.fullmatch(between):
            low, high = values
            if low < high:
                alone = not text[: matches[0].start()].strip()
                return (low + high) / 2, [], (low, high, text[matches[1].end() :].strip() if alone else None)
            return _refuse("descending range in scientific notation; ambiguous")
    if len(matches) > 1 or NUMBER_RE.search(rest):
        return _refuse("numbers outside the scientific notation; ambiguous")
    return values[0], [], None


def _uncertainty(text: str) -> _Reading | None:
    match = _PLUS_MINUS.match(text)
    return None if match is None else (float(_plain(match.group("a"))), ["uncertainty dropped"], None)


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
    return (low + high) / 2, notes, (low, high, unit or "")


def _first_number(text: str) -> _Reading:
    """The fallback: one number is the value. Several separated only by spaces or a multiplication sign ("300
    500", "40 x 10 cm") keep the first with a note. A range buried among other numbers, a list ("30, 40"), and a
    number carrying its own unit before another ("550 nm: 85%", "140 nm ATO/25 nm ITO") have no first value
    worth keeping: each is a second quantity beside the first, and they are refused."""
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return _refuse("no number found")
    if len(numbers) == 1:
        return float(_plain(numbers[0])), [], None
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
    return float(_plain(numbers[0])), [f"{len(numbers)} numbers found, first used"], None


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
    units: UnitRegistry,
    *,
    value_text: str | None = None,
    range_ends: tuple[float, float] | None = None,
) -> tuple[float | None, str | None, str | None]:
    """``(canonical value, canonical unit, note)`` in ``units`` (a profile's); the value is None when conversion
    fails.

    The value is multiplied by any scale factor in the header first, then converted as ``value * factor +
    offset``: the header's power of ten counts in the unit it was written in, before a temperature is shifted.
    ``value_text`` is the raw text ``value`` was parsed from. It is only consulted to detect a power of ten
    written twice, once in the value and once in the unit, which no reading can resolve. ``range_ends`` are the
    ends of the range ``value`` was taken from (``range_policy`` lower/upper): a bare number's meaning is then
    decided once for the whole range, so both ends read in the same unit.
    """
    canonical = spec.canonical_unit
    if canonical is None:
        return value, None, None
    if unit_raw is None:
        return _bare_number(spec, value, range_ends)

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
        scaled = None if range_ends is None else (range_ends[0] * scale, range_ends[1] * scale)
        canonical_value, canonical_unit, note = _bare_number(spec, value, scaled)
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


def unit_of_value(
    spec: FieldSpec, unit_raw: str | None, written: str, units: UnitRegistry
) -> tuple[str | None, bool] | None:
    """``(unit, own)``: the unit a number is converted from when its quote writes ``written`` after it, and
    whether that is the quote's own unit rather than ``unit_raw``; None when ``written`` is no unit of the field.

    Units are compared as the registry converts them, never as spellings: "Ω cm", "Ω-cm" and "ohm cm" are all
    "Ω·cm". A written unit that converts exactly as ``unit_raw`` does -- the same factor and offset, a header's
    power of ten included -- changes nothing, and ``unit_raw`` is kept. One that converts otherwise is the more
    specific statement and is converted from: "1.5e-4 Ω·cm" under unit_raw "mΩ·cm" is 1.5e-4 Ω·cm, not 1.5e-7,
    and "1.2 Ω·cm" under a header "×10^-4 Ω·cm" states its own unit, so the header's power of ten is not applied.
    Anything the registry cannot read as a unit of the field ("Ω cm (sample A)", "K" on a ℃ field) is refused
    rather than guessed. A field without a canonical unit has no registry to ask, so there the written unit must
    be spelled as ``unit_raw``."""
    if not clean_unit(written):
        return unit_raw, False
    if spec.canonical_unit is None:
        return (unit_raw, False) if unit_raw and clean_unit(unit_raw) == clean_unit(written) else None
    conversion = _conversion(spec, written, units)
    if conversion is None:
        return None
    if unit_raw and _conversion(spec, unit_raw, units) == conversion:
        return unit_raw, False
    return written, True


def _conversion(spec: FieldSpec, unit: str, units: UnitRegistry) -> tuple[float, float] | None:
    """``(factor, offset)`` that :func:`convert_to_canonical` applies to a value quoted in ``unit``, a header's power
    of ten folded into the factor; None when ``unit`` names no unit of the field (a bare power of ten included)."""
    canonical = spec.canonical_unit
    assert canonical is not None

    def convert(spelling: str) -> tuple[float, float] | None:
        return units.convert(canonical, spelling)

    scale, rest = split_scale_factor(unit, convert)
    if scale is None or not rest:
        return None
    conversion = convert(rest)
    if conversion is None:
        gasless = units.without_ignored_suffix(rest)
        conversion = convert(gasless) if gasless and gasless != rest else None
    if conversion is None:
        return None
    factor, offset = conversion
    return factor * scale, offset


def _apply(value: float, factor: float, offset: float) -> float:
    # A zero offset is not added at all: -0.0 + 0.0 is 0.0, and a factor-only unit keeps the bits it always gave.
    return value * factor + offset if offset else value * factor


def _bare_number(
    spec: FieldSpec, value: float, range_ends: tuple[float, float] | None = None
) -> tuple[float | None, str | None, str | None]:
    canonical = spec.canonical_unit
    match spec.bare_number:
        case "percent_or_fraction":
            # Strictly below 1: a bare "1" is far more often 1 % (1 % O2 in Ar) than a fraction of exactly
            # one, and reading it as 100 % turns a trace admixture into the whole gas. A range is a fraction only
            # when all of it is: "0.8-1.2" is 0.8-1.2 % at either end, never 80 % at one and 1.2 % at the other.
            if range_ends is not None:
                if 0.0 <= range_ends[0] and range_ends[1] < 1.0:
                    return value * 100.0, canonical, "no unit; range < 1 read as a fraction"
                return value, canonical, "no unit; read as percent"
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
    cell (``kinds``) alike, so the two never read one string differently."""
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


@dataclass(frozen=True)
class Reading:
    """What every reader of a numeric value -- the comparison (:func:`normalize_field`) and the dataset cell
    (``kinds``) -- takes from a quote before deciding what its number is. One function builds it, so the two
    never read one string differently."""

    # The value text to parse: number words spelled out, an "after" clause cut off, a bound put back in front.
    text: str
    # The spelled-out text when the quote held a number word, else None.
    number_word: str | None
    # The "after ..." clause moved into the condition (FieldSpec.after_clause), or "".
    clause: str
    # The bound grounding found before the quote in its block (FieldValue.bound), or None.
    bound: str | None
    # set_aside's reading of ``text``: the bare value, a note per thing set aside, and the condition.
    bare: str
    context_notes: tuple[str, ...]
    condition: str
    # The canonical value of a compound duration ("3 h 30 min"), or None.
    compound: float | None


def read_value(field: FieldValue, spec: FieldSpec, units: UnitRegistry) -> Reading:
    """The shared first steps of reading a numeric field's quote; see :class:`Reading`."""
    # A number word is decided here, not in parse_number: only the unit tells "four-inch" from "ten-fold".
    spelled = spell_number_word(field.value_raw, field.unit_raw)
    text, clause = spelled, ""
    if spec.after_clause == "condition":
        # "92.5% after 100 cycles": the number is the value, the clause is what it was measured after. Moved into
        # the condition, it separates "after 50 cycles" from "after 100 cycles" in the comparison and the cell.
        text, clause = split_after_clause(text)
    if field.bound:
        # "90" quoted out of "above 90 %" is read as "above 90": exactly what quoting the bound would have given.
        text = f"{field.bound} {text}"
    bare, context_notes, condition = set_aside(text)
    return Reading(
        text=text,
        number_word=spelled if spelled != field.value_raw else None,
        clause=clause,
        bound=field.bound,
        bare=bare,
        context_notes=tuple(context_notes),
        condition=condition,
        compound=compound_value(spec, bare, units),
    )


def read_range(raw: str, range_unit: Callable[[str], bool]) -> tuple[float, float, str] | None:
    """``(low, high, unit)`` when ``raw`` is one clean range, else None: the one definition of a range with two
    printed ends, for the lanes (:func:`read_number` under ``range_policy`` lower/upper) and the dataset cell
    (``kinds``) alike, so the two never disagree about which quotes have an end.

    A range is what the general reader reads as one (:func:`read_number`): two ascending numbers, both plain or
    both in scientific notation, joined by a range separator. Clean means nothing else is around it but a unit
    after it that ``range_unit`` accepts as the field's ("" when the range has none). A qualifier of
    approximation or a name before "=" may precede it (:func:`set_aside`); a bound ("> 450-500", "below 1.2e-4 -
    1.5e-4"), a condition ("450-500 °C for 2 h"), an "after" clause, a parenthesis ("450-500 (600)"), another unit
    ("450-500 K" on a ℃ field) and an exponent written once for two numbers ("1.2-1.5 × 10^-3") are not."""
    reading = read_number(raw)
    if reading.ends is None or reading.unit is None or (reading.unit and not range_unit(reading.unit)):
        return None
    return reading.ends[0], reading.ends[1], reading.unit


# A one-sided bound written before the number: the signs, and the words grounding recognises before a quote
# (grounding.quoted_bound), so "80" quoted out of "above 80 %" reads the way ">80" does.
_ONE_SIDED = re.compile(
    rf"^(?:(?P<lower>>=|≥|>|{'|'.join(LOWER_BOUND_WORDS)})"
    rf"|(?P<upper><=|≤|<|{'|'.join(UPPER_BOUND_WORDS)}))(?![a-z])\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_NO_INTERVAL = "an interval field needs two ends or a bound"


def read_interval(
    raw: str, range_unit: Callable[[str], bool]
) -> tuple[tuple[float | None, float | None, str] | None, str | None]:
    """``((low, high, unit), None)`` when ``raw`` is one clean range or one clean one-sided bound (">80 %", "at most
    5 nm", the open end None), else ``(None, why)``. ``unit`` is what follows the number, "" when nothing does.

    A bound is held to a range's contract (:func:`read_range`): one number after the bound word, and after it
    nothing but a unit ``range_unit`` accepts. "> 450-500", "> 450 (600)", "< 500 °C for 2 h" and ">80 % at 550
    nm" are refused, not read as the one number they start with."""
    clean = read_range(raw, range_unit)
    if clean is not None:
        return clean, None
    match = _ONE_SIDED.match(_typeset(raw))
    if match is None:
        number, note = parse_number(raw, range_policy="reject")
        return None, _NO_INTERVAL if number is not None else _join([n for n in (note, _NO_INTERVAL) if n])
    reading = read_number(match.group("rest"), range_policy="reject")
    if reading.value is None:
        return None, reading.note
    if reading.unit is None or reading.ends is not None:
        return None, "a bound followed by more than one number and its unit (a condition, a parenthesis); ambiguous"
    if reading.unit and not range_unit(reading.unit):
        return None, f"{reading.unit!r} after the bound is not a unit of the field; ambiguous"
    lower = match.group("lower") is not None
    return ((reading.value, None, reading.unit) if lower else (None, reading.value, reading.unit)), None


# ---- Dates -------------------------------------------------------------------------------------------------

_MONTHS = {
    name: index
    for index, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        1,
    )
    for name in names
}
# "2021", "2021-03", "2021-03-12", "2021/03/12", "2021.03.12": a numeric date is read only year first.
_YEAR_FIRST = re.compile(r"^(?P<y>\d{4})(?:(?P<sep>[-/.])(?P<m>\d{1,2})(?:(?P=sep)(?P<d>\d{1,2}))?)?$")
# The tokens of a date written with its month's name: "12 March 2021", "March 12th, 2021", "Mar. 2021".
_DATE_TOKEN = re.compile(r"(?P<word>[a-z]+)\.?|(?P<number>\d+)(?:st|nd|rd|th)?|(?P<gap>[\s,]+)|(?P<other>.)")
_EARLIEST_YEAR, _LATEST_YEAR = 1800, 2100


def read_date(raw: str) -> tuple[str | None, str | None]:
    """``(iso, note)``: the date ``raw`` states as ISO at the precision written -- "2021", "2021-03" or
    "2021-03-12" -- or None with the reason it is refused.

    Year-first numeric dates and dates naming their month are read. Refused, because each could be read two ways
    or names more than one date: a two-digit year ("Mar 21"), an all-numeric date that is not year first
    ("03/04/2021" is March or April), a range ("2019-2021", "March-May 2021"), and a year outside 1800-2100."""
    # A date quoted at the end of a sentence or in parentheses: "(March 2021)", "March 2021.".
    text = normalize_text(raw).casefold().removesuffix(".").strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    numeric = _YEAR_FIRST.fullmatch(text)
    if numeric is not None and numeric.group("sep") == "." and numeric.group("d") is None:
        return None, "a year with one part after a point could be a decimal year; ambiguous"
    if numeric is not None:
        year, month, day = (int(part) if part else None for part in numeric.group("y", "m", "d"))
    elif re.fullmatch(r"[\d\s/.\-]+", text) and re.search(r"\d", text):
        return None, "an all-numeric date that is not year first, or not one date; ambiguous"
    else:
        year = month = day = None
        for token in _DATE_TOKEN.finditer(text):
            word, number = token.group("word"), token.group("number")
            if token.group("other") is not None:
                return None, f"{token.group('other')!r} in a date: a range or not one date; ambiguous"
            if word is not None:
                if word not in _MONTHS or month is not None:
                    return None, f"{word!r} is not the one month of a date"
                month = _MONTHS[word]
            elif number is not None and len(number) == 4 and year is None:
                year = int(number)
            elif number is not None and len(number) <= 2 and day is None:
                day = int(number)
            elif number is not None:
                return None, f"{number!r} is neither the day nor the year of one date; ambiguous"
        if month is None and day is not None:
            return None, "a date needs its month"
        if year is None:
            return None, "no four-digit year (a two-digit year is not read)"
    if not _EARLIEST_YEAR <= year <= _LATEST_YEAR:
        return None, f"year {year} outside {_EARLIEST_YEAR}-{_LATEST_YEAR}"
    try:
        datetime.date(year, month or 1, day or 1)
    except ValueError:
        return None, "no such calendar date"
    if month is None:
        return f"{year:04d}", None
    if day is None:
        return f"{year:04d}-{month:02d}", None
    return f"{year:04d}-{month:02d}-{day:02d}", None


def normalize_field(
    field: FieldValue, spec: FieldSpec, units: UnitRegistry, ctx: KindContext | None = None
) -> FieldValue:
    """``field`` with ``value`` / ``unit`` filled in as its kind reads it (:mod:`paperfacts.kinds`). ``ctx`` holds the
    lane's samples a reference field resolves against; no other kind reads it. None rather than ``NO_CONTEXT`` as the
    default only because the kind rows are imported lazily here."""
    # Imported here, not at the top: the kind rows are built on this module's readers.
    from paperfacts.kinds import NO_CONTEXT, rules_for

    return rules_for(spec).read(field, spec, units, ctx or NO_CONTEXT)


def _normalize_fields(
    fields: tuple[FieldValue, ...], profile: DomainProfile, ctx: KindContext
) -> tuple[FieldValue, ...]:
    # Fields outside the schema were dropped at extraction time; this is a defensive second check.
    specs = profile.by_name
    return tuple(normalize_field(f, specs[f.field], profile.units, ctx) if f.field in specs else f for f in fields)


def normalize_lane(lane: LaneExtraction, profile: DomainProfile) -> LaneExtraction:
    """Fill in ``value`` / ``unit`` for every field, in ``profile``'s units, and a reference field's ``ref_id``
    among the lane's own samples. Pure and idempotent: always returns a new object."""
    from paperfacts.kinds import KindContext

    ctx = KindContext(samples=lane.listed())
    paper: PaperRecord | None = None
    if lane.paper is not None:
        paper = lane.paper.model_copy(update={"fields": _normalize_fields(lane.paper.fields, profile, ctx)})
    samples = tuple(
        sample.model_copy(update={"fields": _normalize_fields(sample.fields, profile, ctx)}) for sample in lane.samples
    )
    # Unattributed values are compared now, so they need canonical values like every other; leaving them
    # raw would silently turn every such comparison into "unparsed" and bury real agreements.
    unattributed = _normalize_fields(lane.unattributed, profile, ctx)
    return lane.model_copy(update={"paper": paper, "samples": samples, "unattributed": unattributed})


def drop_implausible(records: ExtractedRecords, profile: DomainProfile) -> ExtractedRecords:
    """Drop every value whose converted number, or an interval's converted end, falls outside its field's
    ``valid_range``, with the reason.

    The range lives in the canonical unit, so this has to run on the converted value, in ``profile``'s units:
    "2 μm" is outside a 500 nm ceiling although its digits are not. A value that cannot be converted is kept,
    since there is no number to judge and the comparison already reports it as unparsed.
    """
    dropped: list[str] = []

    def plausible(value: FieldValue) -> bool:
        spec = profile.by_name.get(value.field)
        if spec is None or spec.describe_range() is None:
            return True
        normalized = normalize_field(value, spec, profile.units)
        # A numeric value is judged on its number, an interval on each end it has.
        ends = normalized.bounds if normalized.bounds is not None else (normalized.value,)
        number = next((end for end in ends if end is not None and not spec.in_range(end)), None)
        if number is None:
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

    paper = records.paper
    if paper is not None:
        paper = paper.model_copy(update={"fields": kept(paper.fields)})
    samples = tuple(sample.model_copy(update={"fields": kept(sample.fields)}) for sample in records.samples)
    unattributed = kept(records.unattributed)
    if not dropped:
        return records
    return records.model_copy(
        update={
            "paper": paper,
            "samples": samples,
            "unattributed": unattributed,
            "dropped": (*records.dropped, *dropped),
        }
    )
