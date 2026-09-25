"""Turn the text the model transcribed into comparable canonical values: text folding, number parsing, unit
conversion, and applying all three to a lane.

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
import unicodedata
from collections.abc import Callable
from functools import cache
from pathlib import Path

from paperfacts.errors import ConfigError
from paperfacts.fields import FIELD_BY_NAME, FIELD_SPECS, FIELDS_SOURCE, FieldSpec
from paperfacts.records import ExtractedRecords, FieldValue, LaneExtraction, TargetRecord, spell_number_word

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
    "•": ".",  # bullet U+2022, read off a chart axis as "Ω•cm"
    "∙": ".",  # bullet operator U+2219
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
    units: lowercasing collides mΩ with MΩ; use :func:`clean_unit` there. Nor for sample ids: it deletes
    Greek letters and folds a case-distinguished suffix; use :func:`paperfacts.records.sample_key` there."""
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
# "85% after 10 cycles", "100 nm after annealing": another state of the sample, not a condition of this value.
_AFTER = re.compile(r"\s+after\s+\S", re.IGNORECASE)
# The words of a condition tail that may name a unit ("2 h", "550 nm", "°C").
_TAIL_WORD = re.compile(r"[^\d\s,;:()\[\]]+")
NUMBER_RE = re.compile(_NUM)
"""Every plain number in a piece of text. Public because the comparison layer reads the numbers out of a
measurement condition ("550 nm") and must use the same notion of "a number" this module parses with."""
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
    r"\sim": "~",
    r"\approx": "≈",
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


# Formatting commands that survive delatex and split what they wrap: MinerU writes the unit Ω·cm as
# "\Omega { \cdot } \mathrm { c m }" and the formula SnO2 as "\mathrm { S n O } _ { 2 }".
_WRAPPED_DIGITS = re.compile(r"(?<=[0-9.] )\s*\{\s*([0-9.])\s*\}")
LATEX_WRAPPERS = re.compile(r"\\(?:mathrm|mathbf|mathit|mathsf|mathcal|text|rm|it|bf|left|right|operatorname)\b")


# LaTeX symbols a unit is written with, restored as the character before anything else is undone: "300
# $^{\circ}$C" and "5 at.\%" otherwise lose the very character a unit is recognised by. The one table for
# retrieval, grounding and unit parsing alike, so the three cannot fold the same text differently.
LATEX_SYMBOLS = {"\\circ": "°", "\\%": "%"}
# A degree sign typeset as a superscript ("^{°}" once \circ is restored) is just a degree sign; left as a
# caret it reads as the start of an exponent, and "500" in "500 ^{\circ}C" as the base of a power.
_RAISED_DEGREE = re.compile(r"\^\s*\{?\s*°\s*\}?")


def delatex(text: str) -> str:
    """Undo the LaTeX MinerU produces for numbers in tables and formulas.

    ``6.4 × 10⁻³`` arrives as ``$6 . 4 \\times 1 0 ^ { - 3 }$``, a space between every character, and the
    prompt's "verbatim" rule keeps it that way. Spaces between digits are collapsed only when the text
    carries a LaTeX marker, so ordinary "10 20" is left alone.
    """
    for command, symbol in LATEX_SYMBOLS.items():
        text = text.replace(command, symbol)
    text = _RAISED_DEGREE.sub("°", text)
    # A digit wrapped in a formatting command ("2 3 \\mathbf { 0 }", MinerU bolding a table cell's last digit)
    # is unwrapped first, so the run of spaced digits below still reads as one number.
    text = _WRAPPED_DIGITS.sub(r"\1", LATEX_WRAPPERS.sub("", text)) if "\\" in text else text
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
    """``(value, note)``: the number ``raw`` spells, or None with the reason it was refused.

    A qualifier ("~", ">", "about") is dropped and recorded first; what is left must then match one of the
    spellings in :data:`_SPELLINGS`, tried in order. Each spelling either claims the text -- with a value, or
    with a refusal -- or passes it on. A refusal is always better than a guess: the comparison turns None
    into AMBIGUOUS, while a wrong number is indistinguishable from a real measurement.
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
    value, reading = _read(text)
    return value, _join([*notes, *reading])


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


_Reading = tuple[float | None, list[str]]


def _read(text: str) -> _Reading:
    for spelling in _SPELLINGS:
        reading = spelling(text)
        if reading is not None:
            return reading
    raise AssertionError("the last spelling always answers")  # pragma: no cover


def _refuse(reason: str) -> _Reading:
    return None, [reason]


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
    return value, ["uncertainty dropped"] if match.group("pm") else []


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
    value, reading = _read(rest)
    return value, ["parenthesized alternative ignored", *reading]


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
            return values[0], ["uncertainty dropped"]
        if not NUMBER_RE.search(rest) and _RANGE_SEPARATOR.fullmatch(between):
            low, high = values
            if low < high:
                return (low + high) / 2, [f"range {low:g}-{high:g} → midpoint"]
            return _refuse("descending range in scientific notation; ambiguous")
    if len(matches) > 1 or NUMBER_RE.search(rest):
        return _refuse("numbers outside the scientific notation; ambiguous")
    return values[0], []


def _uncertainty(text: str) -> _Reading | None:
    match = _PLUS_MINUS.match(text)
    return None if match is None else (float(_plain(match.group("a"))), ["uncertainty dropped"])


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
    return (low + high) / 2, [*notes, f"range {low:g}-{high:g} → midpoint"]


def _first_number(text: str) -> _Reading:
    """The fallback: one number is the value. Several separated only by spaces or a multiplication sign ("300
    500", "40 x 10 cm") keep the first with a note. A range buried among other numbers, a list ("30, 40"), and a
    number carrying its own unit before another ("550 nm: 85%", "140 nm ATO/25 nm ITO") have no first value
    worth keeping: each is a second quantity beside the first, and they are refused."""
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return _refuse("no number found")
    if len(numbers) == 1:
        return float(_plain(numbers[0])), []
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
    return float(_plain(numbers[0])), [f"{len(numbers)} numbers found, first used"]


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
    "Pa": _pressure,
}


def check_canonical_units(specs: tuple[FieldSpec, ...], source: Path) -> None:
    """Refuse a field whose canonical unit nothing here can convert into, naming the field and the file.

    Checked at import rather than as a KeyError buried in normalisation, field by field. It lives here and
    not in fields.py because the converters do, and fields.py cannot import this module.
    """
    for spec in specs:
        if spec.canonical_unit and spec.canonical_unit not in CONVERTERS:
            raise ConfigError(
                f"{source}: field {spec.name!r}: canonical_unit {spec.canonical_unit!r} has no converter; "
                f"known units are {', '.join(CONVERTERS)}"
            )


check_canonical_units(FIELD_SPECS, FIELDS_SOURCE)


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
        return 1 / factor, unit_of(tail)  # "ρ × 10^4 (Ω cm)": ρ multiplied, the unit bracketed apart
    if tail and tail[0] not in _OPENING:
        return factor, unit_of(tail)  # "ρ × 10^-4 Ω·cm": the factor leads the unit written after it
    return None, clean_unit(text)  # "ρ × 10^4": on the quantity, but no unit says so


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
    scale, unit = split_scale_factor(unit_raw, CONVERTERS[canonical])
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
    factor = CONVERTERS[canonical](unit)
    if factor is None:
        return None, None, f"unknown unit {unit_raw!r} for {canonical}"
    return value * factor, canonical, scale_note


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


def compound_value(spec: FieldSpec, text: str) -> float | None:
    """The canonical value of a duration spelled in two of its units, larger first ("3 h 30 min" -> 210), or
    None for anything else. The smaller part must be less than one of the larger unit: "1 h 90 min" is no
    way anyone writes 150 minutes. Both units carry their own factor, so the model's ``unit_raw`` -- which can
    name only one of them -- plays no part. ``text`` is the value alone: callers set aside what surrounds it.

    The one reader of compound durations, for the comparison (:func:`normalize_field`) and for the dataset
    cell (``decide``) alike, so the two never read one string differently."""
    if spec.canonical_unit not in _SUMMED_UNITS:
        return None
    match = _COMPOUND.match(normalize_text(text).strip())
    if match is None:
        return None
    convert = CONVERTERS[spec.canonical_unit]
    big, small = convert(match.group("ua")), convert(match.group("ub"))
    if big is None or small is None or big <= small:
        return None
    part = float(_plain(match.group("b"))) * small
    if part >= big:
        return None
    return float(_plain(match.group("a"))) * big + part


def _names_unit_of(spec: FieldSpec, text: str) -> bool:
    """Whether ``text`` names a unit of the field's own quantity ("2 h" for a time, "°C" for a temperature)."""
    if spec.canonical_unit is None:
        return False
    convert = CONVERTERS[spec.canonical_unit]
    return any(convert(word) is not None for word in _TAIL_WORD.findall(normalize_text(text)))


def normalize_field(field: FieldValue, spec: FieldSpec) -> FieldValue:
    if spec.kind != "numeric":
        # Text and composition fields are compared through normalize_key on the fly.
        return field
    # A number word is decided here, not in parse_number: only the unit tells "four-inch" from "ten-fold".
    spelled = spell_number_word(field.value_raw, field.unit_raw)
    word_note = f"number word {field.value_raw.strip()!r} read as {spelled}" if spelled != field.value_raw else None
    bare, context_notes, condition = set_aside(spelled)
    if condition and _names_unit_of(spec, condition) and not _names_unit_of(spec, bare):
        # "400 °C for 2 h" on annealing_time: the time is in the tail, and the number kept is a temperature.
        note = f"the condition {condition!r} holds this field's quantity and the value does not; ambiguous"
        return field.model_copy(update={"value": None, "unit": None, "normalization_note": note})
    compound = compound_value(spec, bare)
    if compound is not None:
        compound_note = f"compound {bare!r} read as {compound:g} {spec.canonical_unit}"
        note = "; ".join(n for n in (word_note, *context_notes, compound_note) if n)
        return field.model_copy(update={"value": compound, "unit": spec.canonical_unit, "normalization_note": note})
    number, parse_note = parse_number(spelled)
    if number is None:
        return field.model_copy(update={"value": None, "unit": None, "normalization_note": parse_note})
    value, unit, unit_note = convert_to_canonical(spec, number, field.unit_raw, value_text=spelled)
    note = "; ".join(n for n in (word_note, parse_note, unit_note) if n) or None
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
