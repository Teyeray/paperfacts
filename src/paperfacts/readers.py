"""The readers of a quote that is more than one number: a clean range, an interval (a range or a one-sided bound)
and a calendar date. Each is built on :func:`paperfacts.normalize.read_number`'s reading of the quote, so a range is
the same thing to the lanes, the dataset cell and an interval field.

Split from :mod:`paperfacts.normalize` only to keep either module readable; like it, this module decides values
and verdicts, so its source is hashed wherever ``normalize.py`` is (:mod:`paperfacts.keys`).
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable

from paperfacts.grounding import LOWER_BOUND_WORDS, UPPER_BOUND_WORDS
from paperfacts.normalize import NumberReading, read_number, typeset
from paperfacts.text import normalize_text


def read_range(raw: str, range_unit: Callable[[str], bool]) -> tuple[float, float, str] | None:
    """``(low, high, unit)`` when ``raw`` is one clean range, else None: the one definition of a range with two
    printed ends, for the lanes (:func:`read_number` under ``range_policy`` lower/upper) and the dataset cell
    (``kinds``) alike, so the two never disagree about which quotes have an end.

    A range is what the general reader reads as one (:func:`read_number`): two ascending numbers, both plain or
    both in scientific notation, joined by a range separator. Clean means nothing else is around it but a unit
    after it that ``range_unit`` accepts as the field's ("" when the range has none). A qualifier of
    approximation or a name before "=" may precede it (``normalize.set_aside``); a bound ("> 450-500", "below 1.2e-4 -
    1.5e-4"), a condition ("450-500 °C for 2 h"), an "after" clause, a parenthesis ("450-500 (600)"), another unit
    ("450-500 K" on a ℃ field) and an exponent written once for two numbers ("1.2-1.5 × 10^-3") are not."""
    return _clean_ends(read_number(raw), range_unit)


def _clean_ends(reading: NumberReading, range_unit: Callable[[str], bool]) -> tuple[float, float, str] | None:
    """:func:`read_range` of a quote already read: its ends and unit do not depend on the range policy it was read
    under."""
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
    # One reading serves the range and, when the quote is neither a range nor a bound, the reason it is no interval.
    whole = read_number(raw, range_policy="reject")
    clean = _clean_ends(whole, range_unit)
    if clean is not None:
        return clean, None
    match = _ONE_SIDED.match(typeset(raw))
    if match is None:
        number, note = whole.value, whole.note
        return None, _NO_INTERVAL if number is not None else "; ".join(n for n in (note, _NO_INTERVAL) if n)
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
