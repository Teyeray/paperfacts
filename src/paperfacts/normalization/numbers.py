"""Parse the many numeric spellings found in papers into a float: scientific notation, approx/inequality
qualifiers, uncertainty ranges, ranges, parenthesized alternatives, thousands separators.

Returns ``(value, note)``; ``value`` is None when parsing fails, and ``note`` explains why. The note is
written into provenance so "why couldn't the two lanes be compared" stays traceable.
"""

from __future__ import annotations

import re

from paperfacts.normalization.text import normalize_text

_NUM = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+\.\d*|\.\d+|\d+)"
# Scientific notation only matches three explicit spellings ("2108" must never be read as 2x10^8):
#   1.2 x 10^-4 / 1.2x10^-4 / 1.2 x 10-4 (explicit "x 10" survives OCR losing the superscript; ⁻⁴ and
#     <sup>-4</sup> have already been converted to ^-4 by the time this regex runs)
#   10^-4 (no mantissa, so the "^" is mandatory)
#   1.2e-4 / 1.2E-4
_SCI = re.compile(
    rf"(?P<m>{_NUM})\s*x\s*10\s*\^?\s*(?P<e>[-+]?\d+)"
    rf"|10\s*\^\s*(?P<e1>[-+]?\d+)"
    rf"|(?P<e2m>{_NUM})[eE](?P<e2>[-+]?\d+)"
)
_RANGE = re.compile(rf"^(?P<a>{_NUM})\s*(?:-|to|~)\s*(?P<b>{_NUM})$")
_PLUS_MINUS = re.compile(rf"^(?P<a>{_NUM})\s*(?:\+/-|±|\+-)\s*{_NUM}")
_NUMBER = re.compile(_NUM)
# Multi-character qualifiers must come first, or "<=" gets swallowed by the lone "<" in the character class
_QUALIFIERS = re.compile(r"^(?P<q>>=|<=|ca\.?|approx\.?|about|[~≈≃≅≥≤<>])\s*", re.IGNORECASE)


_LATEX_MARKERS = ("$", "\\")
_LATEX_COMMANDS = {r"\times": " x ", r"\cdot": " x ", r"\,": " ", r"\;": " ", "\\ ": " "}
_DIGIT_GAP = re.compile(r"(?<=[0-9.])\s+(?=[0-9.])")
_CARET_GAP = re.compile(r"\^\s*([-+]?)\s*(?=\d)")


def delatex(text: str) -> str:
    """Undo the LaTeX MinerU produces for numbers copied out of tables/formulas, back into plain notation.

    MinerU renders ``6.4 × 10⁻³`` as ``$6 . 4 \\times 1 0 ^ { - 3 }$`` (a space between every character);
    since the prompt asks for a verbatim transcription, that shape ends up unchanged in ``value_raw``.
    Spaces between digits are only collapsed when the text carries a LaTeX marker (``$`` or a backslash
    command), so ordinary text like "10 20" is left alone.
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

    # A parenthesized alternative value (e.g. "12 (60)", "108 (107)"): take the value outside the parens
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
        # Only a < b counts as a range; something like "10-4" isn't (it's probably 10^-4 that lost its
        # superscript), so it falls through to be treated as a plain number below
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
