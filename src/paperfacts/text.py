"""Text folding shared by normalisation, retrieval, grounding and units: one spelling for each character.

A leaf module -- it imports nothing of the package -- so :mod:`paperfacts.units` and :mod:`paperfacts.passages`
can fold text exactly as :mod:`paperfacts.normalize` does without importing it. Its source is hashed wherever
normalize.py's was: a folding rule decides which spellings are the same, and so which values are.
"""

from __future__ import annotations

import re
import unicodedata

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


def is_word_edge(character: str) -> bool:
    """Whether a pattern ending (or starting) on ``character`` may be closed by ``\\b``: a letter or digit of a
    script that separates its words with spaces. Chinese and Japanese run their words together, so ``\\b``
    next to one of their characters demands a break that is never written and the pattern never matches."""
    return character.isalnum() and unicodedata.east_asian_width(character) not in ("W", "F")


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


def clean_unit(unit_raw: str) -> str:
    """Whitespace and decoration stripped, case preserved; no interpretation."""
    return normalize_text(unit_raw).replace(" ", "").rstrip(".")
