"""Text normalization: unify Unicode variants, fold sub/superscripts back to ASCII, and build a key for
equality comparisons.

Order matters: superscript digits must be handled **before** NFKC — NFKC collapses "10⁻⁴" down to "10-4",
losing the fact that it was ever an exponent.
"""

from __future__ import annotations

import re
import unicodedata

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPT_RUN = re.compile(r"[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
# MinerU's Markdown writes sub/superscripts as HTML tags: SnO<sub>2</sub>, 10<sup>-4</sup>
_HTML_SUP = re.compile(r"<sup>\s*([^<]*?)\s*</sup>", re.IGNORECASE)
_HTML_SUB = re.compile(r"<sub>\s*([^<]*?)\s*</sub>", re.IGNORECASE)
# Only list variants NFKC does **not** already handle (OHM SIGN -> Ω, MICRO SIGN -> μ, NBSP -> space are
# already covered by NFKC; don't add them here)
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
    """Normalize spelling while preserving meaning: sub/superscripts -> ASCII, Unicode variants -> common
    characters, whitespace collapsed."""
    text = _HTML_SUP.sub(lambda m: f"^{m.group(1)}", text)
    text = _HTML_SUB.sub(lambda m: m.group(1), text)
    text = _ascii_superscripts(text).translate(_SUBSCRIPTS)
    text = unicodedata.normalize("NFKC", text)
    for source, target in _REPLACEMENTS.items():
        text = text.replace(source, target)
    return _SPACES.sub(" ", text).strip()


def normalize_key(text: str | None) -> str:
    """Build a key for "are these the same" comparisons: lowercase, strip whitespace and decorative
    punctuation. Blank input maps to an empty string.

    Only use this for text/condition/composition content; **never** for comparing units (lowercasing would
    collide mΩ with MΩ) — use :func:`units.clean_unit` for those.
    """
    if not text:
        return ""
    # .lower() turns Ω into ω; convert it back before the whitelist filter, or Ω would be stripped entirely
    return _NON_KEY.sub("", normalize_text(text).lower().replace("ω", "Ω"))
