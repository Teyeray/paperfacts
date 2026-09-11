"""Normalisation: turns the raw text the model transcribed into comparable canonical values.

This layer is pure functions, millisecond-fast, and testable with hundreds of unit tests — the one place
in the whole system that can offer a determinism guarantee. The LLM's only job is to transcribe the source
text verbatim (``value_raw`` / ``unit_raw``); all conversion happens here instead, because LLM unit
conversion is not just occasionally wrong but wrong **silently**, and the two lanes fail in different ways
— which would flood CONFLICT with false positives that have nothing to do with the actual parsers.

:func:`normalization_fingerprint` hashes the source of this layer's four modules into a fingerprint that
feeds the comparison report's cache key: changing any rule automatically invalidates comparison results
already on disk (extraction results are unaffected, since what's stored there is the raw text).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from paperfacts.fingerprint import source_fingerprint
from paperfacts.normalization import numbers, records, text, units
from paperfacts.normalization.records import normalize_field, normalize_lane


@cache
def normalization_fingerprint() -> str:
    """Hash of the four modules' source — more reliable than a hand-maintained version number."""
    return source_fingerprint(*(Path(module.__file__) for module in (text, numbers, units, records)))


__all__ = ["normalization_fingerprint", "normalize_field", "normalize_lane"]
