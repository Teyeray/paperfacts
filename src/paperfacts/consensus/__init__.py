"""Alignment and comparison layer (M2).

Division of labor (fixed by the PRD): **sample identity is judged by the model** (:mod:`matching`, with
justification and a confidence score attached), while **rules only do field-level comparison**
(:mod:`compare`, treating values within tolerance as equal). Fuzzy matching lives exclusively in the
model layer; the rule layer never does semantic fuzziness.
"""

from paperfacts.consensus.compare import (
    AMBIGUOUS_MATCH_CONFIDENCE,
    ComparisonCounts,
    ComparisonReport,
    FieldComparison,
    compare_lanes,
    comparison_key,
)
from paperfacts.consensus.matching import SampleMatch, SampleMatching, match_samples

__all__ = [
    "AMBIGUOUS_MATCH_CONFIDENCE",
    "ComparisonCounts",
    "ComparisonReport",
    "FieldComparison",
    "SampleMatch",
    "SampleMatching",
    "compare_lanes",
    "comparison_key",
    "match_samples",
]
