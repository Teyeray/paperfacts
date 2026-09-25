"""Sample identity matching: exact-pair by sample_key first, then hand the rest to the LLM
(which must supply justification and a confidence score).

When the model fails to produce valid JSON on both attempts, this does **not** pretend "everything is
unmatched": :attr:`SampleMatching.failed` is set to True instead, and the comparison layer uses that to
mark the unmatched samples' fields ambiguous (flagged for review) rather than missing (silently accepted)
— when uncertain, lean toward the lower-risk outcome.
"""

from __future__ import annotations

import logging
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from paperfacts.errors import LlmResponseError
from paperfacts.llm import LlmClient, complete_validated
from paperfacts.profile import DomainProfile
from paperfacts.prompts import matching_system_prompt, matching_user_prompt, repair_prompt
from paperfacts.records import FieldValue, LaneExtraction, SampleRecord, sample_key

logger = logging.getLogger(__name__)


class SampleMatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    a_id: str
    b_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str
    method: Literal["exact", "llm"]


class SampleMatching(BaseModel):
    """The correspondence between the two lanes' sample lists."""

    model_config = ConfigDict(frozen=True)

    pairs: tuple[SampleMatch, ...] = ()
    unmatched_a: tuple[str, ...] = ()
    unmatched_b: tuple[str, ...] = ()
    failed: bool = Field(
        default=False,
        description="the matching model failed to produce valid JSON on both attempts; when true, "
        "'unmatched' does not mean the two lanes truly have no match",
    )
    failure: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    raw_response: str = ""

    @classmethod
    def trivial(cls, pairs: tuple[SampleMatch, ...], rest_a: list[SampleRecord], rest_b: list[SampleRecord]) -> Self:
        return cls(
            pairs=pairs,
            unmatched_a=tuple(s.sample_id for s in rest_a),
            unmatched_b=tuple(s.sample_id for s in rest_b),
        )


# ---- LLM response models -----------------------------------------------------------------


class _ResponsePair(BaseModel):
    model_config = ConfigDict(extra="ignore")

    a: str
    b: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    justification: str = ""


class _MatchingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pairs: list[_ResponsePair] = Field(default_factory=list)
    unmatched_a: list[str] = Field(default_factory=list)
    unmatched_b: list[str] = Field(default_factory=list)


def match_samples(
    lane_a: LaneExtraction,
    lane_b: LaneExtraction,
    client: LlmClient,
    profile: DomainProfile,
    *,
    refresh: bool = False,
) -> SampleMatching:
    exact, rest_a, rest_b = _exact_pairs(lane_a.samples, lane_b.samples)
    if not rest_a or not rest_b:
        return SampleMatching.trivial(exact, rest_a, rest_b)

    user = matching_user_prompt(lane_a.backend, _render(rest_a), lane_b.backend, _render(rest_b))
    try:
        response, raw_text, usage = complete_validated(
            client,
            _MatchingResponse,
            system=matching_system_prompt(profile),
            user=user,
            repair=lambda previous, error: repair_prompt(user, previous, error),
            refresh=refresh,
        )
    except LlmResponseError as exc:
        logger.error("sample matching failed, leaving %d/%d samples unmatched: %s", len(rest_a), len(rest_b), exc)
        return SampleMatching.trivial(exact, rest_a, rest_b).model_copy(update={"failed": True, "failure": str(exc)})

    llm_pairs, seen_a, seen_b = _accept_llm_pairs(
        response, {s.sample_id for s in rest_a}, {s.sample_id for s in rest_b}
    )
    matching = SampleMatching(
        pairs=exact + llm_pairs,
        unmatched_a=tuple(s.sample_id for s in rest_a if s.sample_id not in seen_a),
        unmatched_b=tuple(s.sample_id for s in rest_b if s.sample_id not in seen_b),
        usage=usage,
        raw_response=raw_text,
    )
    logger.info(
        "matched samples exact=%d llm=%d unmatched_a=%d unmatched_b=%d",
        len(exact),
        len(llm_pairs),
        len(matching.unmatched_a),
        len(matching.unmatched_b),
    )
    return matching


def _exact_pairs(
    samples_a: tuple[SampleRecord, ...], samples_b: tuple[SampleRecord, ...]
) -> tuple[tuple[SampleMatch, ...], list[SampleRecord], list[SampleRecord]]:
    """Pair samples directly when their sample_key is identical (deterministic — costs nothing
    and introduces no model noise)."""
    by_key_b = {sample_key(s.sample_id): s for s in samples_b}
    pairs: list[SampleMatch] = []
    used_b: set[str] = set()
    rest_a: list[SampleRecord] = []
    for sample in samples_a:
        key = sample_key(sample.sample_id)
        match = by_key_b.get(key) if key else None
        if match is not None and match.sample_id not in used_b:
            pairs.append(
                SampleMatch(
                    a_id=sample.sample_id,
                    b_id=match.sample_id,
                    confidence=1.0,
                    justification="identical sample_id",
                    method="exact",
                )
            )
            used_b.add(match.sample_id)
        else:
            rest_a.append(sample)
    rest_b = [s for s in samples_b if s.sample_id not in used_b]
    return tuple(pairs), rest_a, rest_b


def _accept_llm_pairs(
    response: _MatchingResponse, ids_a: set[str], ids_b: set[str]
) -> tuple[tuple[SampleMatch, ...], set[str], set[str]]:
    """Accept only pairs where both ids genuinely exist and each id appears at most once; everything else
    is treated as unmatched."""
    pairs: list[SampleMatch] = []
    seen_a: set[str] = set()
    seen_b: set[str] = set()
    for pair in response.pairs:
        if pair.a not in ids_a or pair.b not in ids_b or pair.a in seen_a or pair.b in seen_b:
            logger.warning("ignoring invalid or duplicate pair from LLM: %r ↔ %r", pair.a, pair.b)
            continue
        pairs.append(
            SampleMatch(
                a_id=pair.a,
                b_id=pair.b,
                confidence=pair.confidence,
                justification=pair.justification.strip(),
                method="llm",
            )
        )
        seen_a.add(pair.a)
        seen_b.add(pair.b)
    return tuple(pairs), seen_a, seen_b


def _render(samples: list[SampleRecord]) -> str:
    """A compact sample listing for the matching prompt: id, label, conditions, and each field's raw text."""
    lines = []
    for sample in samples:
        conditions = "; ".join(f"{k}={v}" for k, v in sample.conditions.items()) or "-"
        fields = "; ".join(_render_field(f) for f in sample.fields) or "-"
        lines.append(
            f"- id: {sample.sample_id} | label: {sample.label or '-'} | conditions: {conditions} | fields: {fields}"
        )
    return "\n".join(lines)


def _render_field(field: FieldValue) -> str:
    unit = f" {field.unit_raw}" if field.unit_raw else ""
    condition = f" @{field.condition}" if field.condition else ""
    return f"{field.field}={field.value_raw}{unit}{condition}"
