"""Retry loop for low-confidence extractions flagged by the supervisor."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from paperfacts.errors import SupervisorError

if TYPE_CHECKING:
    from paperfacts.fields import FieldSpec
    from paperfacts.llm import LlmClient
    from paperfacts.records import FieldValue
    from paperfacts.supervisor import SupervisorClient, SupervisorResult

logger = logging.getLogger(__name__)


def retry_with_supervisor(
    field: FieldSpec,
    value: FieldValue,
    passage: str,
    client: LlmClient,
    supervisor: SupervisorClient,
    *,
    max_retries: int = 2,
    retry_threshold: float = 0.6,
) -> tuple[FieldValue, SupervisorResult]:
    """Retry extraction when supervisor score falls below threshold.

    Returns the final value (may be unchanged) and its supervisor result.
    On retry exhaustion, returns original value with low_confidence=True.
    """
    result = supervisor.score_value(field, value, passage, request_critique=False)

    if result.score >= retry_threshold:
        return value, result

    # Score too low — retry with critique
    for attempt in range(max_retries):
        logger.info(
            "supervisor score %.2f < %.2f for field=%s; retry %d/%d",
            result.score,
            retry_threshold,
            field.name,
            attempt + 1,
            max_retries,
        )

        # Get detailed critique
        try:
            result = supervisor.score_value(field, value, passage, request_critique=True)
        except SupervisorError as exc:
            logger.warning("supervisor critique request failed: %s; keeping original value", exc)
            break

        if result.score >= retry_threshold:
            logger.info("retry succeeded: score improved to %.2f", result.score)
            return value, result

        # TODO: Augment extraction prompt with critique and re-ask LLM
        # This requires integration with extract.py's per-field question loop
        # For now, just log and continue
        logger.debug("retry %d failed, score still %.2f: %s", attempt + 1, result.score, result.critique)

    # Retries exhausted — flag as low confidence
    logger.info(
        "supervisor retries exhausted for field=%s; keeping original value with low_confidence=True",
        field.name,
    )
    flagged = value.model_copy(update={"low_confidence": True, "supervisor_score": result.score})
    return flagged, result
