"""Run one lane's Markdown through the model and return a cleaned, grounded LaneExtraction.

Both parser lanes go through this same function, with the same prompt, the same model and temperature 0.
The only thing that differs between them is the Markdown, which is the point: any disagreement downstream
has to come from the parsers, not from prompt noise.

What the model returns is treated as a claim, not a result. Three mechanisms check it:

* **Schema and type cleaning** -- unknown fields, and numeric fields holding words, are dropped.
* **Citation validation** -- ids that are not in the document the model was shown are stripped and audited.
* **Grounding** -- the quoted text has to actually occur in the block it cites. Without this, "traceable to
  a page and a bounding box" only means the model named a real block, not that the value came from it.

Optionally the whole thing runs several times and keeps what a majority of passes agree on; see
:mod:`paperfacts.extraction.passes`.
"""

from __future__ import annotations

import json
import logging
from functools import cache
from pathlib import Path

from paperfacts.errors import PaperFactsError
from paperfacts.extraction.document import (
    CHARS_PER_TOKEN,
    ExtractionDocument,
    build_extraction_document,
    document_fingerprint,
)
from paperfacts.extraction.fields import FIELD_BY_NAME, schema_fingerprint
from paperfacts.extraction.grounding import ground_lane
from paperfacts.extraction.llm import DEFAULT_MAX_TOKENS, LlmClient, complete_validated
from paperfacts.extraction.passes import merge_passes
from paperfacts.extraction.prompts import (
    PROMPT_VERSION,
    extraction_system_prompt,
    extraction_user_prompt,
    repair_prompt,
)
from paperfacts.extraction.records import ExtractedRecords, ExtractionResponse, LaneExtraction, response_to_records
from paperfacts.fingerprint import content_fingerprint, source_fingerprint
from paperfacts.models.artifact import ParsedArtifact

logger = logging.getLogger(__name__)

# deepseek-chat serves a 64K context; leave headroom so an under-estimate cannot silently truncate.
DEFAULT_CONTEXT_TOKENS = 60_000


class ContextBudgetError(PaperFactsError):
    """The paper does not fit in the model's context window."""


# Modules whose logic is baked into a stored extraction: they decide which of the model's claims survive.
# Re-deriving records after changing them is free, because the model's answer is itself cached by payload.
_CLEANING_MODULES = ("records.py", "passes.py")


@cache
def cleaning_fingerprint() -> str:
    """Hash of the code that turns a model response into stored records."""
    folder = Path(__file__).parent
    return source_fingerprint(*(folder / name for name in _CLEANING_MODULES))


def extractor_key(model: str, *, passes: int = 1) -> str:
    """Content hash of everything that determines the extraction result.

    Covers the model, the field schema, the extraction prompt, how the document is rendered for it, and
    the rules that decide which of the model's claims survive. The *matching* prompt is deliberately absent:
    it drives sample pairing, which lives in the comparison report, so tuning it must not throw away the
    expensive per-lane extractions. Change any of them and the cached extraction is
    invalidated automatically, which is safer than remembering to bump a version. ``passes`` is only mixed
    in when it is not the default, so keys written by single-pass runs stay stable.
    """
    material = {
        "model": model,
        "schema": schema_fingerprint(),
        "prompt_version": PROMPT_VERSION,
        "extraction_system": extraction_system_prompt(),
        "document": document_fingerprint(),
        "cleaning": cleaning_fingerprint(),
    }
    if passes != 1:
        material["passes"] = passes
    return content_fingerprint(json.dumps(material, ensure_ascii=False, sort_keys=True))


def extract_lane(
    artifact: ParsedArtifact,
    client: LlmClient,
    *,
    passes: int = 1,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    refresh: bool = False,
) -> LaneExtraction:
    """Extract sample-level facts from one parser lane."""
    if passes < 1:
        raise ValueError(f"passes must be at least 1, got {passes}")
    document = build_extraction_document(artifact)
    system = extraction_system_prompt()
    user = extraction_user_prompt(document.markdown)
    _check_context_budget(system, user, context_tokens=context_tokens)

    results, usage, raw_response = _run_passes(client, system, user, document=document, passes=passes, refresh=refresh)
    records = merge_passes(results)
    lane = LaneExtraction(
        document_id=artifact.document_id,
        backend=artifact.backend,
        extractor_key=extractor_key(client.model, passes=passes),
        model=client.model,
        schema_version=schema_fingerprint(),
        target=records.target,
        samples=records.samples,
        invalid_source_ids=records.invalid_source_ids,
        dropped=records.dropped,
        passes=passes,
        usage=usage,
        raw_response=raw_response,
    )
    lane = ground_lane(lane, document.blocks)
    _log_outcome(lane)
    return lane


def _run_passes(
    client: LlmClient,
    system: str,
    user: str,
    *,
    document: ExtractionDocument,
    passes: int,
    refresh: bool,
) -> tuple[list[ExtractedRecords], dict[str, int], str]:
    """Ask the same question ``passes`` times and clean each answer. Returns the records, summed token
    usage, and the first raw response as evidence."""
    known_ids = frozenset(document.blocks)
    results: list[ExtractedRecords] = []
    usage: dict[str, int] = {}
    raw_response = ""
    for index in range(passes):
        # Identical payload every time; only the cache key differs, so repeats are free but not identical.
        response, text, pass_usage = complete_validated(
            client,
            ExtractionResponse,
            system=system,
            user=user,
            repair=lambda previous, error: repair_prompt(user, previous, error),
            refresh=refresh,
            cache_salt="" if index == 0 else f"pass-{index}",
        )
        results.append(response_to_records(response, known_ids=known_ids, known_fields=FIELD_BY_NAME))
        raw_response = raw_response or text
        for key, value in pass_usage.items():
            usage[key] = usage.get(key, 0) + value
    return results, usage, raw_response


def _log_outcome(lane: LaneExtraction) -> None:
    for entry in lane.dropped:
        logger.warning("dropped extracted value backend=%s: %s", lane.backend, entry)
    ungrounded = lane.ungrounded()
    for value in ungrounded:
        logger.warning(
            "ungrounded value backend=%s field=%s value=%r cited=%s",
            lane.backend,
            value.field,
            value.value_raw,
            ",".join(value.source_ids) or "nothing",
        )
    logger.info(
        "extracted backend=%s doc=%s passes=%d samples=%d values=%d ungrounded=%d invalid_ids=%d dropped=%d usage=%s",
        lane.backend,
        lane.document_id[:16],
        lane.passes,
        len(lane.samples),
        len(lane.values()),
        len(ungrounded),
        len(lane.invalid_source_ids),
        len(lane.dropped),
        lane.usage,
    )


def _check_context_budget(system: str, user: str, *, context_tokens: int) -> None:
    """Fail before spending money when the prompt cannot fit, instead of letting the API truncate it."""
    estimate = int((len(system) + len(user)) / CHARS_PER_TOKEN) + 1
    budget = context_tokens - DEFAULT_MAX_TOKENS
    if estimate > budget:
        raise ContextBudgetError(
            f"the paper needs roughly {estimate} prompt tokens but only {budget} are available "
            f"({context_tokens} of context minus {DEFAULT_MAX_TOKENS} reserved for the reply). "
            "Use a model with a larger context via PAPERFACTS_LLM_MODEL, raise "
            "PAPERFACTS_LLM_CONTEXT_TOKENS if the model allows it, or split the PDF."
        )
