"""Run one lane's blocks through the model and return a cleaned, grounded :class:`LaneExtraction`.

Both lanes go through this same function with the same prompts, model and temperature 0. The only thing
that differs is the text, which is the point: any disagreement downstream comes from the parsers.

Two ways of asking, chosen by ``mode``:

- **document** hands the whole filtered paper over and asks for everything at once. One call, but the model
  has to find nine fields and every sample in fifteen thousand tokens, and it cites the wrong block when it
  loses its place.
- **passage** asks which samples exist, then asks about one field at a time, showing only the blocks
  :mod:`paperfacts.passages` retrieved for that field. Ten small calls instead of one large one: the model
  cannot cite a block it was not shown, each answer is cached on its own, and a paper too long for the
  context window still fits.

What the model returns is a claim, not a result. Cleaning and citation validation happen in
:mod:`paperfacts.records`, grounding in :mod:`paperfacts.grounding`, and attribution to a sample here in
:func:`passage_records`. Optionally the whole thing runs several times and only what a majority of passes
agree on is kept (:func:`merge_passes`).

What the model reads is an LLM input only; the artifact remains the source of truth for the viewer. Every
piece of code that shapes it -- here, ``adapters.render_markdown`` and ``passages`` -- is hashed into
``extractor_key``, so changing any of it invalidates stored extractions.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum

from paperfacts.adapters import render_markdown
from paperfacts.config import (
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_LLM_CONTEXT_TOKENS,
    EXTRACTION_MODES,
    ExtractionMode,
)
from paperfacts.errors import ContextBudgetError
from paperfacts.fields import FIELD_SPECS, FieldSpec
from paperfacts.grounding import block_adjacency, ground_lane, grounding_key
from paperfacts.keys import extractor_key, schema_fingerprint
from paperfacts.llm import LlmClient, complete_validated
from paperfacts.models import Backend, ParsedArtifact, SourceBlock
from paperfacts.normalize import clean_unit, normalize_key
from paperfacts.passages import candidate_blocks, fit_budget, inventory_blocks
from paperfacts.prompts import (
    extraction_system_prompt,
    extraction_user_prompt,
    field_system_prompt,
    field_user_prompt,
    inventory_system_prompt,
    inventory_user_prompt,
    repair_prompt,
)
from paperfacts.records import (
    ExtractedRecords,
    ExtractionResponse,
    FieldResponse,
    FieldValue,
    InventoryResponse,
    InventorySample,
    LaneExtraction,
    ResponseCleaning,
    ResponseValue,
    SampleRecord,
    TargetRecord,
    response_to_records,
)

logger = logging.getLogger(__name__)

# Block types that never hold an extractable value: page furniture and figure blocks (whose content is an
# image path).
NOISE_TYPES: frozenset[str] = frozenset({"unknown", "figure"})
# Everything from this heading onwards is citations, not results.
_END_SECTION = re.compile(r"^(references|reference|bibliography|literature cited)\b", re.IGNORECASE)
# Measured on a real 10-page paper: 71.9K characters billed as 21.4K tokens, rounded down so the guard
# errs towards over-estimating.
CHARS_PER_TOKEN = 3.0
# Room left for everything in a passage-mode prompt that is not an excerpt: the instructions, the sample
# list and the JSON scaffolding. Only used to decide when to trim; the exact check happens after rendering.
PROMPT_OVERHEAD_TOKENS = 2_000


# ---- The document the model reads --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionDocument:
    """The text handed to the model, plus the block texts needed to check its citations afterwards."""

    markdown: str
    blocks: dict[str, str]
    kept_blocks: int
    dropped_blocks: int

    @property
    def estimated_tokens(self) -> int:
        return int(len(self.markdown) / CHARS_PER_TOKEN) + 1


def build_extraction_document(artifact: ParsedArtifact) -> ExtractionDocument:
    """Render the artifact for the prompt, dropping page furniture and the bibliography."""
    kept = _informative_blocks(artifact.blocks)
    document = ExtractionDocument(
        markdown=render_markdown(kept),
        blocks={block.source_id: block.content for block in kept},
        kept_blocks=len(kept),
        dropped_blocks=len(artifact.blocks) - len(kept),
    )
    logger.info(
        "extraction document backend=%s blocks=%d/%d chars=%d ~tokens=%d",
        artifact.backend,
        document.kept_blocks,
        len(artifact.blocks),
        len(document.markdown),
        document.estimated_tokens,
    )
    return document


def _informative_blocks(blocks: tuple[SourceBlock, ...]) -> list[SourceBlock]:
    kept: list[SourceBlock] = []
    for block in blocks:
        if block.type == "title" and _END_SECTION.match(block.content.lstrip("# ").strip()):
            break  # the bibliography and everything after it
        if block.type in NOISE_TYPES or not block.content.strip():
            continue
        kept.append(block)
    return kept


# ---- Extraction ---------------------------------------------------------------------------------------------


def extract_lane(
    artifact: ParsedArtifact,
    client: LlmClient,
    *,
    mode: ExtractionMode,
    passes: int = 1,
    context_tokens: int = DEFAULT_LLM_CONTEXT_TOKENS,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    refresh: bool = False,
) -> LaneExtraction:
    """Extract one parser lane, whole-document or question by question."""
    if passes < 1:
        raise ValueError(f"passes must be at least 1, got {passes}")
    if mode not in EXTRACTION_MODES:
        raise ValueError(f"unknown extraction mode: {mode!r}, expected one of {', '.join(EXTRACTION_MODES)}")
    blocks = _informative_blocks(artifact.blocks)
    document = build_extraction_document(artifact) if mode == "document" else None

    results: list[ExtractedRecords] = []
    usage: dict[str, int] = {}
    raw_response = ""
    for index in range(passes):
        # Every pass asks exactly the same question; only the cache key differs, so a repeat costs a call
        # but never a different prompt.
        cache_salt = "" if index == 0 else f"pass-{index}"
        if document is not None:
            records, pass_usage, text = _extract_whole_document(
                document, client, context_tokens=context_tokens, refresh=refresh, cache_salt=cache_salt
            )
        else:
            records, pass_usage, text = _extract_passages(
                blocks,
                client,
                backend=artifact.backend,
                context_tokens=context_tokens,
                candidate_limit=candidate_limit,
                refresh=refresh,
                cache_salt=cache_salt,
            )
        results.append(records)
        raw_response = raw_response or text
        for key, value in pass_usage.items():
            usage[key] = usage.get(key, 0) + value

    records = _deduplicate(merge_passes(results))
    lane = LaneExtraction(
        document_id=artifact.document_id,
        backend=artifact.backend,
        extractor_key=extractor_key(
            client.model,
            passes=passes,
            mode=mode,
            temperature=client.temperature,
            max_tokens=client.max_tokens,
            candidate_limit=candidate_limit,
        ),
        model=client.model,
        schema_version=schema_fingerprint(),
        target=records.target,
        samples=records.samples,
        invalid_source_ids=records.invalid_source_ids,
        dropped=records.dropped,
        unattributed=records.unattributed,
        passes=passes,
        usage=usage,
        raw_response=raw_response,
    )
    lane = ground_lane(lane, {block.source_id: block.content for block in blocks}, adjacency=block_adjacency(blocks))
    _log_outcome(lane)
    return lane


def _extract_whole_document(
    document: ExtractionDocument,
    client: LlmClient,
    *,
    context_tokens: int,
    refresh: bool,
    cache_salt: str,
) -> tuple[ExtractedRecords, dict[str, int], str]:
    """Document mode: one question carrying the whole filtered paper."""
    system = extraction_system_prompt()
    user = extraction_user_prompt(document.markdown)
    _check_context_budget(system, user, context_tokens=context_tokens, reply_tokens=client.max_tokens)
    response, text, usage = complete_validated(
        client,
        ExtractionResponse,
        system=system,
        user=user,
        repair=lambda previous, error: repair_prompt(user, previous, error),
        refresh=refresh,
        cache_salt=cache_salt,
    )
    return response_to_records(response, known_ids=frozenset(document.blocks)), usage, text


@dataclass(frozen=True)
class FieldHarvest:
    """One field's answer, with the ids it was allowed to cite.

    The permitted ids travel with the answer because they differ per field: each question was shown its own
    retrieved blocks, so "did the model invent this citation?" can only be judged against that set.
    """

    spec: FieldSpec
    values: tuple[ResponseValue, ...]
    known_ids: frozenset[str]


def _extract_passages(
    blocks: Sequence[SourceBlock],
    client: LlmClient,
    *,
    backend: Backend,
    context_tokens: int,
    candidate_limit: int,
    refresh: bool,
    cache_salt: str,
) -> tuple[ExtractedRecords, dict[str, int], str]:
    """Passage mode: one question about the samples, then one question per field."""
    budget_chars = int(max(context_tokens - client.max_tokens - PROMPT_OVERHEAD_TOKENS, 0) * CHARS_PER_TOKEN)
    selection = fit_budget(inventory_blocks(blocks), budget_chars=budget_chars)
    system = inventory_system_prompt()
    user = inventory_user_prompt(render_markdown(selection))
    _check_context_budget(system, user, context_tokens=context_tokens, reply_tokens=client.max_tokens)
    inventory, raw_text, usage = complete_validated(
        client,
        InventoryResponse,
        system=system,
        user=user,
        repair=lambda previous, error: repair_prompt(user, previous, error),
        refresh=refresh,
        cache_salt=cache_salt,
    )
    logger.info("inventory backend=%s samples=%d blocks=%d", backend, len(inventory.samples), len(selection))

    sample_list = _render_sample_list(inventory.samples)
    field_system = field_system_prompt()
    harvests: list[FieldHarvest] = []
    dropped: list[str] = []
    raw_parts = [f"# inventory\n{raw_text}"]
    for spec in FIELD_SPECS:
        candidates = fit_budget(candidate_blocks(spec, blocks, limit=candidate_limit), budget_chars=budget_chars)
        if not candidates:
            # Not an error: a paper that never mentions a target's density simply has none to find. Recorded
            # so that "the model missed it" and "we never asked" stay distinguishable.
            dropped.append(f"{spec.name}: no block in this lane mentions it, so it was not asked about")
            continue
        field_user = field_user_prompt(spec, sample_list, render_markdown(candidates))
        _check_context_budget(field_system, field_user, context_tokens=context_tokens, reply_tokens=client.max_tokens)
        response, text, field_usage = complete_validated(
            client,
            FieldResponse,
            system=field_system,
            user=field_user,
            # The default argument binds this field's question; a bare closure would repair against the last.
            repair=lambda previous, error, question=field_user: repair_prompt(question, previous, error),
            refresh=refresh,
            cache_salt=cache_salt,
        )
        harvests.append(
            FieldHarvest(
                spec=spec,
                values=tuple(response.values),
                known_ids=frozenset(block.source_id for block in candidates),
            )
        )
        raw_parts.append(f"# {spec.name}\n{text}")
        for key, value in field_usage.items():
            usage[key] = usage.get(key, 0) + value

    records = passage_records(
        inventory,
        harvests,
        inventory_ids=frozenset(block.source_id for block in selection),
        dropped=dropped,
    )
    return records, usage, "\n\n".join(raw_parts)


def _deduplicate(records: ExtractedRecords) -> ExtractedRecords:
    """Collapse repeats of the same fact, keeping every citation they brought.

    One field question routinely gets the same number back more than once: quoted from the table, and again
    from the sentence discussing it. They are one fact with two citations. Keeping both inflates every count
    and, worse, hands the comparison layer a duplicate to pair against -- it keeps the first value per
    condition, so a second, genuinely different reading of the same quantity (``10^-2`` against ``10^2``,
    a real OCR disagreement) can be silently dropped behind a duplicate of the first.
    """
    target = records.target
    if target is not None:
        target = target.model_copy(update={"fields": _merge_repeats(target.fields)})
    return records.model_copy(
        update={
            "target": target,
            "samples": tuple(
                sample.model_copy(update={"fields": _merge_repeats(sample.fields)}) for sample in records.samples
            ),
            "unattributed": _merge_repeats(records.unattributed),
        }
    )


def _merge_repeats(values: Sequence[FieldValue]) -> tuple[FieldValue, ...]:
    """One entry per distinct fact, in first-seen order, with the citations of its repeats merged in."""
    merged: dict[ValueKey, FieldValue] = {}
    for value in values:
        key = _value_key(value)
        previous = merged.get(key)
        if previous is None:
            merged[key] = value
            continue
        citations = tuple(dict.fromkeys((*previous.source_ids, *value.source_ids)))
        merged[key] = previous.model_copy(update={"source_ids": citations})
    return tuple(merged.values())


def _render_sample_list(samples: Sequence[InventorySample]) -> str:
    """The sample list as the field questions see it: id, label and the conditions that tell samples apart."""
    if not samples:
        return "(none were identified; use null for sample_id)"
    lines = []
    for sample in samples:
        conditions = "; ".join(f"{name}={value}" for name, value in sample.conditions.items()) or "-"
        lines.append(f"- id: {sample.sample_id} | label: {sample.label or '-'} | conditions: {conditions}")
    return "\n".join(lines)


def passage_records(
    inventory: InventoryResponse,
    harvests: Sequence[FieldHarvest],
    *,
    inventory_ids: frozenset[str],
    dropped: Sequence[str] = (),
) -> ExtractedRecords:
    """Assemble one pass of passage answers into records, placing each value on the sample it names.

    Attribution is by normalised sample id -- the same key that pairs samples across lanes -- so the model
    only has to repeat an id it was given. A sample-level value naming no sample, or one the inventory does
    not have, is kept in ``unattributed`` rather than attached to a plausible neighbour: an unplaced value
    is visible in the report, a misplaced one is indistinguishable from a real measurement.
    """
    cleaning = ResponseCleaning()
    cleaning.dropped.extend(dropped)

    samples: list[SampleRecord] = []
    sample_fields: list[list[FieldValue]] = []
    index_by_key: dict[str, int] = {}
    for item in inventory.samples:
        sample_id = item.sample_id.strip()
        key = normalize_key(sample_id)
        # min_length still admits "  ". A repeat means the model listed one sample twice under one id, and
        # every later value for it would land on the first: worth recording, not worth guessing about.
        if not sample_id:
            cleaning.dropped.append("inventory: a sample was listed with no usable id")
            continue
        if key in index_by_key:
            cleaning.dropped.append(f"inventory: sample {sample_id!r} repeats an id already listed")
            continue
        index_by_key[key] = len(samples)
        samples.append(
            SampleRecord(
                sample_id=sample_id,
                label=item.label.strip(),
                conditions={str(name).strip(): str(value).strip() for name, value in item.conditions.items()},
                source_ids=cleaning.keep_ids(item.source_ids, inventory_ids),
            )
        )
        sample_fields.append([])

    target_fields: list[FieldValue] = []
    target_ids: list[str] = []
    unattributed: list[FieldValue] = []
    for harvest in harvests:
        for item in harvest.values:
            value = cleaning.value(
                harvest.spec,
                value_raw=item.value_raw,
                unit_raw=item.unit_raw,
                condition=item.condition,
                source_ids=item.source_ids,
                note=item.note,
                known_ids=harvest.known_ids,
            )
            if value is None:
                continue
            if not harvest.spec.is_sample_level:
                # The question itself decided the scope, so a stray sample_id on a paper-level field is
                # noise rather than the scope error document mode has to guard against.
                target_fields.append(value)
                target_ids.extend(value.source_ids)
                continue
            index = index_by_key.get(normalize_key(item.sample_id)) if item.sample_id else None
            if index is None:
                unattributed.append(value)
                continue
            sample_fields[index].append(value)

    return ExtractedRecords(
        target=(
            TargetRecord(source_ids=tuple(dict.fromkeys(target_ids)), fields=tuple(target_fields))
            if target_fields
            else None
        ),
        samples=tuple(
            sample.model_copy(update={"fields": tuple(values)})
            for sample, values in zip(samples, sample_fields, strict=True)
        ),
        invalid_source_ids=tuple(sorted(cleaning.invalid)),
        dropped=tuple(cleaning.dropped),
        unattributed=tuple(unattributed),
    )


def _check_context_budget(system: str, user: str, *, context_tokens: int, reply_tokens: int) -> None:
    """Fail before spending money when the prompt cannot fit, instead of letting the API truncate it."""
    estimate = int((len(system) + len(user)) / CHARS_PER_TOKEN) + 1
    budget = context_tokens - reply_tokens
    if estimate > budget:
        raise ContextBudgetError(
            f"the paper needs roughly {estimate} prompt tokens but only {budget} are available "
            f"({context_tokens} of context minus {reply_tokens} reserved for the reply). "
            "Use a model with a larger context via PAPERFACTS_LLM_MODEL, raise "
            "PAPERFACTS_LLM_CONTEXT_TOKENS if the model allows it, or split the PDF."
        )


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
    for value in lane.unattributed:
        logger.warning(
            "unattributed value backend=%s field=%s value=%r: no sample in the inventory matches it",
            lane.backend,
            value.field,
            value.value_raw,
        )
    logger.info(
        "extracted backend=%s doc=%s passes=%d samples=%d values=%d ungrounded=%d unattributed=%d "
        "invalid_ids=%d dropped=%d usage=%s",
        lane.backend,
        lane.document_id[:16],
        lane.passes,
        len(lane.samples),
        len(lane.values()),
        len(ungrounded),
        len(lane.unattributed),
        len(lane.invalid_source_ids),
        len(lane.dropped),
        lane.usage,
    )


# ---- Majority vote over repeated passes ----------------------------------------------------------------------
# Even at temperature 0 the model is not deterministic across runs: repeated extractions drop or add the
# occasional value, and with one pass that noise is indistinguishable from parser disagreement. Each
# surviving value records the fraction of passes that produced it.


class Scope(Enum):
    """The two scopes a value can have that are not a sample.

    A sample's scope is its normalised id, a plain string, so these members cannot collide with one whatever
    a paper calls its samples -- a promise a reserved string like ``"__target__"`` could not make.
    """

    TARGET = "target"
    UNATTRIBUTED = "unattributed"


type ScopeKey = str | Scope
type ValueKey = tuple[str, str, str, str]


def merge_passes(results: Sequence[ExtractedRecords]) -> ExtractedRecords:
    """Keep the values a majority of ``results`` agree on, annotated with their agreement."""
    if len(results) == 1:
        return results[0]
    passes = len(results)
    majority = passes // 2 + 1

    counts: Counter[tuple[ScopeKey, ValueKey]] = Counter()
    exemplars: dict[tuple[ScopeKey, ValueKey], FieldValue] = {}
    sample_counts: Counter[str] = Counter()
    samples: dict[str, SampleRecord] = {}
    target_ids: tuple[str, ...] = ()
    for records in results:
        for key in {(scope, _value_key(value)) for scope, value in _values(records)}:
            counts[key] += 1
        for scope, value in _values(records):
            exemplars.setdefault((scope, _value_key(value)), value)
        for scope in {normalize_key(sample.sample_id) for sample in records.samples}:
            sample_counts[scope] += 1
        for sample in records.samples:
            samples.setdefault(normalize_key(sample.sample_id), sample)
        if records.target is not None and not target_ids:
            target_ids = records.target.source_ids

    kept: dict[ScopeKey, list[FieldValue]] = {}
    dropped = [entry for records in results for entry in records.dropped]
    for (scope, value_key), value in exemplars.items():
        count = counts[(scope, value_key)]
        if count < majority:
            dropped.append(f"{value.field}: only {count}/{passes} passes produced {value.value_raw!r}")
            continue
        kept.setdefault(scope, []).append(value.model_copy(update={"agreement": count / passes}))

    # Sample identity is voted on separately from its values: "this sample exists, under these conditions"
    # is itself a finding, kept even when none of its measurements survived.
    target_fields = tuple(kept.get(Scope.TARGET, ()))
    return ExtractedRecords(
        target=TargetRecord(source_ids=target_ids, fields=target_fields) if target_fields else None,
        samples=tuple(
            sample.model_copy(update={"fields": tuple(kept.get(scope, ()))})
            for scope, sample in samples.items()
            if sample_counts[scope] >= majority
        ),
        invalid_source_ids=tuple(sorted({sid for records in results for sid in records.invalid_source_ids})),
        dropped=tuple(dict.fromkeys(dropped)),
        # An unplaced value is voted on like any other: agreeing three times that it cannot be placed is
        # still agreement about the value itself.
        unattributed=tuple(kept.get(Scope.UNATTRIBUTED, ())),
    )


def _values(records: ExtractedRecords) -> Iterator[tuple[ScopeKey, FieldValue]]:
    for value in records.target.fields if records.target else ():
        yield Scope.TARGET, value
    for sample in records.samples:
        scope = normalize_key(sample.sample_id)
        for value in sample.fields:
            yield scope, value
    for value in records.unattributed:
        yield Scope.UNATTRIBUTED, value


def _value_key(value: FieldValue) -> ValueKey:
    """Identity for voting and for merging repeats: the same number in the same unit under the same
    condition, however it is spelled.

    The unit belongs in the identity: "2.1 μm" and "2.1 nm" are a thousand-fold disagreement, and treating
    them as one value would merge the disagreement away instead of reporting it.
    """
    unit = clean_unit(value.unit_raw) if value.unit_raw else ""
    return value.field, normalize_key(value.condition), grounding_key(value.value_raw), unit
