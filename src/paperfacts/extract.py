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
agree on is kept, which :mod:`paperfacts.voting` decides.

What the model reads is an LLM input only; the artifact remains the source of truth for the viewer. Every
piece of code that shapes it -- here, ``adapters.render_markdown`` and ``passages`` -- is hashed into
``extractor_key``, so changing any of it invalidates stored extractions.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from paperfacts.adapters import render_markdown
from paperfacts.config import (
    DEFAULT_LLM_CONCURRENCY,
    EXTRACTION_MODES,
)
from paperfacts.errors import ContextBudgetError, LlmResponseError
from paperfacts.fields import FieldSpec
from paperfacts.grounding import block_adjacency, ground_lane
from paperfacts.keys import ExtractionOptions, extractor_key, profile_extraction_fingerprint
from paperfacts.llm import LlmClient, complete_validated
from paperfacts.models import Backend, ParsedArtifact, SourceBlock
from paperfacts.normalize import drop_implausible
from paperfacts.passages import candidate_blocks, fit_budget, inventory_blocks
from paperfacts.profile import DomainProfile
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
    FailedQuestion,
    FieldResponse,
    FieldValue,
    InventoryResponse,
    InventorySample,
    LaneExtraction,
    ResponseCleaning,
    ResponseModels,
    ResponseValue,
    TargetRecord,
    clean_samples,
    place_on_every_sample,
    response_models,
    response_to_records,
    sample_key,
)
from paperfacts.threads import ContextThreadPoolExecutor
from paperfacts.voting import deduplicate, merge_passes

logger = logging.getLogger(__name__)

# Block types that never hold an extractable value: page furniture and figure blocks (whose content is an
# image path).
NOISE_TYPES: frozenset[str] = frozenset({"unknown", "figure"})
# Everything from this heading onwards is citations, not results. The heading must be the whole block, with
# at most a section number in front ("6. References", "VI. REFERENCES"): a prefix match cut a paper at a
# section titled "Reference electrode", and without the number "6. References" kept the bibliography --
# either way depending on how one parser wrote the heading, which is a lane asymmetry.
_END_SECTION = re.compile(
    r"(?:(?:\d+|[ivx]+)(?:\.\d+)*\.?\s+)?"
    r"(?:references(?:\s+and\s+notes)?|notes\s+and\s+references|reference\s+list|bibliography|literature\s+cited)"
    r"\s*:?",
    re.IGNORECASE,
)
# The block types a heading arrives as: MinerU labels it a title, PaddleOCR-VL sometimes plain text. The
# whole-block match is what makes admitting text safe, and the plural is required: a lone "Reference" is as
# often a chart legend or a table column as a heading, and a false cut empties the rest of one lane only.
_HEADING_TYPES = frozenset({"title", "text"})
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
    kept = informative_blocks(artifact.blocks)
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


def bibliography_cut(blocks: Sequence[SourceBlock]) -> int | None:
    """The index of the references heading everything from which is citations, or None."""
    for index, block in enumerate(blocks):
        if block.type in _HEADING_TYPES and _END_SECTION.fullmatch(block.content.strip().strip("#* ")):
            return index
    return None


def informative_blocks(blocks: tuple[SourceBlock, ...]) -> list[SourceBlock]:
    """The blocks the model is shown, in reading order: no page furniture, no figures, no bibliography."""
    cut = bibliography_cut(blocks)
    if cut is not None and cut < len(blocks) / 2:
        # A cut early in the paper is either a very short paper or a heading misread as the references.
        logger.warning(
            "references heading %r at block %d of %d: everything after it is left out",
            blocks[cut].content.strip(),
            cut,
            len(blocks),
        )
    body = blocks if cut is None else blocks[:cut]
    return [block for block in body if block.type not in NOISE_TYPES and block.content.strip()]


def _bibliography_audit(blocks: tuple[SourceBlock, ...]) -> tuple[str, ...]:
    """The cut, as the lane's audit states it: which heading, and how much of the parse it removed. The two
    lanes cut at their own headings, so a cut that differs between them is visible where the values are."""
    cut = bibliography_cut(blocks)
    if cut is None:
        return ()
    heading = blocks[cut].content.strip()
    return (f"bibliography: {len(blocks) - cut} of {len(blocks)} blocks from the heading {heading!r} on were not read",)


# ---- Extraction ---------------------------------------------------------------------------------------------


def _add_usage(total: dict[str, int], part: Mapping[str, int]) -> None:
    """Accumulate one call's token counts into the lane's total."""
    for key, value in part.items():
        total[key] = total.get(key, 0) + value


def extract_lane(
    artifact: ParsedArtifact,
    client: LlmClient,
    options: ExtractionOptions,
    *,
    concurrency: int = DEFAULT_LLM_CONCURRENCY,
    refresh: bool = False,
) -> LaneExtraction:
    """Extract one parser lane, whole-document or question by question.

    ``options`` is everything that shapes a request, built once by the caller
    (``ExtractionOptions.from_settings``); the key recorded on the lane is ``extractor_key(options)``, the
    very key the caller stores the lane under and every reader looks it up by. The client sends the sampling
    settings, so it must have been built from the same values; a mismatch is refused here rather than
    stored under a key that does not describe the requests that produced it.

    ``concurrency`` only decides how many of passage mode's field questions wait on the network at once.
    Every request is the one the sequential loop would have sent, so it stays out of ``extractor_key``.

    ``options.inventory_reasoning_effort`` overrides the client's effort for the inventory question alone --
    the one question that reasons for far longer than the field questions after it. ``INHERIT`` leaves the
    request exactly as the client builds it, ``None`` sends that question with no such parameter at all, a
    value sends that effort. Both lanes get the same value, so the disagreement signal stays a comparison of
    two identically-asked lanes.
    """
    if options.passes < 1:
        raise ValueError(f"passes must be at least 1, got {options.passes}")
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    if options.mode not in EXTRACTION_MODES:
        raise ValueError(f"unknown extraction mode: {options.mode!r}, expected one of {', '.join(EXTRACTION_MODES)}")
    sent = (client.model, client.temperature, client.max_tokens, client.reasoning_effort)
    described = (options.model, options.temperature, options.max_tokens, options.reasoning_effort)
    if sent != described:
        raise ValueError(f"the client asks with {sent} but the extraction options describe {described}")
    mode, passes = options.mode, options.passes
    blocks = informative_blocks(artifact.blocks)
    document = build_extraction_document(artifact) if mode == "document" else None

    results: list[ExtractedRecords] = []
    failed: dict[str, FailedQuestion] = {}
    usage: dict[str, int] = {}
    raw_response = ""

    # Passage mode asks which samples exist exactly once for the whole lane, however many passes follow.
    # Repeating that question let the model name the same sample differently in each pass
    # ("ITO-O2-0.0sccm-480C" against "ITO-0.0sccm-480C"), and since a sample id is the scope its values are
    # voted under, no sample reached a majority and two passes returned an empty lane. One inventory makes
    # the sample list rendered into every pass's field questions byte-identical, so the ids match across
    # passes by construction and the sample vote in `merge_passes` becomes unanimous rather than useless.
    # What the passes repeat is what they are meant to measure: the field questions, whose noise is the
    # noise the vote exists to filter.
    inventory = (
        None
        if document is not None
        else _take_inventory(blocks, client, options, backend=artifact.backend, refresh=refresh)
    )
    if inventory is not None:
        # One call, counted once: charging it to every pass would misreport what the lane cost.
        _add_usage(usage, inventory.usage)

    for index in range(passes):
        # Every pass asks exactly the same question; only the cache key differs, so a repeat costs a call
        # but never a different prompt.
        cache_salt = "" if index == 0 else f"pass-{index}"
        if inventory is None:
            records, pass_usage, text = _extract_whole_document(
                document, client, options, refresh=refresh, cache_salt=cache_salt
            )
        else:
            records, pass_usage, text, pass_failed = _extract_passages(
                blocks,
                client,
                options,
                inventory=inventory,
                concurrency=concurrency,
                refresh=refresh,
                cache_salt=cache_salt,
            )
            for question in pass_failed:
                failed.setdefault(question.field, question)
        results.append(records)
        raw_response = raw_response or text
        _add_usage(usage, pass_usage)

    profile = options.profile
    records = drop_implausible(deduplicate(merge_passes(results)), profile)
    lane = LaneExtraction(
        document_id=artifact.document_id,
        backend=artifact.backend,
        extractor_key=extractor_key(options),
        model=client.model,
        profile_fingerprint=profile_extraction_fingerprint(profile),
        target=records.target,
        samples=records.samples,
        invalid_source_ids=records.invalid_source_ids,
        dropped=(*_bibliography_audit(artifact.blocks), *records.dropped),
        unattributed=records.unattributed,
        # Carried as data, not left to the audit text in `dropped`, so the web page can say why the lane is
        # empty without matching prose. The same condition that skips the sample-level questions below.
        no_tco_film=inventory is not None and _deposits_no_film(inventory.response),
        passes=passes,
        # In the profile's field order, whichever pass failed first.
        failed_questions=tuple(failed[spec.name] for spec in profile.fields if spec.name in failed),
        usage=usage,
        raw_response=raw_response,
    )
    lane = ground_lane(lane, {block.source_id: block.content for block in blocks}, adjacency=block_adjacency(blocks))
    _log_outcome(lane)
    return lane


def _extract_whole_document(
    document: ExtractionDocument,
    client: LlmClient,
    options: ExtractionOptions,
    *,
    refresh: bool,
    cache_salt: str,
) -> tuple[ExtractedRecords, dict[str, int], str]:
    """Document mode: one question carrying the whole filtered paper."""
    profile = options.profile
    system = extraction_system_prompt(profile)
    user = extraction_user_prompt(document.markdown)
    _check_context_budget(system, user, options)
    response, text, usage = complete_validated(
        client,
        _response_models(profile).extraction,
        system=system,
        user=user,
        repair=lambda previous, error: repair_prompt(user, previous, error),
        refresh=refresh,
        cache_salt=cache_salt,
    )
    records = response_to_records(response, fields=profile.by_name, known_ids=frozenset(document.blocks))
    return records, usage, text


def _response_models(profile: DomainProfile) -> ResponseModels:
    """The answer shapes under the JSON keys this profile's prompts tell the model to emit."""
    return response_models(profile.prompt.paper_key, profile.prompt.no_samples_key)


@dataclass(frozen=True)
class FieldHarvest:
    """One field's answer, with the ids it was allowed to cite.

    The permitted ids travel with the answer because they differ per field: each question was shown its own
    retrieved blocks, so "did the model invent this citation?" can only be judged against that set.
    """

    spec: FieldSpec
    values: tuple[ResponseValue, ...]
    known_ids: frozenset[str]


@dataclass(frozen=True)
class SampleInventory:
    """Which samples the paper has, asked once and reused by every pass of one lane."""

    response: InventoryResponse
    raw_text: str
    usage: Mapping[str, int]
    source_ids: frozenset[str]


def _budget_chars(options: ExtractionOptions) -> int:
    """How much rendered markdown one question may carry, once the reply and the prompt are paid for."""
    return int(max(options.context_tokens - options.max_tokens - PROMPT_OVERHEAD_TOKENS, 0) * CHARS_PER_TOKEN)


def _take_inventory(
    blocks: Sequence[SourceBlock],
    client: LlmClient,
    options: ExtractionOptions,
    *,
    backend: Backend,
    refresh: bool,
) -> SampleInventory:
    """Ask which samples exist -- once per lane.

    The salt is empty, the salt pass 0 would have used, so a lane re-run with more passes still hits the
    inventory entry the earlier run cached.
    """
    profile = options.profile
    selection = fit_budget(inventory_blocks(blocks, profile.retrieval), budget_chars=_budget_chars(options))
    system = inventory_system_prompt(profile)
    user = inventory_user_prompt(render_markdown(selection))
    _check_context_budget(system, user, options)
    response, raw_text, usage = complete_validated(
        client,
        _response_models(profile).inventory,
        system=system,
        user=user,
        repair=lambda previous, error: repair_prompt(user, previous, error),
        refresh=refresh,
        cache_salt="",
        reasoning_effort=options.inventory_reasoning_effort,
    )
    logger.info("inventory backend=%s samples=%d blocks=%d", backend, len(response.samples), len(selection))
    return SampleInventory(
        response=response,
        raw_text=raw_text,
        usage=usage,
        source_ids=frozenset(block.source_id for block in selection),
    )


def _extract_passages(
    blocks: Sequence[SourceBlock],
    client: LlmClient,
    options: ExtractionOptions,
    *,
    inventory: SampleInventory,
    concurrency: int,
    refresh: bool,
    cache_salt: str,
) -> tuple[ExtractedRecords, dict[str, int], str, tuple[FailedQuestion, ...]]:
    """Passage mode, one pass: one question per field against the lane's single inventory.

    A field question the model answers badly twice (or cuts off) is returned as a :class:`FailedQuestion`
    instead of failing the pass: at temperature 0 such an answer tends to repeat, and it must cost that field,
    not every field of the paper. Any other error -- a transport failure, an exhausted retry budget -- still
    propagates, so an outage fails the lane rather than emptying it.
    """
    budget_chars = _budget_chars(options)
    usage: dict[str, int] = {}

    sample_list = _render_sample_list(inventory.response.samples)
    # What the inventory cited as describing the samples: the recipe paragraph every field question needs.
    sample_blocks = frozenset(source_id for sample in inventory.response.samples for source_id in sample.source_ids)
    profile = options.profile
    field_system = field_system_prompt(profile)

    # Which fields get asked, and with which blocks, is decided here in the profile's field order and nowhere
    # else. Retrieval and the budget check stay on this thread, so the questions -- and the "never asked"
    # reasons recorded beside them -- are the same bytes in the same order whatever `concurrency` is.
    questions: list[tuple[FieldSpec, tuple[SourceBlock, ...], str]] = []
    dropped: list[str] = []
    for spec in profile.fields:
        if spec.is_sample_level and _deposits_no_film(inventory.response):
            # A device paper on purchased ITO glass: asking anyway only harvests the absorber's thickness and
            # the spin-coater's rpm as unattributed values that look like findings. An inventory that is
            # empty for any other reason -- it missed the sample text -- still gets every question, so one
            # missed inventory cannot cost the lane all its sample-level values.
            dropped.append(f"{spec.name}: the inventory found no in-scope sample, so it was not asked about")
            continue
        candidates = fit_budget(
            candidate_blocks(
                spec, blocks, units=profile.units, limit=options.candidate_limit, sample_blocks=sample_blocks
            ),
            budget_chars=budget_chars,
        )
        if not candidates:
            # Not an error: a paper that never mentions a target's density simply has none to find. Recorded
            # so that "the model missed it" and "we never asked" stay distinguishable.
            dropped.append(f"{spec.name}: no block in this lane mentions it, so it was not asked about")
            continue
        field_user = field_user_prompt(
            spec, sample_list, render_markdown(candidates), profile.prompt.implausible_origin
        )
        _check_context_budget(field_system, field_user, options)
        questions.append((spec, tuple(candidates), field_user))

    def ask(
        question: tuple[FieldSpec, tuple[SourceBlock, ...], str],
    ) -> tuple[FieldHarvest | FailedQuestion, str, dict[str, int]]:
        """One field question. Returns everything it produced; it shares no mutable state with its peers."""
        spec, candidates, field_user = question
        try:
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
        except LlmResponseError as exc:
            logger.warning("no valid answer to the %s question; the next run asks it again: %s", spec.name, exc)
            return FailedQuestion(field=spec.name, detail=str(exc)), "", {}
        harvest = FieldHarvest(
            spec=spec,
            values=tuple(response.values),
            known_ids=frozenset(block.source_id for block in candidates),
        )
        return harvest, text, field_usage

    # The questions are independent and the wait is the network, so they overlap. `complete_validated` is a
    # pure function of its arguments and `OpenAICompatibleClient` keeps nothing mutable on `self` -- the
    # httpx.Client underneath is safe for concurrent requests, and cache entries land through
    # `write_text_atomic`, whose temp names are unique. `pool.map` hands the answers back in submission
    # order and re-raises the earliest failure in that order, so the first failure wins exactly as the
    # sequential loop's would, and every usage dict is merged here on one thread rather than by the workers.
    if concurrency == 1 or len(questions) <= 1:
        answers = [ask(question) for question in questions]
    else:
        with ContextThreadPoolExecutor(
            max_workers=min(concurrency, len(questions)), thread_name_prefix="paperfacts-field"
        ) as pool:
            answers = list(pool.map(ask, questions))

    harvests: list[FieldHarvest] = []
    failed: list[FailedQuestion] = []
    raw_parts = [f"# inventory\n{inventory.raw_text}"]
    for harvest, text, field_usage in answers:
        if isinstance(harvest, FailedQuestion):
            failed.append(harvest)
            raw_parts.append(f"# {harvest.field}\n(no valid answer: {harvest.detail})")
            continue
        harvests.append(harvest)
        raw_parts.append(f"# {harvest.spec.name}\n{text}")
        _add_usage(usage, field_usage)

    records = passage_records(
        inventory.response,
        harvests,
        inventory_ids=inventory.source_ids,
        dropped=dropped,
    )
    return records, usage, "\n\n".join(raw_parts), tuple(failed)


def _deposits_no_film(inventory: InventoryResponse) -> bool:
    """The inventory's verdict that the paper has no film of its own, trusted only when it named no sample."""
    return inventory.no_tco_film and not inventory.samples


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

    Attribution is by :func:`~paperfacts.records.sample_key` -- the same key that pairs samples across
    lanes -- so the model only has to repeat an id it was given. A sample-level value naming no sample, or
    one the inventory does not have, is kept in ``unattributed`` rather than attached to a plausible
    neighbour: an unplaced value is visible in the report, a misplaced one is indistinguishable from a real
    measurement. The one exception is a paper with a single sample, where a value naming no sample has only
    one possible owner.

    A value the model flagged ``applies_to_all_samples`` is the other kind of null id: the paper stated it
    once for the whole series ("all films were RF sputtered"), so it is written onto every sample with
    ``series=True``. The flag is the model's explicit claim about the excerpt; the code never infers it.
    """
    cleaning = ResponseCleaning()
    cleaning.dropped.extend(dropped)

    samples, _ = clean_samples(inventory.samples, cleaning, inventory_ids)
    sample_fields: list[list[FieldValue]] = [[] for _ in samples]
    index_by_key = {sample_key(sample.sample_id): index for index, sample in enumerate(samples)}

    target_fields: list[FieldValue] = []
    target_ids: list[str] = []
    unattributed: list[FieldValue] = []
    single_sample_attributed = 0
    series_fanned_out = 0
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
                # The question itself decided the scope, so a stray sample_id or series flag on a paper-level
                # field is noise rather than the scope error document mode has to guard against.
                target_fields.append(value)
                target_ids.extend(value.source_ids)
                continue
            if item.applies_to_all_samples and item.sample_id:
                # An id and the series flag contradict each other. The id is the more specific claim and
                # the one the prompt asks to be copied verbatim, so it wins; the flag is noise.
                logger.debug(
                    "ignoring applies_to_all_samples on %s: the value names sample %r",
                    harvest.spec.name,
                    item.sample_id,
                )
            elif item.applies_to_all_samples:
                series_fanned_out += place_on_every_sample(value, sample_fields, unattributed)
                continue
            index = index_by_key.get(sample_key(item.sample_id)) if item.sample_id else None
            if index is None and not item.sample_id and len(samples) == 1:
                # The prompt allows a null sample_id when the excerpts do not say which sample a value
                # belongs to. With exactly one sample in the inventory there is nothing to say: the lone
                # sample is not a plausible neighbour, it is the only possible owner.
                index = 0
                single_sample_attributed += 1
            if index is None:
                unattributed.append(value)
                continue
            sample_fields[index].append(value)

    if series_fanned_out:
        logger.info(
            "fanned out %d series value(s) stated for the whole sample list to each of %d sample(s)",
            series_fanned_out,
            len(samples),
        )

    if single_sample_attributed:
        logger.info(
            "attributed %d value(s) with no sample_id to the paper's only sample %r",
            single_sample_attributed,
            samples[0].sample_id,
        )

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


def _check_context_budget(system: str, user: str, options: ExtractionOptions) -> None:
    """Fail before spending money when the prompt cannot fit, instead of letting the API truncate it."""
    context_tokens, reply_tokens = options.context_tokens, options.max_tokens
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
