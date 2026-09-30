"""Supervisor: a second model scores a disputed or borderline value against the passage it cites.

The stage is lazy. It never looks at a value both lanes agree on comfortably; it reads the comparison report
and picks out two kinds of comparison (:func:`selection_reason`):

- ``conflict``: the lanes quote different values for one fact. Each side is scored against its own cited
  blocks, and :mod:`paperfacts.decide` settles the cell in favour of a side the supervisor trusted when the
  other was doubted (cell status ``supervised``, never ``agree``: one lane's word, vouched for by a judge).
- ``borderline``: the lanes agree, but the value sits within :data:`BORDERLINE_FRACTION` of an edge of the
  field's plausible range, where a mistaken quantity most often lands. A doubted score does not refuse the
  cell; it is committed as ``agree`` with the critique in its audit trail.

Unlike visual validation (:mod:`paperfacts.validate`), the supervisor *is* told the value and asked whether the
passage supports it: that is the point of a critique, which names what the passage says instead. The score
is a model's opinion, so it only ever breaks a tie the code could not (a conflict) or annotates; it never
overrules an agreement.

The thresholds are applied here, where the score is produced, so the stored :class:`SupervisorScore` carries a
categorical verdict and the decision rules stay pure. The model, the thresholds and this module's source are
hashed into ``comparison_key`` only when the stage is on (:func:`paperfacts.keys.comparison_key`): turning it
on or off renames only the reports it changes.

The stage runs inside ``compare`` (:func:`paperfacts.workflow.compare_document`), on its own client. A request
that fails after its retries scores the value ``error``, which never reads as a doubted one, and a report
holding one is not stored (:func:`paperfacts.dataset.incomplete_reason`): the next run asks again, and only
that request reaches the endpoint, since the answered ones replay from the LLM cache. A side the judge cannot
be shown -- a value citing no block the parse has, or a passage longer than :data:`MAX_PASSAGE_CHARS` -- is
not scored at all: a passage cut short would have the judge doubt a value it was never shown.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from paperfacts.compare import (
    PAPER_SCOPE,
    ComparisonReport,
    FieldComparison,
    Supervision,
    SupervisionReason,
    SupervisorFlag,
    SupervisorScore,
    SupervisorVerdict,
)
from paperfacts.errors import LlmError, LlmOfflineMiss
from paperfacts.fields import FieldSpec
from paperfacts.keys import SupervisorOptions
from paperfacts.llm import LlmClient, complete_validated
from paperfacts.models import Backend
from paperfacts.profile import DomainProfile
from paperfacts.prompts import repair_prompt
from paperfacts.records import FieldValue
from paperfacts.threads import ContextThreadPoolExecutor

logger = logging.getLogger(__name__)

# Sampling for the supervisor's client; fixed here, like the figures stage's, so they are part of the code
# the key hashes rather than knobs.
TEMPERATURE = 0.0
MAX_TOKENS = 600
RETRY_ATTEMPTS = 1
# How close to an edge of the plausible range an agreed value must be to be worth a look, as a fraction of the
# range's width (or of the one end's magnitude when the other is open).
BORDERLINE_FRACTION = 0.1
# The longest passage the judge is shown whole. A cited table can run to pages of HTML; one cut short could
# leave out the row the value sits in, so such a side is left unscored rather than judged on a fragment.
MAX_PASSAGE_CHARS = 12000


class _Reply(BaseModel):
    """The JSON the supervisor is asked for."""

    score: float = Field(ge=0.0, le=1.0)
    flag: SupervisorFlag
    critique: str = ""


def supervisor_system_prompt() -> str:
    """Domain-free: the field's own description reaches the user prompt."""
    return (
        "You verify one value that was extracted from a scientific paper against the passage it was cited from.\n"
        "\n"
        "Return ONLY a JSON object:\n"
        '{"score": <0.0-1.0>, "flag": "<correct|plausible|unit_mismatch|value_not_in_passage>", '
        '"critique": "<what the passage actually says, when the value is not fully supported; else empty>"}\n'
        "\n"
        "Rules:\n"
        '1. "correct": the value and its unit appear in the passage as extracted, for the quantity described and '
        "for the sample named. Score 1.0.\n"
        '2. "plausible": the value appears with a minor spelling or OCR variation, or the passage states it in an '
        "equivalent form. Score 0.7-0.9.\n"
        '3. "unit_mismatch": the number appears but its unit is wrong or missing. Score 0.3-0.6.\n'
        '4. "value_not_in_passage": the passage does not state this value for this quantity and sample (a '
        "different number, a different quantity, another sample's row, or nothing). Score 0.0-0.2.\n"
        "5. Judge only against the passage given. Never use outside knowledge of what the value should be.\n"
        "6. Write the critique whenever the score is below 0.7: quote the number or words the passage gives."
    )


def supervisor_user_prompt(spec: FieldSpec, value: FieldValue, passage: str, *, sample: str | None = None) -> str:
    """``sample`` names the sample the value was attributed to (the comparison's scope), so a value read off a
    neighbouring row of the same table is not scored as correct."""
    unit = value.unit_raw or "(none)"
    condition = f"\nStated condition: {value.condition}" if value.condition else ""
    description = f"\nDescription: {spec.description}" if spec.description else ""
    sample_line = f"\nSample: {sample}" if sample else ""
    return (
        f"Field: {spec.name}{description}{sample_line}\n"
        f"Extracted value: {value.value_raw}\n"
        f"Extracted unit: {unit}{condition}\n"
        f"\nCited passage:\n{passage}\n"
        "\nReturn the JSON object now."
    )


# ---- Selection ---------------------------------------------------------------------------------------------


def is_borderline(spec: FieldSpec, value: FieldValue) -> bool:
    """Whether a normalised value lies within :data:`BORDERLINE_FRACTION` of an edge of the plausible range.

    A field with no range, or a value that did not normalise to a number, is never borderline: there is no
    edge to be near.
    """
    number = value.value
    low, high = spec.valid_range
    if number is None or (low is None and high is None):
        return False
    if low is not None and high is not None:
        margin = (high - low) * BORDERLINE_FRACTION
    else:
        margin = abs(low if high is None else high) * BORDERLINE_FRACTION  # type: ignore[arg-type]
    near_low = low is not None and number <= low + margin
    near_high = high is not None and number >= high - margin
    return near_low or near_high


def selection_reason(comparison: FieldComparison, specs: Mapping[str, FieldSpec]) -> SupervisionReason | None:
    """Why the supervisor should look at ``comparison``, or None when it need not."""
    if comparison.a is None or comparison.b is None:
        return None
    if comparison.status == "conflict":
        return "conflict"
    spec = specs.get(comparison.field)
    if comparison.status == "agree" and spec is not None:
        if is_borderline(spec, comparison.a) or is_borderline(spec, comparison.b):
            return "borderline"
    return None


def passage_for(value: FieldValue, blocks: Mapping[str, str]) -> str | None:
    """The text of the blocks a value cites, in citation order; None when it cites none the lane's parse has, or
    when the whole of it is longer than the judge may be shown."""
    parts = [blocks[source_id] for source_id in value.source_ids if source_id in blocks]
    passage = "\n\n".join(parts)
    if not passage or len(passage) > MAX_PASSAGE_CHARS:
        return None
    return passage


def sample_of(comparison: FieldComparison) -> str | None:
    """The sample a comparison is about, as its scope names it, or None for the paper level."""
    if comparison.scope == PAPER_SCOPE:
        return None
    _entity, _, ids = comparison.scope.partition(":")
    return ids or None


# ---- Scoring -----------------------------------------------------------------------------------------------


def verdict_for(score: float, options: SupervisorOptions) -> SupervisorVerdict:
    if score >= options.vote_threshold:
        return "trusted"
    if score < options.min_confidence:
        return "doubted"
    return "uncertain"


def score_value(
    client: LlmClient,
    spec: FieldSpec,
    value: FieldValue,
    passage: str,
    options: SupervisorOptions,
    *,
    sample: str | None = None,
    refresh: bool = False,
) -> SupervisorScore:
    """One value against its passage. A request that fails is an ``error`` score, never an exception: the judge's
    outage must not fail the paper. An offline-replay miss still propagates, as everywhere else."""
    user = supervisor_user_prompt(spec, value, passage, sample=sample)
    try:
        reply, _raw, _usage = complete_validated(
            client,
            _Reply,
            system=supervisor_system_prompt(),
            user=user,
            repair=lambda previous, error: repair_prompt(user, previous, error),
            refresh=refresh,
        )
    except LlmOfflineMiss:
        raise
    except LlmError as exc:
        logger.warning("supervisor could not score %s=%r: %s", spec.name, value.value_raw, exc)
        return SupervisorScore(score=0.0, flag="error", critique=str(exc)[:500], verdict="error")
    return SupervisorScore(
        score=reply.score, flag=reply.flag, critique=reply.critique.strip(), verdict=verdict_for(reply.score, options)
    )


@dataclass(frozen=True)
class _Job:
    """One side of one selected comparison, ready to be scored."""

    index: int
    side: str
    spec: FieldSpec
    value: FieldValue
    passage: str
    sample: str | None


def supervise_report(
    report: ComparisonReport,
    options: SupervisorOptions,
    client: LlmClient,
    blocks: Mapping[Backend, Mapping[str, str]],
    profile: DomainProfile,
    *,
    refresh: bool = False,
    concurrency: int = 1,
) -> ComparisonReport:
    """The report with every selected comparison scored on both sides. ``blocks`` maps each lane's block ids to
    their text, so a value is judged against what its own parser produced. ``concurrency`` requests are in
    flight at once (scheduling only: it reaches no key); the process-wide in-flight limit still applies."""
    specs = profile.by_name
    reasons: dict[int, SupervisionReason] = {}
    jobs: list[_Job] = []
    for index, comparison in enumerate(report.comparisons):
        reason = selection_reason(comparison, specs)
        spec = specs.get(comparison.field)
        if reason is None or spec is None:
            continue
        reasons[index] = reason
        for side, backend, value in (("a", report.backend_a, comparison.a), ("b", report.backend_b, comparison.b)):
            passage = passage_for(value, blocks.get(backend, {})) if value is not None else None
            if value is not None and passage is not None:
                jobs.append(_Job(index, side, spec, value, passage, sample_of(comparison)))

    def score(job: _Job) -> SupervisorScore:
        return score_value(client, job.spec, job.value, job.passage, options, sample=job.sample, refresh=refresh)

    if concurrency > 1 and len(jobs) > 1:
        with ContextThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="paperfacts-supervisor") as pool:
            scores = list(pool.map(score, jobs))
    else:
        scores = [score(job) for job in jobs]

    sides: dict[int, dict[str, SupervisorScore]] = {index: {} for index in reasons}
    for job, result in zip(jobs, scores, strict=True):
        sides[job.index][job.side] = result
    comparisons = tuple(
        comparison.model_copy(
            update={"supervision": Supervision(reason=reasons[index], a=sides[index].get("a"), b=sides[index].get("b"))}
        )
        if index in reasons
        else comparison
        for index, comparison in enumerate(report.comparisons)
    )
    return report.model_copy(update={"comparisons": comparisons})


def carry_supervision(fresh: ComparisonReport, stored: ComparisonReport) -> ComparisonReport:
    """``fresh`` with the supervision of ``stored`` copied onto the comparisons that are the same fact judged
    the same way. An export rebuilds the report from re-read lanes without a model; the scores it stored are
    still of these values (a comparison is matched by scope, field, condition, status and both sides' quotes),
    so the rebuilt report must not lose them, or the cells they settled would go back to conflicts."""
    kept = {_comparison_identity(c): c.supervision for c in stored.comparisons if c.supervision is not None}
    if not kept:
        return fresh
    comparisons = tuple(
        c.model_copy(update={"supervision": kept[_comparison_identity(c)]})
        if c.supervision is None and _comparison_identity(c) in kept
        else c
        for c in fresh.comparisons
    )
    return fresh.model_copy(update={"comparisons": comparisons})


def _comparison_identity(c: FieldComparison) -> tuple[object, ...]:
    return (c.scope, c.field, c.condition, c.status, _value_identity(c.a), _value_identity(c.b))


def _value_identity(value: FieldValue | None) -> tuple[object, ...] | None:
    if value is None:
        return None
    return (value.value_raw, value.unit_raw, value.condition, tuple(value.source_ids))


def unscored(report: ComparisonReport) -> Sequence[FieldComparison]:
    """The comparisons a supervisor request failed on: their report is not stored, and the next run asks again."""
    return [
        c
        for c in report.comparisons
        if c.supervision is not None
        and any(score is not None and score.verdict == "error" for score in (c.supervision.a, c.supervision.b))
    ]


def supervision_summary(report: ComparisonReport) -> str:
    """One clause for the stage detail: how many comparisons were scored and how the scores fell."""
    scores = [
        score
        for comparison in report.comparisons
        if comparison.supervision is not None
        for score in (comparison.supervision.a, comparison.supervision.b)
        if score is not None
    ]
    looked_at = sum(1 for comparison in report.comparisons if comparison.supervision is not None)
    doubted = sum(1 for score in scores if score.verdict == "doubted")
    errors = sum(1 for score in scores if score.verdict == "error")
    detail = f"supervised {looked_at}"
    notes = [f"{doubted} doubted"] if doubted else []
    if errors:
        notes.append(f"{errors} unscored")
    return detail + (f" ({', '.join(notes)})" if notes else "")
