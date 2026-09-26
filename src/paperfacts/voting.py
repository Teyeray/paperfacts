"""Collapse repeated facts within one pass, and keep only what a majority of passes agree on.

Even at temperature 0 the model is not deterministic across runs: repeated extractions drop or add the
occasional value, and with one pass that noise is indistinguishable from parser disagreement. Running the
lane several times and voting turns the model's own jitter into a measured quantity -- each surviving value
records the fraction of passes that produced it -- so the disagreement left downstream is the parsers'.

Two collapses live here because they share one notion of identity:

- :func:`deduplicate` merges repeats *within* a pass, where two entries with the same field, condition,
  number and unit are one fact quoted twice.
- :func:`merge_passes` votes *across* passes, where the condition is deliberately not part of the identity
  because the model rewords it between passes.

This module is hashed into ``extractor_key`` (see :func:`paperfacts.keys.extraction_code_fingerprint`):
it decides which of the model's claims are stored.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

from paperfacts.grounding import grounding_key
from paperfacts.records import ExtractedRecords, FieldValue, PaperRecord, SampleRecord, sample_key
from paperfacts.text import clean_unit, normalize_key


class Scope(Enum):
    """The two scopes a value can have that are not a sample.

    A sample's scope is its identity, a tuple (entity, sample_key), so these members cannot collide with one
    whatever a paper calls its samples -- a promise a reserved string like ``"__paper__"`` could not make.
    """

    PAPER = "paper"
    UNATTRIBUTED = "unattributed"


# A sample's scope: its entity type and its sample_key, so two entities' samples named alike stay two samples.
type SampleScope = tuple[str, str]
type ScopeKey = SampleScope | Scope


class ValueKey(NamedTuple):
    """A value's full identity within one pass -- see ``_value_key``. ``holds`` is a boolean field's alone; every
    other value leaves it None, so two of their keys are equal exactly when the four parts are."""

    field: str
    condition: str
    quote: str
    unit: str
    holds: bool | None = None


class VoteKey(NamedTuple):
    """The identity the passes vote on: the same number, in the same unit, for the same field (and the same
    ``holds``). The condition is deliberately absent -- see ``_vote_key``."""

    field: str
    quote: str
    unit: str
    holds: bool | None = None


# What a vote is actually cast for: the nth entry a pass gave one voted identity. Rank 1 is the first
# condition a pass reported that number under, rank 2 the second, and so on -- see ``merge_passes``.
type VoteSlot = tuple[ScopeKey, VoteKey, int]


# ---- Repeats within one pass ---------------------------------------------------------------------------------


def deduplicate(records: ExtractedRecords, *, reference_fields: Collection[str]) -> ExtractedRecords:
    """Collapse repeats of the same fact, keeping every citation they brought. ``reference_fields`` names the
    profile's reference fields, whose quotes are sample ids and are compared as such (``_value_key``).

    One field question routinely gets the same number back more than once: quoted from the table, and again
    from the sentence discussing it. They are one fact with two citations. Keeping both inflates every count
    and, worse, hands the comparison layer a duplicate to pair against -- it keeps the first value per
    condition, so a second, genuinely different reading of the same quantity (``10^-2`` against ``10^2``,
    a real OCR disagreement) can be silently dropped behind a duplicate of the first.
    """
    paper = records.paper
    if paper is not None:
        paper = paper.model_copy(update={"fields": _merge_repeats(paper.fields, reference_fields)})
    return records.model_copy(
        update={
            "paper": paper,
            "samples": tuple(
                sample.model_copy(update={"fields": _merge_repeats(sample.fields, reference_fields)})
                for sample in records.samples
            ),
            "unattributed": _merge_repeats(records.unattributed, reference_fields),
        }
    )


def _merge_repeats(values: Sequence[FieldValue], reference_fields: Collection[str]) -> tuple[FieldValue, ...]:
    """One entry per distinct fact, in first-seen order, with the citations of its repeats merged in.

    Order decides which copy is kept, except between a series value and a sample-specific one: there the
    sample-specific copy wins whichever came first.
    """
    merged: dict[ValueKey, FieldValue] = {}
    for value in values:
        key = _value_key(value, reference_fields)
        previous = merged.get(key)
        if previous is None:
            merged[key] = value
            continue
        citations = tuple(dict.fromkeys((*previous.source_ids, *value.source_ids)))
        # When the same fact arrives both as a series value fanned out onto this sample and as a quote
        # naming the sample itself, the sample-specific one is the more precise claim and becomes the
        # exemplar; the series copy only adds its citation.
        exemplar = value if previous.series and not value.series else previous
        merged[key] = exemplar.model_copy(update={"source_ids": citations})
    return tuple(merged.values())


# ---- Majority vote over repeated passes ----------------------------------------------------------------------


@dataclass
class _Tally:
    """Everything one voted slot accumulates: how many passes produced it, the wording kept, and every
    pass's (identity, citations) so the guarded union below can decide which citations it may absorb."""

    exemplar: FieldValue
    votes: int = 0
    supporters: list[tuple[ValueKey, tuple[str, ...]]] = field(default_factory=list)


def merge_passes(results: Sequence[ExtractedRecords], *, reference_fields: Collection[str]) -> ExtractedRecords:
    """Keep the values a majority of ``results`` agree on, annotated with their agreement. ``reference_fields`` as
    in :func:`deduplicate`.

    The vote is on ``_vote_key`` -- field, number, unit -- and not on the condition, because the condition
    is free text the model rewords between passes ("at 550 nm", "550 nm wavelength"). Voting on the full
    key made every paraphrase its own candidate with a single vote, so two passes dropped nearly everything
    they in fact agreed on.

    Conditions are still never merged. A pass that reports one number under several genuinely different
    conditions (85 % at 550 nm and at 600 nm) keeps an entry for each, because the passes vote per *rank*:
    within a pass the entries for one voted identity are ordered as they were reported, and rank n is
    supported by every pass that produced at least n of them. Two passes reporting the number once each
    therefore agree on one entry however differently they word its condition, while a second entry only one
    pass produced fails the majority like any other lone value.

    The wording kept for a rank is the first pass's, so the order of ``results`` is meaningful; the
    citations of every pass that supported the rank are merged into it.
    """
    if len(results) == 1:
        return results[0]
    passes = len(results)
    majority = passes // 2 + 1

    slots: dict[VoteSlot, _Tally] = {}
    # How many entries the most generous pass gave each voted identity; it decides below whether a
    # supporter's citations may be merged into a rank that kept someone else's wording.
    entry_counts: Counter[tuple[ScopeKey, VoteKey]] = Counter()
    sample_counts: Counter[SampleScope] = Counter()
    samples: dict[SampleScope, SampleRecord] = {}
    paper_ids: tuple[str, ...] = ()
    for records in results:
        slot_of: dict[tuple[ScopeKey, ValueKey], VoteSlot] = {}
        ranks: Counter[tuple[ScopeKey, VoteKey]] = Counter()
        cited: dict[VoteSlot, tuple[str, ...]] = {}
        for scope, value in _values(records):
            identity = (scope, _value_key(value, reference_fields))
            slot = slot_of.get(identity)
            if slot is None:
                vote = (scope, _vote_key(identity[1]))
                ranks[vote] += 1
                slot = (*vote, ranks[vote])
                slot_of[identity] = slot
                entry_counts[vote] = max(entry_counts[vote], ranks[vote])
                slots.setdefault(slot, _Tally(exemplar=value)).votes += 1
                cited[slot] = value.source_ids
            else:
                cited[slot] = (*cited[slot], *value.source_ids)
        for identity, slot in slot_of.items():
            slots[slot].supporters.append((identity[1], cited[slot]))
        for scope in {_sample_scope(sample) for sample in records.samples}:
            sample_counts[scope] += 1
        for sample in records.samples:
            samples.setdefault(_sample_scope(sample), sample)
        if records.paper is not None and not paper_ids:
            paper_ids = records.paper.source_ids

    # A pass that supported a rank also supported its citations, including when its wording of the condition
    # was not the one kept: losing the wording must not lose the block it quoted. The union is guarded,
    # though, because rank matching pairs entries by position, and two passes may list the same number's
    # conditions in opposite orders. Merge only when the identity has a single entry everywhere -- there is
    # then no other condition the citation could belong to -- or when the two conditions normalise alike.
    for (scope, vote, _rank), tally in slots.items():
        condition = _value_key(tally.exemplar, reference_fields).condition
        unambiguous = entry_counts[(scope, vote)] == 1
        cited = list(tally.exemplar.source_ids)
        for value_key, source_ids in tally.supporters:
            if unambiguous or condition == value_key.condition:
                cited.extend(source_ids)
        merged = tuple(dict.fromkeys(cited))
        if merged != tally.exemplar.source_ids:
            tally.exemplar = tally.exemplar.model_copy(update={"source_ids": merged})

    kept: dict[ScopeKey, list[FieldValue]] = {}
    dropped = [entry for records in results for entry in records.dropped]
    for (scope, _vote, _rank), tally in slots.items():
        value, votes = tally.exemplar, tally.votes
        if votes < majority:
            dropped.append(f"{value.field}: only {votes}/{passes} passes produced {value.value_raw!r}")
            continue
        kept.setdefault(scope, []).append(value.model_copy(update={"agreement": votes / passes}))

    # Sample identity is voted on separately from its values: "this sample exists, under these conditions"
    # is itself a finding, kept even when none of its measurements survived.
    paper_fields = tuple(kept.get(Scope.PAPER, ()))
    return ExtractedRecords(
        paper=PaperRecord(source_ids=paper_ids, fields=paper_fields) if paper_fields else None,
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
    for value in records.paper.fields if records.paper else ():
        yield Scope.PAPER, value
    for sample in records.samples:
        scope = _sample_scope(sample)
        for value in sample.fields:
            yield scope, value
    for value in records.unattributed:
        yield Scope.UNATTRIBUTED, value


def _sample_scope(sample: SampleRecord) -> SampleScope:
    return sample.entity, sample_key(sample.sample_id)


def _value_key(value: FieldValue, reference_fields: Collection[str]) -> ValueKey:
    """Full identity, used to merge repeats *within* one pass: the same number in the same unit under the
    same condition, however it is spelled.

    Within a pass the condition belongs in the identity -- two conditions are two measurements and must not
    be merged. Across passes it does not; ``_vote_key`` is what the passes vote on.

    A boolean field's ``holds`` is part of it: "doped" quoted as true and as false are two answers, never one
    fact. A reference field's quote is a sample id, keyed by :func:`sample_key` as the sample it names is: "Cat-1"
    and "cat 1" resolve to one listed sample, so they are one answer, where the text key would split the vote.
    """
    unit = clean_unit(value.unit_raw) if value.unit_raw else ""
    quote = sample_key(value.value_raw) if value.field in reference_fields else grounding_key(value.value_raw)
    return ValueKey(value.field, normalize_key(value.condition), quote, unit, value.holds)


def _vote_key(key: ValueKey) -> VoteKey:
    """Condition-free identity: what "the passes agree on this number" means.

    The unit belongs in it: "2.1 μm" and "2.1 nm" are a thousand-fold disagreement, and treating them as
    one value would merge the disagreement away instead of reporting it. The condition does not: it is free
    text the model paraphrases between passes, and counting paraphrases as separate candidates split the
    vote until nothing reached a majority.
    """
    return VoteKey(key.field, key.quote, key.unit, key.holds)
