"""Merge several extraction passes over the same lane by majority vote.

Even at temperature 0 the model is not deterministic across runs: repeated extractions of the same paper
drop or add the occasional value. With one pass that noise is indistinguishable from a genuine
parser-level disagreement, which is exactly the signal this project is built to measure.

Running the same lane N times and keeping only what a majority of passes agree on trades money for a
cleaner signal, so it is off by default and enabled per run. Each surviving value records the fraction of
passes that produced it, so a 2/3 value is still visibly weaker than a 3/3 one.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence

from paperfacts.extraction.grounding import grounding_key
from paperfacts.extraction.records import ExtractedRecords, FieldValue, SampleRecord, TargetRecord
from paperfacts.normalization.text import normalize_key

# The paper-level target record's scope. It is None rather than a reserved string so that no sample id,
# however it normalises, can ever collide with it: the type makes the invariant instead of a comment.
TARGET_SCOPE = None

type ScopeKey = str | None
type ValueKey = tuple[str, str, str]


def merge_passes(results: Sequence[ExtractedRecords]) -> ExtractedRecords:
    """Keep the values that a majority of ``results`` agree on, annotated with their agreement."""
    if len(results) == 1:
        return results[0]
    passes = len(results)
    majority = passes // 2 + 1

    counts: Counter[tuple[ScopeKey, ValueKey]] = Counter()
    exemplars: dict[tuple[ScopeKey, ValueKey], FieldValue] = {}
    sample_counts: Counter[ScopeKey] = Counter()
    samples: dict[ScopeKey, SampleRecord] = {}
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

    # Sample identity is voted on separately from its values: a sample the passes agree on is kept even
    # when none of its measurements survived, because "this sample exists, under these conditions" is
    # itself a finding, and single-pass extraction reports it the same way.
    target_fields = tuple(kept.get(TARGET_SCOPE, ()))
    return ExtractedRecords(
        target=TargetRecord(source_ids=target_ids, fields=target_fields) if target_fields else None,
        samples=tuple(
            sample.model_copy(update={"fields": tuple(kept.get(scope, ()))})
            for scope, sample in samples.items()
            if sample_counts[scope] >= majority
        ),
        invalid_source_ids=tuple(sorted({sid for records in results for sid in records.invalid_source_ids})),
        dropped=tuple(dict.fromkeys(dropped)),
    )


def _values(records: ExtractedRecords) -> Iterator[tuple[ScopeKey, FieldValue]]:
    for value in records.target.fields if records.target else ():
        yield TARGET_SCOPE, value
    for sample in records.samples:
        scope = normalize_key(sample.sample_id)
        for value in sample.fields:
            yield scope, value


def _value_key(value: FieldValue) -> ValueKey:
    """Identity of a value for voting: the same number under the same condition, however it is spelled."""
    return value.field, normalize_key(value.condition), grounding_key(value.value_raw)
