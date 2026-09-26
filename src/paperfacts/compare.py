"""Field-level comparison (the rule layer): the deterministic parts happen here, the fuzzy judgment calls
are left to the model layer.

The unit of comparison is "one value of one field": first pair exactly by normalized measurement condition
(550 nm and a full-spectrum average are two different facts), then pair whatever is left by
value equality — so when the two lanes phrase the same fact's condition differently ("ellipsometric" vs.
"from ellipsometric"), an identical value doesn't get split into two records that each look like they're
missing the other's value. Unequal leftovers are never paired across conditions. Numeric comparison uses
``math.isclose`` (|a-b| <= max(rel_tol*max(|a|,|b|), abs_tol)), with per-field tolerances configured in
:mod:`paperfacts.fields`; text/composition comparison uses
:func:`paperfacts.normalize.same_text`.

``status`` is always the **actual** comparison outcome; sample-pairing confidence is recorded separately in
``match_confidence``. Whether a low-confidence pairing should be escalated for review is a decision left to
downstream consumers — the raw observation is never discarded.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from paperfacts.errors import ProfileMismatchError
from paperfacts.fields import FieldSpec
from paperfacts.keys import (
    ComparisonOptions,
    comparison_key,
    profile_comparison_fingerprint,
    profile_extraction_fingerprint,
)
from paperfacts.kinds import element_key, rules_for
from paperfacts.matching import SampleMatching
from paperfacts.models import Backend
from paperfacts.normalize import (
    NUMBER_RE,
    delatex,
    normalize_key,
    normalize_lane,
    normalize_text,
)
from paperfacts.profile import IMPLICIT_ENTITY
from paperfacts.records import FieldValue, LaneExtraction
from paperfacts.storage import write_text_atomic

FactStatus = Literal["agree", "conflict", "ambiguous", "missing"]
# The scope of the paper-level comparisons.
PAPER_SCOPE = "paper"
# The paper-level scope as files written before round 2 spell it. Unambiguous: every sample scope is prefixed.
_LEGACY_PAPER_SCOPE = "target"


class FieldComparison(BaseModel):
    """The comparison outcome for one fact across both lanes, keeping both sides' candidate values and provenance."""

    model_config = ConfigDict(frozen=True)

    scope: str = Field(
        description='"paper", "unattributed", "<entity>:<a_id>|<b_id>" (matched) or "<entity>:<id>" (unmatched); '
        'the entity of a profile without entity types is "sample"'
    )
    field: str
    condition: str | None = None
    status: FactStatus
    missing_in: Backend | None = Field(
        default=None, description='which lane is missing the value when status == "missing"; None otherwise'
    )
    match_confidence: float | None = Field(
        default=None, description="confidence of the LLM sample pairing; None for paper-level or exact-matched facts"
    )
    a: FieldValue | None = None
    b: FieldValue | None = None
    detail: str = ""

    @field_validator("scope", mode="before")
    @classmethod
    def _current_paper_scope(cls, scope: Any) -> Any:
        return PAPER_SCOPE if scope == _LEGACY_PAPER_SCOPE else scope


class ComparisonCounts(BaseModel):
    """The evaluation counts the PRD asks for.

    Every count is present even at zero, so a consumer never guards against a missing key. The two per-lane
    dictionaries are the exception: a lane with nothing to report is simply absent from them.
    """

    model_config = ConfigDict(frozen=True)

    agree: int = 0
    conflict: int = 0
    ambiguous: int = 0
    missing: int = 0
    total: int = 0
    missing_by_backend: dict[str, int] = Field(default_factory=dict)
    samples_matched: int = 0
    samples_unmatched: int = 0
    low_confidence_matches: int = 0
    matching_failed: bool = False
    unattributed_by_backend: dict[str, int] = Field(
        default_factory=dict,
        description="per lane, values extracted but not placed on a sample; the ones neither lane could "
        "place are paired and compared (see unattributed_compared), the rest take part in no comparison",
    )
    unattributed_compared: int = Field(
        default=0, description="unattributed values both lanes extracted, paired and given a verdict"
    )


class ComparisonReport(BaseModel):
    """The full report of aligning and comparing both lanes for one document."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    extractor_key: str
    comparison_key: str
    backend_a: Backend
    backend_b: Backend
    matchings: dict[str, SampleMatching] = Field(description="the sample matching of each entity type, by its name")
    comparisons: tuple[FieldComparison, ...] = ()
    counts: ComparisonCounts = Field(default_factory=ComparisonCounts)
    # The lanes' artifact_sha256, so a report is tied to the parses it compared (None: unknown, older file).
    artifact_sha256_a: str | None = None
    artifact_sha256_b: str | None = None
    # keys.profile_comparison_fingerprint of the profile the verdicts were reached under (None: an older file).
    profile_fingerprint: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _matchings_from_legacy(cls, data: Any) -> Any:
        # A report written before round 2 holds its one matching as ``matching``: the implicit entity's.
        if isinstance(data, dict) and "matching" in data and "matchings" not in data:
            data = {name: value for name, value in data.items() if name != "matching"} | {
                "matchings": {IMPLICIT_ENTITY: data["matching"]}
            }
        return data

    @model_validator(mode="after")
    def _has_a_matching(self) -> Self:
        # Every profile has at least one entity (a profile without entity types has the implicit one), and every
        # reader starts from their matchings; a report without one is refused at load, not far away.
        if not self.matchings:
            raise ValueError("matchings is empty; a report holds the matching of every entity type")
        return self

    def sample_matching(self) -> SampleMatching:
        """The primary entity's matching: the first entry, which for a profile without entity types is the
        implicit entity's only one."""
        return next(iter(self.matchings.values()))

    def write(self, path: Path) -> None:
        write_text_atomic(path, self.model_dump_json(indent=2))

    @classmethod
    def read(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def compare_lanes(
    lane_a: LaneExtraction,
    lane_b: LaneExtraction,
    matchings: Mapping[str, SampleMatching] | SampleMatching,
    options: ComparisonOptions,
) -> ComparisonReport:
    """Compare two lanes under the sample matching of each entity type, by entity name. A single
    :class:`SampleMatching` stands for the one matching of a profile with one entity, the implicit one included."""
    if lane_a.extractor_key != lane_b.extractor_key:
        raise ValueError(
            f"lanes have different extractor_key ({lane_a.extractor_key} vs {lane_b.extractor_key}); cannot compare"
        )
    profile = options.profile
    if isinstance(matchings, SampleMatching):
        matchings = {profile.primary.name: matchings}
    names = [entity.name for entity in profile.entities]
    if set(matchings) != set(names):
        raise ValueError(f"the matchings are of {', '.join(matchings)}; the profile's entities are {', '.join(names)}")
    # In the profile's order, so the primary entity's matching comes first in the report.
    matchings = {name: matchings[name] for name in names}
    _check_lane_profiles(lane_a, lane_b, profile_extraction_fingerprint(profile))
    # Normalization is a pure, idempotent function, so it is unconditionally redone here: callers never have
    # to remember to normalize first, which rules out "forgot to normalize, so the answer was silently wrong"
    lane_a, lane_b = normalize_lane(lane_a, profile), normalize_lane(lane_b, profile)
    a_name, b_name = lane_a.backend, lane_b.backend
    comparisons: list[FieldComparison] = []

    # Paper-level: does not go through sample pairing
    comparisons += _compare_records(
        PAPER_SCOPE,
        lane_a.paper.fields if lane_a.paper else (),
        lane_b.paper.fields if lane_b.paper else (),
        profile.paper_fields,
        a_name,
        b_name,
    )
    # Each entity's samples, under that entity's matching and against its own fields only.
    for entity in profile.entities:
        comparisons += _compare_entity(
            entity.name, lane_a, lane_b, matchings[entity.name], profile.entity_fields(entity)
        )
    # Unattributed values are extracted by both lanes yet placed on no sample. Where both lanes hold the
    # same unplaced value, agreement is real evidence about the parsers and disagreement a real signal;
    # both were invisible while unattributed values took part in no comparison at all. Only pairs are
    # reported: a value one lane could not place may sit on a sample in the other lane (already reported
    # there), so a one-sided row here would double-count the same fact as MISSING.
    comparisons += _compare_records(
        "unattributed",
        lane_a.unattributed,
        lane_b.unattributed,
        profile.sample_fields,
        a_name,
        b_name,
        emit_one_sided=False,
        pair_leftovers_ambiguous=True,
        detail_prefix="unattributed in both lanes; ",
    )

    return ComparisonReport(
        document_id=lane_a.document_id,
        extractor_key=lane_a.extractor_key,
        comparison_key=comparison_key(options),
        backend_a=a_name,
        backend_b=b_name,
        matchings=dict(matchings),
        comparisons=tuple(comparisons),
        counts=_count(comparisons, tuple(matchings.values()), (lane_a, lane_b), options.ambiguous_match_confidence),
        artifact_sha256_a=lane_a.artifact_sha256,
        artifact_sha256_b=lane_b.artifact_sha256,
        profile_fingerprint=profile_comparison_fingerprint(profile),
    )


def _compare_entity(
    entity: str,
    lane_a: LaneExtraction,
    lane_b: LaneExtraction,
    matching: SampleMatching,
    specs: Sequence[FieldSpec],
) -> list[FieldComparison]:
    """One entity type's sample comparisons, each scoped ``"<entity>:..."``."""
    a_name, b_name = lane_a.backend, lane_b.backend
    comparisons: list[FieldComparison] = []
    # Matched samples: status is the real comparison outcome; pairing confidence is recorded separately
    for pair in matching.pairs:
        sample_a, sample_b = lane_a.sample(pair.a_id, entity), lane_b.sample(pair.b_id, entity)
        if sample_a is None or sample_b is None:
            continue
        comparisons += _compare_records(
            f"{entity}:{pair.a_id}|{pair.b_id}",
            sample_a.fields,
            sample_b.fields,
            specs,
            a_name,
            b_name,
            match_confidence=pair.confidence if pair.method == "llm" else None,
        )
    # Unmatched samples: normally this just means the other lane doesn't have it; when the matching model
    # itself failed we can't tell whether it's really missing, so mark it ambiguous and send it for review
    one_sided: FactStatus = "ambiguous" if matching.failed else "missing"
    reason = f"sample matching failed: {matching.failure}" if matching.failed else None
    for sample_id in matching.unmatched_a:
        if (sample := lane_a.sample(sample_id, entity)) is not None:
            comparisons += _compare_records(
                f"{entity}:{sample_id}",
                sample.fields,
                (),
                specs,
                a_name,
                b_name,
                one_sided=one_sided,
                one_sided_detail=reason,
            )
    for sample_id in matching.unmatched_b:
        if (sample := lane_b.sample(sample_id, entity)) is not None:
            comparisons += _compare_records(
                f"{entity}:{sample_id}",
                (),
                sample.fields,
                specs,
                a_name,
                b_name,
                one_sided=one_sided,
                one_sided_detail=reason,
            )
    return comparisons


def check_profile(found: str | None, expected: str, what: str) -> None:
    """Refuse ``what`` unless it was produced under the profile whose fingerprint is ``expected``.

    Results of two profiles may name the same field with other units or verdict rules, so combining them would
    report agreement that means nothing. A file without a fingerprint predates profiles and is reachable only
    through an explicitly old key; it is refused too rather than trusted.
    """
    if found is None:
        raise ProfileMismatchError(f"{what} was written before profiles; re-run it")
    if found != expected:
        raise ProfileMismatchError(f"{what} comes from profile {found}, not {expected}; re-run it under one profile")


def _check_lane_profiles(lane_a: LaneExtraction, lane_b: LaneExtraction, expected: str) -> None:
    # Two lanes of different profiles are a broken run whatever the options say, so that is reported first.
    found_a, found_b = lane_a.profile_fingerprint, lane_b.profile_fingerprint
    if found_a is not None and found_b is not None and found_a != found_b:
        raise ProfileMismatchError(
            f"the {lane_a.backend} and {lane_b.backend} lanes were extracted under different profiles "
            f"({found_a} vs {found_b}); re-run both under one profile"
        )
    for lane in (lane_a, lane_b):
        check_profile(lane.profile_fingerprint, expected, f"the {lane.backend} lane")


def compare_values(a: FieldValue, b: FieldValue, spec: FieldSpec) -> tuple[FactStatus, str]:
    """Decide the outcome when both sides have a value."""
    return rules_for(spec).compare(a, b, spec)


def _compare_records(
    scope: str,
    fields_a: Sequence[FieldValue],
    fields_b: Sequence[FieldValue],
    specs: Sequence[FieldSpec],
    backend_a: Backend,
    backend_b: Backend,
    *,
    match_confidence: float | None = None,
    one_sided: FactStatus = "missing",
    one_sided_detail: str | None = None,
    emit_one_sided: bool = True,
    pair_leftovers_ambiguous: bool = False,
    detail_prefix: str = "",
) -> list[FieldComparison]:
    """Pair up both sides' values field by field and compare them. When ``fields_b`` is empty this
    naturally degenerates to "everything is only in lane a".

    ``emit_one_sided=False`` suppresses the one-sided rows, for values whose other half may exist under a
    different scope (the unattributed comparison): pairing machinery is shared, reporting is not.

    ``pair_leftovers_ambiguous`` is the companion of that suppression. When both sides are left holding a
    value for the same field, no one-sided row reports it and the fact would vanish altogether -- yet both
    lanes did report the field, with different values under differently worded conditions, which is
    exactly what a reviewer needs to see. One positional row keeps it visible. It is always ambiguous,
    never a conflict: the pairing is positional, so the two readings may equally well be two different
    measurements. A list field is exempt: its leftovers are elements one lane did not read, not two readings
    of one value, so a positional row would claim a disagreement the lanes never had; its unplaced values go
    unreported like any other one-sided unplaced value.
    """
    spec_by_name = {spec.name: spec for spec in specs}
    names = sorted({f.field for f in (*fields_a, *fields_b)} & spec_by_name.keys())
    out: list[FieldComparison] = []
    for name in names:
        spec = spec_by_name[name]
        values_a = [f for f in fields_a if f.field == name]
        values_b = [f for f in fields_b if f.field == name]
        pairs = _pair_values(values_a, values_b, spec)
        # A list's leftovers are elements one lane did not read, never two readings of one value.
        positional = pair_leftovers_ambiguous and spec.cardinality != "many"
        leftover = _split_off_first_leftover_pair(pairs) if positional else None
        for a, b in pairs:
            if a is None or b is None:
                if not emit_one_sided:
                    continue
                present = a if a is not None else b
                assert present is not None
                missing_in = backend_a if a is None else backend_b
                out.append(
                    FieldComparison(
                        scope=scope,
                        field=name,
                        condition=present.condition,
                        status=one_sided,
                        missing_in=missing_in if one_sided == "missing" else None,
                        match_confidence=match_confidence,
                        a=a,
                        b=b,
                        detail=one_sided_detail or f"only in {backend_b if a is None else backend_a}",
                    )
                )
                continue
            status, detail = compare_values(a, b, spec)
            if normalize_key(a.condition) != normalize_key(b.condition):
                # Only an equal-value pair survives stage 2 (see _equal_pairs), so a differently worded
                # condition never turns into a conflict here; it is an agreement with a note saying so.
                detail = f"{detail}; conditions worded differently: {a.condition!r} / {b.condition!r}"
            out.append(
                FieldComparison(
                    scope=scope,
                    field=name,
                    condition=a.condition,
                    status=status,
                    match_confidence=match_confidence,
                    a=a,
                    b=b,
                    detail=detail_prefix + detail,
                )
            )
        if leftover is not None:
            left_a, left_b = leftover
            out.append(
                FieldComparison(
                    scope=scope,
                    field=name,
                    condition=left_a.condition,
                    status="ambiguous",
                    match_confidence=match_confidence,
                    a=left_a,
                    b=left_b,
                    detail=detail_prefix
                    + "both lanes report the field with different values under different conditions "
                    + f"({left_a.condition!r} / {left_b.condition!r})",
                )
            )
    return out


def _split_off_first_leftover_pair(
    pairs: list[tuple[FieldValue | None, FieldValue | None]],
) -> tuple[FieldValue, FieldValue] | None:
    """Remove the first unpaired value of each side from ``pairs`` and return the two, when both sides
    have one; ``None`` (leaving ``pairs`` untouched) when only one side does.

    Order is the order :func:`_pair_values` produced, so the choice is deterministic.
    """
    index_a = next((i for i, (a, b) in enumerate(pairs) if b is None and a is not None), None)
    index_b = next((i for i, (a, b) in enumerate(pairs) if a is None and b is not None), None)
    if index_a is None or index_b is None:
        return None
    left_a = pairs[index_a][0]
    left_b = pairs[index_b][1]
    assert left_a is not None and left_b is not None
    for i in sorted((index_a, index_b), reverse=True):
        pairs.pop(i)
    return left_a, left_b


def _pair_values(
    values_a: Sequence[FieldValue], values_b: Sequence[FieldValue], spec: FieldSpec
) -> list[tuple[FieldValue | None, FieldValue | None]]:
    """Pair up both sides' values for one field, so that every value is accounted for.

    Two stages. Stage 1 pairs values under the same measurement condition, by numeric proximity within
    that condition. Stage 2 pairs whatever is left across differently worded conditions ("ellipsometric"
    against "from ellipsometric", "Alloy target" against "alloy target used for all depositions"), but
    only where the *values* are equal: the condition is the model's own wording and differs between the
    lanes routinely, so equal values under differently worded conditions are one fact. Unequal values are
    not -- pairing those would claim the two conditions describe the same fact, a guess the code is not
    entitled to make -- so they stay one-sided. Anything still unpaired is reported one-sided.

    Every value gets a row. An earlier version indexed each side by condition and kept the first value per
    condition, which silently discarded a second reading of the same quantity -- exactly the case worth
    reporting, since a lane that reads a resistivity as both ``10^-2`` and ``10^2`` has an OCR problem the
    other lane may not share.

    A list field (``cardinality: many``) pairs as a set (:func:`_set_pairs`): its values are several facts that
    hold at once, so pairing "LiOH" with "NiSO4" because they came first would report a conflict the paper never
    made. What one lane has and the other lacks is reported one-sided, so a list never reports ``conflict``.
    """
    if spec.cardinality == "many":
        return _set_pairs(values_a, values_b, spec)
    groups_a, groups_b = _by_condition(values_a), _by_condition(values_b)
    pairs: list[tuple[FieldValue | None, FieldValue | None]] = []
    rest_a: list[FieldValue] = []
    rest_b: list[FieldValue] = []
    for key in sorted(groups_a.keys() | groups_b.keys()):
        group_a, group_b = list(groups_a.get(key, ())), list(groups_b.get(key, ()))
        pairs += _closest_pairs(group_a, group_b, rules_for(spec).distance)
        # Same condition, so whatever remains still describes the same fact: pair it in the order the paper
        # gave, rather than leaving both sides looking like the other is missing a value.
        while group_a and group_b:
            pairs.append((group_a.pop(0), group_b.pop(0)))
        rest_a += group_a
        rest_b += group_b
    pairs += _equal_pairs(rest_a, rest_b, spec)
    pairs += [(a, None) for a in rest_a]
    pairs += [(None, b) for b in rest_b]
    return pairs


def _set_pairs(
    values_a: Sequence[FieldValue], values_b: Sequence[FieldValue], spec: FieldSpec
) -> list[tuple[FieldValue | None, FieldValue | None]]:
    """A list field's pairing: each value of lane A with the first of lane B holding the same element
    (:func:`~paperfacts.kinds.element_key`, the union cell's identity too) under conditions that do not measure
    differently; everything else one-sided. A value naming no category pairs with nothing."""
    rest_b = list(values_b)
    pairs: list[tuple[FieldValue | None, FieldValue | None]] = []
    for a in values_a:
        key = element_key(spec, a.value_raw)
        j = next(
            (
                j
                for j, b in enumerate(rest_b)
                if key is not None
                and element_key(spec, b.value_raw) == key
                and not conditions_measure_differently(a.condition, b.condition)
            ),
            None,
        )
        pairs.append((a, None if j is None else rest_b.pop(j)))
    return pairs + [(None, b) for b in rest_b]


def conditions_measure_differently(condition_a: str | None, condition_b: str | None) -> bool:
    """Whether two conditions name different numbers, and so cannot describe the same measurement.

    Equal values are not enough to pair across conditions when the conditions themselves are numeric:
    85 % at 550 nm and 85 % at 600 nm are two measurements that happen to have come out the same, and
    reporting them as one AGREE would invent agreement the paper never claimed. Numbers are the part of a
    condition the lanes transcribe rather than phrase, so they are the part worth trusting -- wording
    alone ("Alloy target" against "alloy target used for all ATO film depositions") still pairs, and so
    does the same number said differently ("at 550 nm" against "550 nm wavelength"). A condition with no
    number on either side carries nothing to contradict, so it never blocks a pair.
    """
    # As a multiset: "550 nm, 25 °C" and "at 25 °C and 550 nm" name the same numbers in another order.
    numbers_a = sorted(condition_numbers(condition_a))
    numbers_b = sorted(condition_numbers(condition_b))
    return bool(numbers_a) and bool(numbers_b) and numbers_a != numbers_b


_REFERENCE = re.compile(
    r"\b(?:fig(?:ure)?s?|tables?|eqs?|equations?|refs?|sections?)\.?\s*S?\d+[a-z]?(?:\s*(?:,|&|and)\s*(?:S\d+[a-z]?|\d+[a-z])\b)*",
    re.IGNORECASE,
)


def condition_numbers(condition: str | None) -> tuple[float, ...]:
    """The numbers a condition names, in order: "550 nm" -> (550,), "400-800 nm" -> (400, 800).

    The one definition of a condition's numbers: this module pairs values by it and ``decide.py`` picks
    and cross-checks conditions by it, so the two can never parse a condition differently. In order, so a
    caller that cares (a configured preference "400-800") can match exactly; pairing across lanes compares
    them as a multiset (:func:`conditions_measure_differently`). The text is de-LaTeXed first, since MinerU
    spells a table's numbers "4 0 0".
    """
    if not condition:
        return ()
    # "(films after annealing, Fig. 3c)": a figure, table or equation number is where the paper says it, not
    # a number the measurement was taken at; counted, it made "average 400-1100 nm" miss the "400-1100" entry.
    text = _REFERENCE.sub(" ", delatex(normalize_text(condition)))
    numbers: list[float] = []
    for match in NUMBER_RE.finditer(text):
        token = match.group()
        # A "-" straight after a digit is a range separator, not a sign: "400-800" names 400 and 800,
        # not 400 and -800, and reading it as a sign would make one range look unlike the same range
        # written "400 to 800".
        if token[0] in "+-" and text[: match.start()].rstrip().endswith(tuple("0123456789")):
            token = token[1:]
        numbers.append(float(token.replace(",", "")))
    return tuple(numbers)


def _equal_pairs(
    rest_a: list[FieldValue], rest_b: list[FieldValue], spec: FieldSpec
) -> list[tuple[FieldValue | None, FieldValue | None]]:
    """Stage 2: pair leftovers whose values are equal, whatever their conditions say.

    Equality is the same test the report uses: ``compare_values`` returning "agree" -- the field's
    tolerances for a numeric field, ``same_text`` for a text one. Greedy and deterministic:
    lane A's order, first equal partner in lane B. Because only agreement pairs, a stage-2 row is an
    agreement by construction and can never manufacture a conflict.

    Conditions whose numbers disagree are refused outright, even when the values are equal: see
    :func:`conditions_measure_differently`.
    """
    pairs: list[tuple[FieldValue | None, FieldValue | None]] = []
    # Index-based removal: two values can be equal as models, and ``list.remove`` would then drop the
    # wrong one.
    i = 0
    while i < len(rest_a):
        for j, b in enumerate(rest_b):
            if conditions_measure_differently(rest_a[i].condition, b.condition):
                continue
            if compare_values(rest_a[i], b, spec)[0] == "agree":
                pairs.append((rest_a.pop(i), rest_b.pop(j)))
                break
        else:
            i += 1
    return pairs


def _closest_pairs(
    rest_a: list[FieldValue], rest_b: list[FieldValue], distance: Callable[[FieldValue, FieldValue], float | None]
) -> list[tuple[FieldValue | None, FieldValue | None]]:
    """Greedily pair the two closest values by the kind's ``distance``, removing them from both lists as it goes.

    Stops as soon as a side runs out or no pair left has a distance (nothing parsed as a number, or a kind with
    no distance at all), leaving those for the caller to pair in order or report one-sided.
    """
    pairs: list[tuple[FieldValue | None, FieldValue | None]] = []
    while rest_a and rest_b:
        candidates = [
            (gap, i, j)
            for i, a in enumerate(rest_a)
            for j, b in enumerate(rest_b)
            if (gap := distance(a, b)) is not None
        ]
        if not candidates:
            break
        _, i, j = min(candidates)
        pairs.append((rest_a.pop(i), rest_b.pop(j)))
    return pairs


def _by_condition(values: Sequence[FieldValue]) -> dict[str, list[FieldValue]]:
    """Group values by normalized condition, keeping every one of them in the order they were extracted."""
    grouped: dict[str, list[FieldValue]] = {}
    for value in values:
        grouped.setdefault(normalize_key(value.condition), []).append(value)
    return grouped


def _count(
    comparisons: Sequence[FieldComparison],
    matchings: Sequence[SampleMatching],
    lanes: Sequence[LaneExtraction],
    ambiguous_match_confidence: float,
) -> ComparisonCounts:
    """The counts over every entity's matching: pairs and unmatched samples summed, failed if any failed."""
    tally = Counter(c.status for c in comparisons)
    missing_by_backend = Counter(c.missing_in for c in comparisons if c.missing_in is not None)
    pairs = [pair for matching in matchings for pair in matching.pairs]
    return ComparisonCounts(
        agree=tally["agree"],
        conflict=tally["conflict"],
        ambiguous=tally["ambiguous"],
        missing=tally["missing"],
        total=len(comparisons),
        missing_by_backend={str(k): v for k, v in sorted(missing_by_backend.items())},
        samples_matched=len(pairs),
        samples_unmatched=sum(len(matching.unmatched_a) + len(matching.unmatched_b) for matching in matchings),
        low_confidence_matches=sum(1 for p in pairs if p.confidence < ambiguous_match_confidence),
        matching_failed=any(matching.failed for matching in matchings),
        unattributed_by_backend={lane.backend: len(lane.unattributed) for lane in lanes if lane.unattributed},
        unattributed_compared=sum(1 for c in comparisons if c.scope == "unattributed"),
    )
