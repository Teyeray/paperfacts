"""One dataset cell's verdict: which of a sample's candidate values the cell states, or why it states none.

:mod:`paperfacts.dataset` asks this once per sample and field. The answer is a chain of pure steps over one
candidate list, each narrowing it or refusing:

1. **Trust.** Only candidates located in the text and carrying a citation count.
2. **Narrow.** A lane that quotes the field under several conditions has several measurements; a rule must
   pick one (:func:`_one_condition`), or the lane's values must be one number under several wordings.
   Narrowing sees every candidate, bounds and ranges included: a bound at the condition the rules prefer is
   the answer to "what does the paper say there", and a scalar at a less preferred condition is not.
3. **Set aside non-scalars.** Only after narrowing has chosen the measurement: a bound or a range next to a
   scalar at the chosen condition is a weaker statement of it and is dropped with a note; a cell holding only
   such statements is ``non_scalar``. There is no second narrowing over the scalars alone, which would let a
   bound take its own condition out of the running. A quote the paper writes a bound before ("90" out of
   "above 90 %", :attr:`FieldValue.bound`) is a bound like any other. A range is a scalar only under
   ``range_policy`` lower/upper, as the end the field asks for (:mod:`paperfacts.kinds`).
4. **Agree.** Derived once, from the final candidates only: both lanes present, their values within the
   field's tolerance, and no pair across the lanes quoting conditions that measure differently. Never from
   the comparison report's statuses, which may be about a candidate an earlier step set aside.

The report's statuses serve one purpose, as a review gate: a ``conflict`` or ``ambiguous`` comparison refuses
the cell. Once narrowing has chosen conditions, a comparison wholly about candidates it set aside no longer
counts: a conflict between the lanes' 400-1100 nm averages is about a measurement the cell does not state when
their preferred 550 nm values agree. Any other troubled comparison still refuses.

Conditions and source ids of a committed cell are derived from the final candidates, in one place.

A list field (``cardinality: many``) is decided by :func:`decide_many` instead: the same refusals, then the
union of both lanes' elements rather than one value.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from paperfacts.compare import FieldComparison, condition_numbers, conditions_measure_differently
from paperfacts.fields import FieldSpec
from paperfacts.kinds import CellValue, element_key, joined, rules_for
from paperfacts.models import BACKENDS, Backend
from paperfacts.normalize import normalize_key, normalize_text
from paperfacts.records import FieldValue
from paperfacts.units import UnitRegistry

# How a condition says it is an average; used only to break a tie inside one preference entry.
_AVERAGE_WORDS = re.compile(r"\b(?:average[ds]?|avg|mean|avt)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Decision:
    value: CellValue
    status: str
    conditions: str
    sources: str
    detail: str
    # The committed value rests entirely on evidence the paper stated for the whole sample series,
    # never for this sample on its own. False for a rejected decision, which commits to nothing.
    series: bool = False
    # The backends whose trusted, parsed evidence produced the committed value. A reader seeing a
    # single-source cell needs to know which lane it came from; empty for a rejected decision.
    lanes: tuple[Backend, ...] = ()


@dataclass(frozen=True)
class _Candidate:
    backend: Backend
    value: FieldValue
    # The cell value the text states, or None when it states no single scalar (a bound, a range...).
    scalar: CellValue
    # The reading's note, or why there is no scalar.
    note: str | None


def decide(
    spec: FieldSpec,
    evidence: Sequence[tuple[Backend, FieldValue]],
    comparisons: Sequence[FieldComparison],
    *,
    units: UnitRegistry,
    blocked: str | None = None,
    unanswered: bool = False,
    row_sources: frozenset[str] = frozenset(),
) -> Decision:
    """The cell for ``spec`` given every lane's candidates for it, converted in ``units`` (the profile's).

    ``blocked`` is why nothing on this sample may be committed (its sample match failed or is too weak), or
    None. ``unanswered`` says some lane's question about this field got no valid answer: the other lane's value
    would then pass as single_source, as if that lane had read the paper and found nothing, so the cell is
    refused in both. ``row_sources`` are the blocks the rest of the sample's row cites, for
    :func:`_one_condition`.
    """

    def reject(status: str, reason: str) -> Decision:
        return _rejection(evidence, status, reason)

    if (refused := _opening_refusal(evidence, blocked=blocked, unanswered=unanswered)) is not None:
        return refused
    trusted = _trusted(evidence)
    rules = rules_for(spec)
    candidates = [_Candidate(backend, value, *rules.cell(value, spec, units)) for backend, value in trusted]
    # Narrowing sees every candidate, bounds included. Setting a bound aside first and narrowing again would
    # let it take its own condition out of the running, so the scalar's condition would win although no rule
    # chose it ("<100 nm as-deposited" + "95 nm annealed" would commit 95 as the film's thickness).
    narrowed = _narrow(spec, candidates, row_sources) if candidates else None
    troubled = [c for c in comparisons if c.status in {"conflict", "ambiguous"}]
    if narrowed is not None and len(narrowed[0]) < len(candidates):
        # Fail closed: a comparison is ignored only when every side it has is a candidate narrowing set aside.
        # One whose values match no candidate (a stale report, say) still refuses the cell.
        aside = {_identity(c.value) for c in candidates} - {_identity(c.value) for c in narrowed[0]}
        troubled = [c for c in troubled if not _only_about(c, aside)]
    if (refused := _review_refusal(evidence, troubled, comparisons, trusted)) is not None:
        return refused
    details = [_UNTRUSTED_NOTE] if len(trusted) != len(evidence) else []
    several = _several_conditions(candidates)
    if narrowed is None:
        return reject("multiple_conditions", "同一解析通道记录了多种测量条件，无法唯一确定")
    kept, reason = narrowed
    if reason:
        details.append(f"该样品有多种测量条件；{reason}")
    aside = [c for c in kept if c.scalar is None]
    final = [c for c in kept if c.scalar is not None]
    if not final:
        return reject("non_scalar", aside[0].note or "无法生成唯一标量")
    if aside:
        dropped = joined([f"{c.backend}: {_quote(c.value)}".strip() for c in aside])
        details.append(f"已排除不是唯一标量的候选（{dropped}）：{aside[0].note}")
    details.extend(c.note for c in final if c.note)

    # compare.py keeps the first value per condition. Inspect every candidate here so two different
    # same-condition values cannot disappear behind that first one.
    for backend in BACKENDS:
        same_lane = [c.scalar for c in final if c.backend == backend]
        if any(not rules.same(same_lane[0], scalar, spec) for scalar in same_lane[1:]):
            return reject("multiple_values", "同一解析通道在相同条件下记录了多个不同值")
    # Recipe numbers in a condition only stop an agreement the values themselves do not settle: two lanes quoting
    # the very same number agree whatever else their conditions restate; 100 against 104 needs the same state.
    exactly_equal = all(rules.same(final[0].scalar, c.scalar, spec) for c in final)
    if (_numbers_matter(spec) or not exactly_equal) and _lanes_measure_differently(final, strict=several):
        return reject("multiple_conditions", "两个解析通道的数值来自不同的测量条件")
    # Of spellings judged the same, the kind may prefer one (text: the one naming the field's category).
    chosen = min(final, key=lambda c: _preference(spec, c.backend, c.value))
    agreed = len({c.backend for c in final}) == 2 and all(rules.within(chosen.scalar, c.scalar, spec) for c in final)
    if not agreed and any(not rules.same(chosen.scalar, c.scalar, spec) for c in final):
        return reject("multiple_values", "多个候选值未经双路一致确认，无法唯一确定")
    return _commit(spec, chosen, final, agreed=agreed, details=details)


def _commit(
    spec: FieldSpec, chosen: _Candidate, final: Sequence[_Candidate], *, agreed: bool, details: list[str]
) -> Decision:
    """Nothing refused the evidence: record the value, the conditions and blocks it rests on, and how."""
    conditions = joined([c.value.condition or "" for c in final])
    sources = joined(sorted({source for c in final for source in c.value.source_ids}))
    if spec.condition_rule and spec.missing_condition_note_zh and not conditions:
        # A field whose prompt demands a condition: a value without one is kept, but the reader is told. The note
        # is the profile's (the loader refuses a rule without one), so the text stored here is covered by
        # comparison_key.
        details.append(spec.missing_condition_note_zh)
    details.append(f"采用 {chosen.backend}；抽取重复一致率 {chosen.value.agreement:g}；合并重复证据")
    return Decision(
        chosen.scalar,
        "agree" if agreed else "single_source",
        conditions,
        sources,
        joined(details),
        series=all(c.value.series for c in final),
        lanes=tuple(dict.fromkeys(c.backend for c in final)),
    )


def _rejection(evidence: Sequence[tuple[Backend, FieldValue]], status: str, reason: str) -> Decision:
    """A refused cell: no value, with every candidate's quote, condition and blocks in the audit trail."""
    raw = joined([f"{backend}: {_quote(value)}" for backend, value in evidence])
    conditions = joined([value.condition or "" for _, value in evidence])
    sources = joined(sorted({source for _, value in evidence for source in value.source_ids}))
    return Decision(None, status, conditions, sources, joined([reason, raw]))


def decide_cell(
    spec: FieldSpec,
    evidence: Sequence[tuple[Backend, FieldValue]],
    comparisons: Sequence[FieldComparison],
    *,
    units: UnitRegistry,
    blocked: str | None = None,
    unanswered: bool = False,
    row_sources: frozenset[str] = frozenset(),
) -> Decision:
    """The cell of ``spec``: :func:`decide_many` for a list field, :func:`decide` for every other."""
    if spec.cardinality == "many":
        return decide_many(spec, evidence, comparisons, blocked=blocked, unanswered=unanswered)
    return decide(
        spec, evidence, comparisons, units=units, blocked=blocked, unanswered=unanswered, row_sources=row_sources
    )


def _opening_refusal(
    evidence: Sequence[tuple[Backend, FieldValue]], *, blocked: str | None, unanswered: bool
) -> Decision | None:
    """The refusals every cell starts with, in order: a lane's question went unanswered, nothing was extracted,
    or the sample match forbids committing anything."""
    if unanswered:
        return _rejection(evidence, "unanswered", "某一解析通道对该字段的提问未得到有效回答；下次运行会重新提问")
    if not evidence:
        return _rejection(evidence, "missing", "未提取到该字段；留空，不填 0")
    if blocked:
        return _rejection(evidence, "ambiguous", blocked)
    return None


def _review_refusal(
    evidence: Sequence[tuple[Backend, FieldValue]],
    troubled: Sequence[FieldComparison],
    comparisons: Sequence[FieldComparison],
    trusted: Sequence[tuple[Backend, FieldValue]],
) -> Decision | None:
    """The refusals that follow, in order: a troubled comparison, none at all, or no trusted evidence."""
    if troubled:
        status = "conflict" if any(c.status == "conflict" for c in troubled) else "ambiguous"
        return _rejection(evidence, status, "双路比较存在冲突或歧义，需人工复核")
    if not comparisons:
        return _rejection(evidence, "unreviewed", "比较报告没有覆盖该字段")
    if not trusted:
        return _rejection(evidence, "ungrounded", "没有同时通过原文定位且包含有效引用的证据")
    return None


_UNTRUSTED_NOTE = "已排除未定位到原文或缺少有效引用的候选"


def _trusted(evidence: Sequence[tuple[Backend, FieldValue]]) -> list[tuple[Backend, FieldValue]]:
    """The candidates a cell may rest on: located in the text and carrying a citation."""
    return [(backend, value) for backend, value in evidence if value.grounded and value.source_ids]


def _preference(spec: FieldSpec, backend: Backend, value: FieldValue) -> tuple[object, ...]:
    """The chooser among spellings judged the same, smallest first: the kind's preference (text: the one naming
    the field's category), then the most repeated, then MinerU, then the text."""
    return (*rules_for(spec).prefer(value, spec), -value.agreement, BACKENDS.index(backend), value.value_raw)


# ---- A list field ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Element:
    """One trusted value of a list field, as the element it contributes."""

    backend: Backend
    value: FieldValue
    # kinds.element_key: what makes two values one element.
    key: str


def decide_many(
    spec: FieldSpec,
    evidence: Sequence[tuple[Backend, FieldValue]],
    comparisons: Sequence[FieldComparison],
    *,
    blocked: str | None = None,
    unanswered: bool = False,
) -> Decision:
    """The cell of a list field (``cardinality: many``): the union of the elements either lane grounded, each
    traceable to the lanes that read it.

    The refusals are :func:`decide`'s, in its order, and nothing else: a list has no condition to narrow and no
    one value to agree on. The comparison pairs a list as a set (:func:`paperfacts.compare._set_pairs`), so a
    ``conflict`` cannot arise; an ``ambiguous`` one (a failed sample match) still refuses.

    An element is one trusted value, identified by :func:`~paperfacts.kinds.element_key` -- the comparison's
    identity too. A field with categories keeps the category a value names and refuses as an element a value
    naming none: "XRD and XPS" is two categories, which reads as none, so the field line asks for one entry
    each. Of an element's spellings the cell writes the one :func:`decide`'s chooser prefers. The list follows
    the categories' declared order, or else the order first seen, MinerU first. The cell is ``agree`` when both
    lanes hold every element and ``single_source`` otherwise; the detail names each element's lanes, so a union
    never hides which lane an element rests on. Unlike the comparison, grouping ignores conditions: one element
    under two conditions naming different numbers is still one element of the list, even where the report shows
    it one-sided in each lane.
    """
    if (refused := _opening_refusal(evidence, blocked=blocked, unanswered=unanswered)) is not None:
        return refused
    troubled = [c for c in comparisons if c.status in {"conflict", "ambiguous"}]
    trusted = _trusted(evidence)
    if (refused := _review_refusal(evidence, troubled, comparisons, trusted)) is not None:
        return refused
    details = [_UNTRUSTED_NOTE] if len(trusted) != len(evidence) else []
    groups, refused_quotes = _elements(spec, trusted)
    if refused_quotes:
        details.append(f"已排除未对应唯一类别的元素（{joined(refused_quotes)}）：每个元素须单独引用一个类别")
    if not groups:
        return _rejection(evidence, "non_scalar", joined([*details, "没有可作为列表元素的证据"]))
    written = [min(group, key=lambda e: _preference(spec, e.backend, e.value)) for group in groups]
    lanes = [tuple(backend for backend in BACKENDS if any(e.backend == backend for e in group)) for group in groups]
    # A category is written as declared; any other element as its lane quoted it.
    names = [element.key if spec.categories else " ".join(element.value.value_raw.split()) for element in written]
    details.append("列表为两路已定位证据的并集")
    details.append(
        "元素来源：" + "；".join(f"{name}（{', '.join(held)}）" for name, held in zip(names, lanes, strict=True))
    )
    kept = [element.value for group in groups for element in group]
    return Decision(
        names,
        "agree" if all(len(held) == len(BACKENDS) for held in lanes) else "single_source",
        joined([value.condition or "" for value in kept]),
        joined(sorted({source for value in kept for source in value.source_ids})),
        joined(details),
        series=all(value.series for value in kept),
        lanes=tuple(backend for backend in BACKENDS if any(backend in held for held in lanes)),
    )


def _elements(spec: FieldSpec, trusted: Sequence[tuple[Backend, FieldValue]]) -> tuple[list[list[_Element]], list[str]]:
    """The trusted values grouped into elements, in the cell's order, and the quotes refused as an element."""
    groups: dict[str, list[_Element]] = {}
    refused: list[str] = []
    for backend, value in sorted(trusted, key=lambda item: BACKENDS.index(item[0])):
        text = " ".join(value.value_raw.split())
        if not text:
            continue
        key = element_key(spec, text)
        if key is None:
            refused.append(f"{backend}: {text}")
        else:
            groups.setdefault(key, []).append(_Element(backend, value, key))
    ordered = list(groups.values())
    if spec.categories:
        ordered.sort(key=lambda group: spec.categories.index(group[0].key))
    return ordered, refused


def _quote(value: FieldValue) -> str:
    """The quote and its unit for an audit note. A unit the quote already ends with is not written twice: "3
    wt.%" with unit "wt.%" read "3 wt.% wt.%", as if the paper had repeated it."""
    unit = value.unit_raw or ""
    if unit and normalize_text(value.value_raw).endswith(normalize_text(unit)):
        unit = ""
    return f"{value.value_raw} {unit}"


def _only_about(comparison: FieldComparison, aside: set[tuple[object, ...]]) -> bool:
    """Whether a comparison is wholly about candidates narrowing set aside. One with no values at all is about
    nothing known, so it is not."""
    sides = [value for value in (comparison.a, comparison.b) if value is not None]
    return bool(sides) and all(_identity(value) in aside for value in sides)


def _identity(value: FieldValue) -> tuple[object, ...]:
    """What identifies one extracted value between the lane and a comparison of it. Not the whole model: a
    stored report may carry grounding verdicts older than the lane's re-derived ones."""
    # ``holds`` too: two quotes of the same words, one read as true and one as false, are two values.
    return (value.field, value.value_raw, value.unit_raw, value.condition, tuple(value.source_ids), value.holds)


# ---- Narrowing ---------------------------------------------------------------------------------------------


def _several_conditions(candidates: Sequence[_Candidate]) -> bool:
    """Whether some lane quotes the field under more than one condition.

    Per lane only: the two lanes word the same condition differently ("after sputtering" vs "after
    deposition"), so only a lane disagreeing with itself is evidence of several measurements. The key is
    normalize_key, the one compare.py and extract.py judge conditions by, so a condition the comparison report
    called one thing is never two here.
    """
    return any(
        len({normalize_key(c.value.condition) for c in candidates if c.backend == backend}) > 1 for backend in BACKENDS
    )


def _narrow(
    spec: FieldSpec, candidates: Sequence[_Candidate], row_sources: frozenset[str]
) -> tuple[list[_Candidate], str | None] | None:
    """The candidates of the one measurement the cell states, with the reason when there was a choice; or
    None when the lanes hold several measurements and nothing picks one."""
    if not _several_conditions(candidates):
        return list(candidates), None
    chosen = _one_condition(spec, candidates, row_sources)
    if chosen is not None:
        return chosen
    if _one_number(spec, candidates):
        return list(candidates), "同一解析通道的多种条件表述给出同一数值，视为同一测量"
    return None


def _one_number(spec: FieldSpec, candidates: Sequence[_Candidate]) -> bool:
    """Whether each lane's several condition texts are notes on one measurement rather than several.

    "100 nm, by TEM cross-section" and "100 nm, not reduced by the forming gas" are one thickness under two
    wordings. Only the very same number counts: 100 nm as-deposited and 104 nm after annealing are within
    tolerance of each other and are still two states, which a lane holding both under one sample usually means
    an inventory error worth surfacing. For a field measured along a stated axis (:func:`_numbers_matter`),
    conditions naming different numbers are never merged, however equal the values: 85 % at 450 nm and 85 % at
    600 nm are two measurements. Whether the other lane measured the same
    thing is judged afterwards, by :func:`_lanes_measure_differently`.
    """
    for backend in BACKENDS:
        same_lane = [c for c in candidates if c.backend == backend]
        for index, candidate in enumerate(same_lane):
            for other in same_lane[index + 1 :]:
                if candidate.scalar is None or other.scalar is None:
                    return False
                if _numbers_matter(spec) and conditions_measure_differently(
                    candidate.value.condition, other.value.condition
                ):
                    return False
                if not rules_for(spec).same(candidate.scalar, other.scalar, spec):
                    return False
    return True


def _one_condition(
    spec: FieldSpec, candidates: Sequence[_Candidate], row_sources: frozenset[str]
) -> tuple[list[_Candidate], str] | None:
    """Of a sample's several measurements, the one the cell should state, with the reason; or None.

    Tried in order, the first that settles it wins:

    1. The condition stated in a block the rest of the row also cites. "Resistivity of 5.74e-4 Ω·cm and a
       transmittance of 83.5 % (400-1800 nm)" ties one of several transmittances to the rest of its row;
       that is the one a reader expects in the cell.
    2. The field's ``condition_preference``, entry by entry: a condition matches an entry when it names
       exactly the entry's numbers, so "average 400–800 nm" and "from 400 to 800 nm" both match "400-800".
       When an entry matches several conditions in one lane, the one that says average / avg / mean / AVT is
       taken: "average 400-1100 nm" is the measurement a reader compares across papers, a peak or a minimum
       over the same range is not. A peak, a minimum or an unlabelled range never wins this way. An entry
       that still matches two conditions in one lane (550 nm as-deposited and 550 nm annealed) ends the
       search: the next entry would pick a third measurement whose state nobody chose.

    Once a rule chooses, every lane is held to it: a lane keeps only its values the rule picks, so a lane
    quoting a different condition cannot vouch for the one chosen. A rule that picks two conditions in one
    lane, two different conditions across the lanes, or nothing in any, settles nothing and the next is
    tried.
    """
    kept, _ = _held_to(candidates, lambda value: bool(row_sources.intersection(value.source_ids)))
    if kept:
        return kept, "采用与本行其他字段引用同一原文块的条件"
    for entry in spec.condition_preference:
        numbers = condition_numbers(entry)

        def matches(value: FieldValue, numbers: tuple[float, ...] = numbers) -> bool:
            return condition_numbers(value.condition) == numbers

        kept, tied = _held_to(candidates, matches)
        if kept:
            return kept, f"按字段配置的优先条件 {entry} 选取"
        if not tied:
            continue
        kept, _ = _held_to(candidates, lambda value, matches=matches: matches(value) and _is_average(value))
        if kept:
            return kept, f"按字段配置的优先条件 {entry} 选取，取平均值"
        # The entry matched several states and no average settled them; a later entry would choose a third
        # measurement whose state nobody picked.
        return None
    return None


def _is_average(value: FieldValue) -> bool:
    return bool(_AVERAGE_WORDS.search(value.condition or ""))


def _held_to(
    candidates: Sequence[_Candidate], picks: Callable[[FieldValue], bool]
) -> tuple[list[_Candidate] | None, bool]:
    """Each lane's candidates ``picks`` keeps, when every lane keeps one condition and the lanes keep the same
    one; otherwise None. The flag says whether the rule failed on a tie (one lane kept two conditions)
    rather than on picking nothing.

    A lane the rule picks nothing from is not silently dropped: its values whose condition names no number
    (or has none) stay in, so the tolerance check can still find that they contradict the chosen value.

    A rule is judged per lane (rule 1 looks at each lane's own blocks), so each lane can pick one condition
    and still not the other lane's: "85 % @ 550 nm" in one and "85.2 % @ 400-800 nm" in the other are two
    measurements, and committing them as one agreement is the manufactured agreement this module refuses.
    Across lanes the wording differs, so "the same condition" is compare.py's rule: they do not name
    different numbers.
    """
    kept: list[_Candidate] = []
    chosen: dict[Backend, str | None] = {}
    for backend in BACKENDS:
        groups: dict[str, list[_Candidate]] = {}
        for candidate in candidates:
            if candidate.backend == backend and picks(candidate.value):
                groups.setdefault(normalize_key(candidate.value.condition), []).append(candidate)
        if len(groups) > 1:
            return None, True
        for group in groups.values():
            chosen[backend] = group[0].value.condition
            kept += group
    if len(chosen) == 2 and conditions_measure_differently(*chosen.values()):
        return None, False
    if not kept:
        return None, False
    for backend in BACKENDS:
        if backend not in chosen:
            kept += [c for c in candidates if c.backend == backend and not condition_numbers(c.value.condition)]
    return kept, False


# ---- Agreement ---------------------------------------------------------------------------------------------


def _numbers_matter(spec: FieldSpec) -> bool:
    """Whether a number in this field's condition text names the measurement itself.

    A field that declares a ``condition_hint`` is measured along an axis the paper states beside the value --
    transmittance at a wavelength -- so "85 % at 450 nm" and "85 % at 600 nm" are two measurements. Every other
    field's condition text is a description, and its numbers are the sample's recipe restated ("1 h, in 15 %
    H2 forming gas" against "1 h, forming gas, 400 °C"): reading those as separate measurements refuses a value
    both lanes quote identically.
    """
    return spec.condition_hint is not None


def _lanes_measure_differently(final: Sequence[_Candidate], *, strict: bool) -> bool:
    """Whether some pair across the lanes quotes conditions that cannot be one measurement.

    The comparison's rule by default (:func:`conditions_measure_differently`: both name numbers and the
    numbers differ), under which the lanes' paraphrases of one condition still pair. ``strict`` is for a
    sample where a lane quoted several conditions: there the conditions are known to separate measurements,
    so the lanes must name the very same numbers -- "100 nm (TEM)" merged with "100 nm (SEM)" in one lane is no
    evidence for "100 nm after annealing at 500 °C" in the other.
    """
    first, second = ([c.value.condition for c in final if c.backend == backend] for backend in BACKENDS)
    for a in first:
        for b in second:
            if strict and sorted(condition_numbers(a)) != sorted(condition_numbers(b)):
                return True
            if conditions_measure_differently(a, b):
                return True
    return False
