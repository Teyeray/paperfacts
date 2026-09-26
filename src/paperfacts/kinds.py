"""What a field's kind decides: how its quote is read, when two lanes agree, and what a dataset cell holds.

One rules object per :data:`~paperfacts.fields.FieldKind`, looked up with :func:`rules_for`. The stages that
treat kinds differently -- lane normalisation (:func:`paperfacts.normalize.normalize_field`), the comparison
(:func:`paperfacts.compare.compare_values` and its value pairing), the dataset cell (:mod:`paperfacts.decide`)
and the field line of every question (:mod:`paperfacts.prompts`) -- ask the row instead of branching on
``spec.kind``, so a new kind is one new row here rather than a branch in each of them.

Two stages sit below this module and cannot import it: cleaning (:mod:`paperfacts.records`) and retrieval
(:mod:`paperfacts.passages`, which must not reach :mod:`paperfacts.normalize`). Both only need to know which
kinds quote a number, and read that from :data:`paperfacts.fields.DIGIT_KINDS`.

This module decides verdicts and which values survive extraction, so its source is hashed into the
extraction, normalisation and comparison keys.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from paperfacts.fields import RANGE_ENDS, FieldKind, FieldSpec
from paperfacts.normalize import (
    LOOSE_PUNCTUATION,
    NUMBER_ATOM,
    SCALAR,
    Reading,
    canonical_category,
    clean_unit,
    convert_to_canonical,
    delatex,
    normalize_key,
    normalize_text,
    parse_number,
    read_date,
    read_interval,
    read_number,
    read_range,
    read_value,
    same_text,
    unit_of_value,
)
from paperfacts.records import FieldValue, resolve_reference
from paperfacts.units import UnitRegistry

if TYPE_CHECKING:
    from paperfacts.compare import FactStatus
    from paperfacts.profile import EntitySpec

# A list is the cell of an interval ([low, high], None for an open end) or of a field holding several values at
# once (``cardinality: many``, paperfacts.decide.decide_many).
CellValue = str | float | int | bool | list[str | float | None] | None


# What a list field's line adds to its description (TextRules.note).
LIST_NOTE = "Several values may hold at once: report each as its own entry."


def joined(values: Sequence[str]) -> str:
    """Distinct non-empty strings, in order, as one cell."""
    return "; ".join(dict.fromkeys(value for value in values if value))


@dataclass(frozen=True)
class KindContext:
    """What a ``reference`` field is asked, read, compared and decided against beyond its own spec. Every other kind
    ignores it, so a caller with nothing of the kind passes :data:`NO_CONTEXT`.

    Each stage fills the part it holds: the question the entity types a field may name, a lane's normalisation that
    lane's samples, the comparison every entity's matching, the dataset every row id and the lane of the value."""

    # The profile's entity types by name: the referenced one's heading and noun, for the field line's note.
    entities: Mapping[str, EntitySpec] = field(default_factory=dict)
    # One lane's samples of each entity type, sample_key -> sample_id (LaneExtraction.listed).
    samples: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    # Each entity type's matched pairs, (lane A's sample id, lane B's sample id).
    pairs: Mapping[str, frozenset[tuple[str, str]]] = field(default_factory=dict)
    # The dataset row id of a lane's sample: (backend, entity, the lane's sample id) -> the row's sample_id.
    row_ids: Mapping[tuple[str, str, str], str] = field(default_factory=dict)
    # The lane the value decided belongs to; set per candidate by paperfacts.decide.
    backend: str | None = None


NO_CONTEXT = KindContext()


class KindRules(Protocol):
    """The per-kind decisions. Every method is pure."""

    def read(
        self, field: FieldValue, spec: FieldSpec, units: UnitRegistry, ctx: KindContext = NO_CONTEXT
    ) -> FieldValue:
        """``field`` with ``value`` / ``unit`` filled in, in ``units`` (``normalize.normalize_field``)."""
        ...

    def compare(
        self, a: FieldValue, b: FieldValue, spec: FieldSpec, ctx: KindContext = NO_CONTEXT
    ) -> tuple[FactStatus, str]:
        """The verdict on two read values that are both present."""
        ...

    def distance(self, a: FieldValue, b: FieldValue) -> float | None:
        """How far apart two read values under one condition are, for pairing the closest first; None when
        there is no such measure and values under one condition pair in the order the paper gave."""
        ...

    def cell(
        self, value: FieldValue, spec: FieldSpec, units: UnitRegistry, ctx: KindContext = NO_CONTEXT
    ) -> tuple[CellValue, str | None]:
        """``(cell value, note)``, or ``(None, reason)`` when the quote states no single value for a cell."""
        ...

    def same(self, a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
        """Whether two candidate cells state the same thing."""
        ...

    def within(self, a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
        """Whether two candidate cells agree the way the comparison judges agreement."""
        ...

    def prefer(self, value: FieldValue, spec: FieldSpec) -> tuple[object, ...]:
        """The leading sort key when the cell chooses among spellings judged the same (smallest first)."""
        ...

    def note(self, spec: FieldSpec, ctx: KindContext = NO_CONTEXT) -> str:
        """Text appended to the field's description in its field line; "" when the kind adds nothing."""
        ...


# ---- numeric -----------------------------------------------------------------------------------------------

# The words of a condition tail that may name a unit ("2 h", "550 nm", "°C").
_TAIL_WORD = re.compile(r"[^\d\s,;:()\[\]]+")

# "100 nm (± 5 nm)": the uncertainty in parentheses after the unit, read as "100 ± 5 nm" when both units agree.
_PARENTHESISED_UNCERTAINTY = re.compile(
    rf"^(?P<center>{NUMBER_ATOM})\s*(?P<unit>[^\d\s(±][^(±]*?)?\s*\(\s*(?:±|\+/-|\+-)\s*(?P<uncertainty>{NUMBER_ATOM})\s*(?P<again>[^)]*)\)$"
)
# The tilde operator U+223C and its friends are folded to "~" by normalize_text, which runs first.
_APPROX_NOTE = "原文为近似值，保留中心值"
_RANGE_APPROX_NOTE = "原文为近似值"
_APPROX = re.compile(r"^(?:approximately|approx\.?|roughly|around|about|circa|ca\.?|[~≈≃≅])\s*", re.IGNORECASE)


def _unit_key(unit_raw: str | None) -> str:
    return clean_unit(unit_raw) if unit_raw else ""


def _with_after_condition(field: FieldValue, clause: str) -> FieldValue:
    """``field`` with ``clause`` in its condition, unless the condition already says it: normalising twice must
    give the same value."""
    condition = field.condition
    if condition and normalize_key(clause) in normalize_key(condition):
        return field
    return field.model_copy(update={"condition": f"{condition}; {clause}" if condition else clause})


def _names_unit_of(spec: FieldSpec, text: str, units: UnitRegistry) -> bool:
    """Whether ``text`` names a unit of the field's own quantity ("2 h" for a time, "°C" for a temperature)."""
    canonical = spec.canonical_unit
    if canonical is None:
        return False
    return any(units.convert(canonical, word) is not None for word in _TAIL_WORD.findall(normalize_text(text)))


def _is_field_unit(value: FieldValue, spec: FieldSpec, units: UnitRegistry) -> Callable[[str], bool]:
    """Whether a unit written in ``value``'s quote is one of the field's (:func:`unit_of_value`)."""
    return lambda written: unit_of_value(spec, value.unit_raw, written, units) is not None


def _unit_to_convert(
    value: FieldValue, spec: FieldSpec, units: UnitRegistry, written: str | None
) -> tuple[str | None, bool]:
    """``(unit, own)``: what the number of ``value`` is converted from, given the unit its quote writes after it
    (None when that is not known), and whether that is the quote's own unit rather than ``unit_raw``. A written
    unit that is none of the field's leaves ``unit_raw`` in place: refusing it is the caller's to decide."""
    chosen = unit_of_value(spec, value.unit_raw, written, units) if written else None
    return chosen if chosen is not None else (value.unit_raw, False)


def _range_end(
    value: FieldValue, spec: FieldSpec, units: UnitRegistry, reading: Reading, notes: list[str], refusal: str
) -> tuple[CellValue, str | None]:
    """The end of a quoted range the field asks for (``range_policy`` lower/upper), or ``refusal``: why the text
    is no single scalar.

    An end is a number the paper printed, so it may fill a cell; a midpoint was never measured, so under
    ``midpoint`` or ``reject`` a range stays out. Only a clean range has an end (:func:`read_range`, which the
    lanes read ends with too)."""
    if spec.range_policy not in RANGE_ENDS:
        return None, refusal
    clean = read_range(reading.text, _is_field_unit(value, spec, units))
    if clean is None:
        return None, refusal
    low, high, written = clean
    unit_raw, own = _unit_to_convert(value, spec, units, written)
    upper = spec.range_policy == "upper"
    canonical, _, note = convert_to_canonical(
        spec, high if upper else low, unit_raw, units, value_text=reading.text, range_ends=(low, high)
    )
    if canonical is None or not math.isfinite(canonical):
        return None, note or "单位无法转换为标准单位"
    chosen = f"原文为区间 {low:g}–{high:g}，按字段配置取{'上限' if upper else '下限'}"
    return canonical, joined([note or "", _own_unit_note(written, value.unit_raw) if own else "", *notes, chosen])


def _own_unit_note(written: str, unit_raw: str | None) -> str:
    return f"原文数值自带单位 {written!r}，按其换算" + (f"（unit_raw 为 {unit_raw!r}）" if unit_raw else "")


class NumericRules:
    """A number in the field's canonical unit, compared within the field's tolerances."""

    def read(
        self, field: FieldValue, spec: FieldSpec, units: UnitRegistry, ctx: KindContext = NO_CONTEXT
    ) -> FieldValue:
        reading = read_value(field, spec, units)
        lead_notes = []
        if reading.number_word is not None:
            lead_notes.append(f"number word {field.value_raw.strip()!r} read as {reading.number_word}")
        if reading.clause:
            field = _with_after_condition(field, reading.clause)
            lead_notes.append(f"{reading.clause!r} moved into the condition")
        if reading.bound:
            lead_notes.append(f"bound {reading.bound!r} stands before the quote in its cited block")
        lead_note = "; ".join(lead_notes) or None
        bare, condition = reading.bare, reading.condition
        if condition and _names_unit_of(spec, condition, units) and not _names_unit_of(spec, bare, units):
            # "400 °C for 2 h" on annealing_time: the time is in the tail, and the number kept is a temperature.
            note = f"the condition {condition!r} holds this field's quantity and the value does not; ambiguous"
            return field.model_copy(update={"value": None, "unit": None, "normalization_note": note})
        if reading.compound is not None:
            compound_note = f"compound {bare!r} read as {reading.compound:g} {spec.canonical_unit}"
            note = "; ".join(n for n in (lead_note, *reading.context_notes, compound_note) if n)
            return field.model_copy(
                update={"value": reading.compound, "unit": spec.canonical_unit, "normalization_note": note}
            )
        parsed = read_number(
            reading.text, range_policy=spec.range_policy, range_unit=_is_field_unit(field, spec, units)
        )
        if parsed.value is None:
            return field.model_copy(update={"value": None, "unit": None, "normalization_note": parsed.note})
        unit_raw, own = _unit_to_convert(field, spec, units, parsed.unit)
        own_note = f"the value's own unit {parsed.unit!r} converted, not unit_raw {field.unit_raw!r}" if own else None
        # The range an end was taken from decides a bare number's unit for both ends at once.
        value, unit, unit_note = convert_to_canonical(
            spec,
            parsed.value,
            unit_raw,
            units,
            value_text=reading.text,
            range_ends=parsed.ends if spec.range_policy in RANGE_ENDS else None,
        )
        note = "; ".join(n for n in (lead_note, parsed.note, own_note, unit_note) if n) or None
        return field.model_copy(update={"value": value, "unit": unit, "normalization_note": note})

    def compare(
        self, a: FieldValue, b: FieldValue, spec: FieldSpec, ctx: KindContext = NO_CONTEXT
    ) -> tuple[FactStatus, str]:
        if a.value is None or b.value is None:
            # At least one side failed to parse as a number: if the raw text (including unit, case-sensitive)
            # is identical on both sides, that still counts as agreement; otherwise there's no way to judge
            if normalize_key(a.value_raw) == normalize_key(b.value_raw) and _unit_key(a.unit_raw) == _unit_key(
                b.unit_raw
            ):
                return "agree", "identical raw text (not parsed as a number)"
            return (
                "ambiguous",
                f"unparsed: {a.normalization_note or a.value_raw!r} vs {b.normalization_note or b.value_raw!r}",
            )
        if a.unit != b.unit:
            return "ambiguous", f"units differ after normalization: {a.unit} vs {b.unit}"
        if math.isclose(a.value, b.value, rel_tol=spec.rel_tol, abs_tol=spec.abs_tol):
            return (
                "agree",
                f"{a.value:g} ≈ {b.value:g} {a.unit or ''} (rel_tol={spec.rel_tol:g}, abs_tol={spec.abs_tol:g})",
            )
        return "conflict", f"{a.value:g} vs {b.value:g} {a.unit or ''}"

    def distance(self, a: FieldValue, b: FieldValue) -> float | None:
        if a.value is None or b.value is None:
            return None
        return abs(a.value - b.value)

    def cell(
        self, value: FieldValue, spec: FieldSpec, units: UnitRegistry, ctx: KindContext = NO_CONTEXT
    ) -> tuple[CellValue, str | None]:
        # The comparison's reading (normalize.read_value), so the cell and the report agree on what the quote says;
        # what follows is only the cell's stricter demand of one exact scalar.
        reading = read_value(value, spec, units)
        if reading.bound:
            return None, f"原文在所引数值前写有界限 {reading.bound!r}，不是唯一精确标量"
        text = delatex(normalize_text(reading.text)).strip()
        approx = _APPROX.match(text)
        if approx:
            text = text[approx.end() :].strip()
        word = []
        if reading.number_word is not None:
            word.append(f"原文为英文数词 {value.value_raw.strip()!r}，读作 {reading.number_word}")
        clause = [f"{reading.clause!r} 已计入测量条件"] if reading.clause else []

        def context(approx_note: str) -> list[str]:
            # A centre value is kept of an approximate scalar, an end of an approximate range.
            return [*word, *([approx_note] if approx else []), *clause]

        notes = context(_APPROX_NOTE)
        if reading.compound is not None:
            if reading.condition:
                notes.append(f"条件 {reading.condition!r} 不计入数值")
            compound_note = f"原文为复合时长 {reading.bare!r}，合计 {reading.compound:g} {spec.canonical_unit}"
            return reading.compound, joined([*notes, compound_note])
        parenthesised = _PARENTHESISED_UNCERTAINTY.fullmatch(text)
        if parenthesised:
            center, unit, uncertainty, again = parenthesised.group("center", "unit", "uncertainty", "again")
            if clean_unit(unit or "") == clean_unit(again):
                text = f"{center} ± {uncertainty} {unit or ''}"
        match = SCALAR.fullmatch(text)
        if match is None:
            return _range_end(
                value,
                spec,
                units,
                reading,
                context(_RANGE_APPROX_NOTE),
                "不是唯一精确标量（含上下界、区间、尺寸组合或无法解析的文字）",
            )
        tail = match.group("tail").strip()
        chosen = unit_of_value(spec, value.unit_raw, tail, units)
        if chosen is None:
            return _range_end(
                value,
                spec,
                units,
                reading,
                context(_RANGE_APPROX_NOTE),
                "含多个数值、范围、上下界或附加条件，不能取中点或第一个数",
            )
        # No range_policy: "center" is one number. A range reaches a cell only through _range_end.
        number, _ = parse_number(match.group("center"))
        if number is None or not math.isfinite(number):
            return None, "数值不可解析或非有限数"
        unit_raw, own = chosen
        canonical, _, note = convert_to_canonical(spec, number, unit_raw, units, value_text=match.group("center"))
        if canonical is None or not math.isfinite(canonical):
            return None, note or "单位无法转换为标准单位"
        notes.insert(0, note or "")
        if own:
            notes.insert(1, _own_unit_note(tail, value.unit_raw))
        if match.group("uncertainty"):
            notes.append(f"原文不确定度 ±{match.group('uncertainty')} {unit_raw or ''}；保留中心值")
        return canonical, joined(notes) or None

    def same(self, a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)
        return False

    def within(self, a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=spec.rel_tol, abs_tol=spec.abs_tol)
        return False

    def prefer(self, value: FieldValue, spec: FieldSpec) -> tuple[object, ...]:
        return ()

    def note(self, spec: FieldSpec, ctx: KindContext = NO_CONTEXT) -> str:
        return ""


# ---- text and composition ----------------------------------------------------------------------------------


class TextRules:
    """Text as quoted, equal as :func:`paperfacts.normalize.same_text` judges it: across spacing, case and a
    lost hyphen, or by the category both name. A composition is compared the same way."""

    def read(
        self, field: FieldValue, spec: FieldSpec, units: UnitRegistry, ctx: KindContext = NO_CONTEXT
    ) -> FieldValue:
        # Compared through same_text on the fly.
        return field

    def compare(
        self, a: FieldValue, b: FieldValue, spec: FieldSpec, ctx: KindContext = NO_CONTEXT
    ) -> tuple[FactStatus, str]:
        if same_text(spec, a.value_raw, b.value_raw):
            category = canonical_category(spec.categories, a.value_raw) or canonical_category(
                spec.categories, b.value_raw
            )
            return "agree", f"both name {category}" if category else "identical after text normalization"
        return "conflict", f"{a.value_raw!r} vs {b.value_raw!r}"

    def distance(self, a: FieldValue, b: FieldValue) -> float | None:
        return None

    def cell(
        self, value: FieldValue, spec: FieldSpec, units: UnitRegistry, ctx: KindContext = NO_CONTEXT
    ) -> tuple[CellValue, str | None]:
        return value.value_raw.strip(), None

    def same(self, a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
        """Text is judged as the comparison judges it (:func:`same_text`): "DC and RF" and "DC and RF magnetron
        co-sputtering" are one category, "rfmagnetron" is "rf-magnetron" with its hyphen lost."""
        if not (isinstance(a, str) and isinstance(b, str)):
            return False
        return same_text(spec, a, b)

    def within(self, a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
        return self.same(a, b, spec)

    def prefer(self, value: FieldValue, spec: FieldSpec) -> tuple[object, ...]:
        # Of spellings judged the same, one naming the field's category is the cell: "rf-magnetron sputtering" is
        # RF, its twin that lost the hyphen names nothing.
        return (canonical_category(spec.categories, value.value_raw) is None,)

    def note(self, spec: FieldSpec, ctx: KindContext = NO_CONTEXT) -> str:
        if spec.cardinality != "many":
            return ""
        # One entry per value: a quote naming two categories ("XRD and XPS") names none, and is refused in the cell.
        named = f" Name each with one of: {', '.join(spec.prompt_categories)}." if spec.prompt_categories else ""
        return f" {LIST_NOTE}{named}"


def _unparsed(a: FieldValue, b: FieldValue) -> tuple[FactStatus, str]:
    return "ambiguous", f"unparsed: {a.normalization_note or a.value_raw!r} vs {b.normalization_note or b.value_raw!r}"


# ---- boolean -----------------------------------------------------------------------------------------------


class BooleanRules:
    """A yes/no the paper states in words. The model quotes the words and says with ``holds`` whether they affirm
    the field; the code never reads a negation itself. Cleaning drops an answer without ``holds``."""

    def read(
        self, field: FieldValue, spec: FieldSpec, units: UnitRegistry, ctx: KindContext = NO_CONTEXT
    ) -> FieldValue:
        return field

    def compare(
        self, a: FieldValue, b: FieldValue, spec: FieldSpec, ctx: KindContext = NO_CONTEXT
    ) -> tuple[FactStatus, str]:
        if a.holds is None or b.holds is None:
            return "ambiguous", f"no yes/no: {a.holds} vs {b.holds}"
        if a.holds == b.holds:
            return "agree", f"both {'affirm' if a.holds else 'deny'} it"
        return "conflict", f"{a.value_raw!r} ({a.holds}) vs {b.value_raw!r} ({b.holds})"

    def distance(self, a: FieldValue, b: FieldValue) -> float | None:
        # Equal answers pair first, so two lanes listing the same two answers in another order still agree.
        return None if a.holds is None or b.holds is None else float(a.holds != b.holds)

    def cell(
        self, value: FieldValue, spec: FieldSpec, units: UnitRegistry, ctx: KindContext = NO_CONTEXT
    ) -> tuple[CellValue, str | None]:
        if value.holds is None:
            return None, "原文未给出是/否"
        return value.holds, None

    def same(self, a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
        return isinstance(a, bool) and isinstance(b, bool) and a == b

    def within(self, a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
        return self.same(a, b, spec)

    def prefer(self, value: FieldValue, spec: FieldSpec) -> tuple[object, ...]:
        return ()

    def note(self, spec: FieldSpec, ctx: KindContext = NO_CONTEXT) -> str:
        return (
            " A yes/no field: quote in value_raw the words that state it, and set holds to true when they affirm it,"
            " false when they deny it. Report nothing when the paper does not say."
        )


# ---- date --------------------------------------------------------------------------------------------------


class DateRules:
    """A calendar date as ISO at the precision the paper wrote (:func:`paperfacts.normalize.read_date`)."""

    def read(
        self, field: FieldValue, spec: FieldSpec, units: UnitRegistry, ctx: KindContext = NO_CONTEXT
    ) -> FieldValue:
        if field.bound:
            # "2021" quoted out of "up to 2021" is no exact date.
            note = f"bound {field.bound!r} stands before the date in its cited block; not one date"
            return field.model_copy(update={"iso_date": None, "normalization_note": note})
        iso, note = read_date(field.value_raw)
        return field.model_copy(update={"iso_date": iso, "normalization_note": note})

    def compare(
        self, a: FieldValue, b: FieldValue, spec: FieldSpec, ctx: KindContext = NO_CONTEXT
    ) -> tuple[FactStatus, str]:
        if a.iso_date is None or b.iso_date is None:
            return _unparsed(a, b)
        if a.iso_date == b.iso_date:
            return "agree", f"both {a.iso_date}"
        shorter, longer = sorted((a.iso_date, b.iso_date), key=len)
        if longer.startswith(f"{shorter}-"):
            # "2021-03" against "2021-03-12": the same date at two precisions, or two dates in one month.
            return "ambiguous", f"{a.iso_date} vs {b.iso_date}: one is stated more precisely"
        return "conflict", f"{a.iso_date} vs {b.iso_date}"

    def distance(self, a: FieldValue, b: FieldValue) -> float | None:
        # Equal dates pair first, so two lanes listing the same dates in another order still agree.
        return None if a.iso_date is None or b.iso_date is None else float(a.iso_date != b.iso_date)

    def cell(
        self, value: FieldValue, spec: FieldSpec, units: UnitRegistry, ctx: KindContext = NO_CONTEXT
    ) -> tuple[CellValue, str | None]:
        if value.bound:
            return None, f"原文在所引日期前写有界限 {value.bound!r}，不是确切日期"
        iso, note = read_date(value.value_raw)
        return iso, None if iso is not None else note

    def same(self, a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
        return isinstance(a, str) and a == b

    def within(self, a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
        return self.same(a, b, spec)

    def prefer(self, value: FieldValue, spec: FieldSpec) -> tuple[object, ...]:
        return ()

    def note(self, spec: FieldSpec, ctx: KindContext = NO_CONTEXT) -> str:
        return " Quote the date exactly as written."


# ---- interval ----------------------------------------------------------------------------------------------


def _ends_close(a: Sequence[float | None], b: Sequence[float | None], rel_tol: float, abs_tol: float) -> bool:
    """Each end within tolerance of the other's, an open end equal only to an open end."""
    return all(
        (x is None and y is None)
        or (x is not None and y is not None and math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol))
        for x, y in zip(a, b, strict=True)
    )


class IntervalRules:
    """A range with two printed ends ("2.8-4.3 V"), or a one-sided bound (">80 %"), both ends in the canonical
    unit. The quote is read through :func:`read_value`, so a bound grounding found before it counts: "80" quoted
    out of "above 80 %" is (80, None). A bare number is no interval and is refused."""

    def read(
        self, field: FieldValue, spec: FieldSpec, units: UnitRegistry, ctx: KindContext = NO_CONTEXT
    ) -> FieldValue:
        def refused(note: str | None) -> FieldValue:
            return field.model_copy(update={"bounds": None, "unit": None, "normalization_note": note})

        reading = read_value(field, spec, units)
        interval, why = read_interval(reading.text, _is_field_unit(field, spec, units))
        if interval is None:
            return refused(why)
        low, high, written = interval
        # The model is asked for the unit in unit_raw; an interval that writes it only in the quote still has one.
        unit_raw, _ = _unit_to_convert(field, spec, units, written)
        # A range decides a bare number's unit for both ends at once; a bound has one end.
        range_ends = (low, high) if low is not None and high is not None else None
        ends: list[float | None] = []
        canonical = None
        for end in (low, high):
            if end is None:
                ends.append(None)
                continue
            converted, canonical, note = convert_to_canonical(
                spec, end, unit_raw, units, value_text=reading.text, range_ends=range_ends
            )
            if converted is None:
                return refused(note)
            ends.append(converted)
        bounds: tuple[float | None, float | None] = (ends[0], ends[1])
        if range_ends is not None:
            note = f"range {range_ends[0]:g}-{range_ends[1]:g}"
        else:
            note = f"{'lower' if high is None else 'upper'} bound {low if high is None else high:g}"
        if reading.bound:
            note = f"bound {reading.bound!r} stands before the quote in its cited block; {note}"
        return field.model_copy(update={"bounds": bounds, "unit": canonical, "normalization_note": note})

    def compare(
        self, a: FieldValue, b: FieldValue, spec: FieldSpec, ctx: KindContext = NO_CONTEXT
    ) -> tuple[FactStatus, str]:
        if a.bounds is None or b.bounds is None:
            return _unparsed(a, b)
        if a.unit != b.unit:
            return "ambiguous", f"units differ after normalization: {a.unit} vs {b.unit}"
        shown = f"{interval_text(a.bounds)} vs {interval_text(b.bounds)} {a.unit or ''}".rstrip()
        if _ends_close(a.bounds, b.bounds, spec.rel_tol, spec.abs_tol):
            return "agree", f"{shown} (rel_tol={spec.rel_tol:g}, abs_tol={spec.abs_tol:g})"
        return "conflict", shown

    def distance(self, a: FieldValue, b: FieldValue) -> float | None:
        if a.bounds is None or b.bounds is None:
            return None
        if [end is None for end in a.bounds] != [end is None for end in b.bounds]:
            return None
        return sum(abs(x - y) for x, y in zip(a.bounds, b.bounds, strict=True) if x is not None and y is not None)

    def cell(
        self, value: FieldValue, spec: FieldSpec, units: UnitRegistry, ctx: KindContext = NO_CONTEXT
    ) -> tuple[CellValue, str | None]:
        # The comparison's reading, so the cell and the report agree on what the quote says.
        read = self.read(value, spec, units)
        if read.bounds is None:
            return None, read.normalization_note or "不是区间或单侧界限"
        return list(read.bounds), None

    def same(self, a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
        return _is_interval(a) and _is_interval(b) and _ends_close(a, b, 1e-12, 0.0)  # type: ignore[arg-type]

    def within(self, a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
        return _is_interval(a) and _is_interval(b) and _ends_close(a, b, spec.rel_tol, spec.abs_tol)  # type: ignore[arg-type]

    def prefer(self, value: FieldValue, spec: FieldSpec) -> tuple[object, ...]:
        return ()

    def note(self, spec: FieldSpec, ctx: KindContext = NO_CONTEXT) -> str:
        return (
            ' Quote the whole range as written, both ends and the unit (e.g. "2.8-4.3 V"), or a one-sided bound'
            ' (">80 %").'
        )


def _is_interval(value: CellValue) -> bool:
    return isinstance(value, list) and len(value) == 2


def interval_text(bounds: Sequence[float | None]) -> str:
    """An interval as one text: "2.8–4.3", "≥ 80", "≤ 5". The one formatter of an interval, for the comparison's
    note and the workbook cell (:func:`paperfacts.workbook.format_cell`); the web's ``tsv.js`` ``intervalText``
    writes the same."""
    low, high = bounds
    if high is None:
        return f"≥ {plain_number(low)}"
    if low is None:
        return f"≤ {plain_number(high)}"
    return f"{plain_number(low)}–{plain_number(high)}"


def plain_number(number: float | None) -> str:
    """``number`` as JavaScript's ``String(number)`` writes it: the shortest text that reads back as the same float,
    positional from 1e-6 up to 1e21, never cut to a number of significant digits ("0.000123456789", "80", "1e-8")."""
    if number is None:
        return ""
    if number.is_integer() and abs(number) < 1e21:
        return str(int(number))
    text = repr(number)
    if "e" in text and 1e-6 <= abs(number) < 1e21:
        return format(Decimal(text), "f")
    return re.sub(r"e([+-])0*(\d)", r"e\1\2", text)


# ---- reference ---------------------------------------------------------------------------------------------


class ReferenceRules:
    """The id of a sample of another entity type (``FieldSpec.references``): the catalyst a reaction test ran on.

    The model copies the id from that entity's sample list, which its question shows; grounding resolves it among
    the lane's samples (:func:`paperfacts.grounding.ground_lane`) and reading records which one (``ref_id``). Two
    lanes agree when that entity's matching pairs the samples they name, and the cell is the id of the dataset row
    those samples became, so it names a row of the referenced entity's sheet."""

    def read(
        self, field: FieldValue, spec: FieldSpec, units: UnitRegistry, ctx: KindContext = NO_CONTEXT
    ) -> FieldValue:
        ref_id = resolve_reference(ctx.samples.get(spec.references or "", {}), field.value_raw)
        note = None if ref_id is not None else f"names no listed {spec.references} of this lane"
        return field.model_copy(update={"ref_id": ref_id, "normalization_note": note})

    def compare(
        self, a: FieldValue, b: FieldValue, spec: FieldSpec, ctx: KindContext = NO_CONTEXT
    ) -> tuple[FactStatus, str]:
        if a.ref_id is None or b.ref_id is None:
            return _unparsed(a, b)
        pairs = ctx.pairs.get(spec.references or "", frozenset())
        if (a.ref_id, b.ref_id) in pairs:
            return "agree", f"both name {spec.references} {a.ref_id} | {b.ref_id}, which its matching pairs"
        if any(pair_a == a.ref_id for pair_a, _ in pairs) and any(pair_b == b.ref_id for _, pair_b in pairs):
            return "conflict", f"{a.ref_id!r} and {b.ref_id!r} are two {spec.references} samples"
        # One of them was paired with nothing: whether it is the other's sample is not known.
        return "ambiguous", f"{a.ref_id!r} vs {b.ref_id!r}: the {spec.references} matching does not pair them"

    def distance(self, a: FieldValue, b: FieldValue) -> float | None:
        return None

    def cell(
        self, value: FieldValue, spec: FieldSpec, units: UnitRegistry, ctx: KindContext = NO_CONTEXT
    ) -> tuple[CellValue, str | None]:
        if value.ref_id is None:
            return None, f"引用的样品 {value.value_raw!r} 不在本通道的{spec.references}列表中"
        row_id = ctx.row_ids.get((ctx.backend or "", spec.references or "", value.ref_id))
        if row_id is None:
            return None, f"引用的{spec.references}样品 {value.ref_id!r} 没有数据行"
        return row_id, None

    def same(self, a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
        return isinstance(a, str) and a == b

    def within(self, a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
        return self.same(a, b, spec)

    def prefer(self, value: FieldValue, spec: FieldSpec) -> tuple[object, ...]:
        return ()

    def note(self, spec: FieldSpec, ctx: KindContext = NO_CONTEXT) -> str:
        referenced = ctx.entities[spec.references or ""].prompt
        return (
            f" Copy the id of the referenced {referenced.sample_singular} exactly from the list"
            f' "{referenced.sample_list_heading}" below.'
        )


RULES: Mapping[FieldKind, KindRules] = MappingProxyType(
    {
        "numeric": NumericRules(),
        "composition": TextRules(),
        "text": TextRules(),
        "boolean": BooleanRules(),
        "date": DateRules(),
        "interval": IntervalRules(),
        "reference": ReferenceRules(),
    }
)


def rules_for(spec: FieldSpec) -> KindRules:
    return RULES[spec.kind]


_WHITESPACE = re.compile(r"\s+")


def element_key(spec: FieldSpec, raw: str) -> str | None:
    """What identifies one element of a list field (``cardinality: many``), for the comparison's set pairing and
    the union cell alike; None when a field with categories gets a value naming none of them.

    The loose half of :func:`same_text` without its Greek deletion: spacing and a hyphen or period not before a
    digit are dropped, so OCR's "Ni(NO3)2 · 6H2O", "nickel (II) nitrate", "coprecipitation" and "Nickel nitrate."
    are the one element their tidy spellings are, while "10-20" and "1.5" keep their punctuation. Greek letters
    stay, because in a union two elements judged one lose one of them: "α-Al2O3" and "γ-Al2O3" are two. A
    composition keeps its case ("Co3O4" is not "CO3O4"); text folds it ("Ethanol" is "ethanol")."""
    if spec.categories:
        return canonical_category(spec.categories, raw)
    text = LOOSE_PUNCTUATION.sub("", _WHITESPACE.sub("", normalize_text(raw)))
    return text if spec.kind == "composition" else text.casefold()
