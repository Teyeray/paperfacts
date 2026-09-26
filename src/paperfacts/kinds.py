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
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from paperfacts.fields import RANGE_ENDS, FieldKind, FieldSpec
from paperfacts.normalize import (
    Reading,
    canonical_category,
    clean_unit,
    convert_to_canonical,
    delatex,
    normalize_key,
    normalize_text,
    parse_number,
    read_range,
    read_value,
    same_text,
)
from paperfacts.records import FieldValue
from paperfacts.units import UnitRegistry

if TYPE_CHECKING:
    from paperfacts.compare import FactStatus

# A list is the cell of a field holding several values at once; no kind produces one yet.
CellValue = str | float | int | bool | list[str | float | None] | None


def joined(values: Sequence[str]) -> str:
    """Distinct non-empty strings, in order, as one cell."""
    return "; ".join(dict.fromkeys(value for value in values if value))


class KindRules(Protocol):
    """The per-kind decisions. Every method is pure."""

    def read(self, field: FieldValue, spec: FieldSpec, units: UnitRegistry) -> FieldValue:
        """``field`` with ``value`` / ``unit`` filled in, in ``units`` (``normalize.normalize_field``)."""
        ...

    def compare(self, a: FieldValue, b: FieldValue, spec: FieldSpec) -> tuple[FactStatus, str]:
        """The verdict on two read values that are both present."""
        ...

    def distance(self, a: FieldValue, b: FieldValue) -> float | None:
        """How far apart two read values under one condition are, for pairing the closest first; None when
        there is no such measure and values under one condition pair in the order the paper gave."""
        ...

    def cell(self, value: FieldValue, spec: FieldSpec, units: UnitRegistry) -> tuple[CellValue, str | None]:
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

    def note(self, spec: FieldSpec) -> str:
        """Text appended to the field's description in its field line; "" when the kind adds nothing."""
        ...


# ---- numeric -----------------------------------------------------------------------------------------------

# The words of a condition tail that may name a unit ("2 h", "550 nm", "°C").
_TAIL_WORD = re.compile(r"[^\d\s,;:()\[\]]+")

_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+\.\d*|\.\d+|\d+)"
_ATOM = rf"(?:{_NUMBER}\s*x\s*10\s*\^?\s*[-+]?\d+|10\s*\^\s*[-+]?\d+|{_NUMBER}(?:[eE][-+]?\d+)?)"
_SCALAR = re.compile(rf"^(?P<center>{_ATOM})(?:\s*(?:±|\+/-|\+-|\\pm)\s*(?P<uncertainty>{_ATOM}))?(?P<tail>.*)$")
# "100 nm (± 5 nm)": the uncertainty in parentheses after the unit, read as "100 ± 5 nm" when both units agree.
_PARENTHESISED_UNCERTAINTY = re.compile(
    rf"^(?P<center>{_ATOM})\s*(?P<unit>[^\d\s(±][^(±]*?)?\s*\(\s*(?:±|\+/-|\+-)\s*(?P<uncertainty>{_ATOM})\s*(?P<again>[^)]*)\)$"
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


def _value_units(value: FieldValue, spec: FieldSpec) -> tuple[str, ...]:
    """The units a quote may carry in its own text: the one the model transcribed and the field's canonical one."""
    return tuple(unit for unit in (value.unit_raw, spec.canonical_unit) if unit)


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
    clean = read_range(reading.text, _value_units(value, spec))
    if clean is None:
        return None, refusal
    low, high, _ = clean
    upper = spec.range_policy == "upper"
    canonical, _, note = convert_to_canonical(
        spec, high if upper else low, value.unit_raw, units, value_text=reading.text, range_ends=(low, high)
    )
    if canonical is None or not math.isfinite(canonical):
        return None, note or "单位无法转换为标准单位"
    chosen = f"原文为区间 {low:g}–{high:g}，按字段配置取{'上限' if upper else '下限'}"
    return canonical, joined([note or "", *notes, chosen])


class NumericRules:
    """A number in the field's canonical unit, compared within the field's tolerances."""

    def read(self, field: FieldValue, spec: FieldSpec, units: UnitRegistry) -> FieldValue:
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
        own_units = _value_units(field, spec)
        number, parse_note = parse_number(reading.text, range_policy=spec.range_policy, range_units=own_units)
        if number is None:
            return field.model_copy(update={"value": None, "unit": None, "normalization_note": parse_note})
        # The range an end was taken from decides a bare number's unit for both ends at once.
        clean = read_range(reading.text, own_units) if spec.range_policy in RANGE_ENDS else None
        value, unit, unit_note = convert_to_canonical(
            spec, number, field.unit_raw, units, value_text=reading.text, range_ends=clean[:2] if clean else None
        )
        note = "; ".join(n for n in (lead_note, parse_note, unit_note) if n) or None
        return field.model_copy(update={"value": value, "unit": unit, "normalization_note": note})

    def compare(self, a: FieldValue, b: FieldValue, spec: FieldSpec) -> tuple[FactStatus, str]:
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

    def cell(self, value: FieldValue, spec: FieldSpec, units: UnitRegistry) -> tuple[CellValue, str | None]:
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
        match = _SCALAR.fullmatch(text)
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
        if tail and clean_unit(tail) not in {clean_unit(unit) for unit in _value_units(value, spec)}:
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
        canonical, _, note = convert_to_canonical(spec, number, value.unit_raw, units, value_text=match.group("center"))
        if canonical is None or not math.isfinite(canonical):
            return None, note or "单位无法转换为标准单位"
        notes.insert(0, note or "")
        if match.group("uncertainty"):
            notes.append(f"原文不确定度 ±{match.group('uncertainty')} {value.unit_raw or ''}；保留中心值")
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

    def note(self, spec: FieldSpec) -> str:
        return ""


# ---- text and composition ----------------------------------------------------------------------------------


class TextRules:
    """Text as quoted, equal as :func:`paperfacts.normalize.same_text` judges it: across spacing, case and a
    lost hyphen, or by the category both name. A composition is compared the same way."""

    def read(self, field: FieldValue, spec: FieldSpec, units: UnitRegistry) -> FieldValue:
        # Compared through same_text on the fly.
        return field

    def compare(self, a: FieldValue, b: FieldValue, spec: FieldSpec) -> tuple[FactStatus, str]:
        if same_text(spec, a.value_raw, b.value_raw):
            category = canonical_category(spec.categories, a.value_raw) or canonical_category(
                spec.categories, b.value_raw
            )
            return "agree", f"both name {category}" if category else "identical after text normalization"
        return "conflict", f"{a.value_raw!r} vs {b.value_raw!r}"

    def distance(self, a: FieldValue, b: FieldValue) -> float | None:
        return None

    def cell(self, value: FieldValue, spec: FieldSpec, units: UnitRegistry) -> tuple[CellValue, str | None]:
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

    def note(self, spec: FieldSpec) -> str:
        return ""


RULES: Mapping[FieldKind, KindRules] = MappingProxyType(
    {"numeric": NumericRules(), "composition": TextRules(), "text": TextRules()}
)


def rules_for(spec: FieldSpec) -> KindRules:
    return RULES[spec.kind]
