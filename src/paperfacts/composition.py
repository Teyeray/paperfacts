"""A composition quote read into components, converted between weight and atomic percent, and written out in
one standard sentence: what the page shows for a composition column whose field declares ``atomic_basis``.

A composition is stored and compared as the paper's own words ("A:B = 90:10 wt%", "A/B 95:5 wt.%", "2 wt% B-doped
A"), so two papers that mean the same mix read differently. This module reads a quote into components with amounts
and a unit, converts them, and writes "90 wt% A and 10 wt% B" in each unit the page can show. It is display code,
unhashed like :mod:`paperfacts.readings` and :mod:`paperfacts.columns`: the stored cell, the comparison and the
workbook keep the quote, and the server hands the page both readings beside the column (:func:`with_readings`).
A quote this cannot read has no reading, and the page shows it as written.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from paperfacts.columns import CompositionReading, FieldColumn
from paperfacts.fields import AtomicBasis

# Conventional atomic weights (IUPAC 2021), g/mol, for every element a formula may name.
ATOMIC_WEIGHTS: Mapping[str, float] = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "Ne": 20.18, "Na": 22.99, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06,
    "Cl": 35.45, "Ar": 39.95, "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.63,
    "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 97.0, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41,
    "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.6, "I": 126.9, "Xe": 131.29, "Cs": 132.91, "Ba": 137.33,
    "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24, "Pm": 145.0, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25,
    "Tb": 158.93, "Dy": 162.5, "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05, "Lu": 174.97, "Hf": 178.49,
    "Ta": 180.95, "W": 183.84, "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209.0, "At": 210.0, "Rn": 222.0, "Fr": 223.0, "Ra": 226.0,
    "Ac": 227.0, "Th": 232.04, "Pa": 231.04, "U": 238.03,
}  # fmt: skip

# The units the page can show, in the order its switch cycles them.
SHOWN_UNITS: tuple[str, ...] = ("wt%", "at%")

# The units a composition may be stated in, folded from the spellings papers use ("wt.%", "wt %", "atom%").
_UNIT_SPELLINGS = (
    (re.compile(r"^(?:wt|weight)\.?%$", re.IGNORECASE), "wt%"),
    (re.compile(r"^(?:at|atom|atomic)\.?%$", re.IGNORECASE), "at%"),
    (re.compile(r"^(?:mol|mole|molar)\.?%$", re.IGNORECASE), "mol%"),
)

Atoms = dict[str, float]

# ---- formulas ---------------------------------------------------------------------------------------------

_FORMULA_TOKEN = re.compile(r"[A-Z][a-z]?|\d+(?:\.\d+)?|[()\[\]]")
_HYDRATE_DOT = re.compile(r"[·•⋅]\s*")
_LEADING_COUNT = re.compile(r"^\d+(?:\.\d+)?")


def atoms_of(formula: str) -> Atoms | None:
    """The atoms a formula names, symbol to count, or None when it is not a formula ("ITO" is I and T, and there
    is no element T). Brackets with a multiplier are read, and a hydrate's parts, written with a middle dot and an
    optional coefficient ("CuSO4·5H2O"); a decimal subscript ("In1.9Sn0.1O3") is a count like any other. A
    coefficient before the first part ("2In2O3") is not a formula."""
    atoms: Atoms = {}
    first, *hydrate = _HYDRATE_DOT.split(str(formula))
    if not _part_atoms(first, atoms, 1.0):
        return None
    for part in hydrate:
        lead = _LEADING_COUNT.match(part)
        if not _part_atoms(part[lead.end() :] if lead else part, atoms, float(lead.group()) if lead else 1.0):
            return None
    return atoms or None


def _part_atoms(text: str, into: Atoms, times: float) -> bool:
    """One hydrate part, ``times`` over, added into ``into``; False when it is not a formula. A count applies to
    what came just before it (an element or a bracketed group), so that is held back until the next token says
    there is none."""
    stack: list[Atoms] = [{}]
    held: tuple[str, Atoms | None] | None = None  # (symbol, None) or ("", group)

    def flush(count: float) -> None:
        nonlocal held
        if held is None:
            return
        symbol, group = held
        if group is None:
            _add(stack[-1], symbol, count)
        else:
            _merge(stack[-1], group, count)
        held = None

    consumed = 0
    for match in _FORMULA_TOKEN.finditer(text):
        if match.start() != consumed:
            return False
        consumed = match.end()
        token = match.group()
        if token in "([":
            flush(1.0)
            stack.append({})
        elif token in ")]":
            flush(1.0)
            if len(stack) < 2:
                return False
            held = ("", stack.pop())
        elif token[0].isdigit():
            if held is None:
                return False
            flush(float(token))
        else:
            flush(1.0)
            if token not in ATOMIC_WEIGHTS:
                return False
            held = (token, None)
    flush(1.0)
    if consumed != len(text) or len(stack) != 1:
        return False
    _merge(into, stack[0], times)
    return True


def _add(into: Atoms, symbol: str, count: float) -> None:
    into[symbol] = into.get(symbol, 0.0) + count


def _merge(into: Atoms, source: Atoms, times: float) -> None:
    for symbol, count in source.items():
        _add(into, symbol, count * times)


def molar_mass(atoms: Mapping[str, float]) -> float:
    return sum(ATOMIC_WEIGHTS[symbol] * count for symbol, count in atoms.items())


def _counted_atoms(atoms: Mapping[str, float], basis: AtomicBasis | None) -> list[tuple[str, float]] | None:
    """The atoms an atomic percent counts under ``basis``; None when the formula has none. ``cations``: every atom
    but oxygen, so an oxide's share is its metal's and a pure element is itself."""
    counted = [(symbol, count) for symbol, count in atoms.items() if symbol != "O"] if basis == "cations" else []
    return counted or None


# ---- reading a quote --------------------------------------------------------------------------------------

# A quote as formulas, numbers, units, separators, punctuation and other words, in order. A formula may hold
# bracketed groups ("Ca(OH)2") but never an unbalanced bracket, so "(90:10 wt%)" stays punctuation.
_TOKEN = re.compile(
    r"(?P<unit>(?i:wt|weight|at|atom|atomic|mol|mole|molar)\.?\s*%)"
    r"|(?P<number>\d+(?:\.\d+)?)"
    r"|(?P<formula>[A-Z][A-Za-z0-9·•⋅]*(?:[(\[][A-Za-z0-9]*[)\]][A-Za-z0-9]*)*)"
    r"|(?P<sep>[:/])"
    r"|(?P<punct>[()\[\],;=])"
    r"|(?P<word>[^\s:/()\[\],;=]+)"
)


@dataclass(frozen=True)
class _Token:
    type: str
    value: str
    number: float = 0.0
    atoms: Atoms | None = None


def _tokenize(quote: str) -> list[_Token] | None:
    text = quote.strip()
    tokens: list[_Token] = []
    consumed = 0
    for match in _TOKEN.finditer(text):
        if text[consumed : match.start()].strip():
            return None
        consumed = match.end()
        kind = match.lastgroup or "word"
        value = match.group()
        if kind == "unit":
            tokens.append(_Token("unit", _fold_unit(value) or ""))
        elif kind == "number":
            tokens.append(_Token("number", value, number=float(value)))
        elif kind == "formula":
            atoms = atoms_of(value)
            tokens.append(_Token("formula", value, atoms=atoms) if atoms else _Token("word", value))
        elif kind in ("sep", "word"):
            tokens.append(_Token(kind, value))
    return None if text[consumed:].strip() else tokens


def _fold_unit(text: str) -> str | None:
    spelled = re.sub(r"\s+", "", text)
    return next((unit for pattern, unit in _UNIT_SPELLINGS if pattern.match(spelled)), None)


def _runs(tokens: Sequence[_Token], kind: str) -> list[tuple[int, int, list[_Token]]]:
    """Runs of one token type joined by separators, as (start, end, tokens): "A:B:C" or "90/10"."""
    found: list[tuple[int, int, list[_Token]]] = []
    index = 0
    while index < len(tokens):
        if tokens[index].type != kind:
            index += 1
            continue
        run = [tokens[index]]
        end = index
        while end + 2 < len(tokens) and tokens[end + 1].type == "sep" and tokens[end + 2].type == kind:
            run.append(tokens[end + 2])
            end += 2
        found.append((index, end, run))
        index = end + 1
    return found


@dataclass(frozen=True)
class Component:
    formula: str
    atoms: Mapping[str, float]
    amount: float
    # The paper's own digits, kept when they are shown as written; None once normalised or inferred.
    stated: str | None


@dataclass(frozen=True)
class Composition:
    unit: str
    components: tuple[Component, ...]


def parse_composition(quote: str) -> Composition | None:
    """The quote as components with amounts in one unit, or None when it does not state one mix in one unit.

    Two shapes are read: a ratio, "A:B (=) 90:10 wt%" (amounts normalised to 100 when they are a ratio such as
    9:1), and terms, "90 wt% A and 10 wt% B" or "2 wt% B-doped A", where one amount-less formula takes the
    balance."""
    tokens = _tokenize(quote)
    if tokens is None:
        return None
    units = {token.value for token in tokens if token.type == "unit"}
    if len(units) != 1 or "" in units:
        return None
    (unit,) = units
    return _ratio_form(tokens, unit) or _term_form(tokens, unit)


def _component(token: _Token, amount: float, stated: str | None) -> Component:
    assert token.atoms is not None
    return Component(formula=token.value, atoms=token.atoms, amount=amount, stated=stated)


def _ratio_form(tokens: Sequence[_Token], unit: str) -> Composition | None:
    formulas = [run for run in _runs(tokens, "formula") if len(run[2]) >= 2]
    numbers = [run for run in _runs(tokens, "number") if len(run[2]) >= 2]
    if len(formulas) != 1 or len(numbers) != 1:
        return None
    (_, names_end, names), (amounts_start, _, amounts) = formulas[0], numbers[0]
    if len(names) != len(amounts) or amounts_start < names_end:
        return None
    # Every number in the quote must be part of the mix: with one more ("A:B 90:10 wt% at 5 Pa") the quote says
    # something this does not read, and is shown as written.
    if sum(token.type == "number" for token in tokens) != len(amounts):
        return None
    total = sum(token.number for token in amounts)
    if total <= 0:
        return None
    scale = 1.0 if abs(total - 100) <= 1 else 100 / total
    return Composition(
        unit=unit,
        components=tuple(
            _component(name, amount.number * scale, amount.value if scale == 1.0 else None)
            for name, amount in zip(names, amounts, strict=True)
        ),
    )


def _term_form(tokens: Sequence[_Token], unit: str) -> Composition | None:
    used: set[int] = set()
    components: list[Component] = []
    for index, token in enumerate(tokens):
        if token.type != "number":
            continue
        # "90 wt% A" or "A 90 wt%" or "A (90 wt%)": the unit sits right after the number, the formula on either side.
        if index + 1 >= len(tokens) or tokens[index + 1].type != "unit":
            return None
        if index + 2 < len(tokens) and tokens[index + 2].type == "formula" and index + 2 not in used:
            formula_index = index + 2
        elif index > 0 and tokens[index - 1].type == "formula" and index - 1 not in used:
            formula_index = index - 1
        else:
            return None
        used.update((index, index + 1, formula_index))
        components.append(_component(tokens[formula_index], token.number, token.value))
    if not components:
        return None
    hosts = [token for index, token in enumerate(tokens) if token.type == "formula" and index not in used]
    total = sum(component.amount for component in components)
    if len(hosts) == 1 and total < 100:
        components.append(_component(hosts[0], 100 - total, None))
    elif hosts:
        return None
    if abs(sum(component.amount for component in components) - 100) > 1:
        return None
    return Composition(unit=unit, components=tuple(components))


# ---- converting and writing out ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Part:
    label: str
    amount: float
    stated: str | None


def convert(composition: Composition, unit: str, basis: AtomicBasis | None) -> tuple[Part, ...] | None:
    """The composition's parts in ``unit`` ("wt%", "at%" or "mol%"). The stated amounts give the relative moles of
    each formula unit (wt%: over the molar mass; mol%: as they are; at%: over the atoms ``basis`` counts), and the
    moles give the amounts in the other unit. An at% is labelled by the atoms it counts whether stated or
    converted ("In" for In2O3), every other unit by the formula. None when a component has no counted atom. A
    component's digits are kept only in the unit the paper used."""
    counted = [_counted_atoms(component.atoms, basis) for component in composition.components]
    if any(atoms is None for atoms in counted):
        return None

    def per_mole(index: int, in_unit: str) -> float:
        if in_unit == "wt%":
            return molar_mass(composition.components[index].atoms)
        if in_unit == "mol%":
            return 1.0
        return sum(count for _, count in counted[index] or ())

    components = composition.components
    moles = [component.amount / per_mole(index, composition.unit) for index, component in enumerate(components)]
    shares = [value * per_mole(index, unit) for index, value in enumerate(moles)]
    total = sum(shares)
    if total <= 0:
        return None
    same = composition.unit == unit
    return tuple(
        Part(
            label="+".join(symbol for symbol, _ in counted[index] or ()) if unit == "at%" else component.formula,
            amount=100 * shares[index] / total,
            stated=component.stated if same else None,
        )
        for index, component in enumerate(composition.components)
    )


def _amount_text(part: Part) -> str:
    return part.stated if part.stated is not None else f"{part.amount:.1f}".rstrip("0").rstrip(".")


def sentence(unit: str, parts: Sequence[Part]) -> str:
    """ "90 wt% A and 10 wt% B"; three or more as "a, b and c"."""
    texts = [f"{_amount_text(part)} {unit} {part.label}" for part in parts]
    if len(texts) <= 2:
        return " and ".join(texts)
    return f"{', '.join(texts[:-1])} and {texts[-1]}"


def readings(quote: str, basis: AtomicBasis | None) -> dict[str, CompositionReading] | None:
    """What the page shows for ``quote`` in each unit it can switch to, or None when the quote cannot be read or
    converted. A reading is ``computed`` whenever a number in it is not the paper's own: a converted unit, an
    inferred balance ("2 wt% B-doped A" says nothing about the 98), a ratio normalised to 100."""
    parsed = parse_composition(quote)
    if parsed is None:
        return None
    shown: dict[str, CompositionReading] = {}
    for unit in SHOWN_UNITS:
        parts = convert(parsed, unit, basis)
        if parts is None:
            return None
        computed = any(part.stated is None for part in parts)
        shown[unit] = CompositionReading(text=sentence(unit, parts), computed=computed)
    return shown


def with_readings(columns: Sequence[FieldColumn], rows: Iterable[Mapping[str, object]]) -> tuple[FieldColumn, ...]:
    """``columns`` with each switchable composition column (a field declaring ``atomic_basis``) carrying the
    readings of every quote ``rows`` hold in it, keyed by the quote; the other columns as they are."""
    switchable = [column for column in columns if column.atomic_basis is not None]
    if not switchable:
        return tuple(columns)
    quotes: dict[str, set[str]] = {column.name: set() for column in switchable}
    for row in rows:
        for name, found in quotes.items():
            value = row.get(name)
            found.update(item for item in (value if isinstance(value, list) else [value]) if isinstance(item, str))
    by_name = {
        column.name: {
            quote: read for quote in sorted(quotes[column.name]) if (read := readings(quote, column.atomic_basis))
        }
        for column in switchable
    }
    return tuple(
        column.model_copy(update={"compositions": by_name[column.name]}) if column.name in by_name else column
        for column in columns
    )
