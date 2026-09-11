"""The target field table: what to extract, in what unit, and how close counts as the same.

Fields fall into three groups: the sputtering target (paper-level -- a paper usually has one), the
deposition process (sample-level), and film characterisation (sample-level).

One table drives four things: the field descriptions given to the model, unit conversion, what to do with
a bare number that has no unit, and the numeric tolerance used when comparing the two lanes. Its contents
are hashed by :func:`schema_fingerprint` into both cache keys, so changing any cell invalidates exactly
the caches that depended on it.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from functools import cache
from typing import Literal

from paperfacts.fingerprint import content_fingerprint

FieldGroup = Literal["target", "process", "film"]
# numeric: a number with a unit; composition: a chemical formula; text: anything else
FieldKind = Literal["numeric", "composition", "text"]
# What a bare number with no unit means. Declared per field so normalisation never special-cases a name.
BareNumberPolicy = Literal["reject", "assume_canonical", "percent_or_fraction"]

# A human label; the machine-readable version is schema_fingerprint()
SCHEMA_LABEL = "tco-v1"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    group: FieldGroup
    kind: FieldKind
    description: str
    keywords: tuple[str, ...]
    canonical_unit: str | None = None
    # Numeric tolerance: |a-b| <= max(rel_tol * max(|a|,|b|), abs_tol)
    rel_tol: float = 0.0
    abs_tol: float = 0.0
    condition_hint: str | None = None
    bare_number: BareNumberPolicy = "reject"

    @property
    def is_sample_level(self) -> bool:
        return self.group != "target"


FIELD_SPECS: tuple[FieldSpec, ...] = (
    # ---- Sputtering target (paper-level) ----
    FieldSpec(
        name="component",
        group="target",
        kind="composition",
        description=(
            "Chemical composition of the sputtering TARGET (not the film), e.g. 'SnO2:Ta (2 wt% Ta2O5)', "
            "'ITO 90:10 wt%', 'Sn/Ta 95:5 wt%'."
        ),
        keywords=("target", "component", "composition", "wt%", "at%"),
    ),
    FieldSpec(
        name="resistance",
        group="target",
        kind="numeric",
        description="Electrical resistivity of the sputtering target itself (not the film).",
        keywords=("target resistivity", "target resistance"),
        canonical_unit="Ω·cm",
        rel_tol=0.05,
    ),
    FieldSpec(
        name="density",
        group="target",
        kind="numeric",
        description="Relative density of the sputtering target as a percentage of theoretical density.",
        keywords=("relative density", "target density"),
        canonical_unit="%",
        abs_tol=0.5,
        bare_number="percent_or_fraction",
    ),
    FieldSpec(
        name="inch",
        group="target",
        kind="numeric",
        description="Size (diameter or length) of the sputtering target.",
        keywords=("inch", "target size", "diameter"),
        canonical_unit="inch",
        rel_tol=0.01,
        abs_tol=0.05,
    ),
    # ---- Deposition process (sample-level) ----
    FieldSpec(
        name="sputtering_time",
        group="process",
        kind="numeric",
        description="Duration of the sputtering deposition for this sample.",
        keywords=("sputtering time", "deposition time", "duration"),
        canonical_unit="min",
        rel_tol=0.02,
        abs_tol=1.0,
    ),
    # ---- Film characterisation (sample-level) ----
    FieldSpec(
        name="sheet_resistance",
        group="film",
        kind="numeric",
        description="Sheet resistance of the film (Ω/sq).",
        keywords=("sheet resistance", "Ω/sq", "ohm/sq", "Rs"),
        canonical_unit="Ω/sq",
        rel_tol=0.02,
    ),
    FieldSpec(
        name="resistivity",
        group="film",
        kind="numeric",
        description=(
            "Electrical resistivity of the deposited film, as opposed to the `resistance` of the target "
            "it was sputtered from."
        ),
        keywords=("resistivity", "specific resistance", "ρ", "Ω cm"),
        canonical_unit="Ω·cm",
        rel_tol=0.05,
    ),
    FieldSpec(
        name="transmittance",
        group="film",
        kind="numeric",
        description="Optical transmittance of the film (%). Always record the wavelength or range it refers to.",
        keywords=("transmittance", "transparency", "UV-vis", "%T"),
        canonical_unit="%",
        abs_tol=1.0,
        condition_hint="wavelength or spectral range, e.g. '550 nm' or 'average 400-800 nm'",
        bare_number="percent_or_fraction",
    ),
    FieldSpec(
        name="thickness",
        group="film",
        kind="numeric",
        description="Film thickness (e.g. from SEM cross section or profilometry).",
        keywords=("thickness", "SEM cross section", "nm", "µm"),
        canonical_unit="nm",
        rel_tol=0.05,
    ),
)

FIELD_BY_NAME: dict[str, FieldSpec] = {spec.name: spec for spec in FIELD_SPECS}
TARGET_FIELDS: tuple[FieldSpec, ...] = tuple(spec for spec in FIELD_SPECS if spec.group == "target")
SAMPLE_FIELDS: tuple[FieldSpec, ...] = tuple(spec for spec in FIELD_SPECS if spec.is_sample_level)


@cache
def schema_fingerprint() -> str:
    """Hash of the whole table, tolerances and bare-number policies included.

    A hand-maintained version number eventually gets forgotten; a fingerprint does not.
    """
    return content_fingerprint(
        json.dumps([dataclasses.asdict(spec) for spec in FIELD_SPECS], ensure_ascii=False, sort_keys=True)
    )
