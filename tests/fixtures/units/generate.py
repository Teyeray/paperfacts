"""Record how the built-in units read a fixed set of spellings, for ``tests/test_unit_identity.py``.

    PYTHONPATH=src uv run python tests/fixtures/units/generate.py

Writes ``identity.json`` next to this script. It was first run on the commit before the built-in converters and
retrieval patterns moved into ``units.py`` (importing ``passages.UNIT_PATTERNS``, as the retrieval table was
called there), so the file is what those tables did before the move; the test pins that the moved tables still
do exactly that. Re-run it only for an intended change to a built-in unit, and review
the JSON diff: that diff is the whole behavioural change.

Three tables, every canonical unit against every spelling:

- ``factors``: the factor a converter returns for a cleaned unit (None when it refuses it);
- ``conversions``: ``normalize.convert_to_canonical`` of the value 2 with that spelling as ``unit_raw`` (scale
  factors, gas suffixes and bare numbers included) in the TCO profile's units, as ``[value, unit, note]``;
- ``retrieval``: whether a unit's retrieval pattern finds the spelling in lower-cased text after a number.
"""

from __future__ import annotations

import json
from pathlib import Path

from paperfacts.config import Settings
from paperfacts.fields import FieldSpec
from paperfacts.normalize import convert_to_canonical
from paperfacts.profile_loader import load_profile, profile_path
from paperfacts.units import BUILTIN_CONVERTERS, BUILTIN_RETRIEVAL

HERE = Path(__file__).parent
# The shipped TCO profile's units, from the built-in settings: the built-in tables and the gas names it sets aside,
# which is what every conversion was read with when this file was recorded.
UNITS = load_profile(profile_path(Settings())).units

# Every key of every built-in table, the spellings the anchored patterns accept, and near misses: wrong case,
# wrong prefix, OCR damage, units of other quantities.
SPELLINGS = (
    # sheet resistance
    *("Ω/sq", "ohm/sq", "Ohm/sq", "ohms/sq", "OHM/SQ", "Ω/□", "Ω□", "Ω sq", "Ω per square", "Ω/square", "Ω/sq."),
    *("Ω.sq-1", "Ω sq^-1", "Ωsq-1", "kΩ/sq", "KΩ/sq", "MΩ/sq", "mΩ/sq", "μΩ/sq", "nΩ/sq", "Ω/L", "Ω", "GΩ/sq"),
    # resistivity
    *("Ω·cm", "Ω.cm", "Ωcm", "Ω cm", "Ω-cm", "ohm-cm", "ohm cm", "Ohm.cm", "Ωxcm", "Ω*cm", "mΩ·cm", "mΩ.cm"),
    *("μΩ·cm", "μΩ cm", "kΩ·cm", "MΩ·cm", "nΩ·cm", "Ω·m", "Ω·CM"),
    # length and distance
    *("nm", "NM", "Nm", "μm", "um", "µm", "mm", "cm", "m", "å", "Å", "angstrom", "Angstrom", "pm", "km", "inch"),
    *("inches", "in", '"', "''", "″", "′′", "in."),
    # time
    *("min", "mins", "minute", "minutes", "Min", "MIN", "h", "hr", "hrs", "hour", "hours", "H", "s", "sec"),
    *("seconds", "second", "S", "ms", "d", "day"),
    # percent
    *("%", "percent", "Percent", "vol.%", "at.%", "wt%", "at%"),
    # temperature
    *("°C", "℃", "C", "c", "°c", "K", "k", "°F", "degC"),
    # flow
    *("sccm", "SCCM", "Sccm", "cm3/min", "cm^3/min", "slm", "ml/min"),
    # rotation
    *("rpm", "RPM", "r/min", "rev/min", "R/min", "rps"),
    # power
    *("W", "w", "kW", "KW", "mW", "MW", "μW", "nW", "GW", "Wh", "W/cm2"),
    # pressure
    *("Pa", "pa", "PA", "mPa", "MPa", "mpa", "hPa", "kPa", "KPa", "mbar", "bar", "Bar", "torr", "Torr", "mtorr"),
    *("mTorr", "mTORR", "atm", "psi"),
    # gas names after a unit
    *("Pa Ar", "Pa (Ar)", "mTorr (O2)", "sccm O2", "sccm(Ar)", "sccm air", "W Ar"),
    # power-of-ten headers
    *("×10^-4 Ω·cm", "(10^-4 Ω cm)", "ρ (×10^-4 Ω cm)", "ρ × 10^4 (Ω cm)", "ρ × 10^-4 Ω·cm", "ρ (×10^-4) (Ω cm)"),
    *("10^2 ohm/sq", "x10^3 Ω/sq", "ρ ×10^-4 (Ω cm)", "ρ 10^4 Ω cm", "ρ × 10^4"),
    # nothing, or noise
    *("", " ", ".", "-", "x", "a.u.", "eV", "V", "A"),
)


def main() -> None:
    factors = {
        canonical: {unit: convert(unit) for unit in SPELLINGS} for canonical, convert in BUILTIN_CONVERTERS.items()
    }
    conversions = {}
    for canonical in BUILTIN_CONVERTERS:
        # The bare-number policy only matters for an empty unit; "reject" is the table's default.
        spec = FieldSpec(
            name="probe", group="film", kind="numeric", description="probe", keywords=(), canonical_unit=canonical
        )
        conversions[canonical] = {unit: list(convert_to_canonical(spec, 2.0, unit, UNITS)) for unit in SPELLINGS}
    texts = [f"of 2 {unit.lower()} here" for unit in SPELLINGS] + [f"2{unit.lower()}" for unit in SPELLINGS]
    retrieval = {
        canonical: {text: pattern.search(text) is not None for text in texts}
        for canonical, pattern in BUILTIN_RETRIEVAL.items()
    }
    payload = {"spellings": list(SPELLINGS), "factors": factors, "conversions": conversions, "retrieval": retrieval}
    (HERE / "identity.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
