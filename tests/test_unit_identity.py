"""The built-in units read every recorded spelling exactly as they did before they moved into ``units.py``.

``tests/fixtures/units/identity.json`` was recorded by ``generate.py`` there, on the commit before the move: the
factor each built-in converter gives, what ``convert_to_canonical`` makes of each spelling (scale factors, gas
names and bare numbers included), and which retrieval pattern finds it in running text. A converter or a pattern
that reads any of them differently fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperfacts.fields import FieldSpec
from paperfacts.normalize import convert_to_canonical
from paperfacts.units import BUILTIN_CONVERTERS, BUILTIN_RETRIEVAL
from support.profiles import shipped_profile

RECORDED = json.loads((Path(__file__).parent / "fixtures" / "units" / "identity.json").read_text(encoding="utf-8"))


def test_the_recording_covers_every_built_in_unit():
    assert RECORDED["factors"].keys() == BUILTIN_CONVERTERS.keys()
    assert RECORDED["retrieval"].keys() == BUILTIN_RETRIEVAL.keys()
    assert sum(len(table) for table in RECORDED["factors"].values()) >= 200


@pytest.mark.parametrize("canonical", list(BUILTIN_CONVERTERS))
def test_a_built_in_converter_gives_the_recorded_factors(canonical):
    convert = BUILTIN_CONVERTERS[canonical]

    assert {unit: convert(unit) for unit in RECORDED["spellings"]} == RECORDED["factors"][canonical]


@pytest.mark.parametrize("canonical", list(BUILTIN_CONVERTERS))
def test_conversion_to_a_built_in_unit_is_the_recorded_one(canonical):
    spec = FieldSpec(
        name="probe", group="film", kind="numeric", description="probe", keywords=(), canonical_unit=canonical
    )

    # The TCO profile's units: the built-ins and the gas names the recording set aside.
    units = shipped_profile().units

    converted = {unit: list(convert_to_canonical(spec, 2.0, unit, units)) for unit in RECORDED["spellings"]}

    assert converted == RECORDED["conversions"][canonical]


@pytest.mark.parametrize("canonical", list(BUILTIN_RETRIEVAL))
def test_a_built_in_retrieval_pattern_finds_what_it_found(canonical):
    pattern = BUILTIN_RETRIEVAL[canonical]
    recorded = RECORDED["retrieval"][canonical]

    assert {text: pattern.search(text) is not None for text in recorded} == recorded
