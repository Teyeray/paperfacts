"""``validation_key``: what a stored validation depends on, and what it deliberately does not."""

from __future__ import annotations

import dataclasses

from paperfacts.config import DEFAULT_VLM_MODEL, Settings
from paperfacts.keys import (
    FINGERPRINT_LENGTH,
    comparison_key,
    extractor_key_for,
    validation_code_fingerprint,
    validation_key,
    validation_key_for,
)


def test_the_key_is_a_short_hex_fingerprint():
    key = validation_key("qwen3-vl")
    assert len(key) == FINGERPRINT_LENGTH
    assert all(c in "0123456789abcdef" for c in key)


def test_the_model_is_in_the_key():
    assert validation_key("qwen3-vl-8b-instruct") != validation_key("qwen3-vl-32b-instruct")


def test_settings_at_their_baseline_give_the_same_key_as_the_bare_model():
    # An unedited config.json must keep the filenames it has: every knob at its built-in value is left out.
    assert validation_key_for(Settings()) == validation_key(DEFAULT_VLM_MODEL)


def test_every_knob_that_changes_what_the_model_sees_or_how_it_is_judged_changes_the_key():
    baseline = validation_key_for(Settings())
    for change in (
        {"vlm_temperature": 0.3},
        {"vlm_max_tokens": 1024},
        {"vlm_crop_dpi": 150},
        {"vlm_crop_padding": 0.03},
        {"vlm_crop_max_pixels": 500_000},
        {"vlm_policy": "all"},
    ):
        assert validation_key_for(dataclasses.replace(Settings(), **change)) != baseline, change


def test_scheduling_and_endpoint_settings_are_not_in_the_key():
    # Where the model runs and how many requests are in flight change no answer.
    baseline = validation_key_for(Settings())
    moved = dataclasses.replace(
        Settings(), vlm_base_url="http://gpu7:8090/v1", vlm_concurrency=1, vlm_timeout_s=9.0, vlm_api_key="k"
    )
    assert validation_key_for(moved) == baseline


def test_the_validation_key_is_independent_of_the_other_two_keys():
    # A VLM setting must never rename an extraction or a comparison: those files did not change.
    baseline = (extractor_key_for(Settings()), comparison_key())
    changed = dataclasses.replace(Settings(), vlm_model="other", vlm_crop_dpi=300, vlm_policy="all")
    assert (extractor_key_for(changed), comparison_key()) == baseline


def test_the_code_fingerprint_covers_the_matcher_the_stage_judges_with():
    # Not a behavioural test, a reminder: grounding.py and normalize.py decide the verdict, so a more
    # lenient fold there is a different verdict and must rename the stored report.
    assert len(validation_code_fingerprint()) == FINGERPRINT_LENGTH
