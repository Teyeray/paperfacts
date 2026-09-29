"""``validation_key`` is its own key: it must re-ask the vision model and rename nothing else.

The rules under test are the ones CLAUDE.md states: the three keys stay apart, a setting at its built-in
baseline is left out of the material, and everything that changes what the model is shown or how its reading
is judged is in.
"""

from __future__ import annotations

from paperfacts.config import Settings
from paperfacts.keys import (
    comparison_key_for,
    extractor_key_for,
    figure_key_for,
    validation_key,
    validation_key_for,
)
from support.profiles import make_profile, shipped_profile

MODEL = "some-vision-model"


def _key(profile=None, **changes: object) -> str:
    return validation_key(profile or shipped_profile(), MODEL, **changes)  # type: ignore[arg-type]


# ---- The baseline rule ------------------------------------------------------------------------------------


def test_settings_at_their_baseline_are_left_out() -> None:
    """An unedited checkout keeps the filenames it has, so passing the baselines explicitly changes nothing."""
    assert _key() == _key(
        temperature=0.0,
        max_tokens=4096,
        crop_dpi=200,
        crop_padding=0.01,
        crop_max_pixels=2_000_000,
        policy="tables",
        context_blocks=1,
        fill_blanks=True,
    )


def test_the_key_is_stable_across_calls() -> None:
    assert _key() == _key()


# ---- What must change it ---------------------------------------------------------------------------------


def test_the_model_changes_it() -> None:
    assert _key() != validation_key(shipped_profile(), "another-vision-model")


def test_the_policy_changes_it() -> None:
    """Which values were checked is part of what the report says."""
    assert len({_key(policy="tables"), _key(policy="disputed"), _key(policy="all")}) == 3


def test_the_context_window_changes_it() -> None:
    """A wider window is a different picture."""
    assert _key(context_blocks=1) != _key(context_blocks=2)
    assert _key(context_blocks=1) != _key(context_blocks=0)


def test_the_crop_geometry_changes_it() -> None:
    base = _key()
    assert base != _key(crop_dpi=300)
    assert base != _key(crop_padding=0.05)
    assert base != _key(crop_max_pixels=1_000_000)


def test_the_sampling_settings_change_it() -> None:
    base = _key()
    assert base != _key(temperature=0.2)
    assert base != _key(max_tokens=8192)


def test_the_fill_switch_changes_it() -> None:
    """A report with fills says more than one without."""
    assert _key(fill_blanks=True) != _key(fill_blanks=False)


def test_the_profile_changes_it() -> None:
    """Which values are put in front of the model is decided by field, so another field table is another key."""
    other = make_profile({"prompt.domain_subject": "electrolyte formulations"})
    assert _key() != _key(other)


def test_a_field_edit_changes_it() -> None:
    """A field's description is what the region's question names, so editing one re-asks the model."""
    plain = make_profile()
    edited = make_profile({"fields.1.description": "the coating's measured thickness, in nanometres"})
    assert _key(plain) != _key(edited)


# ---- The keys stay apart ---------------------------------------------------------------------------------


def test_validation_settings_rename_nothing_else() -> None:
    """The whole point of a third key: a prompt tweak re-asks the VLM and leaves extractions and comparisons
    where they are. Their keys read no ``vlm_*`` setting, so changing one cannot move their files."""
    profile = shipped_profile()
    base = Settings()
    tuned = Settings(
        vlm_enabled=True,
        vlm_model="a-different-model",
        vlm_policy="all",
        vlm_context_blocks=3,
        vlm_crop_dpi=300,
        vlm_fill_blanks=False,
    )

    assert extractor_key_for(base, profile) == extractor_key_for(tuned, profile)
    assert comparison_key_for(base, profile) == comparison_key_for(tuned, profile)
    assert figure_key_for(base, profile) == figure_key_for(tuned, profile)
    assert validation_key_for(base, profile) != validation_key_for(tuned, profile)


def test_the_key_a_run_writes_is_the_key_a_reader_computes() -> None:
    settings = Settings(vlm_model=MODEL, vlm_policy="all", vlm_context_blocks=2)
    assert validation_key_for(settings, shipped_profile()) == _key(policy="all", context_blocks=2)


def test_the_three_keys_are_different_values() -> None:
    profile, settings = shipped_profile(), Settings()
    keys = {
        extractor_key_for(settings, profile),
        comparison_key_for(settings, profile),
        validation_key_for(settings, profile),
    }
    assert len(keys) == 3
