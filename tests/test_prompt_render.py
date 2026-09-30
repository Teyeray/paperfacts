"""Rendering the prompt templates from a profile: one pass, and the markers computed from its groups and fields.

The TCO bytes are pinned by the prompt snapshot; these tests are about what a different profile gets, and
about text that must reach the model as written even when it looks like a marker.
"""

from __future__ import annotations

import dataclasses

import pytest

from paperfacts import figures
from paperfacts.profile import COMPUTED_MARKERS, MARKER, FigureSlots, GroupSpec, PromptSlots
from paperfacts.prompts import (
    article_note,
    document_article_note,
    extraction_system_prompt,
    extraction_user_prompt,
    field_system_prompt,
    inventory_system_prompt,
    inventory_user_prompt,
    matching_system_prompt,
    quoted_names,
    render,
)
from support.profiles import make_profile

SYSTEM_PROMPTS = (extraction_system_prompt, inventory_system_prompt, field_system_prompt, matching_system_prompt)
KNOWN_MARKERS = {item.name for item in dataclasses.fields(PromptSlots)} | set(COMPUTED_MARKERS)
SAMPLE_GROUP = {"name": "coating", "level": "sample"}


# ---- One pass ----------------------------------------------------------------------------


def test_an_inserted_value_is_never_scanned_again():
    assert render("{a} and {b}", {"a": "{b}", "b": "x"}) == "{b} and x"


def test_a_marker_with_no_value_and_a_json_brace_are_left_as_they_stand():
    assert render('{"values": []} {unknown}', {"values": "no"}) == '{"values": []} {unknown}'


def test_a_field_description_that_looks_like_a_marker_reaches_the_model_as_written():
    # Descriptions are not checked for markers, so only the single pass keeps them literal.
    profile = make_profile({"fields.1.description": "Coating thickness, not {paper_key} or {fields}."})

    assert "Coating thickness, not {paper_key} or {fields}." in extraction_system_prompt(profile)


def test_a_slot_that_looks_like_a_marker_reaches_the_model_as_written():
    # The loader refuses such a slot; this is the renderer's own guarantee behind that check.
    profile = make_profile()
    slots = dataclasses.replace(profile.prompt, domain_subject="coatings {sample_plural}", sample_plural="{fields}")
    profile = dataclasses.replace(profile, prompt=slots)

    assert "ONE scientific paper about coatings {sample_plural}." in extraction_system_prompt(profile)
    assert "list the {fields} the paper reports" in inventory_system_prompt(profile)
    assert "produced a list of {fields} from each." in matching_system_prompt(profile)


@pytest.mark.parametrize("prompt", SYSTEM_PROMPTS, ids=lambda prompt: prompt.__name__)
def test_every_marker_of_every_template_is_filled(prompt):
    left = set(MARKER.findall(prompt(make_profile()))) & KNOWN_MARKERS

    assert not left


def test_the_chart_slots_are_inserted_verbatim():
    slots = FigureSlots(
        subject="coatings {caption}",
        property_noun="coating properties",
        chart_definition="a property against a condition",
        axis_example='axis "Thickness [nm]"',
    )

    prompt = figures.user_prompt("Fig. 1 {fields}", (), slots)

    assert "a scientific paper about coatings {caption}." in prompt
    assert "<<<Fig. 1 {fields}>>>" in prompt
    assert "Only these coating properties are of interest" in prompt


# ---- The JSON keys and the generic defaults ------------------------------------------------


def test_the_json_keys_come_from_the_slots():
    profile = make_profile({"prompt.paper_key": "precursor", "prompt.no_samples_key": "no_coating"})

    assert '  "precursor": {"source_ids"' in extraction_system_prompt(profile)
    assert 'under "precursor", with `applies_to_all_samples: true`' in extraction_system_prompt(profile)
    assert '  "no_coating": <true|false>' in inventory_system_prompt(profile)
    assert "6. `no_coating` is true ONLY when the excerpts show that" in inventory_system_prompt(profile)


def test_a_profile_with_only_the_required_slots_reads_generically():
    profile = make_profile()

    assert "You extract scientific facts from ONE scientific paper about sol-gel coatings." in (
        extraction_system_prompt(profile)
    )
    assert "   A sample is a coating this paper prepares itself. If the paper reports no such sample," in (
        extraction_system_prompt(profile)
    )
    assert "   Every field describes the coating or its precursor.\n" in field_system_prompt(profile)
    assert "produced a list of samples from each." in matching_system_prompt(profile)


# ---- Group lists ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        pytest.param((), "", id="none"),
        pytest.param(("film",), '"film"', id="one"),
        pytest.param(("process", "film"), '"process" or "film"', id="two"),
        pytest.param(("a", "b", "c"), '"a", "b" or "c"', id="three"),
    ],
)
def test_group_names_are_quoted_and_joined(names, expected):
    assert quoted_names(tuple(GroupSpec(name=name, level="sample") for name in names)) == expected


def test_the_series_rule_names_the_sample_groups():
    two = make_profile(
        {"groups": [{"name": "precursor", "level": "paper"}, SAMPLE_GROUP, {"name": "cell", "level": "sample"}]}
    )

    assert '(group "coating") that the paper states once' in extraction_system_prompt(make_profile())
    assert '(group "coating" or "cell") that the paper states once' in extraction_system_prompt(two)


# ---- The generated paper-level rule (rule 5) ------------------------------------------------


def test_the_generated_rule_with_no_paper_group_says_the_key_carries_only_a_series_value():
    profile = make_profile({"groups": [SAMPLE_GROUP], "fields.0.group": "coating"})

    prompt = extraction_system_prompt(profile)

    assert (
        '\n5. No field is paper-level. Use "paper" only for a whole-series value as rule 10 describes; otherwise'
        " set it to null.\n"
    ) in prompt
    assert "(group )" not in prompt


def test_the_generated_rule_with_one_paper_group():
    assert (
        '\n5. Paper-level fields (group "precursor") belong to the paper, not to a sample: put them under "paper"'
        ' only, never under a sample. Use null for "paper" only if the paper states none of them.\n'
    ) in extraction_system_prompt(make_profile())


def test_the_generated_rule_with_two_paper_groups():
    profile = make_profile(
        {"groups": [{"name": "precursor", "level": "paper"}, {"name": "supplier", "level": "paper"}, SAMPLE_GROUP]}
    )

    assert '\n5. Paper-level fields (group "precursor" or "supplier") belong to the paper' in (
        extraction_system_prompt(profile)
    )


def test_a_profile_that_writes_its_own_rule_gets_it_verbatim():
    profile = make_profile({"prompt.paper_level_rule": 'Precursor fields go under "paper".'})

    assert '\n5. Precursor fields go under "paper".\n' in extraction_system_prompt(profile)


# ---- The condition rule (rule 8) ------------------------------------------------------------


def test_with_no_condition_rule_both_modes_get_the_generic_one():
    rule = "\n8. When a field's line names a condition, always fill `condition` with it.\n"

    assert rule in extraction_system_prompt(make_profile())
    assert rule in field_system_prompt(make_profile())


def test_each_field_with_a_condition_rule_gets_its_sentence():
    profile = make_profile(
        {
            "fields.0.condition_hint": "grade",
            "fields.0.condition_rule": "the purity grade",
            "fields.1.condition_hint": "method",
            "fields.1.condition_rule": "the measuring method",
            "fields.0.missing_condition_note_zh": "未注明纯度等级",
            "fields.1.missing_condition_note_zh": "未注明测量方法",
        }
    )

    rule = (
        "\n8. For `precursor_purity` always fill `condition` with the purity grade. For `coating_thickness` always"
        " fill `condition` with the measuring method.\n"
    )
    assert rule in extraction_system_prompt(profile)
    assert rule in field_system_prompt(profile)


# ---- Named values in the field line --------------------------------------------------------------


def test_a_field_with_named_values_names_its_phrases_but_never_their_numbers():
    profile = make_profile({"fields.1.named_values": {"monolayer": 1, "bulk-like": 1000}})

    prompt = extraction_system_prompt(profile)

    assert (
        ' It may also be stated in words rather than as a number ("monolayer", "bulk-like"): then copy those words'
        " exactly as written into value_raw and leave unit_raw null." in prompt
    )
    assert "1000" not in prompt


def test_a_field_without_named_values_renders_its_line_as_before():
    assert "stated in words" not in extraction_system_prompt(make_profile())


# ---- The article-type note -------------------------------------------------------------------------


def test_the_article_note_names_the_type_the_hint_and_the_no_samples_key():
    profile = make_profile({"prompt.no_samples_key": "no_coating", "prompt.article_type_hint": "Only our coatings."})

    note = article_note(profile, "review")

    assert note == (
        "Note on this paper: its front matter marks it as a review. Only our coatings. If that leaves no sample,"
        ' set "no_coating" to true and "samples" empty.'
    )
    assert inventory_user_prompt("MD", note) == f"{note}\n\n{inventory_user_prompt('MD')}"


def test_document_mode_has_its_own_note_without_the_no_samples_key():
    profile = make_profile({"prompt.no_samples_key": "no_coating"})

    note = document_article_note(profile, "review")

    assert note.endswith(' If that leaves no sample, return "samples" empty.')
    assert PromptSlots.article_type_hint in note and "no_coating" not in note
    assert extraction_user_prompt("MD", note) == f"{note}\n\n{extraction_user_prompt('MD')}"


def test_no_article_type_leaves_both_user_prompts_unchanged():
    profile = make_profile()

    assert article_note(profile, None) == document_article_note(profile, None) == ""
    assert inventory_user_prompt("MD", "") == (
        "Paper excerpts (Markdown with provenance markers):\n\nMD\n\nReturn the JSON object now."
    )
    assert (
        extraction_user_prompt("MD") == "Paper (Markdown with provenance markers):\n\nMD\n\nReturn the JSON object now."
    )
