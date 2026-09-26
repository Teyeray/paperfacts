"""Entity types (round 2 spec §5, backend part): several named kinds of sample in one profile.

A profile without ``entities`` has one implicit entity, ``sample``, whose slots are ``prompt``'s, so every request,
file and verdict of TCO stays what it was (the snapshot, payload and fingerprint pins hold that). This covers the
loader rules, the per-entity inventory, attribution, fan-out, vote, matching and comparison, the rows, the refusal
of document mode where a profile is loaded to run, and a two-entity document end to end with its offline re-export.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from paperfacts.batch import export_document
from paperfacts.compare import ComparisonReport, compare_lanes
from paperfacts.config import Settings
from paperfacts.dataset import incomplete_reason
from paperfacts.errors import ConfigError
from paperfacts.extract import FieldHarvest, SampleInventory, passage_records
from paperfacts.keys import (
    ComparisonOptions,
    ExtractionOptions,
    comparison_key,
    extractor_key,
    profile_extraction_fingerprint,
)
from paperfacts.matching import SampleMatch, SampleMatching, match_samples
from paperfacts.models import BACKENDS, DocumentGeometry, DocumentInput
from paperfacts.profile import IMPLICIT_ENTITY, DomainProfile
from paperfacts.profile_loader import parse_profile
from paperfacts.prompts import (
    field_system_prompt,
    field_user_prompt,
    inventory_system_prompt,
    matching_system_prompt,
)
from paperfacts.records import (
    ExtractedRecords,
    InventoryResponse,
    InventorySample,
    LaneExtraction,
    ResponseValue,
    SampleRecord,
)
from paperfacts.report import render_report
from paperfacts.storage import DataLayout
from paperfacts.voting import merge_passes
from paperfacts.workflow import check_mode, load_run_profile, run_document
from support.extraction import make_field
from support.factories import DOC_ID, RawOutputFactory, paddle_page_entry
from support.llm import FakeLlmClient
from support.profiles import DELETE, entity_profile_data, make_entity_profile, make_profile

PROFILE = make_entity_profile()
COATING, WEAR = PROFILE.entities
SPECS = PROFILE.by_name


def refused(changes: dict[str, Any]) -> str:
    data = entity_profile_data()
    for dotted, value in changes.items():
        *parents, leaf = dotted.split(".")
        node: Any = data
        for part in parents:
            node = node[int(part)] if isinstance(node, list) else node[part]
        key: Any = int(leaf) if isinstance(node, list) else leaf
        if value is DELETE:
            del node[key]
        else:
            node[key] = value
    with pytest.raises(ConfigError) as caught:
        parse_profile(data, Path("profiles/demo.json"))
    return str(caught.value)


# ---- The profile surface -------------------------------------------------------------------------------------


def test_a_profile_without_entities_has_the_implicit_one_with_its_own_slots():
    profile = make_profile()

    (entity,) = profile.entities
    assert entity.name == IMPLICIT_ENTITY and profile.declared_entities == ()
    assert entity.prompt is profile.prompt and entity.retrieval is profile.retrieval
    assert all(spec.entity is None for spec in profile.fields)
    assert [group.entity for group in profile.groups] == [None, None]


def test_declared_entities_resolve_their_overrides_over_the_profile():
    assert PROFILE.primary is COATING and [entity.name for entity in PROFILE.entities] == ["coating", "wear_test"]
    assert WEAR.label_zh == "磨损测试"
    assert WEAR.prompt.sample_plural == "wear tests" and WEAR.prompt.sample_list_heading == "Wear tests"
    # What an entity does not override is the profile's.
    assert WEAR.prompt.domain_subject == PROFILE.prompt.domain_subject == "sol-gel coatings"
    assert WEAR.overrides == {
        "sample_definition",
        "sample_plural",
        "sample_singular",
        "sample_list_heading",
        "matching_justification_example",
    }
    # Retrieval: the keys it gives replace the profile's, the others are inherited.
    assert WEAR.retrieval.condition_keywords == ("tested", "wear")
    assert WEAR.retrieval.condition_unit_pattern == PROFILE.retrieval.condition_unit_pattern
    assert COATING.retrieval == PROFILE.retrieval


def test_a_field_takes_its_entity_from_its_group_and_a_paper_field_has_none():
    assert {name: spec.entity for name, spec in SPECS.items()} == {
        "precursor_purity": None,
        "coating_thickness": "coating",
        "solvent": "coating",
        "test_temperature": "wear_test",
        "wear_mode": "wear_test",
    }
    assert PROFILE.entity_of(SPECS["precursor_purity"]) is COATING  # asked beside the primary entity
    assert [spec.name for spec in PROFILE.entity_fields(WEAR)] == ["test_temperature", "wear_mode"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param({"entities": []}, "entities must be a list of 1 to 5", id="none"),
        pytest.param(
            {"entities": [{"name": f"e{i}", "prompt": {"sample_definition": "x"}} for i in range(6)]},
            "entities must be a list of 1 to 5",
            id="six",
        ),
        pytest.param({"entities.1.name": "coating"}, "more than one entity named 'coating'", id="duplicate"),
        pytest.param({"entities.1.name": "paper"}, "the name 'paper' is reserved", id="paper"),
        pytest.param({"entities.1.name": "unattributed"}, "the name 'unattributed' is reserved", id="unattributed"),
        pytest.param({"entities.1.name": "Wear"}, "name must match", id="not-an-identifier"),
        pytest.param({"entities.1.prompt.domain_subject": "x"}, "may not override domain_subject", id="slot"),
        pytest.param({"entities.1.prompt.paper_key": "x"}, "may not override paper_key", id="answer-key"),
        pytest.param({"entities.1.prompt.matching_nonsense": "x"}, "unknown key(s) matching_nonsense", id="matching"),
        pytest.param({"entities.1.prompt.sample_plural": ""}, "sample_plural must be a non-empty string", id="empty"),
        pytest.param({"entities.1.prompt.sample_plural": "{fields}"}, "template marker {fields}", id="marker"),
        pytest.param(
            {"entities.1.prompt.sample_definition": DELETE}, "needs its own sample_definition", id="definition"
        ),
        pytest.param({"entities.1.colour": "red"}, "unknown key(s) colour", id="unknown-key"),
        pytest.param({"entities.1.retrieval.condition_unit_pattern": "(a+)+"}, "repeats a group", id="retrieval"),
        pytest.param({"groups.2.entity": DELETE}, "entity must name one of the declared entities", id="unnamed"),
        pytest.param({"groups.2.entity": "reactor"}, "entity must name one of the declared", id="unknown-entity"),
        pytest.param({"groups.0.entity": "coating"}, "a paper-level group belongs to the paper", id="paper-group"),
        pytest.param(
            {"groups.2.entity": "coating"}, "entity 'wear_test' has no sample group", id="entity-without-group"
        ),
        pytest.param({"fields.3.entity": "coating"}, "unknown key(s) entity", id="derived-not-written"),
        pytest.param({"fields.3.name": "entity"}, "the name is reserved", id="reserved-field-name"),
    ],
)
def test_the_loader_refuses_a_malformed_entity_declaration(changes, message):
    assert message in refused(changes)


def test_a_group_naming_an_entity_without_entities_is_refused():
    data = entity_profile_data()
    del data["entities"]

    with pytest.raises(ConfigError, match="the profile declares no entities"):
        parse_profile(data, Path("profiles/demo.json"))


def test_a_single_declared_entity_needs_no_sample_definition_of_its_own():
    data = entity_profile_data()
    data["entities"] = [{"name": "coating"}]
    data["groups"] = data["groups"][:2]
    data["fields"] = data["fields"][:3]

    profile = parse_profile(data, Path("profiles/demo.json"))

    assert [entity.name for entity in profile.entities] == ["coating"] and profile.primary.overrides == frozenset()


# ---- Prompts ---------------------------------------------------------------------------------------------------


def test_the_implicit_entity_renders_every_prompt_as_the_profile_does():
    profile = make_profile()
    implicit = profile.primary
    for render in (inventory_system_prompt, field_system_prompt, matching_system_prompt):
        assert render(profile, implicit) == render(profile)
    assert field_user_prompt(SPECS["solvent"], "- S1", "text", "x").count("Samples this paper reports:") == 1


def test_each_entity_is_asked_in_its_own_words_with_its_own_condition_rules():
    wear_inventory, coating_inventory = inventory_system_prompt(PROFILE, WEAR), inventory_system_prompt(PROFILE)
    assert "list the wear tests the paper reports" in wear_inventory
    assert "A wear test is one set of test conditions applied to one coating." in wear_inventory
    assert "list the coatings the paper reports" in coating_inventory
    # Rule 8 names the fields asked with the prompt: the wear test's rule is not the coatings' business.
    assert "For `test_temperature` always fill `condition`" in field_system_prompt(PROFILE, WEAR)
    assert "test_temperature" not in field_system_prompt(PROFILE, COATING)
    assert "both are the test at 300 °C" in matching_system_prompt(PROFILE, WEAR)
    assert "both are the test at 300 °C" not in matching_system_prompt(PROFILE, COATING)
    question = field_user_prompt(SPECS["wear_mode"], "- T1", "text", "x", WEAR.prompt.sample_list_heading)
    assert "Wear tests this paper reports:\n- T1" in question


# ---- Attribution, fan-out and the vote, per entity -------------------------------------------------------------


def _inventory(entity, *ids: str) -> SampleInventory:
    response = InventoryResponse(samples=[InventorySample(sample_id=sample_id) for sample_id in ids])
    return SampleInventory(response, "", {}, frozenset(), entity)


def _harvest(field: str, *values: dict[str, Any]) -> FieldHarvest:
    return FieldHarvest(
        spec=SPECS[field],
        values=tuple(ResponseValue.model_validate({"source_ids": ["b1"], **value}) for value in values),
        known_ids=frozenset({"b1"}),
    )


def _fields(records: ExtractedRecords) -> dict[tuple[str, str], list[tuple[str, str]]]:
    return {
        (sample.entity, sample.sample_id): [(value.field, value.value_raw) for value in sample.fields]
        for sample in records.samples
    }


def test_a_value_lands_on_the_sample_of_its_own_entity_even_when_another_shares_its_id():
    records = passage_records(
        [_inventory(COATING, "S1", "S2"), _inventory(WEAR, "S1")],
        [
            _harvest("coating_thickness", {"sample_id": "S1", "value_raw": "100"}),
            _harvest("test_temperature", {"sample_id": "S1", "value_raw": "300"}),
            # A wear test named like no wear test stays unplaced, never attached to the coating of that name.
            _harvest("wear_mode", {"sample_id": "S2", "value_raw": "sliding"}),
        ],
        PROFILE,
    )

    assert _fields(records) == {
        ("coating", "S1"): [("coating_thickness", "100")],
        ("coating", "S2"): [],
        ("wear_test", "S1"): [("test_temperature", "300")],
    }
    assert [(value.field, value.value_raw) for value in records.unattributed] == [("wear_mode", "sliding")]


def test_a_series_value_fans_out_within_its_entity_and_a_lone_sample_owns_a_null_id():
    records = passage_records(
        [_inventory(COATING, "S1", "S2"), _inventory(WEAR, "T1")],
        [
            _harvest("solvent", {"value_raw": "ethanol", "applies_to_all_samples": True}),
            # The wear test is the only one of its entity, so a value naming no sample is its.
            _harvest("wear_mode", {"value_raw": "sliding"}),
            # Two coatings: a coating value naming none has no single owner.
            _harvest("coating_thickness", {"value_raw": "100"}),
        ],
        PROFILE,
    )

    assert _fields(records) == {
        ("coating", "S1"): [("solvent", "ethanol")],
        ("coating", "S2"): [("solvent", "ethanol")],
        ("wear_test", "T1"): [("wear_mode", "sliding")],
    }
    assert all(value.series for sample in records.samples[:2] for value in sample.fields)
    assert [value.field for value in records.unattributed] == ["coating_thickness"]


def test_the_vote_keeps_two_entities_samples_of_one_name_apart():
    one_pass = ExtractedRecords(
        paper=None,
        samples=(
            SampleRecord(sample_id="S1", entity="coating", fields=(make_field("solvent", "ethanol"),)),
            SampleRecord(sample_id="S1", entity="wear_test", fields=(make_field("wear_mode", "sliding"),)),
        ),
        invalid_source_ids=(),
        dropped=(),
    )

    merged = merge_passes([one_pass, one_pass, one_pass])

    assert _fields(merged) == {
        ("coating", "S1"): [("solvent", "ethanol")],
        ("wear_test", "S1"): [("wear_mode", "sliding")],
    }


def test_an_entity_is_written_to_the_lane_only_off_the_implicit_one():
    implicit = SampleRecord(sample_id="S1").model_dump(mode="json")
    named = SampleRecord(sample_id="S1", entity="wear_test").model_dump(mode="json")

    assert "entity" not in implicit and named["entity"] == "wear_test"
    assert SampleRecord.model_validate(implicit).entity == IMPLICIT_ENTITY


# ---- Matching, comparison and report ---------------------------------------------------------------------------


def _lane(backend: str, samples: tuple[SampleRecord, ...]) -> LaneExtraction:
    return LaneExtraction(
        document_id=DOC_ID,
        backend=backend,  # type: ignore[arg-type]
        extractor_key="k",
        model="m",
        profile_fingerprint=profile_extraction_fingerprint(PROFILE),
        samples=samples,
    )


LANE_A = _lane(
    "mineru",
    (
        SampleRecord(sample_id="S1", entity="coating", fields=(make_field("solvent", "ethanol"),)),
        SampleRecord(sample_id="S1", entity="wear_test", fields=(make_field("wear_mode", "sliding"),)),
    ),
)
LANE_B = _lane(
    "paddleocr_vl",
    (
        SampleRecord(sample_id="S1", entity="coating", fields=(make_field("solvent", "ethanol"),)),
        SampleRecord(sample_id="run 1", entity="wear_test", fields=(make_field("wear_mode", "rolling"),)),
    ),
)
OPTIONS = ComparisonOptions(profile=PROFILE, ambiguous_match_confidence=0.6)


def test_matching_pairs_only_the_samples_of_its_entity_with_its_own_prompt():
    answer = json.dumps({"pairs": [{"a": "S1", "b": "run 1", "confidence": 0.9, "justification": "same test"}]})
    client = FakeLlmClient([answer])

    coating = match_samples(LANE_A, LANE_B, client, PROFILE)
    wear = match_samples(LANE_A, LANE_B, client, PROFILE, entity=WEAR)

    # The coatings pair by id with no model; the wear tests ask, under the wear tests' own matching prompt.
    assert [(p.a_id, p.b_id, p.method) for p in coating.pairs] == [("S1", "S1", "exact")]
    assert [(p.a_id, p.b_id, p.method) for p in wear.pairs] == [("S1", "run 1", "llm")]
    assert client.systems == [matching_system_prompt(PROFILE, WEAR)]
    assert "run 1" in client.users[0] and "ethanol" not in client.users[0]


def _matchings(wear_failed: bool = False) -> dict[str, SampleMatching]:
    wear = SampleMatching(
        pairs=(SampleMatch(a_id="S1", b_id="run 1", confidence=0.9, justification="j", method="llm"),)
    )
    if wear_failed:
        wear = SampleMatching(unmatched_a=("S1",), unmatched_b=("run 1",), failed=True, failure="bad json")
    coating = SampleMatching(
        pairs=(SampleMatch(a_id="S1", b_id="S1", confidence=1.0, justification="j", method="exact"),)
    )
    return {"coating": coating, "wear_test": wear}


def test_each_entity_is_compared_under_its_own_matching_and_scope():
    report = compare_lanes(LANE_A, LANE_B, _matchings(), OPTIONS)

    assert [(c.scope, c.field, c.status) for c in report.comparisons] == [
        ("coating:S1|S1", "solvent", "agree"),
        ("wear_test:S1|run 1", "wear_mode", "conflict"),
    ]
    assert list(report.matchings) == ["coating", "wear_test"]
    assert report.counts.samples_matched == 2 and report.counts.samples_unmatched == 0
    assert not report.counts.matching_failed
    # Written and read back under both entities.
    assert ComparisonReport.model_validate_json(report.model_dump_json()) == report


def test_a_failed_matching_of_any_entity_marks_the_run_incomplete_and_names_it():
    report = compare_lanes(LANE_A, LANE_B, _matchings(wear_failed=True), OPTIONS)

    assert report.counts.matching_failed and report.counts.samples_unmatched == 2
    assert {c.status for c in report.comparisons if c.scope.startswith("wear_test:")} == {"ambiguous"}
    assert incomplete_reason({"mineru": LANE_A, "paddleocr_vl": LANE_B}, report) == "sample matching failed (wear_test)"


def test_matchings_must_be_those_of_the_profile_entities():
    with pytest.raises(ValueError, match="the profile's entities are coating, wear_test"):
        compare_lanes(LANE_A, LANE_B, {"coating": _matchings()["coating"]}, OPTIONS)
    with pytest.raises(ValueError, match="the profile's entities are coating, wear_test"):
        compare_lanes(LANE_A, LANE_B, SampleMatching(), OPTIONS)


def test_the_report_prints_each_entity_matching_under_its_name():
    lines = list(render_report(compare_lanes(LANE_A, LANE_B, _matchings(), OPTIONS)))

    assert lines[1:5] == [
        "  [coating]",
        "  match S1 ↔ S1  conf=1.00 (exact): j",
        "  [wear_test]",
        "  match S1 ↔ run 1  conf=0.90 (llm): j",
    ]


def test_an_empty_matchings_is_refused_at_load():
    report = compare_lanes(LANE_A, LANE_B, _matchings(), OPTIONS).model_dump(mode="json")

    with pytest.raises(ValidationError, match="matchings is empty"):
        ComparisonReport.model_validate(report | {"matchings": {}})


# ---- Document mode is refused where a profile is loaded to run --------------------------------------------------


@pytest.fixture
def profile_file(tmp_path: Path) -> Path:
    path = tmp_path / "profiles" / "demo.json"
    path.parent.mkdir()
    path.write_text(json.dumps(entity_profile_data()), encoding="utf-8")
    return path


def test_document_mode_is_refused_when_the_profile_is_loaded_to_run(profile_file: Path, tmp_path: Path):
    settings = Settings(repo_root=tmp_path, profile=str(profile_file), extraction_mode="document")

    with pytest.raises(ConfigError, match=r"extraction\.mode"):
        load_run_profile(settings)
    # A reader (prompts, fields, profiles --check) still loads it.
    profile = load_run_profile(settings, to_run=False)
    with pytest.raises(ConfigError, match="only passage mode"):
        check_mode(profile, settings)
    check_mode(profile, dataclasses.replace(settings, extraction_mode="passage"))
    # Past the entry points it is only asserted.
    with pytest.raises(AssertionError, match="document mode"):
        ExtractionOptions.from_settings(settings, profile)


def test_the_cli_refuses_an_extracting_command_and_notes_the_mode_on_check(profile_file: Path, monkeypatch):
    from typer.testing import CliRunner

    from paperfacts.cli import app

    monkeypatch.setenv("PAPERFACTS_EXTRACTION_MODE", "document")
    runner = CliRunner()

    refused_run = runner.invoke(app, ["extract", str(profile_file), "--profile", str(profile_file)])
    assert refused_run.exit_code == 1 and "extraction.mode" in refused_run.output
    printed = runner.invoke(app, ["prompts", "--profile", str(profile_file)])
    assert printed.exit_code == 0, printed.output
    assert "===== inventory system prompt (wear_test) =====" in printed.output
    checked = runner.invoke(app, ["profiles", "--check", str(profile_file)])
    assert checked.exit_code == 0, checked.output
    assert "note: entity types coating, wear_test; runs only in passage mode" in checked.output


# ---- Keys ------------------------------------------------------------------------------------------------------


def test_the_keys_hash_each_entity_prompt_and_move_with_an_entity_edit():
    def keys(profile: DomainProfile) -> tuple[str, str]:
        options = ExtractionOptions(profile=profile, model="m", mode="passage")
        return extractor_key(options), comparison_key(
            ComparisonOptions(profile=profile, ambiguous_match_confidence=0.6)
        )

    base = keys(PROFILE)
    renamed = make_entity_profile({"entities.1.name": "abrasion", "groups.2.entity": "abrasion"})
    reordered = make_entity_profile({"entities": list(reversed(entity_profile_data()["entities"]))})
    relabelled = make_entity_profile({"entities.1.label_zh": "摩擦测试"})

    assert keys(renamed)[0] != base[0] and keys(renamed)[1] != base[1]
    assert keys(reordered)[0] != base[0]
    assert keys(relabelled) == base


# ---- A two-entity document, end to end, and its re-export ------------------------------------------------------

TITLE = "Wear of sol-gel coatings"
PARAGRAPHS = (
    "Two coatings S1 and S2 were deposited from an ethanol solvent; the coating thickness is 100 nm for S1 and "
    "200 nm for S2. The precursor purity was 99.9 %.",
    "Coating S1 was tested at 300 °C against a steel ball in sliding wear.",
)
# Per lane and entity, the samples its inventory lists: the lanes name them differently, so both entities are
# matched by the model, and the MinerU lane names a coating and a wear test alike.
INVENTORY = {
    ("mineru", "coatings"): ("S1", "S2"),
    ("mineru", "wear tests"): ("S1",),
    ("paddleocr_vl", "coatings"): ("coat-1", "coat-2"),
    ("paddleocr_vl", "wear tests"): ("run-1",),
}
# Per field, one (value_raw, unit_raw, condition) per listed sample, in list order.
ANSWERS = {
    "precursor_purity": (("99.9", "%", None),),
    "coating_thickness": (("100", "nm", None), ("200", "nm", None)),
    "solvent": (("ethanol", None, None), ("ethanol", None, None)),
    "test_temperature": (("300", "°C", "steel ball"),),
    "wear_mode": (("sliding", None, None),),
}
_SOURCE = re.compile(r"<!-- source: (\S+) -->")
_LISTED_ID = re.compile(r"^- id: (.+?) \| label:", re.MULTILINE)
_FIELD = re.compile(r"\AField to extract:\n- `([^`]+)`")
_PLURAL = re.compile(r"list the (.+?) the paper reports")


def _respond(system: str, user: str) -> str:
    lane = "paddleocr_vl" if "paddleocr_vl_" in user else "mineru"
    if user.startswith("Paper excerpts"):
        plural = _PLURAL.search(system)
        assert plural is not None
        cited = _SOURCE.findall(user)[:1]
        samples = [{"sample_id": sid, "source_ids": cited} for sid in INVENTORY[(lane, plural.group(1))]]
        return json.dumps({"samples": samples, "no_samples_in_scope": False})
    if user.startswith("List A (parser:"):
        list_a, list_b = user.split("\n\nList B (parser:", 1)
        pairs = [
            {"a": a, "b": b, "confidence": 0.9, "justification": "same position"}
            for a, b in zip(_LISTED_ID.findall(list_a), _LISTED_ID.findall(list_b), strict=True)
        ]
        return json.dumps({"pairs": pairs})
    field = _FIELD.match(user)
    assert field is not None, user[:200]
    question, excerpts = user.split("Excerpts (Markdown with provenance markers):", 1)
    values = []
    for sample_id, (value_raw, unit_raw, condition) in zip(
        _LISTED_ID.findall(question), ANSWERS[field.group(1)], strict=False
    ):
        cited = [
            match.group(1)
            for block in re.split(r"(?=<!-- source: )", excerpts)
            if (match := _SOURCE.match(block)) and value_raw in block
        ][:1]
        values.append(
            {
                "sample_id": sample_id,
                "value_raw": value_raw,
                "unit_raw": unit_raw,
                "condition": condition,
                "source_ids": cited,
            }
        )
    return json.dumps({"values": values})


def _seed_parses(document: DocumentInput, layout: DataLayout, geometry: DocumentGeometry) -> None:
    blocks = (TITLE, *PARAGRAPHS)
    for backend in BACKENDS:
        raw_dir = layout.raw_dir(document.document_id, backend)
        factory = RawOutputFactory(raw_dir.parent, document, geometry)
        if backend == "mineru":
            content = [
                {"type": "text", "page_idx": 0, "bbox": [100, 80 + 150 * i, 900, 200 + 150 * i], "text": text}
                | ({"text_level": 1} if i == 0 else {})
                for i, text in enumerate(blocks)
            ]
            factory.mineru(content, dir_name=raw_dir.name)
        else:
            parsing = [
                {
                    "block_id": i,
                    "block_order": i,
                    "block_label": "doc_title" if i == 0 else "text",
                    "block_bbox": [200, 150 + 350 * i, 1450, 450 + 350 * i],
                    "block_content": text,
                }
                for i, text in enumerate(blocks)
            ]
            page = paddle_page_entry(0, {"res": {"parsing_res_list": parsing}}, (1654, 2339))
            factory.paddle([page], dir_name=raw_dir.name)


def test_a_two_entity_document_runs_end_to_end_and_re_exports_identically(
    profile_file: Path, tmp_path: Path, document: DocumentInput, geometry: DocumentGeometry, monkeypatch
):
    settings = Settings(
        data_root=tmp_path / "data",
        repo_root=tmp_path,
        llm_api_key="sk-test",
        llm_model="fake-model",
        profile=str(profile_file),
    )
    _seed_parses(document, DataLayout(settings.data_root), geometry)
    profile = load_run_profile(settings)
    client = FakeLlmClient(_respond)
    monkeypatch.setattr("paperfacts.workflow.build_llm_client", lambda _settings: client)

    result = run_document(document, settings, profile)

    # Two inventories per lane, one matching per entity, each field asked with its entity's list.
    inventories = [call.system for call in client.calls if call.user.startswith("Paper excerpts")]
    assert sorted(inventories) == sorted([inventory_system_prompt(profile, entity) for entity in profile.entities] * 2)
    matchings = [call.system for call in client.calls if call.user.startswith("List A")]
    assert sorted(matchings) == sorted(matching_system_prompt(profile, entity) for entity in profile.entities)
    headings = {
        _FIELD.match(call.user).group(1): call.user.split("\n\n")[1].split(" this paper reports:")[0]  # type: ignore[union-attr]
        for call in client.calls
        if call.user.startswith("Field to extract:")
    }
    assert headings == {
        "precursor_purity": "Coatings",
        "coating_thickness": "Coatings",
        "solvent": "Coatings",
        "test_temperature": "Wear tests",
        "wear_mode": "Wear tests",
    }

    # A coating and a wear test named alike are two samples, each with its own values.
    mineru = result.lanes["mineru"]
    assert [(s.entity, s.sample_id) for s in mineru.samples] == [
        ("coating", "S1"),
        ("coating", "S2"),
        ("wear_test", "S1"),
    ]
    assert [v.field for v in mineru.sample("S1", "wear_test").fields] == ["test_temperature", "wear_mode"]  # type: ignore[union-attr]

    report = result.report
    assert list(report.matchings) == ["coating", "wear_test"]
    assert report.counts.samples_matched == 3 and report.counts.samples_unmatched == 0
    assert {c.status for c in report.comparisons} == {"agree"}
    assert {c.scope for c in report.comparisons} == {
        "paper",
        "coating:S1|coat-1",
        "coating:S2|coat-2",
        "wear_test:S1|run-1",
    }

    dataset = result.dataset
    assert not dataset.incomplete
    rows = {(row["entity"], row["sample_id"]): row for row in dataset.sample_rows}
    assert set(rows) == {("coating", "S1 | coat-1"), ("coating", "S2 | coat-2"), ("wear_test", "S1 | run-1")}
    wear = rows[("wear_test", "S1 | run-1")]
    assert wear["test_temperature"] == pytest.approx(300.0) and wear["wear_mode"] == "sliding"
    assert wear["precursor_purity"] == pytest.approx(99.9) and "coating_thickness" not in wear
    coating = rows[("coating", "S2 | coat-2")]
    assert coating["coating_thickness"] == pytest.approx(200.0) and "test_temperature" not in coating
    # The paper row is one of the primary entity's rows.
    assert dataset.paper_row["entity"] == "coating"
    assert {row["entity"] for row in dataset.quality_rows if row["field"] == "wear_mode"} == {"wear_test"}

    # The offline re-export re-compares under every stored matching: the same verdicts and table, and no wear test
    # comes out unmatched as a false "missing".
    exported = export_document(document, settings, profile)
    stored = ComparisonReport.read(
        DataLayout(settings.data_root).comparison_path(
            document.document_id, report.extractor_key, report.comparison_key
        )
    )
    assert stored.comparisons == report.comparisons and stored.matchings == report.matchings
    assert "missing" not in {c.status for c in stored.comparisons}
    assert [dict(row) for row in exported.sample_rows] == [dict(row) for row in dataset.sample_rows]
    assert [dict(row) for row in exported.quality_rows] == [dict(row) for row in dataset.quality_rows]
    assert dict(exported.paper_row) == dict(dataset.paper_row)
