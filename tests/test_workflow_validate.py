"""The validate stage inside the pipeline: skipped or run, cached under three keys, and threaded into the
export -- without a real model, a real parser or a real paper.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from paperfacts.compare import compare_lanes
from paperfacts.config import Settings
from paperfacts.grounding import ground_lane
from paperfacts.keys import comparison_key, extractor_key_for, validation_key_for
from paperfacts.matching import SampleMatch, SampleMatching
from paperfacts.models import BACKENDS, DocumentInput, NormalizedBBox
from paperfacts.normalize import normalize_lane
from paperfacts.storage import DataLayout
from paperfacts.validate import ValidationReport
from paperfacts.workflow import read_validation, run_document, validate_document
from support.extraction import make_artifact, make_field, make_lane, make_sample
from support.factories import make_block
from support.llm import FakeVisionClient
from test_workflow_run import install_fake_pipeline

BOX = NormalizedBBox(x1=0.1, y1=0.4, x2=0.9, y2=0.6)


def reading(text: str) -> str:
    return json.dumps({"transcription": text, "legible": True})


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_root=tmp_path / "data", repo_root=tmp_path, llm_api_key="sk-test")


@pytest.fixture
def vlm_settings(settings: Settings) -> Settings:
    return dataclasses.replace(settings, vlm_enabled=True, vlm_crop_dpi=72)


def store_artifacts(document: DocumentInput, settings: Settings) -> dict:
    """Both lanes' artifacts on disk, where ``validate_document`` loads them from; one table block each."""
    layout = DataLayout(settings.data_root)
    artifacts = {}
    for backend in BACKENDS:
        artifact = make_artifact(
            [
                make_block(
                    page=0,
                    order=1,
                    type="table",
                    backend=backend,
                    document_id=document.document_id,
                    content="Rs 12.5",
                    bbox=BOX,
                )
            ],
            backend=backend,
            document_id=document.document_id,
        )
        artifact.write(layout.artifact_path(document.document_id, backend))
        artifacts[backend] = artifact
    return artifacts


def conflicting_lanes(document: DocumentInput, artifacts: dict):
    key = extractor_key_for(Settings())
    lanes = {}
    for backend, raw in (("mineru", "12.5"), ("paddleocr_vl", "125")):
        lane = make_lane(
            backend=backend,
            document_id=document.document_id,
            extractor_key=key,
            samples=[
                make_sample(
                    "A", [make_field("sheet_resistance", raw, unit_raw="Ω/sq", source_ids=[f"{backend}_p0_b1"])]
                )
            ],
        )
        blocks = {block.source_id: block.content for block in artifacts[backend].blocks}
        lanes[backend] = normalize_lane(ground_lane(lane, blocks))
    matching = SampleMatching(
        pairs=(SampleMatch(a_id="A", b_id="A", confidence=1.0, justification="", method="exact"),)
    )
    return lanes, compare_lanes(lanes["mineru"], lanes["paddleocr_vl"], matching)


# ---- run_document: the stage is skipped or run ----------------------------------------------------------------


def test_with_the_vlm_off_the_stage_is_skipped_and_no_client_is_built(monkeypatch, document, settings):
    install_fake_pipeline(monkeypatch)
    monkeypatch.setattr("paperfacts.workflow.build_vlm_client", lambda s: pytest.fail("no VLM client when disabled"))

    marks: list[tuple[str, str, str]] = []
    result = run_document(document, settings, on_stage=lambda s, st, d: marks.append((s, st, d)))

    assert ("validate", "skipped", "vlm.enabled is false") in marks
    assert result.validation is None
    assert result.dataset.validation_key == ""
    assert result.dataset_json_path.name == f"{result.dataset.extractor_key}.{result.dataset.comparison_key}.json"


def test_with_the_vlm_on_the_stage_runs_between_compare_and_export(monkeypatch, document, vlm_settings):
    install_fake_pipeline(monkeypatch)
    client = FakeVisionClient([])
    monkeypatch.setattr("paperfacts.workflow.build_vlm_client", lambda s: client)
    monkeypatch.setattr("paperfacts.workflow.load_artifact", lambda d, b, s: make_artifact(backend=b))

    marks: list[tuple[str, str, str]] = []
    result = run_document(document, vlm_settings, on_stage=lambda s, st, d: marks.append((s, st, d)))

    names = [name for name, status, _ in marks if status in {"running", "skipped"}]
    assert names.index("compare") < names.index("validate") < names.index("export")
    assert ("validate", "done", "confirmed 0 · contradicted 0 · illegible 0 · not checked 0") in marks
    assert result.validation is not None and result.validation.values == ()
    assert client.closed, "the VLM client is opened for the stage and closed after it"
    # The dataset now carries the third key, in its contents and in its file name.
    assert result.dataset.validation_key == validation_key_for(vlm_settings)
    assert result.dataset_json_path.name.endswith(f".{validation_key_for(vlm_settings)}.json")


# ---- validate_document: cached under all three keys ----------------------------------------------------------


def test_the_verdicts_are_stored_under_the_three_keys_and_replayed(document, vlm_settings):
    artifacts = store_artifacts(document, vlm_settings)
    lanes, report = conflicting_lanes(document, artifacts)
    client = FakeVisionClient(lambda call: reading("Rs | 12.5 Ω/sq"))

    first = validate_document(document, vlm_settings, client, lanes=lanes, report=report)
    second = validate_document(document, vlm_settings, client, lanes=lanes, report=report)

    assert client.call_count == 2, "two values checked once; the second run read the file"
    assert first == second
    path = DataLayout(vlm_settings.data_root).validation_path(
        document.document_id, report.extractor_key, report.comparison_key, validation_key_for(vlm_settings)
    )
    assert path.is_file()
    assert ValidationReport.read(path) == first
    assert {v.backend: v.verdict for v in first.values} == {"mineru": "confirmed", "paddleocr_vl": "contradicted"}
    assert read_validation(DataLayout(vlm_settings.data_root), document.document_id, vlm_settings) == first


def test_force_re_asks_the_model(document, vlm_settings):
    artifacts = store_artifacts(document, vlm_settings)
    lanes, report = conflicting_lanes(document, artifacts)
    client = FakeVisionClient(lambda call: reading("Rs | 12.5 Ω/sq"))

    validate_document(document, vlm_settings, client, lanes=lanes, report=report)
    validate_document(document, vlm_settings, client, lanes=lanes, report=report, force=True)

    assert client.call_count == 4
    assert all(call.refresh for call in client.calls[2:])


def test_a_changed_validation_setting_is_another_file(document, vlm_settings):
    artifacts = store_artifacts(document, vlm_settings)
    lanes, report = conflicting_lanes(document, artifacts)
    client = FakeVisionClient(lambda call: reading("x"))
    validate_document(document, vlm_settings, client, lanes=lanes, report=report)

    wider = dataclasses.replace(vlm_settings, vlm_crop_padding=0.05)
    validate_document(document, wider, client, lanes=lanes, report=report)

    layout = DataLayout(vlm_settings.data_root)
    stored = sorted(p.name for p in layout.doc_dir(document.document_id).joinpath("validations").iterdir())
    assert len(stored) == 2
    assert client.call_count == 4


def test_read_validation_is_none_when_the_stage_is_off_or_has_not_run(document, settings, vlm_settings):
    layout = DataLayout(settings.data_root)
    assert read_validation(layout, document.document_id, settings) is None
    assert read_validation(layout, document.document_id, vlm_settings) is None
    assert comparison_key()  # the keys used for the lookup are the ordinary ones, computed the same way
