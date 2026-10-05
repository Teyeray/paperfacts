"""The independent experiment runner freezes inputs before any visual request."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

from paperfacts.compare import ComparisonReport
from paperfacts.errors import LlmOfflineMiss
from paperfacts.keys import profile_comparison_fingerprint, profile_extraction_fingerprint
from paperfacts.llm import OfflineMisses, OpenAICompatibleClient
from paperfacts.matching import SampleMatching
from paperfacts.models import DocumentInput, ParsedArtifact
from paperfacts.pdf import read_geometry
from paperfacts.profile_loader import parse_profile
from paperfacts.records import LaneExtraction
from paperfacts.storage import DocumentIdentity, PartInfo, parts_sha256
from support.factories import make_blank_pdf
from support.profiles import profile_data
from support.vision import FakeVisionClient

SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "visual_evidence.py"


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("paperfacts_visual_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # A real source tree makes code-drift checks deterministic while other tasks edit this checkout.
    source_root = tmp_path / "source"
    source_file = source_root / "src" / "paperfacts" / "visual_evidence.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("PROMPT = 'frozen synthetic prompt'\n")
    monkeypatch.setattr(module, "REPO", source_root)
    return module


def inputs(tmp_path: Path, *, si: bool = False):
    profile_path = tmp_path / "demo.json"
    profile_json = profile_data()
    profile_path.write_text(json.dumps(profile_json))
    profile = parse_profile(profile_json, profile_path)
    document = DocumentInput.from_path(make_blank_pdf(tmp_path / "paper.pdf"))
    parts = (
        PartInfo(name="main.pdf", sha256="a" * 64, first_page=0, pages=1),
        PartInfo(name="si.pdf", sha256="b" * 64, first_page=1, pages=1),
    )
    document_id = parts_sha256([part.sha256 for part in parts]) if si else document.document_id
    artifact_paths, lane_paths = {}, {}
    hashes = {}
    for backend in ("mineru", "paddleocr_vl"):
        artifact = ParsedArtifact(
            document_id=document_id,
            backend=backend,
            backend_version="synthetic",
            pages=read_geometry(document.pdf_path).pages,
            blocks=(),
        )
        artifact_path = tmp_path / f"{backend}.artifact.json"
        artifact.write(artifact_path)
        hashes[backend] = artifact.content_hash()
        lane = LaneExtraction(
            document_id=document_id,
            backend=backend,
            extractor_key="frozen-extractor",
            model="synthetic",
            profile_fingerprint=profile_extraction_fingerprint(profile),
            artifact_sha256=artifact.content_hash(),
        )
        lane_path = tmp_path / f"{backend}.lane.json"
        lane.write(lane_path)
        artifact_paths[backend], lane_paths[backend] = str(artifact_path), str(lane_path)
    comparison_path = tmp_path / "comparison.json"
    ComparisonReport(
        document_id=document_id,
        extractor_key="frozen-extractor",
        comparison_key="frozen-comparison",
        backend_a="mineru",
        backend_b="paddleocr_vl",
        matchings={"sample": SampleMatching()},
        artifact_sha256_a=hashes["mineru"],
        artifact_sha256_b=hashes["paddleocr_vl"],
        profile_fingerprint=profile_comparison_fingerprint(profile),
    ).write(comparison_path)
    entry = dict(
        id=document_id,
        pdf_path=str(document.pdf_path),
        artifact_paths=artifact_paths,
        lane_paths=lane_paths,
        comparison_path=str(comparison_path),
    )
    if si:
        identity_path = tmp_path / "identity.json"
        identity = DocumentIdentity(
            sha256=document_id, name="main + SI", uploaded=True, created_at="2026-10-05T00:00:00+00:00", parts=parts
        )
        identity_path.write_text(identity.model_dump_json())
        entry["identity_path"] = str(identity_path)
    gold_path = tmp_path / "reviewed-gold.json"
    gold_path.write_text('{"coverage":"synthetic-only"}')
    config = dict(
        profile_path=str(profile_path),
        data_root=str(tmp_path / "data"),
        model="fake-vl",
        base_url="https://example.invalid/v1",
        documents=[entry],
        render={"dpi": 72, "max_pixels": 10000},
        limit=2,
        metadata={"split": "synthetic", "history": "never a real-paper evaluation"},
        reviewed_gold_paths=[str(gold_path)],
    )
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config))
    return config_path, config, document


def test_freeze_is_idempotent_and_refuses_a_changed_setup(experiment, tmp_path):
    config_path, config, _ = inputs(tmp_path)
    run_root = tmp_path / "run"
    manifest = experiment.freeze(config_path, run_root)
    before = (run_root / "manifest.json").read_bytes()
    assert manifest.profile_hash
    assert len(manifest.files) == 9  # config, profile, PDF, two artifacts, two lanes, comparison, gold
    assert str(config_path.resolve()) in manifest.files
    experiment.freeze(config_path, run_root)
    assert (run_root / "manifest.json").read_bytes() == before
    config["limit"] = 3
    config_path.write_text(json.dumps(config))
    with pytest.raises(FileExistsError, match="new run"):
        experiment.freeze(config_path, run_root)
    assert (run_root / "manifest.json").read_bytes() == before


def test_freeze_refuses_incomplete_baseline_without_starting_a_run(experiment, tmp_path):
    config_path, config, _ = inputs(tmp_path)
    lane_path = Path(config["documents"][0]["lane_paths"]["mineru"])
    lane = json.loads(lane_path.read_text())
    lane["failed_questions"] = [{"field": "coating_thickness", "detail": "synthetic unanswered question"}]
    lane_path.write_text(json.dumps(lane))
    with pytest.raises(ValueError, match="incomplete"):
        experiment.freeze(config_path, tmp_path / "run")
    assert not (tmp_path / "run" / "manifest.json").exists()


@pytest.mark.parametrize("drift", ["artifact", "pdf", "profile", "gold", "code"])
def test_input_or_code_drift_refuses_before_any_request(experiment, tmp_path, drift):
    config_path, config, _ = inputs(tmp_path)
    run_root = tmp_path / "run"
    experiment.freeze(config_path, run_root)
    paths = dict(
        artifact=config["documents"][0]["artifact_paths"]["mineru"],
        pdf=config["documents"][0]["pdf_path"],
        profile=config["profile_path"],
        gold=config["reviewed_gold_paths"][0],
        code=experiment.REPO / "src/paperfacts/visual_evidence.py",
    )
    path = Path(paths[drift])
    path.write_bytes(path.read_bytes() + b"\n")
    client = FakeVisionClient({"outcome": "no_facts", "reason": "blank page", "observations": []})
    with pytest.raises(ValueError, match="drift"):
        experiment.run(run_root, client=client)
    assert client.calls == []


def test_a_later_documents_drift_stops_the_entire_run(experiment, tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    config_path, config, _ = inputs(first_dir)
    _, second, _ = inputs(second_dir)
    config["documents"].extend(second["documents"])
    config_path.write_text(json.dumps(config))
    run_root = tmp_path / "run"
    experiment.freeze(config_path, run_root)
    Path(second["documents"][0]["comparison_path"]).write_text("{}")
    client = FakeVisionClient({"outcome": "no_facts", "reason": "blank page", "observations": []})
    with pytest.raises(ValueError, match="drift"):
        experiment.run(run_root, client=client)
    assert client.calls == []


def test_si_identity_is_preserved_and_only_two_strategy_reports_are_written(experiment, tmp_path):
    config_path, config, document = inputs(tmp_path, si=True)
    run_root = tmp_path / "run"
    manifest = experiment.freeze(config_path, run_root)
    client = FakeVisionClient({"outcome": "no_facts", "reason": "blank page", "observations": []})
    paths = experiment.run(run_root, client=client)
    assert {path.name for path in paths} == {
        f"{config['documents'][0]['id'][:16]}.{strategy}.json" for strategy in ("risk_only", "balanced")
    }
    assert manifest.identities[config["documents"][0]["id"]]["parts"][1]["first_page"] == 1
    reports = [json.loads(path.read_text()) for path in paths]
    assert all(report["document_id"] == config["documents"][0]["id"] for report in reports)
    assert all(report["pdf_sha256"] == document.sha256 for report in reports)
    assert document.sha256 != config["documents"][0]["id"]
    assert client.calls


def test_offline_cache_miss_never_reaches_http(experiment, tmp_path):
    config_path, config, _ = inputs(tmp_path)
    run_root = tmp_path / "run"
    experiment.freeze(config_path, run_root)
    requests = []

    def forbidden(request):
        requests.append(request)
        raise AssertionError("offline execution reached HTTP")

    with OpenAICompatibleClient(
        config["base_url"],
        "offline-unused",
        config["model"],
        timeout_s=120,
        temperature=0,
        max_tokens=4096,
        reasoning_effort=None,
        offline=True,
        cache_dir=tmp_path / "empty-cache",
        client=httpx.Client(transport=httpx.MockTransport(forbidden)),
        misses=OfflineMisses(),
        retry_attempts=2,
    ) as client:
        with pytest.raises(LlmOfflineMiss):
            experiment.run(run_root, client=client)
    assert requests == []


def test_invalid_existing_identity_fails_before_freezing(experiment, tmp_path):
    config_path, config, _ = inputs(tmp_path, si=True)
    identity_path = Path(config["documents"][0]["identity_path"])
    identity = json.loads(identity_path.read_text())
    identity["sha256"] = "f" * 64
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="identity"):
        experiment.freeze(config_path, tmp_path / "run")
    assert not (tmp_path / "run" / "manifest.json").exists()


def test_existing_reports_are_preserved_without_another_request(experiment, tmp_path):
    config_path, _, _ = inputs(tmp_path)
    run_root = tmp_path / "run"
    experiment.freeze(config_path, run_root)
    answer = {"outcome": "no_facts", "reason": "blank page", "observations": []}
    paths = experiment.run(run_root, client=FakeVisionClient(answer))
    before = {path: path.read_bytes() for path in paths}
    client = FakeVisionClient(answer)
    assert experiment.run(run_root, client=client) == paths
    assert client.calls == []
    assert all(path.read_bytes() == data for path, data in before.items())


def test_manifest_config_edits_cannot_bypass_frozen_file_checks(experiment, tmp_path):
    config_path, _, _ = inputs(tmp_path)
    run_root = tmp_path / "run"
    experiment.freeze(config_path, run_root)
    manifest_path = run_root / "manifest.json"
    changed = json.loads(manifest_path.read_text())
    changed["config"]["model"] = "edited-model"
    manifest_path.write_text(json.dumps(changed))
    client = FakeVisionClient({}, model="edited-model")
    with pytest.raises(ValueError, match="drift"):
        experiment.run(run_root, client=client)
    assert client.calls == []


def test_default_run_reads_no_credentials_and_uses_an_offline_two_attempt_client(experiment, tmp_path, monkeypatch):
    config_path, _, _ = inputs(tmp_path)
    run_root = tmp_path / "run"
    experiment.freeze(config_path, run_root)
    made = []
    requests = []

    def forbidden_settings():
        raise AssertionError("offline replay read credential settings")

    def forbidden_http(request):
        requests.append(request)
        raise AssertionError("offline replay reached HTTP")

    def make_client(*args, **kwargs):
        client = OpenAICompatibleClient(
            *args, **kwargs, client=httpx.Client(transport=httpx.MockTransport(forbidden_http))
        )
        made.append(client)
        return client

    monkeypatch.setattr(experiment.Settings, "from_env", forbidden_settings)
    monkeypatch.setattr(experiment, "OpenAICompatibleClient", make_client)
    with pytest.raises(LlmOfflineMiss):
        experiment.run(run_root)
    assert len(made) == 1 and made[0].offline and made[0].retry_attempts == 2
    assert made[0].cache_dir == tmp_path / "data" / "llm_cache"
    assert requests == []
    assert any(state.status == "failed" for state in experiment.status(run_root))


def test_a_cached_run_replays_without_any_http(experiment, tmp_path):
    config_path, config, _ = inputs(tmp_path)
    first_root, replay_root = tmp_path / "seeded-run", tmp_path / "replayed-run"
    experiment.freeze(config_path, first_root)
    calls = []

    def local_response(request):
        calls.append(request)
        text = json.dumps({"outcome": "no_facts", "reason": "blank page", "observations": []})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    kwargs = dict(
        timeout_s=120,
        cache_dir=tmp_path / "cache",
        temperature=0,
        max_tokens=4096,
        reasoning_effort=None,
        retry_attempts=2,
    )
    # The online code path is served entirely by an in-process transport, never a real endpoint.
    with OpenAICompatibleClient(
        config["base_url"],
        "fake-unused",
        config["model"],
        **kwargs,
        client=httpx.Client(transport=httpx.MockTransport(local_response)),
    ) as client:
        experiment.run(first_root, online=True, client=client)
    assert calls
    experiment.freeze(config_path, replay_root)

    def no_http(request):
        raise AssertionError("a cache replay reached HTTP")

    with OpenAICompatibleClient(
        config["base_url"],
        "offline-unused",
        config["model"],
        **kwargs,
        offline=True,
        client=httpx.Client(transport=httpx.MockTransport(no_http)),
    ) as client:
        outputs = experiment.run(replay_root, client=client)
    readings = [reading for path in outputs for reading in json.loads(path.read_text())["readings"]]
    assert readings and all(attempt["cached"] for reading in readings for attempt in reading["attempts"])


def test_freeze_rejects_a_nonempty_directory_without_a_manifest(experiment, tmp_path):
    config_path, _, _ = inputs(tmp_path)
    run_root = tmp_path / "run"
    run_root.mkdir()
    previous = run_root / "old-result.json"
    previous.write_text("previous private output")
    with pytest.raises(FileExistsError, match="new run"):
        experiment.freeze(config_path, run_root)
    assert previous.read_text() == "previous private output"
    assert not (run_root / "manifest.json").exists()


def test_an_injected_online_client_cannot_bypass_offline_default(experiment, tmp_path):
    config_path, config, _ = inputs(tmp_path)
    run_root = tmp_path / "run"
    experiment.freeze(config_path, run_root)
    requests = []

    def forbidden(request):
        requests.append(request)
        raise AssertionError("offline run reached HTTP through an injected client")

    with OpenAICompatibleClient(
        config["base_url"],
        "unused",
        config["model"],
        timeout_s=1,
        client=httpx.Client(transport=httpx.MockTransport(forbidden)),
    ) as client:
        with pytest.raises(ValueError, match="offline"):
            experiment.run(run_root, client=client)
    assert requests == []


@pytest.mark.parametrize(
    "change",
    [
        {"base_url": "https://other.invalid/v1"},
        {"temperature": 0.5},
        {"max_tokens": 100},
        {"retry_attempts": 3},
        {"reasoning_effort": "high"},
    ],
)
def test_an_injected_real_client_must_match_frozen_request_parameters(experiment, tmp_path, change):
    config_path, config, _ = inputs(tmp_path)
    run_root = tmp_path / "run"
    experiment.freeze(config_path, run_root)
    kwargs = dict(
        base_url=config["base_url"],
        model=config["model"],
        timeout_s=120,
        temperature=0,
        max_tokens=4096,
        reasoning_effort=None,
        retry_attempts=2,
    )
    kwargs.update(change)
    with OpenAICompatibleClient(api_key="offline-unused", offline=True, **kwargs) as client:
        with pytest.raises(ValueError, match="frozen"):
            experiment.run(run_root, client=client)


def test_si_same_identity_and_geometry_does_not_reuse_crops_of_different_pdf_bytes(experiment, tmp_path):
    import ctypes

    import pypdfium2 as pdfium

    from paperfacts.pdf import _PDFIUM_LOCK
    from support.factories import PAGE_SIZES_PT

    config_path, config, _ = inputs(tmp_path, si=True)
    first_root, second_root = tmp_path / "first-run", tmp_path / "second-run"
    experiment.freeze(config_path, first_root)
    response = {"outcome": "no_facts", "reason": "synthetic answer", "observations": []}
    first_client = FakeVisionClient(response)
    first_reports = experiment.run(first_root, client=first_client)
    changed_pdf = tmp_path / "changed-source.pdf"
    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument.new()
        try:
            for size in PAGE_SIZES_PT:
                page = pdf.new_page(*size)
                try:
                    obj = pdfium.PdfTextObj(pdfium.raw.FPDFPageObj_NewTextObj(pdf, b"Helvetica", 30), pdf=pdf)
                    encoded = ctypes.create_string_buffer("coating thickness 999".encode("utf-16-le") + b"\0\0")
                    assert pdfium.raw.FPDFText_SetText(obj, ctypes.cast(encoded, ctypes.POINTER(ctypes.c_ushort)))
                    obj.set_matrix(pdfium.PdfMatrix(e=60, f=80))
                    page.insert_obj(obj)
                    page.gen_content()
                finally:
                    page.close()
            pdf.save(str(changed_pdf))
        finally:
            pdf.close()
    config["documents"][0]["pdf_path"] = str(changed_pdf)
    changed_config = tmp_path / "changed-config.json"
    changed_config.write_text(json.dumps(config))
    experiment.freeze(changed_config, second_root)
    second_client = FakeVisionClient(response)
    second_reports = experiment.run(second_root, client=second_client)

    first = json.loads(next(path for path in first_reports if "balanced" in path.name).read_text())
    second = json.loads(next(path for path in second_reports if "balanced" in path.name).read_text())
    assert first["document_id"] == second["document_id"]
    assert first["pdf_sha256"] != second["pdf_sha256"]
    first_crop = first["readings"][0]["attempts"][0]["crop"]
    second_crop = second["readings"][0]["attempts"][0]["crop"]
    assert first_crop["path"] != second_crop["path"]
    assert first_client.calls[0].image_png != second_client.calls[0].image_png


def test_failed_state_survives_restart_and_explicit_retry_preserves_history(experiment, tmp_path):
    from paperfacts.errors import LlmError

    config_path, _, _ = inputs(tmp_path)
    root = tmp_path / "run"
    experiment.freeze(config_path, root)
    first = FakeVisionClient(LlmError("synthetic failure"))
    experiment.run(root, client=first)
    states = experiment.status(root)
    assert any(s.status == "failed" for s in states)
    files = {p: p.read_bytes() for p in root.glob("*.json")}
    idle = FakeVisionClient({})
    experiment.run(root, client=idle)
    assert idle.calls == []
    retry = FakeVisionClient({"outcome": "no_facts", "reason": "synthetic blank", "observations": []})
    experiment.run(root, client=retry, retry=True)
    assert retry.calls and all(c.refresh for c in retry.calls)
    assert all(p.read_bytes() == data for p, data in files.items())
    assert any(s.attempt == 2 for s in experiment.status(root))


def test_corrupt_report_is_stale_and_does_not_trigger_implicit_reread(experiment, tmp_path):
    config_path, _, _ = inputs(tmp_path)
    root = tmp_path / "run"
    experiment.freeze(config_path, root)
    paths = experiment.run(
        root, client=FakeVisionClient({"outcome": "no_facts", "reason": "blank", "observations": []})
    )
    paths[0].write_text("{}")
    assert experiment.status(root)[0].status == "stale"
    idle = FakeVisionClient({})
    experiment.run(root, client=idle)
    assert not idle.calls


def test_partial_readings_are_persisted_before_later_failure(experiment, tmp_path):
    from paperfacts.errors import LlmError

    config_path, _, _ = inputs(tmp_path)
    root = tmp_path / "run"
    experiment.freeze(config_path, root)
    count = 0

    def answer(user, image):
        nonlocal count
        count += 1
        return {"outcome": "no_facts", "reason": "blank", "observations": []} if count % 2 else LlmError("synthetic")

    experiment.run(root, client=FakeVisionClient(answer))
    assert any(s.status == "partial" for s in experiment.status(root))


def test_offline_export_reloads_snapshots_without_client_and_keeps_originals(experiment, tmp_path, monkeypatch):
    from paperfacts.visual_snapshot import ExperimentSnapshot

    config_path, _, _ = inputs(tmp_path)
    root = tmp_path / "run"
    experiment.freeze(config_path, root)
    experiment.run(root, client=FakeVisionClient({"outcome": "no_facts", "reason": "blank", "observations": []}))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"revision": "synthetic-v1", "fields": ["coating_thickness"]}))
    before = {p: p.read_bytes() for p in root.glob("*.json")}

    def forbidden(*args, **kwargs):
        raise AssertionError("export constructed a client")

    monkeypatch.setattr(experiment, "OpenAICompatibleClient", forbidden)
    paths = experiment.export_experiment(root, policy, root / "export-1")
    assert paths[-1].name == "dataset.xlsx"
    loaded = ExperimentSnapshot.model_validate_json(paths[0].read_bytes())
    assert loaded.provenance["source"]["digest"]
    assert all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(FileExistsError):
        experiment.export_experiment(root, policy, root / "export-1")
    with pytest.raises(ValueError, match="under"):
        experiment.export_experiment(root, policy, tmp_path / "outside")


def test_export_missing_or_stale_report_keeps_baseline(experiment, tmp_path):
    from paperfacts.visual_snapshot import ExperimentSnapshot

    config_path, _, _ = inputs(tmp_path)
    root = tmp_path / "run"
    experiment.freeze(config_path, root)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"revision": "v1", "fields": ["coating_thickness"]}))
    missing = experiment.export_experiment(root, policy, root / "missing")
    assert ExperimentSnapshot.model_validate_json(missing[0].read_bytes()).report_status == "missing"
    paths = experiment.run(
        root, client=FakeVisionClient({"outcome": "no_facts", "reason": "blank", "observations": []})
    )
    balanced = next(path for path in paths if ".balanced." in path.name)
    balanced.write_text("{}")
    stale = experiment.export_experiment(root, policy, root / "stale")
    snapshot = ExperimentSnapshot.model_validate_json(stale[0].read_bytes())
    assert snapshot.report_status == "stale" and snapshot.audit is None


def test_interrupted_attempt_needs_explicit_retry(experiment, tmp_path):
    from paperfacts.models import sha256_of_file
    from paperfacts.storage import VisualEvidenceLayout

    config_path, config, _ = inputs(tmp_path)
    root = tmp_path / "run"
    experiment.freeze(config_path, root)
    layout = VisualEvidenceLayout(root)
    for strategy in ("risk_only", "balanced"):
        state = experiment.ExperimentState(
            document_id=config["documents"][0]["id"],
            strategy=strategy,
            attempt=1,
            fingerprint=sha256_of_file(layout.manifest_path()),
            status="running",
        )
        layout.state_path(state.document_id, strategy, 1).write_text(state.model_dump_json())
    idle = FakeVisionClient({})
    assert experiment.run(root, client=idle) == ()
    assert not idle.calls and all(s.status == "interrupted" for s in experiment.status(root))
    retry = FakeVisionClient({"outcome": "no_facts", "reason": "blank", "observations": []})
    experiment.run(root, client=retry, retry=True)
    assert retry.calls and all(s.attempt == 2 for s in experiment.status(root))
