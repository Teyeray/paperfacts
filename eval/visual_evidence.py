"""Freeze and replay an explicit visual-evidence experiment, without rerunning either parser lane.

This entry point establishes reproducibility, not G1/G2 scoring. See eval/README.md for the input contract,
output lifecycle, and the distinction between synthetic engineering checks and reviewed real-paper evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pypdfium2 import PdfiumError

from paperfacts.compare import ComparisonReport
from paperfacts.config import Settings
from paperfacts.crops import CropStore
from paperfacts.errors import LlmError, LlmOfflineMiss
from paperfacts.keys import profile_comparison_fingerprint, profile_extraction_fingerprint
from paperfacts.llm import OfflineMisses, OpenAICompatibleClient, VisionClient
from paperfacts.models import DocumentInput, ParsedArtifact, sha256_of_file
from paperfacts.pdf import read_geometry, read_native_pages
from paperfacts.profile import DomainProfile
from paperfacts.profile_loader import parse_profile
from paperfacts.records import LaneExtraction
from paperfacts.storage import (
    DataLayout,
    DocumentIdentity,
    VisualEvidenceLayout,
    document_key,
    parts_sha256,
    write_bytes_if_absent,
    write_text_atomic,
)
from paperfacts.visual_evidence import VisualEvidenceReport

REPO = Path(__file__).resolve().parents[1]
STRATEGIES = ("risk_only", "balanced")
BACKENDS = ("mineru", "paddleocr_vl")


class InputModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class FrozenDocument(InputModel):
    id: str
    pdf_path: Path
    artifact_paths: dict[str, Path]
    lane_paths: dict[str, Path]
    comparison_path: Path
    identity_path: Path | None = None
    dataset_path: Path | None = None

    @field_validator("artifact_paths", "lane_paths")
    @classmethod
    def both_backends(cls, paths: dict[str, Path]) -> dict[str, Path]:
        if set(paths) != set(BACKENDS):
            raise ValueError("paths must name exactly mineru and paddleocr_vl")
        return paths

    def input_paths(self) -> tuple[Path, ...]:
        optional = tuple(path for path in (self.identity_path, self.dataset_path) if path is not None)
        return (
            self.pdf_path,
            *self.artifact_paths.values(),
            *self.lane_paths.values(),
            self.comparison_path,
            *optional,
        )


class RenderConfig(InputModel):
    dpi: int = Field(default=200, gt=0, strict=True)
    max_pixels: int | None = Field(default=2_000_000, gt=0, strict=True)


class ExperimentConfig(InputModel):
    profile_path: Path
    data_root: Path
    documents: tuple[FrozenDocument, ...] = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url: str
    render: RenderConfig = Field(default_factory=RenderConfig)
    limit: int = Field(default=4, ge=1, le=4, strict=True)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0, strict=True)
    timeout_s: float = Field(default=120.0, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    reviewed_gold_paths: tuple[Path, ...] = ()

    @field_validator("base_url")
    @classmethod
    def endpoint_without_credentials(cls, value: str) -> str:
        url = urlsplit(value)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query:
            raise ValueError("base_url must be an HTTP endpoint without credentials or query parameters")
        return value.rstrip("/")


class FrozenManifest(InputModel):
    format: Literal[1] = 1
    created_at: str
    config_path: Path
    config: ExperimentConfig
    files: dict[str, str]
    source: dict[str, Any]
    profile_hash: str
    identities: dict[str, dict[str, Any] | None]
    model_config_snapshot: dict[str, Any]


@dataclass(frozen=True)
class LoadedDocument:
    document: DocumentInput
    artifacts: tuple[ParsedArtifact, ...]
    lanes: tuple[LaneExtraction, ...]
    comparison: ComparisonReport
    identity: DocumentIdentity | None


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


def _source_snapshot() -> dict[str, Any]:
    # Hash complete source bytes, including dirty/untracked modules, rather than trusting HEAD alone.
    paths = list((REPO / "src" / "paperfacts").rglob("*.py"))
    paths += [
        path
        for name in ("eval/visual_evidence.py", "eval/visual_evidence_score.py", "pyproject.toml", "uv.lock")
        if (path := REPO / name).is_file()
    ]
    files = {str(path.relative_to(REPO)): sha256_of_file(path) for path in sorted(paths)}
    revision = None
    dirty = None
    if (REPO / ".git").exists():
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--",
                    "src/paperfacts",
                    "eval/visual_evidence.py",
                    "eval/visual_evidence_score.py",
                    "pyproject.toml",
                    "uv.lock",
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    return {
        "revision": revision,
        "dirty": dirty,
        "files": files,
        "digest": hashlib.sha256(_json_bytes(files)).hexdigest(),
        "prompt_sha256": files.get("src/paperfacts/visual_evidence.py"),
    }


def _resolve_config(path: Path) -> ExperimentConfig:
    config = ExperimentConfig.model_validate_json(path.read_text(encoding="utf-8"))

    def absolute(value: Path) -> Path:
        return (path.parent / value).resolve()

    documents = []
    for entry in config.documents:
        updates = {
            name: absolute(getattr(entry, name))
            for name in ("pdf_path", "comparison_path", "identity_path", "dataset_path")
            if getattr(entry, name) is not None
        }
        updates.update(
            {
                name: {backend: absolute(value) for backend, value in getattr(entry, name).items()}
                for name in ("artifact_paths", "lane_paths")
            }
        )
        documents.append(entry.model_copy(update=updates))
    return config.model_copy(
        update={
            "profile_path": absolute(config.profile_path),
            "data_root": absolute(config.data_root),
            "documents": tuple(documents),
            "reviewed_gold_paths": tuple(map(absolute, config.reviewed_gold_paths)),
        }
    )


def _load_inputs(config: ExperimentConfig) -> tuple[DomainProfile, tuple[LoadedDocument, ...]]:
    # parse_profile bypasses the server's process-lifetime profile cache after disk hashes are verified.
    profile = parse_profile(json.loads(config.profile_path.read_text(encoding="utf-8")), config.profile_path)
    loaded = []
    seen = set()
    for entry in config.documents:
        identity = (
            DocumentIdentity.model_validate_json(entry.identity_path.read_text(encoding="utf-8"))
            if entry.identity_path is not None
            else None
        )
        pdf_sha = sha256_of_file(entry.pdf_path)
        document_id = identity.sha256 if identity else pdf_sha
        if entry.id not in (document_id, document_key(document_id)):
            raise ValueError("document identity does not match the explicitly selected input")
        key = document_key(document_id)
        VisualEvidenceLayout(Path(".")).report_path(document_id, "balanced")
        if key in seen:
            raise ValueError("duplicate document key in experiment")
        seen.add(key)
        geometry = read_geometry(entry.pdf_path)
        if identity and identity.parts:
            if parts_sha256([part.sha256 for part in identity.parts]) != document_id:
                raise ValueError("SI identity does not match its part hashes")
            position = 0
            for part in identity.parts:
                if part.first_page != position or part.pages < 1:
                    raise ValueError("SI identity page ranges must be contiguous and nonempty")
                position += part.pages
            if position != geometry.page_count:
                raise ValueError("SI identity page ranges do not cover the actual PDF")
        elif document_id != pdf_sha:
            raise ValueError("single-part identity does not match PDF bytes")
        artifacts = tuple(ParsedArtifact.read(entry.artifact_paths[backend]) for backend in BACKENDS)
        lanes = tuple(LaneExtraction.read(entry.lane_paths[backend]) for backend in BACKENDS)
        comparison = ComparisonReport.read(entry.comparison_path)
        for backend, artifact, lane in zip(BACKENDS, artifacts, lanes, strict=True):
            if lane.failed_questions:
                raise ValueError("incomplete A/B baseline: resolve failed extraction questions before freezing")
            if artifact.document_id != document_id or lane.document_id != document_id:
                raise ValueError("artifact/lane document identity mismatch")
            if artifact.backend != backend or lane.backend != backend:
                raise ValueError("artifact/lane backend mismatch")
            if artifact.pages != geometry.pages:
                raise ValueError("artifact geometry does not match actual PDF")
            if lane.artifact_sha256 != artifact.content_hash():
                raise ValueError("lane artifact fingerprint is missing or mismatched")
            if lane.profile_fingerprint != profile_extraction_fingerprint(profile):
                raise ValueError("lane profile fingerprint is missing or mismatched")
            if lane.extractor_key != comparison.extractor_key:
                raise ValueError("comparison and lane extractor keys differ")
        if comparison.document_id != document_id or (comparison.backend_a, comparison.backend_b) != BACKENDS:
            raise ValueError("comparison document identity/backend mismatch")
        if comparison.profile_fingerprint != profile_comparison_fingerprint(profile):
            raise ValueError("comparison profile fingerprint is missing or mismatched")
        if (comparison.artifact_sha256_a, comparison.artifact_sha256_b) != tuple(a.content_hash() for a in artifacts):
            raise ValueError("comparison artifact fingerprints are missing or mismatched")
        document = DocumentInput(
            document_id=document_id,
            sha256=document_id,
            pdf_path=entry.pdf_path,
            display_name=identity.name if identity else None,
        )
        loaded.append(LoadedDocument(document, artifacts, lanes, comparison, identity))
    return profile, tuple(loaded)


def _all_input_files(config: ExperimentConfig, config_path: Path) -> dict[str, str]:
    paths = [config_path.resolve(), config.profile_path, *config.reviewed_gold_paths]
    paths.extend(path for document in config.documents for path in document.input_paths())
    return {str(path): sha256_of_file(path) for path in sorted(set(paths))}


def freeze(config_path: Path, run_root: Path) -> FrozenManifest:
    """Create one immutable setup, or return the identical manifest without rewriting it."""
    config_path = config_path.resolve()
    path = VisualEvidenceLayout(run_root).manifest_path()
    if run_root.exists() and not path.exists() and any(run_root.iterdir()):
        raise FileExistsError("run directory already contains outputs; choose a new run directory")
    config = _resolve_config(config_path)
    files = _all_input_files(config, config_path)
    profile, documents = _load_inputs(config)
    manifest = FrozenManifest(
        created_at=datetime.now(UTC).isoformat(),
        config_path=config_path,
        config=config,
        files=files,
        source=_source_snapshot(),
        profile_hash=profile.content_hash,
        identities={
            item.document.document_id: item.identity.model_dump(mode="json") if item.identity else None
            for item in documents
        },
        model_config_snapshot={
            "model": config.model,
            "base_url": config.base_url,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "reasoning_effort": None,
            "timeout_s": config.timeout_s,
            "retry_attempts": 2,
        },
    )
    if not write_bytes_if_absent(path, manifest.model_dump_json(indent=2).encode("utf-8")):
        existing = FrozenManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if existing.model_dump(exclude={"created_at"}) != manifest.model_dump(exclude={"created_at"}):
            raise FileExistsError("frozen setup differs; choose a new run directory")
        return existing
    return manifest


def _verify(manifest: FrozenManifest) -> tuple[DomainProfile, tuple[LoadedDocument, ...]]:
    for name, digest in manifest.files.items():
        path = Path(name)
        if not path.is_file() or sha256_of_file(path) != digest:
            raise ValueError(f"frozen input drift: {name}")
    if _resolve_config(manifest.config_path) != manifest.config:
        raise ValueError("frozen configuration drift between manifest and original config")
    if _all_input_files(manifest.config, manifest.config_path) != manifest.files:
        raise ValueError("frozen input inventory drift")
    if _source_snapshot() != manifest.source:
        raise ValueError("frozen code drift; freeze a new run before requesting images")
    profile, documents = _load_inputs(manifest.config)
    if profile.content_hash != manifest.profile_hash:
        raise ValueError("frozen profile drift")
    return profile, documents


class ExperimentState(InputModel):
    document_id: str
    strategy: str
    attempt: int = 0
    fingerprint: str
    status: Literal["pending", "running", "complete", "partial", "failed", "interrupted", "stale"]
    reason: str = ""
    report_sha256: str | None = None


def _states(layout: VisualEvidenceLayout, manifest: FrozenManifest) -> tuple[ExperimentState, ...]:
    fingerprint = sha256_of_file(layout.manifest_path())
    states = []
    for entry in manifest.config.documents:
        document_id = next(key for key in manifest.identities if entry.id in (key, document_key(key)))
        for strategy in STRATEGIES:
            saved = [
                ExperimentState.model_validate_json(path.read_bytes())
                for path in layout.state_paths(document_id, strategy)
            ]
            state = (
                max(saved, key=lambda s: s.attempt)
                if saved
                else ExperimentState(
                    document_id=document_id, strategy=strategy, fingerprint=fingerprint, status="pending"
                )
            )
            if state.document_id != document_id or state.strategy != strategy:
                raise ValueError("saved state identity mismatch")
            if state.fingerprint != fingerprint:
                state = state.model_copy(update={"status": "stale", "reason": "manifest fingerprint changed"})
            elif state.status == "running":
                state = state.model_copy(
                    update={
                        "status": "interrupted",
                        "reason": "previous attempt did not finish; explicit retry required",
                    }
                )
            elif state.report_sha256:
                path = layout.attempt_report_path(document_id, strategy, state.attempt)
                if not path.is_file() or sha256_of_file(path) != state.report_sha256:
                    state = state.model_copy(update={"status": "stale", "reason": "saved report missing or changed"})
            elif state.status == "pending" and layout.report_path(document_id, strategy).exists():
                state = state.model_copy(update={"status": "stale", "reason": "legacy report has no lifecycle receipt"})
            states.append(state)
    return tuple(states)


def status(run_root: Path) -> tuple[ExperimentState, ...]:
    """Read state without constructing a client; frozen input drift is an explicit error."""
    layout = VisualEvidenceLayout(run_root)
    manifest = FrozenManifest.model_validate_json(layout.manifest_path().read_bytes())
    _verify(manifest)
    return _states(layout, manifest)


def _report_status(report: VisualEvidenceReport) -> tuple[str, str]:
    failures = sum(r.status == "error" for r in report.readings)
    unresolved = sum(r.status in ("illegible", "needs_context", "needs_zoom") for r in report.readings)
    if failures and failures == len(report.readings):
        return "failed", "all selected regions failed"
    if failures or unresolved or report.selection.issues or report.selection.unselected_pages:
        return "partial", "some regions failed, remain unresolved, or were outside the budget"
    return "complete", "all selected regions processed"


def run(
    run_root: Path, *, online: bool = False, client: VisionClient | None = None, retry: bool = False
) -> tuple[Path, ...]:
    """Reuse saved attempts; retry only non-complete states, preserving all earlier reports."""
    layout = VisualEvidenceLayout(run_root)
    manifest = FrozenManifest.model_validate_json(layout.manifest_path().read_bytes())
    profile, documents = _verify(manifest)
    from paperfacts.visual_candidates import select_candidates
    from paperfacts.visual_evidence import run_evidence

    states = _states(layout, manifest)
    work = [s for s in states if s.status == "pending" or (retry and s.status != "complete")]

    def outputs() -> tuple[Path, ...]:
        return tuple(
            layout.attempt_report_path(s.document_id, s.strategy, s.attempt)
            for s in _states(layout, manifest)
            if s.report_sha256 and s.status in ("complete", "partial", "failed")
        )

    if not work:
        return outputs()
    config = manifest.config
    misses = OfflineMisses()
    owned = client is None
    if client is None:
        key = Settings.from_env().require_llm_api_key() if online else "offline-unused"
        client = OpenAICompatibleClient(
            config.base_url,
            key,
            config.model,
            timeout_s=config.timeout_s,
            cache_dir=DataLayout(config.data_root).llm_cache_dir(),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            reasoning_effort=None,
            retry_attempts=2,
            offline=not online,
            misses=misses,
        )
    elif client.model != config.model:
        raise ValueError("injected client model differs from the frozen configuration")
    elif not online and not getattr(client, "offline", True):
        raise ValueError("offline execution requires an offline client")
    if not owned:
        for name, expected in manifest.model_config_snapshot.items():
            if hasattr(client, name) and getattr(client, name) != expected:
                raise ValueError(f"injected client {name} differs from the frozen configuration")
    offline_miss = False
    try:
        for previous in work:
            item = next(d for d in documents if d.document.document_id == previous.document_id)
            attempt = previous.attempt + 1
            while (
                layout.attempt_report_path(previous.document_id, previous.strategy, attempt).exists()
                or layout.state_path(previous.document_id, previous.strategy, attempt).exists()
            ):
                attempt += 1
            state = previous.model_copy(
                update={
                    "attempt": attempt,
                    "status": "running",
                    "reason": "",
                    "report_sha256": None,
                    "fingerprint": sha256_of_file(layout.manifest_path()),
                }
            )
            state_path = layout.state_path(state.document_id, state.strategy, attempt)
            write_text_atomic(state_path, state.model_dump_json(indent=2))
            try:
                pages = read_native_pages(item.document.pdf_path)
                crops = CropStore(
                    DataLayout(config.data_root),
                    item.document.document_id,
                    item.document.pdf_path,
                    dpi=config.render.dpi,
                    max_pixels=config.render.max_pixels,
                    page_cache=True,
                    source_pdf_sha256=manifest.files[str(item.document.pdf_path)],
                )
                selection = select_candidates(
                    pages=pages,
                    artifacts={a.backend: a for a in item.artifacts},
                    comparison=item.comparison,
                    lanes=item.lanes,
                    profile=profile,
                    limit=config.limit,
                    strategy=state.strategy,
                )
                report = run_evidence(
                    document=item.document,
                    selection=selection,
                    profile=profile,
                    lanes=item.lanes,
                    comparison=item.comparison,
                    crops=crops,
                    client=client,
                    refresh=retry and previous.status != "pending",
                    keep_offline_misses=True,
                )
                path = layout.attempt_report_path(state.document_id, state.strategy, attempt)
                if not write_bytes_if_absent(path, report.model_dump_json(indent=2).encode()):
                    raise FileExistsError("attempt output already exists")
                outcome, reason = _report_status(report)
                state = state.model_copy(
                    update={"status": outcome, "reason": reason, "report_sha256": sha256_of_file(path)}
                )
                offline_miss |= any(a.error == "LlmOfflineMiss" for r in report.readings for a in r.attempts)
            except (LlmError, OSError, ValueError, PdfiumError) as exc:
                state = state.model_copy(update={"status": "failed", "reason": type(exc).__name__})
                offline_miss |= isinstance(exc, LlmOfflineMiss)
            write_text_atomic(state_path, state.model_dump_json(indent=2))
    finally:
        if owned:
            client.close()
    if offline_miss:
        raise LlmOfflineMiss("offline cache misses recorded; explicit retry required, no online fallback attempted")
    return outputs()


def export_experiment(
    run_root: Path, policy_path: Path, output_root: Path, *, strategy: str = "balanced"
) -> tuple[Path, ...]:
    """Build a new independent export from current frozen A/B and matching saved C; no clients."""
    from paperfacts.dataset import DocumentDataset
    from paperfacts.visual_adoption import AdoptionPolicy
    from paperfacts.visual_evidence import VisualEvidenceReport, visual_prompt
    from paperfacts.visual_snapshot import ExperimentSnapshot, build_snapshot
    from paperfacts.workbook import write_dataset

    if not output_root.resolve().is_relative_to(run_root.resolve()):
        raise ValueError("experimental exports must be under their run directory")
    if output_root.exists():
        raise FileExistsError("export directory already exists; choose a new name")
    if strategy not in STRATEGIES:
        raise ValueError("unknown strategy")
    layout = VisualEvidenceLayout(run_root)
    manifest = FrozenManifest.model_validate_json(layout.manifest_path().read_bytes())
    profile, documents = _verify(manifest)
    states = {(state.document_id, state.strategy): state for state in _states(layout, manifest)}
    policy = AdoptionPolicy.model_validate_json(policy_path.read_bytes())
    snapshots = []
    for item in documents:
        state = states[(item.document.document_id, strategy)]
        report_path = layout.attempt_report_path(state.document_id, strategy, max(state.attempt, 1))
        evidence = (
            VisualEvidenceReport.model_validate_json(report_path.read_bytes())
            if state.report_sha256 and state.status in ("complete", "partial")
            else None
        )
        article_types = {lane.article_type for lane in item.lanes}
        if len(article_types) != 1:
            raise ValueError("baseline article types differ")
        prompt_sha = hashlib.sha256(visual_prompt(profile, next(iter(article_types))).encode()).hexdigest()
        snapshot = build_snapshot(
            document=item.document,
            lanes=item.lanes,
            comparison=item.comparison,
            profile=profile,
            policy=policy,
            pdf_sha256=manifest.files[str(item.document.pdf_path)],
            evidence=evidence,
            expected_model=manifest.config.model,
            expected_prompt_sha256=prompt_sha,
            expected_strategy=strategy,
        )
        if state.status in ("stale", "failed", "interrupted"):
            snapshot = snapshot.model_copy(
                update={"report_status": "stale", "reason": f"{state.status}: {state.reason}"}
            )
        snapshot = snapshot.model_copy(
            update={
                "provenance": {
                    "manifest_sha256": sha256_of_file(layout.manifest_path()),
                    "files": manifest.files,
                    "source": manifest.source,
                    "model": manifest.model_config_snapshot,
                    "policy_sha256": sha256_of_file(policy_path),
                    "state": state.model_dump(mode="json"),
                    "report_path": str(report_path.resolve()) if evidence else None,
                    "metadata": manifest.config.metadata,
                }
            }
        )
        snapshots.append(snapshot)
    # All documents validated before any export is created. A partial write is retained for diagnosis.
    output_root.mkdir(parents=True, exist_ok=False)
    output = VisualEvidenceLayout(output_root)
    paths = []
    restored = []
    for snapshot in snapshots:
        path = output.snapshot_path(snapshot.dataset.document_id)
        snapshot.write(path)
        paths.append(path)
        saved = ExperimentSnapshot.model_validate_json(path.read_bytes())
        restored.append(DocumentDataset.from_payload(saved.dataset))
    write_dataset(
        restored,
        output.workbook_path(),
        profile,
        article_types={item.document.document_id: item.lanes[0].article_type for item in documents},
    )
    return (*paths, output.workbook_path())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_command = commands.add_parser("freeze", help="freeze explicit files and code, without model calls")
    freeze_command.add_argument("--config", type=Path, required=True)
    freeze_command.add_argument("--run-dir", type=Path, required=True)
    run_command = commands.add_parser(
        "run", help="offline replay; --online explicitly permits cache misses on the endpoint"
    )
    run_command.add_argument("--run-dir", type=Path, required=True)
    run_command.add_argument("--online", action="store_true")
    run_command.add_argument(
        "--retry", action="store_true", help="retry non-complete states with refresh; retain history"
    )
    state_command = commands.add_parser("status", help="inspect persisted attempts without requests")
    state_command.add_argument("--run-dir", type=Path, required=True)
    export_command = commands.add_parser(
        "export", help="write a new independent experimental JSON/Excel copy, no requests"
    )
    export_command.add_argument("--run-dir", type=Path, required=True)
    export_command.add_argument("--policy", type=Path, required=True)
    export_command.add_argument("--output-dir", type=Path, required=True)
    export_command.add_argument("--strategy", choices=STRATEGIES, default="balanced")
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            manifest = freeze(args.config, args.run_dir)
            manifest_path = VisualEvidenceLayout(args.run_dir).manifest_path()
            print(f"Frozen {len(manifest.config.documents)} documents: {manifest_path}")
        elif args.command == "export":
            for path in export_experiment(args.run_dir, args.policy, args.output_dir, strategy=args.strategy):
                print(path)
        elif args.command == "status":
            print(json.dumps([s.model_dump(mode="json") for s in status(args.run_dir)], indent=2))
        else:
            paths = run(args.run_dir, online=args.online, retry=args.retry)
            print(
                f"Available {len(paths)} reports; inspect status for partial/failed attempts. "
                "Scientific acceptance pending."
            )
    except (ValueError, OSError, LlmOfflineMiss) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
