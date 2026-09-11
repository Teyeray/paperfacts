"""The workflow: parser -> adapter -> disk.

``build_parser`` is the single point where "am I on a workstation or a server" is decided, so all four
combinations are pinned down here. ``parse_document`` runs against a fake parser that emits prepared
native output into tmp_path, so none of these tests need mineru, paddleocr, a subprocess or a network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from paperfacts.adapters import ADAPTERS
from paperfacts.config import Settings
from paperfacts.models import BACKENDS, Backend, DocumentInput, ParsedArtifact, RawParseOutput
from paperfacts.models.geometry import DocumentGeometry
from paperfacts.parsers.base import ParserError
from paperfacts.parsers.http_parser import MinerUHttpParser, PaddleHttpParser
from paperfacts.parsers.subprocess_parser import SubprocessParser
from paperfacts.storage.paths import DataLayout
from paperfacts.workflow import PARSER_BUILDERS, build_parser, load_artifact, parse_document
from support.factories import RawOutputFactory, paddle_page_entry

# ---- build_parser -------------------------------------------------------------------


def test_build_parser_uses_the_subprocess_runner_when_no_url_is_configured(tmp_path: Path):
    settings = Settings(repo_root=tmp_path, uv_bin="/opt/uv")

    parser = build_parser("mineru", settings)

    assert isinstance(parser, SubprocessParser)
    assert parser.backend == "mineru"
    assert parser.script == tmp_path / "runners/mineru_runner.py"
    assert parser.command_prefix == ("/opt/uv", "run", "--locked", "--script")


def test_build_parser_passes_the_render_dpi_to_the_paddle_runner(tmp_path: Path):
    # Both paths must rasterise at the same DPI or their pixel coordinates are not comparable.
    settings = Settings(repo_root=tmp_path, paddle_render_dpi=144)

    parser = build_parser("paddleocr_vl", settings)

    assert isinstance(parser, SubprocessParser)
    assert parser.script == tmp_path / "runners/paddle_runner.py"
    assert parser.extra_args == ("--dpi", "144")


def test_build_parser_switches_to_http_when_the_mineru_url_is_set():
    parser = build_parser("mineru", Settings(mineru_url="http://gpu01:8000"))

    assert isinstance(parser, MinerUHttpParser)
    assert parser.base_url == "http://gpu01:8000"


def test_build_parser_switches_to_http_when_the_paddle_url_is_set():
    parser = build_parser("paddleocr_vl", Settings(paddle_url="http://gpu01:8080", paddle_render_dpi=150))

    assert isinstance(parser, PaddleHttpParser)
    assert parser.base_url == "http://gpu01:8080"
    assert parser.render_dpi == 150


def test_build_parser_treats_each_backend_url_independently():
    # With only mineru_url configured, paddle must still go through a subprocess.
    settings = Settings(mineru_url="http://gpu01:8000")

    assert isinstance(build_parser("mineru", settings), MinerUHttpParser)
    assert isinstance(build_parser("paddleocr_vl", settings), SubprocessParser)


def test_build_parser_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        build_parser("tesseract", Settings())


def test_the_parser_registry_and_the_adapter_registry_cover_the_same_backends():
    """The two registries must stay symmetric: every parseable backend has an adapter.

    A gap on either side means either "the parse succeeded but nothing can convert it" or "this adapter is
    dead code", and both only surface at runtime.
    """
    assert set(PARSER_BUILDERS) == set(ADAPTERS) == set(BACKENDS)


@pytest.mark.parametrize("backend", BACKENDS)
def test_subprocess_parsers_get_the_configured_subprocess_timeout(tmp_path: Path, backend):
    # The timeout has to survive the trip from Settings to the parser; a broken wire silently means None.
    settings = Settings(repo_root=tmp_path, subprocess_timeout_s=123.0)

    parser = build_parser(backend, settings)

    assert isinstance(parser, SubprocessParser)
    assert parser.timeout_s == 123.0


@pytest.mark.parametrize(("backend", "url_field"), [("mineru", "mineru_url"), ("paddleocr_vl", "paddle_url")])
def test_http_parsers_get_the_configured_http_timeout(backend, url_field):
    settings = Settings(**{url_field: "http://gpu01:8000", "http_timeout_s": 45.0})

    parser = build_parser(backend, settings)

    assert isinstance(parser, MinerUHttpParser | PaddleHttpParser)
    assert parser.timeout_s == 45.0


def test_build_parser_keeps_the_url_verbatim_and_lets_the_parser_normalise_it():
    # Settings does not strip trailing slashes; the parser owns that, so it is fixed in one place.
    settings = Settings(mineru_url="http://gpu01:8000/")

    assert settings.mineru_url == "http://gpu01:8000/"
    assert build_parser("mineru", settings).base_url == "http://gpu01:8000"


# ---- parse_document -----------------------------------------------------------------


class FakeParser:
    """Plant prepared native output into out_dir, standing in for an expensive real parser run."""

    def __init__(self, backend: Backend, source_dir: Path, *, cache_hit: bool = False) -> None:
        self.backend = backend
        self.source_dir = source_dir
        self.cache_hit = cache_hit
        self.calls: list[tuple[Path, bool]] = []

    def parse(self, document: DocumentInput, out_dir: Path, *, force: bool = False) -> RawParseOutput:
        import shutil

        self.calls.append((out_dir, force))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(self.source_dir, out_dir)
        return RawParseOutput.load(out_dir, self.backend, cache_hit=self.cache_hit)


@pytest.fixture
def fake_mineru_raw(raw_output_factory: RawOutputFactory, mineru_content_list: list[dict[str, Any]]) -> RawParseOutput:
    return raw_output_factory.mineru(mineru_content_list, dir_name="prepared_mineru")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_root=tmp_path / "data", repo_root=tmp_path)


def install_fake_parser(monkeypatch, parser: FakeParser) -> FakeParser:
    monkeypatch.setattr("paperfacts.workflow.build_parser", lambda backend, settings: parser)
    return parser


def test_parse_document_writes_artifact_markdown_and_sources(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    layout = DataLayout(settings.data_root)

    artifact, report = parse_document(document, "mineru", settings)

    assert layout.artifact_path(document.document_id, "mineru").is_file()
    assert layout.markdown_path(document.document_id, "mineru").read_text(encoding="utf-8") == artifact.markdown
    sources = json.loads(layout.sources_path(document.document_id, "mineru").read_text(encoding="utf-8"))
    assert len(sources) == len(artifact.blocks)
    assert sources[0]["source_id"] == artifact.blocks[0].source_id
    assert report.artifact_path == layout.artifact_path(document.document_id, "mineru")


def test_parse_document_report_summarises_the_artifact(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))

    artifact, report = parse_document(document, "mineru", settings)

    assert report.backend == "mineru"
    assert report.backend_version == "3.4.5"
    assert report.cache_hit is False
    assert report.page_count == artifact.page_count == 2
    assert report.block_count == len(artifact.blocks) == 11
    assert report.type_counts == artifact.type_counts()
    assert report.runtime_s >= 0.0


def test_parse_document_propagates_the_cache_hit_flag(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir, cache_hit=True))

    _, report = parse_document(document, "mineru", settings)

    assert report.cache_hit is True


def test_parse_document_hands_the_raw_directory_from_the_layout_to_the_parser(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    parser = install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))

    parse_document(document, "mineru", settings, force=True)

    expected = DataLayout(settings.data_root).raw_dir(document.document_id, "mineru")
    assert parser.calls == [(expected, True)]


def test_parse_document_written_artifact_round_trips(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))

    artifact, _ = parse_document(document, "mineru", settings)

    assert ParsedArtifact.read(DataLayout(settings.data_root).artifact_path(document.document_id, "mineru")) == artifact


def test_parse_document_geometry_comes_from_the_pdf_not_from_meta(
    monkeypatch,
    document: DocumentInput,
    settings: Settings,
    fake_mineru_raw: RawParseOutput,
    geometry: DocumentGeometry,
):
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))

    artifact, _ = parse_document(document, "mineru", settings)

    assert artifact.pages == geometry.pages


def test_parse_document_lets_parser_errors_surface(monkeypatch, document: DocumentInput, settings: Settings):
    class ExplodingParser:
        backend: Backend = "mineru"

        def parse(self, document: DocumentInput, out_dir: Path, *, force: bool = False) -> RawParseOutput:
            raise ParserError("mineru", "run", "exit code 1")

    monkeypatch.setattr("paperfacts.workflow.build_parser", lambda backend, settings: ExplodingParser())

    with pytest.raises(ParserError):
        parse_document(document, "mineru", settings)


def test_parse_document_works_for_the_paddle_backend(
    monkeypatch,
    document: DocumentInput,
    settings: Settings,
    raw_output_factory: RawOutputFactory,
    paddle_page_wrapped: dict[str, Any],
):
    prepared = raw_output_factory.paddle(
        [paddle_page_entry(0, paddle_page_wrapped, (1653, 2339))],
        dir_name="prepared_paddle",
    )
    install_fake_parser(monkeypatch, FakeParser("paddleocr_vl", prepared.out_dir))

    artifact, report = parse_document(document, "paddleocr_vl", settings)

    assert report.backend == "paddleocr_vl"
    assert artifact.backend == "paddleocr_vl"
    assert DataLayout(settings.data_root).markdown_path(document.document_id, "paddleocr_vl").is_file()


# ---- load_artifact ------------------------------------------------------------------


def test_load_artifact_reads_back_what_parse_document_wrote(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    artifact, _ = parse_document(document, "mineru", settings)

    assert load_artifact(document, "mineru", settings) == artifact


def test_load_artifact_raises_file_not_found_before_any_parse(document: DocumentInput, settings: Settings):
    # The message has to name the command that produces the file; a bare path leaves the user stuck.
    with pytest.raises(FileNotFoundError, match="paperfacts parse"):
        load_artifact(document, "mineru", settings)


def test_load_artifact_is_per_backend(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    parse_document(document, "mineru", settings)

    with pytest.raises(FileNotFoundError):
        load_artifact(document, "paddleocr_vl", settings)


# ---- Write ordering ---------------------------------------------------------------------


def test_artifact_json_is_the_last_of_the_three_files_to_be_written(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    """artifact.json must be written last.

    ``load_artifact`` looks only at artifact.json, so its existence is the claim that parsed/ is complete.
    Written before markdown and sources, one interrupted run leaves an artifact that loads fine while its
    companion files are missing or stale. This records the actual write order rather than inspecting the
    wreckage afterwards.
    """
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    layout = DataLayout(settings.data_root)
    tracked = {
        layout.markdown_path(document.document_id, "mineru"): "markdown",
        layout.sources_path(document.document_id, "mineru"): "sources",
        layout.artifact_path(document.document_id, "mineru"): "artifact",
    }
    order: list[str] = []
    original_write_text = Path.write_text

    def record(self, data, *args, **kwargs):
        if self in tracked:
            order.append(tracked[self])
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", record)

    parse_document(document, "mineru", settings)

    assert order == ["markdown", "sources", "artifact"]


def test_a_crash_before_the_artifact_is_written_leaves_nothing_loadable(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    layout = DataLayout(settings.data_root)

    def explode(self, path: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ParsedArtifact, "write", explode)

    with pytest.raises(OSError, match="disk full"):
        parse_document(document, "mineru", settings)

    assert not layout.artifact_path(document.document_id, "mineru").exists()
    with pytest.raises(FileNotFoundError):
        load_artifact(document, "mineru", settings)


def test_a_crash_while_writing_sources_leaves_no_artifact_json(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    layout = DataLayout(settings.data_root)
    sources_path = layout.sources_path(document.document_id, "mineru")
    original_write_text = Path.write_text

    def explode_on_sources(self, data, *args, **kwargs):
        if self == sources_path:
            raise OSError("disk full")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", explode_on_sources)

    with pytest.raises(OSError, match="disk full"):
        parse_document(document, "mineru", settings)

    assert layout.markdown_path(document.document_id, "mineru").is_file()
    assert not layout.artifact_path(document.document_id, "mineru").exists()


def test_all_three_parsed_files_exist_after_a_successful_run(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    layout = DataLayout(settings.data_root)

    parse_document(document, "mineru", settings)

    assert layout.markdown_path(document.document_id, "mineru").is_file()
    assert layout.sources_path(document.document_id, "mineru").is_file()
    assert layout.artifact_path(document.document_id, "mineru").is_file()


# ---- identity ----------------------------------------------------------------------


def test_parse_document_writes_the_document_identity_first(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    """identity.json exists from the moment the directory does; the library reads nothing else for the
    full sha256 and the display name."""
    from paperfacts.storage.identity import read_identity

    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))

    parse_document(document, "mineru", settings)

    identity = read_identity(DataLayout(settings.data_root), document.document_id)
    assert identity is not None
    assert identity.sha256 == document.sha256
    assert identity.name == document.pdf_path.name
    assert identity.source_path == str(document.pdf_path)
    assert identity.uploaded is False
