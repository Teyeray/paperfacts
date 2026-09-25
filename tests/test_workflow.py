"""The workflow: parser -> adapter -> disk.

``build_parser`` is the single point where "am I on a workstation or a server" is decided, so all four
combinations are pinned down here. ``parse_document`` runs against a fake parser that emits prepared
native output into tmp_path, so none of these tests need mineru, paddleocr, a subprocess or a network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from paperfacts.adapters import render_markdown
from paperfacts.config import Settings
from paperfacts.errors import ParserError
from paperfacts.models import BACKENDS, Backend, DocumentGeometry, DocumentInput, ParsedArtifact, RawParseOutput
from paperfacts.parsers import MinerUHttpParser, PaddleHttpParser, SubprocessParser
from paperfacts.storage import DataLayout, read_identity
from paperfacts.workflow import build_parser, load_artifact, parse_document
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

    def is_cached(self, out_dir: Path) -> bool:
        return (out_dir / "meta.json").is_file()

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


def test_parse_document_writes_artifact_and_markdown(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    layout = DataLayout(settings.data_root)

    artifact, report = parse_document(document, "mineru", settings)

    assert layout.artifact_path(document.document_id, "mineru").is_file()
    markdown = layout.markdown_path(document.document_id, "mineru").read_text(encoding="utf-8")
    assert markdown == render_markdown(artifact.blocks)
    assert f"<!-- source: {artifact.blocks[0].source_id} -->" in markdown
    assert report.artifact_path == layout.artifact_path(document.document_id, "mineru")
    assert report.markdown_path == layout.markdown_path(document.document_id, "mineru")


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


# ---- falling back to the stored artifact when the raw output is gone -----------------


def _drop_raw_output(settings: Settings, document: DocumentInput, backend: Backend = "mineru") -> None:
    import shutil

    shutil.rmtree(DataLayout(settings.data_root).raw_dir(document.document_id, backend))


def test_parse_document_uses_the_stored_artifact_when_the_raw_output_is_gone(
    monkeypatch, caplog, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    parser = install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    first, _ = parse_document(document, "mineru", settings)
    _drop_raw_output(settings, document)
    parser.calls.clear()

    with caplog.at_level("WARNING"):
        artifact, report = parse_document(document, "mineru", settings)

    assert parser.calls == []  # a re-parse would cost GPU minutes for output we already have
    assert artifact == first
    assert report.cache_hit is True
    assert report.from_artifact is True
    assert report.runtime_s == 0.0
    assert "mineru" in caplog.text and "--force" in caplog.text


def test_parse_document_rerenders_the_markdown_when_only_the_artifact_survives(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    artifact, _ = parse_document(document, "mineru", settings)
    markdown_path = DataLayout(settings.data_root).markdown_path(document.document_id, "mineru")
    markdown_path.unlink()
    _drop_raw_output(settings, document)

    parse_document(document, "mineru", settings)

    assert markdown_path.read_text(encoding="utf-8") == render_markdown(artifact.blocks)


def test_parse_document_parses_when_both_the_raw_output_and_the_artifact_are_gone(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    parser = install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    parse_document(document, "mineru", settings)
    layout = DataLayout(settings.data_root)
    layout.artifact_path(document.document_id, "mineru").unlink()
    _drop_raw_output(settings, document)
    parser.calls.clear()

    _, report = parse_document(document, "mineru", settings)

    assert parser.calls == [(layout.raw_dir(document.document_id, "mineru"), False)]
    assert report.from_artifact is False


def test_parse_document_still_re_adapts_when_the_raw_output_is_present(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    # The whole point of keeping raw output: an adapter change must be re-applied from it.
    parser = install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir, cache_hit=True))
    parse_document(document, "mineru", settings)
    layout = DataLayout(settings.data_root)
    layout.artifact_path(document.document_id, "mineru").write_text("{}", encoding="utf-8")
    parser.calls.clear()

    artifact, report = parse_document(document, "mineru", settings)

    assert parser.calls == [(layout.raw_dir(document.document_id, "mineru"), False)]
    assert report.from_artifact is False
    assert ParsedArtifact.read(layout.artifact_path(document.document_id, "mineru")) == artifact


def test_parse_document_force_re_parses_even_with_a_stored_artifact(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    parser = install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    parse_document(document, "mineru", settings)
    _drop_raw_output(settings, document)
    parser.calls.clear()

    _, report = parse_document(document, "mineru", settings, force=True)

    assert parser.calls == [(DataLayout(settings.data_root).raw_dir(document.document_id, "mineru"), True)]
    assert report.from_artifact is False


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
    Written before the markdown, one interrupted run leaves an artifact that loads fine while its
    companion file is missing or stale. This records the actual write order rather than inspecting the
    wreckage afterwards.
    """
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    layout = DataLayout(settings.data_root)
    tracked = {
        layout.markdown_path(document.document_id, "mineru"): "markdown",
        layout.artifact_path(document.document_id, "mineru"): "artifact",
    }
    order: list[str] = []
    original_write_text = Path.write_text

    def record(self, data, *args, **kwargs):
        # The artifact is written atomically, so its content lands in a `.<name>.<hex>.tmp`
        # sibling; resolve that back to the file it becomes.
        target = self if self in tracked else self.with_name(".".join(self.name.split(".")[1:-2]))
        if target in tracked:
            order.append(tracked[target])
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", record)

    parse_document(document, "mineru", settings)

    assert order == ["markdown", "artifact"]


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


def test_a_crash_while_writing_the_markdown_leaves_no_artifact_json(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    layout = DataLayout(settings.data_root)
    markdown_path = layout.markdown_path(document.document_id, "mineru")
    original_write_text = Path.write_text

    def explode_on_markdown(self, data, *args, **kwargs):
        # The Markdown is written atomically, so the write that fails is the one to its temp file beside it.
        if self.parent == markdown_path.parent and markdown_path.name in self.name:
            raise OSError("disk full")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", explode_on_markdown)

    with pytest.raises(OSError, match="disk full"):
        parse_document(document, "mineru", settings)

    assert not layout.artifact_path(document.document_id, "mineru").exists()


def test_both_parsed_files_exist_after_a_successful_run(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    layout = DataLayout(settings.data_root)

    parse_document(document, "mineru", settings)

    assert layout.markdown_path(document.document_id, "mineru").is_file()
    assert layout.artifact_path(document.document_id, "mineru").is_file()


# ---- identity ----------------------------------------------------------------------


def test_parse_document_writes_the_document_identity_first(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    """identity.json exists from the moment the directory does; the library reads nothing else for the
    full sha256 and the display name."""
    install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))

    parse_document(document, "mineru", settings)

    identity = read_identity(DataLayout(settings.data_root), document.document_id)
    assert identity is not None
    assert identity.sha256 == document.sha256
    assert identity.name == document.pdf_path.name
    assert identity.source_path == str(document.pdf_path)
    assert identity.uploaded is False


# ---- a document whose PDF is gone ---------------------------------------------------


def test_parse_document_serves_the_stored_artifact_without_ever_opening_the_pdf(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    """Parsed on another machine, copied here without the PDF: the artifact is enough."""
    parser = install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    first, _ = parse_document(document, "mineru", settings)
    _drop_raw_output(settings, document)

    def explode(_path: Path) -> DocumentGeometry:
        raise AssertionError("the PDF must not be opened when the stored artifact stands in")

    monkeypatch.setattr("paperfacts.workflow.read_geometry", explode)
    gone = document.model_copy(update={"pdf_path": settings.data_root / "elsewhere" / "gone.pdf"})
    parser.calls.clear()

    artifact, report = parse_document(gone, "mineru", settings)

    assert artifact == first
    assert report.from_artifact is True
    assert parser.calls == []


def test_parse_document_without_a_pdf_fails_with_an_honest_parser_error(
    monkeypatch, document: DocumentInput, settings: Settings, fake_mineru_raw: RawParseOutput
):
    # Forced (or simply never parsed): a real parse needs the file, so say which one is missing.
    parser = install_fake_parser(monkeypatch, FakeParser("mineru", fake_mineru_raw.out_dir))
    parse_document(document, "mineru", settings)
    _drop_raw_output(settings, document)
    gone = document.model_copy(update={"pdf_path": settings.data_root / "elsewhere" / "gone.pdf"})
    parser.calls.clear()

    with pytest.raises(ParserError, match="PDF not available") as caught:
        parse_document(gone, "mineru", settings, force=True)

    assert caught.value.stage == "input"
    assert parser.calls == []
