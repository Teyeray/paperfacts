"""End-to-end integration tests against the real parsers (skipped by default).

Only runs when ``--run-parser`` is passed, because it needs a real MinerU / PaddleOCR-VL
environment and model weights::

    uv run pytest tests/test_parsers_integration.py --run-parser -s

It verifies the one thing unit tests can't cover: **both parsers produce structurally consistent
output for the same real paper** (same page count, non-empty blocks, bboxes all within [0, 1],
unique and resolvable source_ids). Content differences are expected and not asserted on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paperfacts.config import Settings
from paperfacts.models import BACKENDS, META_FILENAME, DocumentInput, ParsedArtifact, ParserMeta
from paperfacts.storage import DataLayout
from paperfacts.workflow import parse_document

# Local corpus of papers (copyrighted PDFs, not checked into git). Any *.pdf under template_files/ will
# do, at any depth. Skip when there is none rather than failing on someone else's machine.
CORPUS_DIR = Path(__file__).resolve().parents[1] / "template_files"

pytestmark = pytest.mark.parser


@pytest.fixture(scope="module")
def corpus_pdf() -> Path:
    candidates = sorted(CORPUS_DIR.rglob("*.pdf"))
    if not candidates:
        pytest.skip(f"local corpus is empty: {CORPUS_DIR}")
    return candidates[0]


@pytest.fixture(scope="module")
def run(
    corpus_pdf: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[dict[str, ParsedArtifact], DataLayout, str]:
    """Run both parsers once each; data lands under tmp_path so it never pollutes the repo's data/."""
    settings = Settings(data_root=tmp_path_factory.mktemp("parser-integration"))
    document = DocumentInput.from_path(corpus_pdf)
    results: dict[str, ParsedArtifact] = {}
    for backend in BACKENDS:
        artifact, report = parse_document(document, backend, settings)
        print(f"[{backend}] pages={report.page_count} blocks={report.block_count} types={report.type_counts}")
        results[backend] = artifact
    return results, DataLayout(settings.data_root), document.document_id


@pytest.fixture(scope="module")
def artifacts(run) -> dict[str, ParsedArtifact]:
    return run[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_meta_json_written_on_disk_satisfies_the_contract(run, backend):
    """The meta.json produced by a real run must validate against ParserMeta.

    A runner can't import the main package, so the contract can only be pinned down by
    validation: when a runner changes a field, the fixtures in unit tests might still be stale,
    and this is the test that catches it immediately.
    """
    _, layout, document_id = run
    meta_path = layout.raw_dir(document_id, backend) / META_FILENAME

    meta = ParserMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))

    assert meta.parser == backend
    assert meta.source.parsed_page_count > 0
    assert len(meta.pages) >= meta.source.parsed_page_count


def test_the_paddle_meta_records_render_pixel_sizes_for_every_parsed_page(run):
    """The Paddle adapter's normalization depends entirely on these two numbers.

    A missing page means that page's bbox can't be computed.
    """
    _, layout, document_id = run
    meta_path = layout.raw_dir(document_id, "paddleocr_vl") / META_FILENAME

    meta = ParserMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))

    assert all(page.width_px and page.height_px and page.json_path for page in meta.pages)


@pytest.mark.parametrize("backend", BACKENDS)
def test_each_backend_produces_blocks(artifacts: dict[str, ParsedArtifact], backend):
    assert artifacts[backend].blocks, f"{backend} produced no blocks at all"


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_bbox_is_normalised(artifacts: dict[str, ParsedArtifact], backend):
    for block in artifacts[backend].blocks:
        assert 0.0 <= block.bbox.x1 < block.bbox.x2 <= 1.0, block.source_id
        assert 0.0 <= block.bbox.y1 < block.bbox.y2 <= 1.0, block.source_id


@pytest.mark.parametrize("backend", BACKENDS)
def test_source_ids_are_unique_and_resolvable(artifacts: dict[str, ParsedArtifact], backend):
    artifact = artifacts[backend]
    ids = [block.source_id for block in artifact.blocks]

    assert len(set(ids)) == len(ids)
    for block in artifact.blocks:
        assert artifact.block(block.source_id) is block


def test_both_backends_agree_on_the_page_count(artifacts: dict[str, ParsedArtifact]):
    # Page count is the one thing that must match exactly between the two backends; a mismatch
    # means one of them dropped or added a page, invalidating every downstream alignment.
    assert artifacts["mineru"].page_count == artifacts["paddleocr_vl"].page_count


def test_both_backends_stay_within_the_page_count(artifacts: dict[str, ParsedArtifact]):
    for backend, artifact in artifacts.items():
        pages = {block.page for block in artifact.blocks}
        assert max(pages) < artifact.page_count, backend
