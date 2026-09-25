"""pytest hooks and shared fixtures.

Only things pytest injects implicitly live here; constructors (blank PDFs, SourceBlocks, faked
parser output) live in :mod:`support.factories` and are imported explicitly by test files.

Two ground rules for unit tests:

1. **Never touch a real parser**: PDFs are generated on the fly, parser native output is faked
   by hand, so tests depend on none of mineru / paddleocr / torch / model weights / the network;
2. **Integration tests that need a real environment are skipped by default**: cases marked
   ``@pytest.mark.parser`` only run when ``--run-parser`` is passed, and ``@pytest.mark.e2e`` (a
   headless browser) is deselected unless ``-m`` names it, as in ``pytest -m e2e``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from paperfacts.config import Settings
from paperfacts.llm import OFFLINE_MISSES
from paperfacts.models import DocumentGeometry, DocumentInput
from paperfacts.pdf import read_geometry
from paperfacts.profile import DomainProfile, load_profile, profile_path
from support.factories import FIXTURES_DIR, RawOutputFactory, make_blank_pdf

# ---- Command-line switch: --run-parser ----------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-parser",
        action="store_true",
        default=False,
        help="run integration tests that need a real MinerU / PaddleOCR-VL environment and model weights",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip every parser-tagged case unless --run-parser is passed; leave out every e2e case unless -m names it."""
    if "e2e" not in (config.getoption("markexpr") or ""):
        browser = [item for item in items if "e2e" in item.keywords]
        if browser:
            config.hook.pytest_deselected(items=browser)
            items[:] = [item for item in items if "e2e" not in item.keywords]
    if config.getoption("--run-parser"):
        return
    skip_parser = pytest.mark.skip(reason="needs a real parser environment; pass --run-parser to run it")
    for item in items:
        if "parser" in item.keywords:
            item.add_marker(skip_parser)


# ---- The process-wide offline miss record ---------------------------------------------


@pytest.fixture(autouse=True)
def no_offline_misses_carried_over():
    """Every client reports into one module-level record; a test that misses must not leave its misses to
    the next one's summary."""
    OFFLINE_MISSES.clear()
    yield
    OFFLINE_MISSES.clear()


# ---- The domain profile ---------------------------------------------------------------


@pytest.fixture(scope="session")
def tco_profile() -> DomainProfile:
    """The shipped TCO profile, from the built-in settings rather than the environment, so a developer's
    PAPERFACTS_PROFILE cannot change what a test runs against. Synthetic profiles come from
    :func:`support.profiles.make_profile`."""
    return load_profile(profile_path(Settings()))


# ---- PDF / document / geometry -------------------------------------------------------


@pytest.fixture
def two_page_pdf(tmp_path: Path) -> Path:
    """A two-page (595x842 + 612x792) blank PDF."""
    return make_blank_pdf(tmp_path / "sample.pdf")


@pytest.fixture
def document(two_page_pdf: Path) -> DocumentInput:
    """The DocumentInput matching ``two_page_pdf`` (document_id = content sha256)."""
    return DocumentInput.from_path(two_page_pdf)


@pytest.fixture
def geometry(two_page_pdf: Path) -> DocumentGeometry:
    """Page geometry read from that same PDF, guaranteeing the sizes an adapter sees match the
    real PDF."""
    return read_geometry(two_page_pdf)


# ---- Faked parser native output -------------------------------------------------------


@pytest.fixture
def raw_output_factory(tmp_path: Path, document: DocumentInput, geometry: DocumentGeometry) -> RawOutputFactory:
    return RawOutputFactory(tmp_path, document, geometry)


@pytest.fixture
def mineru_content_list() -> list[dict[str, Any]]:
    """A hand-written MinerU content_list fixture (see tests/fixtures/mineru_content_list_min.json)."""
    return json.loads((FIXTURES_DIR / "mineru_content_list_min.json").read_text(encoding="utf-8"))


@pytest.fixture
def paddle_page_wrapped() -> dict[str, Any]:
    """A hand-written PaddleOCR-VL single-page fixture, in ``save_to_json``'s
    ``{"res": {...}}`` wrapped form."""
    return json.loads((FIXTURES_DIR / "paddle_page_min.json").read_text(encoding="utf-8"))
