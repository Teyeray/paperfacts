"""The only contract between a runner and the main package: the type definitions for
``meta.json``, and a handle pointing at one parser run's output.

There are three writers: ``runners/mineru_runner.py`` and ``runners/paddle_runner.py``
(subprocesses that **must not** import this module), and :mod:`paperfacts.parsers.http_parser`
(which constructs :class:`ParserMeta` directly). The readers are :meth:`RawParseOutput.load` and
the two adapters.

A runner never imports the main package, so the contract can only be pinned down by
**validation**, not shared code:

- unit tests run a fixture of real runner output (``tests/fixtures/mineru_real_sample``) through
  this model;
- integration tests (``--run-parser``) run the same validation against a freshly produced ``meta.json``.

If a runner's fields change, one of these two will fail first. ``extra="allow"`` lets a runner
write extra fields it cares about (``backend`` / ``framework`` / ``render_dpi`` / ``vl_backend``
...); this module only defines the parts the main package actually depends on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from paperfacts.models.artifact import Backend

# The contract filename a runner writes in its output directory; it is written last, only on
# full success, so "exists" means "usable".
META_FILENAME = "meta.json"


class SourceMeta(BaseModel):
    """Which PDF was parsed, and which pages were parsed."""

    model_config = ConfigDict(frozen=True, extra="allow")

    pdf: str
    sha256: str = Field(min_length=64, max_length=64)
    page_count: int = Field(ge=0, description="total page count of the PDF")
    parsed_page_count: int = Field(
        ge=0, description="number of pages actually parsed this run (after --start-page/--end-page cropping)"
    )
    page_range: tuple[int | None, int | None] = (0, None)

    @property
    def page_offset(self) -> int:
        """After MinerU crops with ``--start-page``, ``page_idx`` restarts from 0; the real
        page number = page_idx + this offset."""
        return self.page_range[0] or 0


class PageMeta(BaseModel):
    """One page's geometry. The first three fields are written by both runners; pixel size and
    per-page files are only written by the PaddleOCR-VL runner, which renders per page."""

    model_config = ConfigDict(frozen=True, extra="allow", populate_by_name=True)

    index: int = Field(ge=0)
    width_pt: float = Field(gt=0)
    height_pt: float = Field(gt=0)
    width_px: int | None = Field(default=None, gt=0)
    height_px: int | None = Field(default=None, gt=0)
    image: str | None = None
    json_path: str | None = Field(default=None, alias="json")
    markdown_dir: str | None = None


class ParserMeta(BaseModel):
    """Top-level structure of ``meta.json``."""

    model_config = ConfigDict(frozen=True, extra="allow")

    parser: Backend
    parser_version: str = "unknown"
    source: SourceMeta
    pages: tuple[PageMeta, ...] = Field(
        default=(),
        description=(
            "the pages actually parsed this run (not the whole PDF); consistent across both "
            "runners and the HTTP parser — the PaddleOCR-VL adapter iterates over this"
        ),
    )
    files: dict[str, str] = Field(
        default_factory=dict, description="table of native file paths relative to the output directory"
    )
    runner: dict[str, Any] = Field(
        default_factory=dict, description="record of the run environment, for troubleshooting only"
    )


class RawParseOutput(BaseModel):
    """A handle to one parser run's native output directory + validated ``meta.json``.

    This is the handoff object between the parser layer (subprocess / HTTP) and the adapter
    layer: a parser is only responsible for putting ``meta.json`` and its native files into
    ``out_dir``, and an adapter only ever reads from here.
    """

    model_config = ConfigDict(frozen=True)

    backend: Backend
    out_dir: Path
    meta: ParserMeta
    cache_hit: bool = False

    @property
    def backend_version(self) -> str:
        return self.meta.parser_version

    @classmethod
    def load(cls, out_dir: Path, backend: Backend, *, cache_hit: bool = False) -> Self:
        """Read and validate ``out_dir/meta.json``; raises ``ValidationError`` (a subclass of
        ``ValueError``) if the structure doesn't match."""
        meta_path = out_dir / META_FILENAME
        if not meta_path.is_file():
            raise FileNotFoundError(f"{meta_path} does not exist, parser output is incomplete")
        meta = ParserMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        if meta.parser != backend:
            raise ValueError(f"{meta_path} belongs to parser={meta.parser!r}, expected {backend!r}")
        return cls(backend=backend, out_dir=out_dir, meta=meta, cache_hit=cache_hit)
