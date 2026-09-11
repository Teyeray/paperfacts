"""Shared skeleton for both adapters: accumulate blocks, assign source_id, record skipped
blocks, assemble the final artifact.

An adapter's input is a :class:`~paperfacts.models.raw_output.RawParseOutput` (native output
directory + validated meta), and its output is a :class:`~paperfacts.models.artifact.ParsedArtifact`.
Adapters are **pure functions**: they only read files, never call a model, never touch the
network, so a hand-written fixture JSON is enough to test them fully, with no need to install
torch or paddle.
"""

from __future__ import annotations

import logging
from pathlib import Path

from paperfacts.adapters.markdown import build_markdown
from paperfacts.models.artifact import Backend, BlockType, ParsedArtifact, SourceBlock, make_source_id
from paperfacts.models.geometry import DocumentGeometry, NormalizedBBox
from paperfacts.models.raw_output import RawParseOutput

logger = logging.getLogger(__name__)


class BlockCollector:
    """Accumulate SourceBlocks per page, auto-assigning the in-page reading order ``order``
    and ``source_id``.

    A bad box (zero area, flipped) never fails the whole document: the adapter catches the
    ``ValueError`` on the spot and calls :meth:`skip`, recording the reason in ``skipped`` for
    logging and run-notes statistics. This is the compromise between "never silently drop" and
    "never fail the whole document over one bad block". Page-level errors (e.g. an invalid pixel
    size for the whole page in meta) should **not** go through skip: that is a contract
    violation and must be raised directly.
    """

    def __init__(self, document_id: str, backend: Backend) -> None:
        self.document_id = document_id
        self.backend = backend
        self._blocks: list[SourceBlock] = []
        self._next_order: dict[int, int] = {}
        self.skipped: list[str] = []
        self.unknown_labels: dict[str, int] = {}

    def skip(self, *, page: int, raw_label: str | None, reason: str) -> None:
        """Record and warn about one skipped block."""
        entry = f"page={page} label={raw_label}: {reason}"
        self.skipped.append(entry)
        logger.warning("skip block backend=%s %s", self.backend, entry)

    def add(
        self,
        *,
        page: int,
        bbox: NormalizedBBox,
        type: BlockType,
        content: str,
        raw_label: str | None,
        raw_backend_id: str | None = None,
        confidence: float | None = None,
    ) -> SourceBlock:
        order = self._next_order.get(page, 0)
        self._next_order[page] = order + 1
        block = SourceBlock(
            source_id=make_source_id(self.backend, page, order),
            document_id=self.document_id,
            backend=self.backend,
            page=page,
            order=order,
            bbox=bbox,
            type=type,
            content=content,
            raw_label=raw_label,
            raw_backend_id=raw_backend_id,
            confidence=confidence,
        )
        self._blocks.append(block)
        return block

    def note_unknown_label(self, label: str) -> None:
        """A native label that doesn't map to a unified type: count it and warn only on the
        first occurrence, to avoid flooding the log."""
        if label not in self.unknown_labels:
            logger.warning("unknown block label backend=%s label=%r -> unknown", self.backend, label)
        self.unknown_labels[label] = self.unknown_labels.get(label, 0) + 1

    @property
    def blocks(self) -> tuple[SourceBlock, ...]:
        return tuple(self._blocks)


def assemble_artifact(collector: BlockCollector, *, raw: RawParseOutput, geometry: DocumentGeometry) -> ParsedArtifact:
    """Render the Markdown with provenance markers, and package the blocks, page geometry,
    and native output location into a ParsedArtifact."""
    markdown, blocks = build_markdown(collector.blocks)
    artifact = ParsedArtifact(
        document_id=collector.document_id,
        backend=collector.backend,
        backend_version=raw.backend_version,
        pages=geometry.pages,
        markdown=markdown,
        blocks=blocks,
        raw_output_dir=raw.out_dir,
    )
    logger.info(
        "adapted backend=%s doc=%s blocks=%d skipped=%d types=%s",
        artifact.backend,
        artifact.document_id[:16],
        len(blocks),
        len(collector.skipped),
        artifact.type_counts(),
    )
    return artifact


def native_path(raw: RawParseOutput, relative: str) -> Path:
    """Resolve a relative path from meta.json into an absolute path, and confirm the file exists."""
    path = raw.out_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"{raw.backend} native output is missing file: {path}")
    return path


def native_file(raw: RawParseOutput, key: str) -> Path:
    """Look up a native file by its key in ``meta.files``; a missing key means this isn't
    output from the matching runner."""
    relative = raw.meta.files.get(key)
    if relative is None:
        raise FileNotFoundError(f"{raw.backend} meta.json files has no {key!r}: {sorted(raw.meta.files)}")
    return native_path(raw, relative)
