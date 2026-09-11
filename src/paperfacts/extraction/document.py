"""Build the Markdown that is actually sent to the extraction model.

The parser artifact is optimised for provenance: it keeps every block the parser emitted, including
running headers, page numbers, figure image paths and the bibliography. None of that carries facts, but
all of it costs context and gives the model more chances to cite the wrong block. This module renders a
smaller document from the same artifact, keeping the ``<!-- source: id -->`` markers so citations still
resolve, and reports what it dropped.

The rendered text is *only* an LLM input. The artifact remains the source of truth for the viewer, so the
``markdown[start:end] == content`` invariant is not maintained here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from paperfacts.fingerprint import source_fingerprint
from paperfacts.models.artifact import ParsedArtifact, SourceBlock

logger = logging.getLogger(__name__)

# Block types that never contain an extractable value: page furniture (headers, footers, page numbers,
# "Check for updates" badges) and figure blocks, whose content is the image path, not the figure.
NOISE_TYPES: frozenset[str] = frozenset({"unknown", "figure"})
# Everything from this heading onwards is citations, not results.
_END_SECTION = re.compile(r"^(references|reference|bibliography|literature cited)\b", re.IGNORECASE)
# Measured against a real 10-page paper: 71.9K characters of prompt billed as 21.4K tokens. Rounded down
# so the context guard errs towards over-estimating the prompt.
CHARS_PER_TOKEN = 3.0


@dataclass(frozen=True)
class ExtractionDocument:
    """The text handed to the model, plus what is needed to check its citations afterwards."""

    markdown: str
    blocks: dict[str, str]
    kept_blocks: int
    dropped_blocks: int

    @property
    def estimated_tokens(self) -> int:
        return int(len(self.markdown) / CHARS_PER_TOKEN) + 1


def build_extraction_document(artifact: ParsedArtifact) -> ExtractionDocument:
    """Render ``artifact`` for the extraction prompt, dropping page furniture and the bibliography."""
    kept = _informative_blocks(artifact.blocks)
    lines: list[str] = []
    page: int | None = None
    for block in kept:
        if block.page != page:
            page = block.page
            lines.append(f"<!-- page: {page} -->\n")
        lines.append(f"<!-- source: {block.source_id} -->\n{block.content}\n")
    document = ExtractionDocument(
        markdown="\n".join(lines),
        blocks={block.source_id: block.content for block in kept},
        kept_blocks=len(kept),
        dropped_blocks=len(artifact.blocks) - len(kept),
    )
    logger.info(
        "extraction document backend=%s blocks=%d/%d chars=%d ~tokens=%d",
        artifact.backend,
        document.kept_blocks,
        len(artifact.blocks),
        len(document.markdown),
        document.estimated_tokens,
    )
    return document


def _informative_blocks(blocks: tuple[SourceBlock, ...]) -> list[SourceBlock]:
    kept: list[SourceBlock] = []
    for block in blocks:
        if block.type == "title" and _END_SECTION.match(block.content.lstrip("# ").strip()):
            break  # the bibliography and anything after it
        if block.type in NOISE_TYPES or not block.content.strip():
            continue
        kept.append(block)
    return kept


@cache
def document_fingerprint() -> str:
    """Hash of this module's source.

    Changing how the document is rendered changes what the model sees, so it has to invalidate the
    extraction cache the same way changing the prompt does. Hashing the source means that happens
    automatically instead of depending on someone bumping a constant.
    """
    return source_fingerprint(Path(__file__))
