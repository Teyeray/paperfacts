"""Generate Markdown with provenance markers from SourceBlocks (design doc §8).

A parser's native Markdown cannot be mapped back to blocks segment by segment, so we **do not
reuse it** and instead regenerate it from ``SourceBlock``:

.. code-block:: markdown

    <!-- page: 7 -->

    <!-- source: mineru_p7_b12 -->
    At 25 °C, LiFePO4 exhibited an ionic conductivity of 1.2 × 10^-4 S/cm.

    <!-- source: mineru_p7_b13 -->
    ## 3.2 Electrical properties

While generating it, each block's ``markdown_start`` / ``markdown_end`` is filled in to preserve
the invariant:

    ``markdown[block.markdown_start : block.markdown_end] == block.content``

This invariant lets "the LLM cited some span of the Markdown" be traced precisely back to a
source_id, and from there to a page + bbox.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from paperfacts.models.artifact import SourceBlock

SOURCE_MARKER_TEMPLATE = "<!-- source: {source_id} -->"
PAGE_MARKER_TEMPLATE = "<!-- page: {page} -->"
_SOURCE_MARKER_RE = re.compile(r"<!-- source: ([A-Za-z0-9_]+) -->")
BLOCK_SEPARATOR = "\n\n"


def render_block(block: SourceBlock) -> tuple[str, int]:
    """Render one block into a Markdown fragment, returning ``(fragment, content's starting
    offset within the fragment)``.

    The offset lets each type carry its own syntax prefix (``## `` for titles, ``![…](`` for
    figures) while ``build_markdown`` can still record the exact span of ``content``.
    """
    content = block.content
    match block.type:
        case "title":
            prefix = "## "
            return prefix + content, len(prefix)
        case "figure":
            prefix = f"![{block.source_id}]("
            return f"{prefix}{content})", len(prefix)
        case "formula":
            prefix = "$$\n"
            return f"{prefix}{content}\n$$", len(prefix)
        case _:
            # text / table (HTML) / caption / unknown pass through unchanged
            return content, 0


def build_markdown(blocks: Iterable[SourceBlock]) -> tuple[str, tuple[SourceBlock, ...]]:
    """Sort by (page, order) to generate the whole Markdown document, returning the block list
    with spans filled in."""
    ordered = sorted(blocks, key=lambda b: (b.page, b.order))
    parts: list[str] = []
    spanned: list[SourceBlock] = []
    cursor = 0
    current_page: int | None = None

    for block in ordered:
        if block.page != current_page:
            page_chunk = PAGE_MARKER_TEMPLATE.format(page=block.page) + BLOCK_SEPARATOR
            parts.append(page_chunk)
            cursor += len(page_chunk)
            current_page = block.page

        marker = SOURCE_MARKER_TEMPLATE.format(source_id=block.source_id) + "\n"
        body, offset = render_block(block)
        start = cursor + len(marker) + offset
        end = start + len(block.content)
        chunk = marker + body + BLOCK_SEPARATOR
        parts.append(chunk)
        cursor += len(chunk)
        spanned.append(block.with_markdown_span(start, end))

    return "".join(parts), tuple(spanned)


def source_ids_in(markdown: str) -> list[str]:
    """List every source_id in the Markdown in order of appearance (used by the retrieval
    layer and by tests)."""
    return _SOURCE_MARKER_RE.findall(markdown)
