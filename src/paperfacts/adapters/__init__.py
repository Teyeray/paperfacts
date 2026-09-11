"""Adapter layer: parser native output -> SourceBlock -> Markdown with provenance markers.

Only :func:`convert` is exposed publicly; it dispatches to the concrete adapter based on
``raw.backend``. To add a new parser: write an ``adapters/<name>.py`` providing a ``convert``
with the same signature, and register it in :data:`ADAPTERS`.
"""

from __future__ import annotations

from collections.abc import Callable

from paperfacts.adapters import mineru, paddle
from paperfacts.models.artifact import Backend, DocumentInput, ParsedArtifact
from paperfacts.models.geometry import DocumentGeometry
from paperfacts.models.raw_output import RawParseOutput

Adapter = Callable[[RawParseOutput, DocumentInput, DocumentGeometry], ParsedArtifact]

ADAPTERS: dict[Backend, Adapter] = {
    "mineru": mineru.convert,
    "paddleocr_vl": paddle.convert,
}


def convert(raw: RawParseOutput, document: DocumentInput, geometry: DocumentGeometry) -> ParsedArtifact:
    """Convert one parser's native output into the unified artifact."""
    try:
        adapter = ADAPTERS[raw.backend]
    except KeyError as exc:
        raise ValueError(f"no adapter registered for backend={raw.backend!r}") from exc
    return adapter(raw, document, geometry)


__all__ = ["ADAPTERS", "Adapter", "convert"]
