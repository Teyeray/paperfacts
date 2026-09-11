"""Core data models. All modules import from here, to avoid deep import paths scattered everywhere."""

from paperfacts.models.artifact import (
    BACKENDS,
    Backend,
    BlockType,
    DocumentInput,
    ParsedArtifact,
    SourceBlock,
    make_source_id,
    sha256_of_file,
)
from paperfacts.models.geometry import DocumentGeometry, NormalizedBBox, PageGeometry
from paperfacts.models.raw_output import (
    META_FILENAME,
    PageMeta,
    ParserMeta,
    RawParseOutput,
    SourceMeta,
)

__all__ = [
    "BACKENDS",
    "META_FILENAME",
    "Backend",
    "BlockType",
    "DocumentGeometry",
    "DocumentInput",
    "NormalizedBBox",
    "PageGeometry",
    "PageMeta",
    "ParsedArtifact",
    "ParserMeta",
    "RawParseOutput",
    "SourceBlock",
    "SourceMeta",
    "make_source_id",
    "sha256_of_file",
]
