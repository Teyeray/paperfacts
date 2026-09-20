"""Everything about the data directory: the single source of truth for paths, atomic writes, and the
document's identity file.

Nowhere else builds a ``data/...`` path by hand. Directories are organised by document, so ``cd``-ing into
one shows every intermediate state of one paper:

.. code-block:: text

    data/docs/<first 16 hex of sha256>/
    ├── identity.json                 full sha256, display name, origin; written when the directory is created
    ├── source.pdf                    the uploaded PDF (web uploads only)
    ├── raw/<backend>/                the parser's native output, plus meta.json
    ├── parsed/<backend>.md           Markdown with <!-- source: id --> markers, for eyeballing
    ├── parsed/<backend>.artifact.json   the complete ParsedArtifact
    ├── facts/<backend>.<extractor_key>.json           one lane's extraction, the model's own wording
    ├── comparisons/<extractor_key>.<comparison_key>.json   the two-lane comparison report
    ├── validations/<extractor_key>.<comparison_key>.<validation_key>.json   the VLM's readings and verdicts
    ├── crops/<page>_<bbox>_<dpi>dpi.png                    the page regions the VLM was shown
    ├── datasets/<extractor_key>.<comparison_key>[.<validation_key>].json   the consolidated table, for the web UI
    ├── overlays/<backend>/page_000.png                bbox overlays
    └── pages/<dpi>dpi/page_000.png                    page renders for the web viewer

    data/llm_cache/<sha256>.json      content-addressed cache of LLM requests, shared across documents

A file's existence means its content is complete. ``meta.json`` and ``artifact.json`` earn that by being
written last; page renders, ``identity.json`` and the uploaded PDF are written atomically (a temp file in
the same directory, then ``os.replace``) because concurrent web requests may write them.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from paperfacts.models import Backend, DocumentInput

DOC_DIR_ID_LENGTH = 16


def document_key(sha256: str) -> str:
    """The document directory name and the public short id: the first 16 hex characters of the sha256."""
    return sha256[:DOC_DIR_ID_LENGTH]


def is_document_key(value: str) -> bool:
    return len(value) == DOC_DIR_ID_LENGTH and all(c in "0123456789abcdef" for c in value)


def overlay_page_name(page: int) -> str:
    """Matches the runner's ``page_000.png`` naming so overlays and renders can be compared side by side."""
    return f"page_{page:03d}.png"


@dataclass(frozen=True)
class DataLayout:
    """Computes paths; never creates directories. Accepts a full sha256 or the 16-character key."""

    root: Path

    def docs_root(self) -> Path:
        return self.root / "docs"

    def doc_dir(self, document_id: str) -> Path:
        return self.docs_root() / document_key(document_id)

    def identity_path(self, document_id: str) -> Path:
        return self.doc_dir(document_id) / "identity.json"

    def source_pdf(self, document_id: str) -> Path:
        return self.doc_dir(document_id) / "source.pdf"

    def raw_dir(self, document_id: str, backend: Backend) -> Path:
        return self.doc_dir(document_id) / "raw" / backend

    def markdown_path(self, document_id: str, backend: Backend) -> Path:
        return self.doc_dir(document_id) / "parsed" / f"{backend}.md"

    def artifact_path(self, document_id: str, backend: Backend) -> Path:
        return self.doc_dir(document_id) / "parsed" / f"{backend}.artifact.json"

    def extraction_path(self, document_id: str, backend: Backend, extractor_key: str) -> Path:
        return self.doc_dir(document_id) / "facts" / f"{backend}.{extractor_key}.json"

    def comparison_path(self, document_id: str, extractor_key: str, comparison_key: str) -> Path:
        return self.doc_dir(document_id) / "comparisons" / f"{extractor_key}.{comparison_key}.json"

    def overlay_dir(self, document_id: str, backend: Backend) -> Path:
        return self.doc_dir(document_id) / "overlays" / backend

    def page_cache_dir(self, document_id: str, dpi: int) -> Path:
        return self.doc_dir(document_id) / "pages" / f"{dpi}dpi"

    def llm_cache_dir(self) -> Path:
        return self.root / "llm_cache"

    def dataset_path(self, document_id: str) -> Path:
        return self.doc_dir(document_id) / "dataset.xlsx"

    def validation_path(self, document_id: str, extractor_key: str, comparison_key: str, validation_key: str) -> Path:
        # All three keys: which values were checked follows from the extraction and the comparison, and
        # how they were checked from the validation settings. Change any of them and this is another file.
        return self.doc_dir(document_id) / "validations" / f"{extractor_key}.{comparison_key}.{validation_key}.json"

    def crops_dir(self, document_id: str) -> Path:
        return self.doc_dir(document_id) / "crops"

    def crop_path(self, document_id: str, page: int, bbox_key: str, dpi: int) -> Path:
        """One rendered region. Named by what it shows, not by which value asked for it, so two values cited
        from the same table share one file and one render."""
        return self.crops_dir(document_id) / f"p{page:03d}_{bbox_key}_{dpi}dpi.png"

    def dataset_json_path(
        self, document_id: str, extractor_key: str, comparison_key: str, validation_key: str | None = None
    ) -> Path:
        # Keyed like comparison_path: a dataset built with another model or field table is a different
        # file, so the browser can never be served a consolidated table the current settings disown. A
        # dataset that consulted the VLM's verdicts adds their key too; one that did not keeps the two-key
        # name it always had, so an installation without a VLM configured keeps every filename.
        stem = f"{extractor_key}.{comparison_key}" + (f".{validation_key}" if validation_key else "")
        return self.doc_dir(document_id) / "datasets" / f"{stem}.json"

    def batch_dataset_path(self) -> Path:
        return self.root / "exports" / "paperfacts.xlsx"


# ---- Atomic writes -----------------------------------------------------------------------------------


def write_atomic(path: Path, write: Callable[[Path], None]) -> None:
    """``write(tmp)`` fills a temp file; on success it replaces ``path`` atomically.

    The temp name is unique per call, not per process: web worker threads may write the same target at once
    (two tabs uploading the same PDF), and a shared temp name would make the later ``os.replace`` fail.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    write_atomic(path, lambda tmp: tmp.write_bytes(data))


def write_text_atomic(path: Path, text: str) -> None:
    write_atomic(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


# ---- Document identity -------------------------------------------------------------------------------


class DocumentIdentity(BaseModel):
    """What the directory name alone cannot tell: the full sha256, a display name, and where the PDF came from."""

    model_config = ConfigDict(frozen=True)

    sha256: str = Field(min_length=64, max_length=64)
    name: str = Field(description="the uploaded filename, or the PDF filename given on the CLI")
    source_path: str | None = Field(default=None, description="original CLI path; stale after a machine change")
    uploaded: bool = Field(default=False, description="a web upload, so source.pdf exists in the directory")
    created_at: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_identity(layout: DataLayout, document_id: str) -> DocumentIdentity | None:
    path = layout.identity_path(document_id)
    if not path.is_file():
        return None
    return DocumentIdentity.model_validate_json(path.read_text(encoding="utf-8"))


def write_identity(layout: DataLayout, identity: DocumentIdentity) -> DocumentIdentity:
    write_text_atomic(layout.identity_path(identity.sha256), identity.model_dump_json(indent=2))
    return identity


def ensure_identity(
    layout: DataLayout, document: DocumentInput, *, name: str | None = None, uploaded: bool = False
) -> DocumentIdentity:
    """Idempotent: return the existing identity unchanged, or write a new one."""
    existing = read_identity(layout, document.document_id)
    if existing is not None:
        return existing
    return write_identity(
        layout,
        DocumentIdentity(
            sha256=document.sha256,
            name=name or document.display_filename,
            source_path=None if uploaded else str(document.pdf_path),
            uploaded=uploaded,
            created_at=_now_iso(),
        ),
    )


def mark_uploaded(layout: DataLayout, identity: DocumentIdentity, *, name: str) -> DocumentIdentity:
    """A CLI-created document later uploaded through the web now has a source.pdf: switch to the uploaded name."""
    if identity.uploaded:
        return identity
    return write_identity(layout, identity.model_copy(update={"uploaded": True, "name": name}))
