"""Document identity: everything about "which paper is this" beyond the directory name (the first 16
hex characters of the sha256), stored in the document directory's ``identity.json``.

Every code path that creates a document directory (CLI parsing, web upload) must write this file
first; every consumer afterwards (the web document library, reports) reads only this one place,
instead of falling back to digging the sha256 or filename out of meta.json / the artifact.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperfacts.models import BACKENDS, META_FILENAME, ParserMeta
from paperfacts.models.artifact import DocumentInput
from paperfacts.storage.atomic import write_text_atomic
from paperfacts.storage.paths import DataLayout

logger = logging.getLogger(__name__)
# Before identity.json existed, web uploads recorded the filename and sha in this file; read only
# when recovering a legacy directory.
LEGACY_UPLOAD_META = "source.json"


class DocumentIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    sha256: str = Field(min_length=64, max_length=64)
    name: str = Field(description="Display name: the original uploaded filename, or the PDF filename given on the CLI")
    source_path: str | None = Field(
        default=None, description="Original PDF path from CLI processing; stale after a machine change, fallback only"
    )
    uploaded: bool = Field(
        default=False,
        description="Whether this came via a web upload (source.pdf then exists in the document directory)",
    )
    created_at: str


def read_identity(layout: DataLayout, document_id: str) -> DocumentIdentity | None:
    """Read the identity.

    A directory created before ``identity.json`` existed is recovered from legacy files once and
    persisted; every later read then sees only that file.
    """
    path = layout.identity_path(document_id)
    if path.is_file():
        return DocumentIdentity.model_validate_json(path.read_text(encoding="utf-8"))
    recovered = _recover_legacy_identity(layout, document_id)
    if recovered is not None:
        write_identity(layout, recovered)
        logger.info("recovered identity.json for legacy document dir %s", path.parent.name)
    return recovered


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
            name=name or document.pdf_path.name,
            source_path=None if uploaded else str(document.pdf_path),
            uploaded=uploaded,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        ),
    )


def mark_uploaded(layout: DataLayout, identity: DocumentIdentity, *, name: str) -> DocumentIdentity:
    """Upgrade a CLI-created identity to "uploaded" when the same document is later uploaded via the
    web: the directory now has source.pdf, so the identity switches to the uploaded filename.
    """
    if identity.uploaded:
        return identity
    return write_identity(layout, identity.model_copy(update={"uploaded": True, "name": name}))


def _recover_legacy_identity(layout: DataLayout, document_id: str) -> DocumentIdentity | None:
    """Reconstruct identity from traces left before identity.json existed: a web upload's
    ``source.json``, or either backend's ``raw/<backend>/meta.json``.

    This is a one-time compatibility path, not the normal read order — a successful recovery is
    written out as identity.json, and the normal path only ever looks at that one file.
    """
    doc_dir = layout.doc_dir(document_id)
    legacy_upload = doc_dir / LEGACY_UPLOAD_META
    if legacy_upload.is_file():
        try:
            meta = json.loads(legacy_upload.read_text(encoding="utf-8"))
            return DocumentIdentity(
                sha256=meta["sha256"],
                name=meta.get("name") or f"{doc_dir.name}.pdf",
                uploaded=True,
                created_at=meta.get("uploaded_at") or _mtime_iso(legacy_upload),
            )
        except (ValueError, KeyError, TypeError) as exc:  # bad file: treat as absent, fall through to meta.json
            logger.warning("ignoring unreadable %s in %s: %s", LEGACY_UPLOAD_META, doc_dir.name, exc)
    for backend in BACKENDS:
        meta_path = layout.raw_dir(document_id, backend) / META_FILENAME
        if not meta_path.is_file():
            continue
        try:
            source = ParserMeta.model_validate_json(meta_path.read_text(encoding="utf-8")).source
        except ValidationError as exc:
            logger.warning("ignoring unreadable %s: %s", meta_path, exc)
            continue
        return DocumentIdentity(
            sha256=source.sha256,
            name=Path(source.pdf).name or f"{doc_dir.name}.pdf",
            source_path=source.pdf,
            created_at=_mtime_iso(meta_path),
        )
    return None


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")
