"""Everything about the data directory: the single source of truth for paths, atomic writes, the
document's identity file, and which PDF a stored document runs from.

Nowhere else builds a ``data/...`` path by hand. Directories are organised by document, so ``cd``-ing into
one shows every intermediate state of one paper:

.. code-block:: text

    data/docs/<first 16 hex of sha256>/
    ├── identity.json                 full sha256, display name, origin, the parts of an SI upload; written when
    │                                 the directory is created
    ├── source.pdf                    the uploaded PDF (web uploads only; main text + SI merged into one)
    ├── raw/<backend>/                the parser's native output, plus meta.json
    ├── parsed/<backend>.md           Markdown with <!-- source: id --> markers, for eyeballing
    ├── parsed/<backend>.artifact.json   the complete ParsedArtifact
    ├── facts/<backend>.<extractor_key>.json           one lane's extraction, the model's own wording
    ├── comparisons/<extractor_key>.<comparison_key>.json   the two-lane comparison report
    ├── datasets/<extractor_key>.<comparison_key>.json      the consolidated per-sample table, for the web UI
    ├── figures/<profile>/<figure_key>.json        values a vision model read off charts (opt-in stage)
    ├── crops/page_000.<bbox>.<dpi>dpi.max-<pixels|unlimited>px.png   rendered page regions
    ├── overlays/<backend>/page_000.png                bbox overlays
    └── pages/<dpi>dpi/page_000.png                    page renders for the web viewer

    data/llm_cache/<sha256>.json      content-addressed cache of LLM requests, shared across documents

VisualEvidenceLayout takes an explicit ``output/visual-evidence/<YYYY-MM-DD>-<run-id>/`` root and keeps
``manifest.json`` and ``<document key>.<strategy>.json`` there. Its naming and retention rules live in
``eval/README.md``; computing paths never creates a run directory or removes an older run.
Experiment crops add ``.pdf-<actual PDF sha256>`` before ``.png`` in the existing crops directory, since
the byte identity of a saved SI merge can differ while the logical document ID stays the same.

A file's existence means its content is complete. ``meta.json`` earns that by being written last into a
staging directory; every other stored file (artifacts, extractions, comparisons, datasets, readings,
Markdown, overlays, page renders, ``identity.json``, the uploaded PDF) is written atomically (a temp file in
the same directory, then ``os.replace``), because the web server reads them while a job may be writing them
and a run killed mid-write must leave the previous file, not a torn one.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    # Annotations only: `models` writes its artifacts through this module's atomic helpers, so importing it
    # at runtime here would be a cycle.
    from paperfacts.models import Backend, DocumentInput

DOC_DIR_ID_LENGTH = 16
# Prefixed into the material of parts_sha256, so the id of a merged upload can never collide with a file's own
# sha256 and a later change of the scheme gets a new prefix rather than silently re-identifying documents.
PARTS_ID_PREFIX = b"paperfacts-parts-v1\n"


def document_key(sha256: str) -> str:
    """The document directory name and the public short id: the first 16 hex characters of the sha256."""
    return sha256[:DOC_DIR_ID_LENGTH]


def parts_sha256(hex_shas: Sequence[str]) -> str:
    """The identity sha256 of a document uploaded with SI: ``sha256(PARTS_ID_PREFIX + "\\n".join(hex_shas))``, the
    main part's sha256 first, then each SI part's in upload order.

    A single-file upload is identified by its bytes; a merged one cannot be, because pypdfium2's save is not
    byte-deterministic (pdfium writes a fresh trailer ``/ID`` per save -- measured: three saves of the same
    parts, three hashes), so hashing the merged bytes would mint a new document on every upload of the same
    pair. The parts' own hashes are stable, and their order matters because the page order does.
    """
    if not hex_shas:
        raise ValueError("a document is made of at least one part")
    return hashlib.sha256(PARTS_ID_PREFIX + "\n".join(hex_shas).encode("ascii")).hexdigest()


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

    def figures_path(self, document_id: str, figure_key: str, profile: str) -> Path:
        # Per profile: two profiles with the same chart slots and figure fields share a figure_key, and one file
        # would be re-tagged by whichever wrote it last.
        return self.figures_dir(document_id, profile) / f"{figure_key}.json"

    def figures_dir(self, document_id: str, profile: str) -> Path:
        return self.legacy_figures_dir(document_id) / profile

    def legacy_figures_dir(self, document_id: str) -> Path:
        """Where readings were stored before they were kept per profile, all of them under the TCO profile."""
        return self.doc_dir(document_id) / "figures"

    def legacy_figures_path(self, document_id: str, figure_key: str) -> Path:
        return self.legacy_figures_dir(document_id) / f"{figure_key}.json"

    def crops_dir(self, document_id: str) -> Path:
        return self.doc_dir(document_id) / "crops"

    def crop_path(
        self,
        document_id: str,
        page: int,
        bbox_key: str,
        dpi: int,
        *,
        max_pixels: int | None = None,
        source_pdf_sha256: str | None = None,
    ) -> Path:
        # Not per profile, unlike figures_path: a crop is pixels cut out of a page, and which fields a profile
        # wants out of it changes nothing about the bytes. Two profiles asking for the same box share one file.
        # Even an unlimited render gets a suffix: legacy files never recorded their limit and cannot be reused.
        limit = "unlimited" if max_pixels is None else str(max_pixels)
        if source_pdf_sha256 is not None and (
            len(source_pdf_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_pdf_sha256)
        ):
            raise ValueError("source_pdf_sha256 must be a lowercase SHA-256 digest")
        source = f".pdf-{source_pdf_sha256}" if source_pdf_sha256 is not None else ""
        return self.crops_dir(document_id) / f"page_{page:03d}.{bbox_key}.{dpi}dpi.max-{limit}px{source}.png"

    def overlay_dir(self, document_id: str, backend: Backend) -> Path:
        return self.doc_dir(document_id) / "overlays" / backend

    def page_cache_dir(self, document_id: str, dpi: int) -> Path:
        return self.doc_dir(document_id) / "pages" / f"{dpi}dpi"

    def llm_cache_dir(self) -> Path:
        return self.root / "llm_cache"

    def dataset_path(self, document_id: str, profile_name: str) -> Path:
        # Named after the profile, so one document run under two profiles keeps both workbooks. A workbook is
        # not key-stamped: it is whatever the profile's last run wrote.
        return self.doc_dir(document_id) / "exports" / f"{profile_name}.xlsx"

    def dataset_json_path(self, document_id: str, extractor_key: str, comparison_key: str) -> Path:
        # Keyed like comparison_path: a dataset built with another model or field table is a different
        # file, so the browser can never be served a consolidated table the current settings disown.
        return self.doc_dir(document_id) / "datasets" / f"{extractor_key}.{comparison_key}.json"

    def batch_dataset_path(self, profile_name: str) -> Path:
        return self.root / "exports" / f"{profile_name}.xlsx"


@dataclass(frozen=True)
class VisualEvidenceLayout:
    """Paths under one explicit, ignored experiment run directory; never creates directories."""

    root: Path

    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def report_path(self, document_id: str, strategy: str) -> Path:
        if len(document_id) not in (DOC_DIR_ID_LENGTH, 64) or any(c not in "0123456789abcdef" for c in document_id):
            raise ValueError("document_id must be a lowercase SHA-256 or 16-character document key")
        if strategy not in ("risk_only", "balanced"):
            raise ValueError("strategy must be risk_only or balanced")
        return self.root / f"{document_key(document_id)}.{strategy}.json"

    def attempt_report_path(self, document_id: str, strategy: str, attempt: int) -> Path:
        if attempt < 1:
            raise ValueError("attempt must be positive")
        base = self.report_path(document_id, strategy)
        return base if attempt == 1 else base.with_suffix(f".retry-{attempt:04d}.json")

    def state_path(self, document_id: str, strategy: str, attempt: int) -> Path:
        return self.attempt_report_path(document_id, strategy, attempt).with_suffix(".state.json")

    def state_paths(self, document_id: str, strategy: str) -> tuple[Path, ...]:
        base = self.report_path(document_id, strategy)
        return tuple(sorted(self.root.glob(f"{base.stem}*.state.json")))

    def snapshot_path(self, document_id: str) -> Path:
        return self.root / f"{self.report_path(document_id, 'balanced').name.split('.')[0]}.snapshot.json"

    def workbook_path(self) -> Path:
        return self.root / "dataset.xlsx"


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


def write_bytes_if_absent(path: Path, data: bytes) -> bool:
    """Write ``data`` to ``path`` atomically unless a file is already there; whether this call wrote it.

    The temp file is hard-linked into place, which fails when the target exists, so of two concurrent writers
    the first wins and the second leaves the first's bytes alone. For a merged upload's ``source.pdf``: the
    merge is not byte-deterministic, and the parse caches belong to the bytes that were written first; and for a
    new ``identity.json`` (:func:`ensure_identity`). The data root must be on a filesystem with hard links: on one
    without (FAT, some network or FUSE mounts) ``os.link`` raises ``OSError`` and nothing is written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        return True
    finally:
        tmp.unlink(missing_ok=True)


def write_text_atomic(path: Path, text: str) -> None:
    write_atomic(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


# ---- Document identity -------------------------------------------------------------------------------


class PartInfo(BaseModel):
    """One of the PDFs an SI upload was merged from, and where its pages sit in ``source.pdf``."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="the part's uploaded filename")
    sha256: str = Field(min_length=64, max_length=64, description="the part's own bytes")
    first_page: int = Field(ge=0, description="0-based index of its first page in the merged PDF")
    pages: int = Field(ge=0)


class DocumentIdentity(BaseModel):
    """What the directory name alone cannot tell: the full sha256, a display name, and where the PDF came from.

    ``sha256`` is the PDF's content hash for a single-file upload or a CLI document, and :func:`parts_sha256`
    of the parts for an upload with SI (then ``parts`` records each part's own hash). It is written once, when
    the directory is created, and ``parts`` is never back-filled: a document created before SI uploads existed,
    or by the CLI, has none.
    """

    model_config = ConfigDict(frozen=True)

    sha256: str = Field(min_length=64, max_length=64)
    name: str = Field(description="the uploaded filename, or the PDF filename given on the CLI")
    source_path: str | None = Field(default=None, description="original CLI path; stale after a machine change")
    uploaded: bool = Field(default=False, description="a web upload, so source.pdf exists in the directory")
    created_at: str
    parts: tuple[PartInfo, ...] | None = Field(
        default=None, description="the main text then each SI file of an upload with SI; absent otherwise"
    )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_identity(layout: DataLayout, document_id: str) -> DocumentIdentity | None:
    path = layout.identity_path(document_id)
    if not path.is_file():
        return None
    return DocumentIdentity.model_validate_json(path.read_text(encoding="utf-8"))


def _identity_json(identity: DocumentIdentity) -> str:
    # ``parts`` is left out of a document that has none, so the file of an ordinary document reads as before.
    exclude = None if identity.parts is not None else {"parts"}
    return identity.model_dump_json(indent=2, exclude=exclude)


def write_identity(layout: DataLayout, identity: DocumentIdentity) -> DocumentIdentity:
    write_text_atomic(layout.identity_path(identity.sha256), _identity_json(identity))
    return identity


def ensure_identity(
    layout: DataLayout,
    document: DocumentInput,
    *,
    name: str | None = None,
    uploaded: bool = False,
    parts: Sequence[PartInfo] | None = None,
) -> DocumentIdentity:
    """Idempotent: return the existing identity unchanged, or write a new one. ``parts`` (an upload with SI) is
    recorded at creation only; an existing identity keeps whatever it has.

    The first writer wins (:func:`write_bytes_if_absent`): two concurrent first uploads of the same document both
    find no identity, but only one creates the file, and the other returns that one's record rather than its own,
    so no caller is ever handed an identity that is not the one on disk."""
    existing = read_identity(layout, document.document_id)
    if existing is not None:
        return existing
    identity = DocumentIdentity(
        sha256=document.sha256,
        name=name or document.display_filename,
        source_path=None if uploaded else str(document.pdf_path),
        uploaded=uploaded,
        created_at=_now_iso(),
        parts=None if parts is None else tuple(parts),
    )
    if write_bytes_if_absent(layout.identity_path(document.sha256), _identity_json(identity).encode("utf-8")):
        return identity
    return read_identity(layout, document.document_id) or identity


def mark_uploaded(layout: DataLayout, identity: DocumentIdentity, *, name: str) -> DocumentIdentity:
    """A CLI-created document later uploaded through the web now has a source.pdf: switch to the uploaded name."""
    if identity.uploaded:
        return identity
    return write_identity(layout, identity.model_copy(update={"uploaded": True, "name": name}))


# ---- Stored documents: what a rerun and the web library decide from what is on disk ------------------------
# Runtime imports of `models` below are local for the reason given at the top of this module.


def stored_pdf(layout: DataLayout, document_id: str, identity: DocumentIdentity | None = None) -> Path | None:
    """The PDF a stored document is processed from. A web-uploaded source.pdf wins; a CLI-processed document
    falls back to the original path recorded in its identity, which is only valid on the machine that ran it.
    A caller that already read the identity passes it, so one request reads ``identity.json`` once."""
    uploaded = layout.source_pdf(document_id)
    if uploaded.is_file():
        return uploaded
    identity = identity or read_identity(layout, document_id)
    if identity and identity.source_path and Path(identity.source_path).is_file():
        return Path(identity.source_path)
    return None


def has_cached_parse(layout: DataLayout, document_id: str) -> bool:
    """Whether both lanes' artifacts are on disk, so the pipeline can run without ever opening the PDF."""
    from paperfacts.models import BACKENDS

    return all(layout.artifact_path(document_id, backend).is_file() for backend in BACKENDS)


def is_runnable(layout: DataLayout, document_id: str, identity: DocumentIdentity | None = None) -> bool:
    """Whether :func:`paperfacts.workflow.run_document` can be asked to process this stored document at all.

    A PDF is only needed for a real parse. A document parsed on another machine arrives with both artifacts
    and no PDF, and extraction, comparison and export need nothing else. A caller that already read the
    identity passes it, as with :func:`stored_pdf`.
    """
    return stored_pdf(layout, document_id, identity) is not None or has_cached_parse(layout, document_id)


def stored_document(layout: DataLayout, document_id: str) -> DocumentInput:
    """The :class:`DocumentInput` that reprocesses a stored document. Raises ``FileNotFoundError`` when it
    has no identity, or neither a PDF nor a parse of both lanes."""
    from paperfacts.models import DocumentInput

    identity = read_identity(layout, document_id)
    if identity is None:
        raise FileNotFoundError(
            f"Document {document_id} has no identity.json: please re-upload or reprocess via the CLI"
        )
    pdf = stored_pdf(layout, document_id, identity)
    if pdf is None:
        if not has_cached_parse(layout, document_id):
            raise FileNotFoundError(
                f"Document {document_id} has no available PDF (not a web upload, and the original path is gone)"
            )
        # Both artifacts are stored, so nothing downstream opens the file. A path is carried anyway, pointing
        # at where this document's PDF would live if it had one: the export names its workbook from
        # ``display_name`` (set below), so no path has to be invented to name it.
        pdf = layout.source_pdf(identity.sha256)
    return DocumentInput(document_id=identity.sha256, pdf_path=pdf, sha256=identity.sha256, display_name=identity.name)


def document_for_path(layout: DataLayout, path: Path) -> DocumentInput:
    """The :class:`DocumentInput` for a PDF named on the command line or found by a batch.

    A stored document's own ``source.pdf`` (under ``layout``, with an identity) is taken as that document, so a
    batch over ``data/docs`` reprocesses what is there instead of minting a second document: a merged upload's
    bytes do not hash to its id (:func:`parts_sha256`), and even an ordinary one would be re-hashed for nothing.
    Any other path is hashed as :meth:`DocumentInput.from_path` does.
    """
    from paperfacts.models import DocumentInput

    resolved = path.resolve()
    if resolved.name == "source.pdf" and resolved.parent.parent == layout.docs_root().resolve():
        key = resolved.parent.name
        if is_document_key(key) and read_identity(layout, key) is not None:
            return stored_document(layout, key)
    return DocumentInput.from_path(path)
