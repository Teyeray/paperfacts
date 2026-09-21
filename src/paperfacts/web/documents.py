"""Document library: what's under ``data/docs/``, how far each document got, and where uploaded
PDFs live.

The public document_id is the directory name (first 16 hex chars of the sha256, see
:func:`paperfacts.storage.document_key`); the full sha256, display name, and source are read from the
document directory's ``identity.json``.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperfacts.compare import ComparisonCounts, ComparisonReport
from paperfacts.config import Settings
from paperfacts.dataset import CellValue, DatasetPayload, DocumentDataset, FieldColumn
from paperfacts.keys import comparison_key, extractor_key_for, validation_key_for
from paperfacts.models import BACKENDS, Backend, DocumentInput, ParsedArtifact
from paperfacts.pdf import render_page_cached
from paperfacts.records import LaneExtraction
from paperfacts.storage import (
    DataLayout,
    DocumentIdentity,
    document_key,
    ensure_identity,
    is_document_key,
    mark_uploaded,
    read_identity,
    write_bytes_atomic,
)
from paperfacts.validate import ValidationReport
from paperfacts.workflow import read_lane, read_validation

logger = logging.getLogger(__name__)


class DocumentSummary(BaseModel):
    """One row, shared by the document list and the detail page."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="directory name (first 16 hex chars of the sha256)")
    name: str
    pdf_available: bool
    parsed: dict[str, bool]
    extracted: dict[str, bool]
    compared: bool
    counts: ComparisonCounts | None = None
    uploaded_at: str | None = None


class CorpusRow(BaseModel):
    """One paper on the home view's library-wide table: its selected sample row and how many it had."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    name: str
    paper_row: dict[str, CellValue]
    sample_count: int


class CorpusPayload(BaseModel):
    """The home view's table. The field list travels once at the top level rather than on every row --
    it is the same list for every document, because a dataset written under a different field table
    lives under a different extractor_key and is simply not read here."""

    model_config = ConfigDict(frozen=True)

    fields: tuple[FieldColumn, ...] = ()
    rows: tuple[CorpusRow, ...] = ()


class Library:
    """Read-only queries over ``data/docs``, plus upload registration. Cache keys follow the current Settings."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.layout = DataLayout(settings.data_root)
        # The same key the pipeline writes under, or the browser looks for a file nothing ever wrote.
        self.extractor_key = extractor_key_for(settings)
        self.comparison_key = comparison_key()
        # None when the stage is off: the dataset then lives under its two-key name, as it always did.
        self.validation_key = validation_key_for(settings) if settings.vlm_enabled else None

    # ---- listing and detail ----------------------------------------------------------------

    def list(self) -> list[DocumentSummary]:
        root = self.layout.docs_root()
        if not root.is_dir():
            return []
        keys = []
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            if not is_document_key(path.name):
                # a hand-made scratch directory, rsync leftovers, etc.: skip anything that isn't a
                # document, so one stray directory can't take down the whole listing
                logger.warning("ignoring non-document directory under docs/: %s", path.name)
                continue
            keys.append(path.name)
        summaries = [self.summary(key) for key in keys]
        return sorted(summaries, key=lambda s: (s.uploaded_at or "", s.name), reverse=True)

    def exists(self, document_id: str) -> bool:
        """Whether this document's directory exists; a malformed id raises ``KeyError`` (also the
        first line of defense against path traversal)."""
        self._require_key(document_id)
        return self.layout.doc_dir(document_id).is_dir()

    def summary(self, document_id: str) -> DocumentSummary:
        """How far processing got. An unseen but well-formed id gets an "empty" summary; existence
        itself is answered by :meth:`exists`."""
        self._require_key(document_id)
        identity = self.identity(document_id)
        report = self.report(document_id)
        return DocumentSummary(
            document_id=document_id,
            name=identity.name if identity else document_id,
            pdf_available=self.pdf_path(document_id) is not None,
            parsed={b: self.layout.artifact_path(document_id, b).is_file() for b in BACKENDS},
            extracted={b: self.layout.extraction_path(document_id, b, self.extractor_key).is_file() for b in BACKENDS},
            compared=report is not None,
            counts=report.counts if report else None,
            uploaded_at=identity.created_at if identity and identity.uploaded else None,
        )

    def identity(self, document_id: str) -> DocumentIdentity | None:
        return read_identity(self.layout, document_id)

    # ---- artifacts -----------------------------------------------------------------------

    def report(self, document_id: str) -> ComparisonReport | None:
        path = self.layout.comparison_path(document_id, self.extractor_key, self.comparison_key)
        return ComparisonReport.read(path) if path.is_file() else None

    def extraction(self, document_id: str, backend: Backend) -> LaneExtraction | None:
        # The same read path as the CLI, so the browser never shows a stale grounding or normalisation.
        return read_lane(self.layout, document_id, backend, self.extractor_key)

    def validation(self, document_id: str) -> ValidationReport | None:
        """The VLM's verdicts under the current keys; None when the stage is off or has not run."""
        return read_validation(self.layout, document_id, self.settings)

    def dataset(self, document_id: str) -> DatasetPayload | None:
        """The consolidated per-sample table, or ``None`` until the export ran under the current keys.

        Validation happens here, at the disk boundary: a file in the wrong shape raises
        ``ValidationError`` rather than travelling on as an untyped dict.

        With the VLM on, the table is looked for under all three keys first -- the one an export that
        consulted the verdicts wrote -- and then under the two-key name, so a document whose validation
        stage has not run yet still shows the table its comparison produced.
        """
        candidates = [self.layout.dataset_json_path(document_id, self.extractor_key, self.comparison_key)]
        if self.validation_key is not None:
            candidates.insert(
                0,
                self.layout.dataset_json_path(
                    document_id, self.extractor_key, self.comparison_key, self.validation_key
                ),
            )
        for path in candidates:
            if path.is_file():
                return DatasetPayload.model_validate_json(path.read_text(encoding="utf-8"))
        return None

    def corpus(self) -> CorpusPayload:
        """The library-wide results table: one row per document that has a dataset under the current keys.

        A document without a dataset is absent, not an empty row: the home view shows what has been
        mined, not what is missing.
        """
        fields: tuple[FieldColumn, ...] = ()
        rows: list[CorpusRow] = []
        for summary, dataset in self._corpus_entries():
            if not fields:
                fields = dataset.fields
            rows.append(
                CorpusRow(
                    document_id=summary.document_id,
                    name=summary.name,
                    paper_row=dataset.paper_row,
                    sample_count=len(dataset.sample_rows),
                )
            )
        return CorpusPayload(fields=fields, rows=tuple(rows))

    def corpus_datasets(self) -> list[DocumentDataset]:
        """The same documents as :meth:`corpus`, rebuilt as datasets so the whole library can be exported
        as one workbook."""
        return [DocumentDataset.from_payload(dataset) for _, dataset in self._corpus_entries()]

    def _corpus_entries(self) -> Iterator[tuple[DocumentSummary, DatasetPayload]]:
        """Every document that has a usable dataset under the current keys, in library order.

        The corpus spans every document, so one corrupt, unreadable or wrongly shaped file must cost
        exactly that one row -- never the whole table. A single-document read still raises, because
        there the caller asked for *that* file and deserves the error.
        """
        for summary in self.list():
            try:
                dataset = self.dataset(summary.document_id)
            except (OSError, ValidationError):
                logger.warning("ignoring unusable dataset for doc=%s in the corpus listing", summary.document_id)
                continue
            if dataset is None:
                continue
            yield summary, dataset

    def dataset_excel(self, document_id: str) -> Path | None:
        """The workbook ``run`` wrote for this document. Unlike the JSON it is not key-stamped, so it is
        whatever the last run produced -- good enough for a download, never for the table on screen."""
        path = self.layout.dataset_path(document_id)
        return path if path.is_file() else None

    def artifact(self, document_id: str, backend: Backend) -> ParsedArtifact | None:
        path = self.layout.artifact_path(document_id, backend)
        return ParsedArtifact.read(path) if path.is_file() else None

    # ---- PDF and DocumentInput --------------------------------------------------------

    def pdf_path(self, document_id: str) -> Path | None:
        """A web-uploaded source.pdf wins; a CLI-processed document falls back to the original
        path recorded in its identity (only valid on the same machine)."""
        uploaded = self.layout.source_pdf(document_id)
        if uploaded.is_file():
            return uploaded
        identity = self.identity(document_id)
        if identity and identity.source_path and Path(identity.source_path).is_file():
            return Path(identity.source_path)
        return None

    def page_image(self, document_id: str, page: int, *, dpi: int) -> Path:
        """A rendered image of one page (cached). Raises ``FileNotFoundError`` when there's no
        PDF, ``IndexError`` when the page number is out of range."""
        pdf = self.pdf_path(document_id)
        if pdf is None:
            raise FileNotFoundError("This document has no available PDF to render pages from")
        return render_page_cached(pdf, page, dpi=dpi, cache_dir=self.layout.page_cache_dir(document_id, dpi))

    def has_cached_parse(self, document_id: str) -> bool:
        """Whether both lanes' artifacts are on disk, so the pipeline can run without ever opening the PDF."""
        self._require_key(document_id)
        return all(self.layout.artifact_path(document_id, backend).is_file() for backend in BACKENDS)

    def runnable(self, document_id: str) -> bool:
        """Whether ``run_document`` can be asked to process this document at all.

        A PDF is only needed for a real parse. A document parsed on another machine arrives here with both
        artifacts and no PDF, and extraction, comparison and export need nothing else.
        """
        return self.pdf_path(document_id) is not None or self.has_cached_parse(document_id)

    def document(self, document_id: str) -> DocumentInput:
        identity = self.identity(document_id)
        if identity is None:
            raise FileNotFoundError(
                f"Document {document_id} has no identity.json: please re-upload or reprocess via the CLI"
            )
        pdf = self.pdf_path(document_id)
        if pdf is None:
            if not self.has_cached_parse(document_id):
                raise FileNotFoundError(
                    f"Document {document_id} has no available PDF (not a web upload, and the original path is gone)"
                )
            # Both artifacts are stored, so nothing downstream opens the file. A path is carried anyway,
            # pointing at where this document's PDF would live if it had one: the export names its
            # workbook from ``display_name`` (set below), so no path has to be invented to name it.
            pdf = self.layout.source_pdf(identity.sha256)
        return DocumentInput(
            document_id=identity.sha256, pdf_path=pdf, sha256=identity.sha256, display_name=identity.name
        )

    def register_upload(self, filename: str, data: bytes) -> DocumentInput:
        """Store an uploaded PDF in its document directory (re-uploading the same content is
        idempotent), and return a DocumentInput that's ready to process.

        Identity is written first (the one place the full sha256 gets recorded), then the PDF is
        written **atomically**: dying partway through never leaves a document with a PDF but no
        discoverable sha. The idempotency check doesn't just look at whether the file exists — a
        file left truncated by a previous half-finished write gets repaired too.
        """
        sha = hashlib.sha256(data).hexdigest()
        pdf = self.layout.source_pdf(sha)
        name = Path(filename).name or f"{document_key(sha)}.pdf"
        document = DocumentInput(document_id=sha, pdf_path=pdf, sha256=sha)
        identity = ensure_identity(self.layout, document, name=name, uploaded=True)
        if not pdf.is_file() or pdf.stat().st_size != len(data):
            write_bytes_atomic(pdf, data)
        identity = mark_uploaded(self.layout, identity, name=name)
        logger.info("registered upload doc=%s name=%s bytes=%d", document_key(sha), filename, len(data))
        # The stored PDF is always source.pdf, so the name a user sees can only come from the identity --
        # the one already on disk, so a re-upload of the same bytes returns exactly what the first did.
        return document.model_copy(update={"display_name": identity.name})

    # ---- internal -----------------------------------------------------------------------

    @staticmethod
    def _require_key(document_id: str) -> None:
        if not is_document_key(document_id):
            raise KeyError(f"Invalid document_id: {document_id!r}")
