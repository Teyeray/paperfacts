"""Document library: what's under ``data/docs/``, how far each document got, and where uploaded
PDFs live.

The public document_id is the directory name (first 16 hex chars of the sha256, see
:func:`paperfacts.storage.document_key`); the full sha256, display name, and source are read from the
document directory's ``identity.json``.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperfacts.columns import FieldColumn, field_columns
from paperfacts.compare import ComparisonCounts, ComparisonReport
from paperfacts.config import Settings
from paperfacts.dataset import DatasetPayload, DocumentDataset
from paperfacts.keys import comparison_key_for, extractor_key_for, figure_key_for
from paperfacts.kinds import CellValue
from paperfacts.models import BACKENDS, Backend, DocumentInput, ParsedArtifact
from paperfacts.pdf import merge_pdfs, render_page_cached
from paperfacts.profile import DomainProfile
from paperfacts.records import LaneExtraction
from paperfacts.storage import (
    DataLayout,
    DocumentIdentity,
    PartInfo,
    document_key,
    ensure_identity,
    has_cached_parse,
    is_document_key,
    is_runnable,
    mark_uploaded,
    parts_sha256,
    read_identity,
    stored_document,
    stored_pdf,
    write_bytes_atomic,
    write_bytes_if_absent,
)
from paperfacts.stored import article_type, is_finished, stored_comparison, stored_dataset, stored_stages
from paperfacts.ui_copy import article_type_zh
from paperfacts.workflow import Stage, read_lane

logger = logging.getLogger(__name__)


class DocumentSummary(BaseModel):
    """One row, shared by the document list and the detail page."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="directory name (first 16 hex chars of the sha256)")
    name: str
    pdf_available: bool
    runnable: bool = Field(description="a rerun can be queued: there is a PDF, or a stored parse of both lanes")
    parsed: dict[str, bool]
    extracted: dict[str, bool]
    compared: bool
    stages: tuple[Stage, ...] = Field(
        default=(), description="every pipeline stage in execution order, as far as the files on disk show"
    )
    counts: ComparisonCounts | None = None
    uploaded_at: str | None = None
    article_type: str | None = Field(
        default=None, description="what the stored lanes were told the paper is (extract.detect_article_type)"
    )
    article_type_zh: str | None = Field(default=None, description="its label on the page (ui_copy.ARTICLE_TYPE_ZH)")
    parts: tuple[PartInfo, ...] | None = Field(
        default=None, description="an upload with SI: the main text then each SI file, with their page ranges"
    )
    # Set by the route, which knows every served profile; the library knows only its own.
    profiles_done: tuple[str, ...] = Field(
        default=(), description="the served profiles this document is finished under, the one asked about included"
    )


def _file_stamp(path: Path) -> tuple[int, int, int] | None:
    """What changes when a file is replaced, or None when it is absent. The inode is in it because every
    stored file is replaced atomically, by a rename, which always gives it a new one: a same-size rewrite
    within the filesystem's mtime granularity would otherwise leave the stamp unchanged."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_ino, stat.st_mtime_ns, stat.st_size)


class DuplicateDocument(BaseModel):
    """Another document holding the same main text as an upload: the main PDF alone when the upload carried SI,
    or a document merged from it and its SI when the upload is the main PDF alone. The upload goes ahead either
    way (there is no delete); the page only says so."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    name: str
    has_si: bool = Field(description="the other document is the one with SI")


class CorpusRow(BaseModel):
    """One paper on the home view's library-wide table: its selected sample row, and every sample row so
    the table can expand the paper in place without a request per document."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    name: str
    paper_row: dict[str, CellValue]
    sample_count: int
    sample_rows: tuple[dict[str, CellValue], ...] = ()


class CorpusPayload(BaseModel):
    """The home view's table. The field list travels once at the top level rather than on every row --
    it is the same list for every document, because a dataset written under a different field table
    lives under a different extractor_key and is simply not read here."""

    model_config = ConfigDict(frozen=True)

    fields: tuple[FieldColumn, ...] = ()
    rows: tuple[CorpusRow, ...] = ()


class Library:
    """Read-only queries over ``data/docs``, plus upload registration. Cache keys follow the Settings and the
    profile it is built with, which must be the ones the jobs run under."""

    def __init__(self, settings: Settings, profile: DomainProfile) -> None:
        self.settings = settings
        self.layout = DataLayout(settings.data_root)
        # The same key the pipeline writes under, or the browser looks for a file nothing ever wrote.
        self.profile = profile
        self.extractor_key = extractor_key_for(settings, self.profile)
        self.comparison_key = comparison_key_for(settings, self.profile)
        self.figure_key = figure_key_for(settings, self.profile)
        self.columns: tuple[FieldColumn, ...] = field_columns(profile)
        self._counts_cache: dict[Path, tuple[tuple[tuple[int, int, int] | None, ...], ComparisonCounts | None]] = {}
        self._counts_lock = threading.Lock()

    # ---- listing and detail ----------------------------------------------------------------

    def list(self) -> list[DocumentSummary]:
        summaries = [self.summary(key) for key in self.document_ids()]
        return sorted(summaries, key=lambda s: (s.uploaded_at or "", s.name), reverse=True)

    def document_ids(self) -> list[str]:
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
        return keys

    def exists(self, document_id: str) -> bool:
        """Whether this document's directory exists; a malformed id raises ``KeyError`` (also the
        first line of defense against path traversal)."""
        self._require_key(document_id)
        return self.layout.doc_dir(document_id).is_dir()

    def summary(self, document_id: str) -> DocumentSummary:
        """How far processing got. An unseen but well-formed id gets an "empty" summary; existence
        itself is answered by :meth:`exists`, which is what every other route checks."""
        self._require_key(document_id)
        identity = self.identity(document_id)
        stages = stored_stages(
            self.layout,
            document_id,
            extractor_key=self.extractor_key,
            comparison_key=self.comparison_key,
            figure_key=self.figure_key,
            profile=self.profile.name,
            figures_enabled=self.settings.figures_enabled,
        )
        status = {stage.name: stage.status for stage in stages}
        counts = self._counts(document_id)
        pdf = stored_pdf(self.layout, document_id, identity)
        kind = article_type(self.layout, document_id, self.extractor_key)
        return DocumentSummary(
            document_id=document_id,
            name=identity.name if identity else document_id,
            pdf_available=pdf is not None,
            runnable=is_runnable(self.layout, document_id, identity),
            parsed={b: status[f"parse:{b}"] == "done" for b in BACKENDS},
            extracted={b: status[f"extract:{b}"] == "done" for b in BACKENDS},
            compared=counts is not None,
            stages=stages,
            counts=counts,
            uploaded_at=identity.created_at if identity and identity.uploaded else None,
            article_type=kind,
            article_type_zh=article_type_zh(kind),
            parts=identity.parts if identity else None,
        )

    def _counts(self, document_id: str) -> ComparisonCounts | None:
        """The report's tally, parsed once per version of the file.

        The list and every summary need only these few numbers, and a report is the largest file a document
        has; reparsing all of them on every refresh made the library cost grow with the size of every paper.
        """
        path = self.layout.comparison_path(document_id, self.extractor_key, self.comparison_key)
        if not path.is_file():
            return None
        # The artifacts are part of the stamp: a re-parse makes the stored report one of an earlier parse,
        # which stored_comparison refuses, and the cached tally must not outlive that.
        stamp = tuple(
            _file_stamp(candidate)
            for candidate in (
                path,
                *(self.layout.artifact_path(document_id, backend) for backend in BACKENDS),
            )
        )
        with self._counts_lock:
            cached = self._counts_cache.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        report = stored_comparison(self.layout, document_id, self.extractor_key, self.comparison_key)
        counts = report.counts if report is not None else None
        with self._counts_lock:
            self._counts_cache[path] = (stamp, counts)
        return counts

    def identity(self, document_id: str) -> DocumentIdentity | None:
        return read_identity(self.layout, document_id)

    # ---- artifacts -----------------------------------------------------------------------

    def report(self, document_id: str) -> ComparisonReport | None:
        # Through workflow, so a comparison of an earlier parse counts as absent here too.
        return stored_comparison(self.layout, document_id, self.extractor_key, self.comparison_key)

    def extraction(self, document_id: str, backend: Backend) -> LaneExtraction | None:
        # The same read path as the CLI, so the browser never shows a stale grounding or normalisation.
        return read_lane(self.layout, document_id, backend, self.extractor_key, self.profile)

    def dataset(self, document_id: str) -> DatasetPayload | None:
        """The consolidated per-sample table, or ``None`` until the export ran under the current keys.

        Validation happens here, at the disk boundary: a file in the wrong shape raises
        ``ValidationError`` rather than travelling on as an untyped dict. The columns are the profile's, not
        the ones stored with the table: a label or description is display text, which re-keys nothing, so an
        edit to it must show without a re-run.
        """
        # Through workflow, so a table of an earlier parse counts as absent here too.
        dataset = stored_dataset(self.layout, document_id, self.extractor_key, self.comparison_key)
        return None if dataset is None else dataset.model_copy(update={"fields": self.columns})

    def corpus(self) -> CorpusPayload:
        """The library-wide results table: one row per document that has a dataset under the current keys.

        A document without a dataset is absent, not an empty row: the home view shows what has been
        mined, not what is missing.
        """
        rows: list[CorpusRow] = []
        for summary, dataset in self._corpus_entries():
            rows.append(
                CorpusRow(
                    document_id=summary.document_id,
                    name=summary.name,
                    paper_row=dataset.paper_row,
                    sample_count=len(dataset.sample_rows),
                    sample_rows=dataset.sample_rows,
                )
            )
        return CorpusPayload(fields=self.columns if rows else (), rows=tuple(rows))

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
        """The workbook ``run`` wrote for this document under this profile. Unlike the JSON it is not
        key-stamped, so it is whatever the last run produced -- good enough for a download, never for the table
        on screen."""
        path = self.layout.dataset_path(document_id, self.profile.name)
        return path if path.is_file() else None

    def artifact(self, document_id: str, backend: Backend) -> ParsedArtifact | None:
        path = self.layout.artifact_path(document_id, backend)
        return ParsedArtifact.read(path) if path.is_file() else None

    # ---- PDF and DocumentInput --------------------------------------------------------

    def pdf_path(self, document_id: str) -> Path | None:
        return stored_pdf(self.layout, document_id)

    def page_image(self, document_id: str, page: int, *, dpi: int) -> Path:
        """A rendered image of one page (cached). Raises ``FileNotFoundError`` when there's no
        PDF, ``IndexError`` when the page number is out of range."""
        pdf = self.pdf_path(document_id)
        if pdf is None:
            raise FileNotFoundError("This document has no available PDF to render pages from")
        return render_page_cached(pdf, page, dpi=dpi, cache_dir=self.layout.page_cache_dir(document_id, dpi))

    # Which PDF a document runs from and whether it can run at all are path rules, kept in storage.py so the
    # web and a rerun cannot disagree; these are the library's names for them.

    def has_cached_parse(self, document_id: str) -> bool:
        self._require_key(document_id)
        return has_cached_parse(self.layout, document_id)

    def runnable(self, document_id: str) -> bool:
        self._require_key(document_id)
        return is_runnable(self.layout, document_id)

    def finished(self, document_id: str) -> bool:
        self._require_key(document_id)
        return is_finished(
            self.layout, document_id, extractor_key=self.extractor_key, comparison_key=self.comparison_key
        )

    def document(self, document_id: str) -> DocumentInput:
        self._require_key(document_id)
        return stored_document(self.layout, document_id)

    def register_upload(self, filename: str, data: bytes, si: Sequence[tuple[str, bytes]] = ()) -> DocumentInput:
        """Store an uploaded PDF in its document directory (re-uploading the same content is
        idempotent), and return a DocumentInput that's ready to process.

        Identity is written first (the one place the full sha256 gets recorded), then the PDF is
        written **atomically**: dying partway through never leaves a document with a PDF but no
        discoverable sha. The idempotency check doesn't just look at whether the file exists — a
        file left truncated by a previous half-finished write gets repaired too.

        With ``si`` (each SI file's name and bytes, in upload order) the parts are merged into one PDF after the
        main text and the document is identified by its parts (:func:`storage.parts_sha256`). Raises
        :class:`UnreadablePdfError` naming the part pdfium cannot open, and ``ValueError`` for a repeated part.
        """
        name = Path(filename).name
        if si:
            return self._register_merged([(name, data), *((Path(n).name, b) for n, b in si)])
        sha = hashlib.sha256(data).hexdigest()
        pdf = self.layout.source_pdf(sha)
        name = name or f"{document_key(sha)}.pdf"
        document = DocumentInput(document_id=sha, pdf_path=pdf, sha256=sha)
        identity = ensure_identity(self.layout, document, name=name, uploaded=True)
        if not pdf.is_file() or pdf.stat().st_size != len(data):
            write_bytes_atomic(pdf, data)
        identity = mark_uploaded(self.layout, identity, name=name)
        logger.info("registered upload doc=%s name=%s bytes=%d", document_key(sha), filename, len(data))
        # The stored PDF is always source.pdf, so the name a user sees can only come from the identity --
        # the one already on disk, so a re-upload of the same bytes returns exactly what the first did.
        return document.model_copy(update={"display_name": identity.name})

    def _register_merged(self, parts: list[tuple[str, bytes]]) -> DocumentInput:
        """The main text and its SI as one document. A ``source.pdf`` already there is never rewritten, not even
        by a repeat of the same upload: the merge is not byte-deterministic, and the parse caches belong to the
        bytes written first. The parts and their page ranges are recorded only when the identity is created."""
        shas = [hashlib.sha256(data).hexdigest() for _, data in parts]
        if len(set(shas)) != len(shas):
            raise ValueError("the same PDF is attached twice")
        sha = parts_sha256(shas)
        pdf = self.layout.source_pdf(sha)
        document = DocumentInput(document_id=sha, pdf_path=pdf, sha256=sha)
        merged: bytes | None = None
        infos: list[PartInfo] | None = None
        if read_identity(self.layout, sha) is None or not pdf.is_file():
            merged, counts = merge_pdfs([data for _, data in parts])
            starts = [sum(counts[:index]) for index in range(len(counts))]
            infos = [
                PartInfo(name=part_name, sha256=part_sha, first_page=first, pages=pages)
                for (part_name, _), part_sha, first, pages in zip(parts, shas, starts, counts, strict=True)
            ]
        name = parts[0][0] or f"{document_key(sha)}.pdf"
        identity = ensure_identity(self.layout, document, name=name, uploaded=True, parts=infos)
        if merged is not None:
            write_bytes_if_absent(pdf, merged)
        logger.info(
            "registered upload with SI doc=%s name=%s parts=%d bytes=%d",
            document_key(sha),
            name,
            len(parts),
            sum(len(data) for _, data in parts),
        )
        return document.model_copy(update={"display_name": identity.name})

    def duplicate_of(self, document_id: str, main_sha256: str) -> DuplicateDocument | None:
        """The other document holding this upload's main text, if any (see :class:`DuplicateDocument`): for an
        upload with SI, the main PDF's own document; for one without, the first document merged from it."""
        uploaded = self.identity(document_id)
        if uploaded is not None and uploaded.parts:
            main = self.identity(document_key(main_sha256))
            if main is not None and main.sha256 == main_sha256:
                return DuplicateDocument(document_id=document_key(main_sha256), name=main.name, has_si=False)
            return None
        for key in self.document_ids():
            if key == document_id:
                continue
            other = self.identity(key)
            if other is not None and other.parts and other.parts[0].sha256 == main_sha256:
                return DuplicateDocument(document_id=key, name=other.name, has_si=True)
        return None

    # ---- internal -----------------------------------------------------------------------

    @staticmethod
    def _require_key(document_id: str) -> None:
        if not is_document_key(document_id):
            raise KeyError(f"Invalid document_id: {document_id!r}")
