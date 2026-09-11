"""The single source of truth for every on-disk path.

Nowhere else in the codebase is allowed to build a ``data/...`` path by hand. Directories are
organized by **document** rather than by pipeline stage, because most research time goes into
eyeballing one paper's failure case, and ``cd``-ing into one directory shows its entire
intermediate state:

.. code-block:: text

    data/docs/<first 16 hex chars of sha256>/
    ├── raw/
    │   ├── mineru/          native output of runners/mineru.py or mineru-api, plus meta.json
    │   └── paddleocr_vl/    native output of runners/paddle.py or paddleocr-vl-api, plus meta.json
    ├── parsed/
    │   ├── mineru.md                 Markdown annotated with <!-- source: … --> markers
    │   ├── mineru.sources.json       list of SourceBlock (sidecar)
    │   ├── mineru.artifact.json      full ParsedArtifact (the two files above plus page geometry,
    │   │                             one read gets everything)
    │   └── paddleocr_vl.*            same, for the other backend
    ├── facts/
    │   └── <backend>.<extractor_key>.json   one lane's sample-level extraction (LaneExtraction,
    │                                        **verbatim**: value/unit are computed by normalization
    │                                        at read time and stored as null on disk)
    ├── comparisons/
    │   └── <extractor_key>.<comparison_key>.json   cross-lane alignment and comparison report
    │                                               (ComparisonReport)
    ├── overlays/
    │   ├── mineru/page_000.png       bbox overlay, for visual acceptance checks
    │   └── paddleocr_vl/page_000.png
    ├── identity.json                 document identity: sha256, display name, source (whichever
    │                                 path creates the document directory writes this first)
    ├── source.pdf                    original PDF from a web upload (absent for CLI-processed docs)
    └── pages/<dpi>dpi/page_000.png   page render cache for the web viewer

    data/llm_cache/<sha256>.json      content-addressed cache of LLM requests

The directory name is the first 16 hex characters of the sha256: unique enough (collision
probability is negligible) while staying short enough to read in a terminal. The full sha256 is
recorded in the artifact / meta.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paperfacts.models.artifact import Backend

DOC_DIR_ID_LENGTH = 16


def document_key(sha256: str) -> str:
    """The document directory name: the first 16 characters of the content sha256, used externally
    (web, URLs) as the short id.
    """
    return sha256[:DOC_DIR_ID_LENGTH]


def is_document_key(value: str) -> bool:
    return len(value) == DOC_DIR_ID_LENGTH and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class DataLayout:
    """Derives per-document, per-backend paths from a data root. Methods only compute paths; they
    never create directories.
    """

    root: Path

    def doc_dir(self, document_id: str) -> Path:
        return self.root / "docs" / document_key(document_id)

    def identity_path(self, document_id: str) -> Path:
        """Document identity (sha256, display name, source); whoever creates the document directory
        must write this first.
        """
        return self.doc_dir(document_id) / "identity.json"

    # ---- native parser output ---------------------------------------------------------

    def raw_dir(self, document_id: str, backend: Backend) -> Path:
        return self.doc_dir(document_id) / "raw" / backend

    # ---- unified artifacts -------------------------------------------------------------

    def parsed_dir(self, document_id: str) -> Path:
        return self.doc_dir(document_id) / "parsed"

    def markdown_path(self, document_id: str, backend: Backend) -> Path:
        return self.parsed_dir(document_id) / f"{backend}.md"

    def sources_path(self, document_id: str, backend: Backend) -> Path:
        return self.parsed_dir(document_id) / f"{backend}.sources.json"

    def artifact_path(self, document_id: str, backend: Backend) -> Path:
        return self.parsed_dir(document_id) / f"{backend}.artifact.json"

    # ---- extraction & alignment (M2) ---------------------------------------------------
    # extractor_key hashes (model, prompt version, field schema): changing the prompt invalidates
    # it automatically, while parser output is reused as-is.

    def extraction_path(self, document_id: str, backend: Backend, extractor_key: str) -> Path:
        return self.doc_dir(document_id) / "facts" / f"{backend}.{extractor_key}.json"

    def comparison_path(self, document_id: str, extractor_key: str, comparison_key: str) -> Path:
        """comparison_key fingerprints the tolerance and normalization rules: changing the tolerance
        skips re-running the LLM, but the comparison must still be recomputed.
        """
        return self.doc_dir(document_id) / "comparisons" / f"{extractor_key}.{comparison_key}.json"

    def llm_cache_dir(self) -> Path:
        """Content-addressed cache of LLM requests (shared across documents: the same prompt is only
        ever paid for once).
        """
        return self.root / "llm_cache"

    # ---- acceptance-check helpers -------------------------------------------------------

    def overlay_dir(self, document_id: str, backend: Backend) -> Path:
        return self.doc_dir(document_id) / "overlays" / backend

    # ---- web UI (uploads and page-render cache) ----------------------------------------
    # A PDF uploaded via the web is stored in the document directory, keeping everything about one
    # document in one place; CLI-processed documents have no source.pdf.

    def docs_root(self) -> Path:
        return self.root / "docs"

    def source_pdf(self, document_id: str) -> Path:
        return self.doc_dir(document_id) / "source.pdf"

    def page_cache_dir(self, document_id: str, dpi: int) -> Path:
        return self.doc_dir(document_id) / "pages" / f"{dpi}dpi"


def overlay_page_name(page: int) -> str:
    """The overlay filename, matching the runner's ``page_000.png`` style so the two can be compared
    side by side.
    """
    return f"page_{page:03d}.png"
