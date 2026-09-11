"""PaperFacts: traceable, sample-level measurements extracted from scientific PDFs.

Package layout:

- ``models``       core data models (geometry, source blocks, the unified artifact); pure pydantic
- ``pdf``          page geometry and rendering, and the only place pypdfium2 is called
- ``parsers``      hand a PDF to MinerU / PaddleOCR-VL, as a subprocess or over HTTP
- ``adapters``     native parser output -> SourceBlock -> Markdown with provenance markers
- ``extraction``   prompt the LLM, then clean, validate and ground what it returns
- ``normalization`` deterministic text, number and unit handling
- ``consensus``    sample matching and field-by-field comparison of the two lanes
- ``storage``      the single source of truth for on-disk paths and atomic writes
- ``verification`` bbox overlays for checking provenance by eye
- ``workflow``     orchestration; ``cli`` and ``web`` are thin layers over it
"""
