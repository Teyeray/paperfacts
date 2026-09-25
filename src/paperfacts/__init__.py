"""PaperFacts: traceable, sample-level measurements extracted from scientific PDFs.

Flat modules, in pipeline order:

- ``models``      page geometry, provenance blocks, the parse artifact, the ``meta.json`` contract
- ``storage``     every on-disk path, atomic writes, the document identity file
- ``threads``     the pipeline's thread pool, which carries the caller's context into every task
- ``pdf``         the only pypdfium2 caller: page geometry and rendering
- ``parsers``     hand a PDF to MinerU / PaddleOCR-VL, as a subprocess or over HTTP
- ``adapters``    native parser output -> blocks -> Markdown with provenance markers
- ``overlay``     block boxes drawn on page images, to check provenance by eye
- ``text``        text folding shared by every stage that compares spellings
- ``units``       built-in unit converters and retrieval patterns, and a profile's declared units
- ``fields``      the target field table: units, tolerances, bare-number policy, each attribute's roles
- ``ui_copy``     a profile's Chinese display copy; in no cache key
- ``profile``     a domain profile: groups, fields, prompt slots, retrieval, units
- ``profile_loader`` reads and checks a profile file into those values; in no cache key
- ``continuation`` paragraphs a page or column break cut in two, linked across the break
- ``prompts``     the extraction and matching prompts
- ``llm``         OpenAI-compatible client with a request cache and one repair round
- ``records``     the model's response schema, the stored records, and the cleaning between them
- ``extract``     the document the model reads and the extraction itself
- ``figures``     a vision model reads property-vs-condition charts; paper-level, never compared
- ``readings``    the figures stage on disk: stored readings, the ones shown (and their rows), reading anew
- ``voting``      repeats collapsed within a pass, majority vote across passes
- ``grounding``   the quoted text must occur in the block it cites
- ``normalize``   number parsing, unit conversion
- ``matching``    which sample in lane A is which sample in lane B
- ``compare``     field-by-field comparison of the two lanes
- ``decide``      one dataset cell's verdict: which candidate the cell states, or why it states none
- ``dataset``     merge source evidence into unique values: the machine-learning rows
- ``columns``     what a reader is told about each of those columns; display only, in no cache key
- ``workbook``    the Excel export of those rows; presentation only, in no cache key
- ``keys``        the cache keys that name stored extractions and comparisons
- ``workflow``    orchestration; ``cli``, ``report`` and ``web`` are thin layers over it
- ``batch``       directory batches and offline re-export, each document through ``workflow``
- ``stored``      what is stored for a document and whether it is still current (keys and parse)
"""
