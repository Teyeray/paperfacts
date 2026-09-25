"""PaperFacts: traceable, sample-level measurements extracted from scientific PDFs.

Flat modules, in pipeline order:

- ``models``      page geometry, provenance blocks, the parse artifact, the ``meta.json`` contract
- ``storage``     every on-disk path, atomic writes, the document identity file
- ``threads``     the pipeline's thread pool, which carries the caller's context into every task
- ``pdf``         the only pypdfium2 caller: page geometry and rendering
- ``parsers``     hand a PDF to MinerU / PaddleOCR-VL, as a subprocess or over HTTP
- ``adapters``    native parser output -> blocks -> Markdown with provenance markers
- ``overlay``     block boxes drawn on page images, to check provenance by eye
- ``fields``      the target field table: units, tolerances, bare-number policy
- ``continuation`` paragraphs a page or column break cut in two, linked across the break
- ``prompts``     the extraction and matching prompts
- ``llm``         OpenAI-compatible client with a request cache and one repair round
- ``records``     the model's response schema, the stored records, and the cleaning between them
- ``extract``     the document the model reads and the extraction itself
- ``figures``     a vision model reads property-vs-condition charts; paper-level, never compared
- ``voting``      repeats collapsed within a pass, majority vote across passes
- ``grounding``   the quoted text must occur in the block it cites
- ``normalize``   text folding, number parsing, unit conversion
- ``matching``    which sample in lane A is which sample in lane B
- ``compare``     field-by-field comparison of the two lanes
- ``dataset``     merge source evidence into unique values and export machine-learning tables to Excel
- ``keys``        the cache keys that name stored extractions and comparisons
- ``workflow``    orchestration; ``cli``, ``report`` and ``web`` are thin layers over it
"""
