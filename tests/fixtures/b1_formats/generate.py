"""Write the stored files of B1 (main 476b76e; its ``src/`` is production's 947f8f6) into this directory.

Round 2 renames persisted attributes (``target`` -> ``paper``, ``no_tco_film`` -> ``no_samples``) and adds new
ones. Every file production already holds must keep loading, and ``extra="ignore"`` on the models means a
forgotten read alias would load an empty paper record rather than fail. These fixtures are the files as B1
writes them, produced by B1's own code rather than typed by hand, so ``tests/test_b1_formats.py`` can prove
they still load with their paper record after the rename:

- ``lane.json`` / ``lane_paddleocr_vl.json``: both lanes of one TCO document, each with a paper-level record
  under ``"target"``, a matched sample and (PaddleOCR-VL only) a sample the other lane lacks;
- ``lane_no_samples.json``: a lane whose inventory said the paper deposits no film (``"no_tco_film": true``);
- ``report.json``: that document's comparison report, with ``"target"`` scopes and an unmatched sample;
- ``dataset.json``: its dataset payload, with a ``"target"`` quality row;
- ``figures.json``: one document's stored chart readings.

The run is ``run_document`` over the two recorded real parses with the repository's ``config.json`` and a
scripted model (the one ``tests/fixtures/payloads/generate_b0.py`` uses, plus one more PaddleOCR-VL sample), and
``read_document_figures`` with a fake vision client over a generated one-figure parse. The generated PDF is not
byte-stable, so a re-run changes the document ids and artifact digests and nothing else. **Never re-generate
after B1**: the files exist to be the old format. If one must be, run it from a checkout of B1:

    PYTHONPATH=src uv run python tests/fixtures/b1_formats/generate.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parents[2]
for path in (TESTS_DIR, TESTS_DIR / "fixtures" / "payloads"):
    if str(path) not in sys.path:  # run as a script; under pytest ``support`` is importable already
        sys.path.insert(0, str(path))

import generate_b0  # noqa: E402

from paperfacts import workflow  # noqa: E402
from paperfacts.config import Settings  # noqa: E402
from paperfacts.keys import comparison_key_for, extractor_key_for, figure_key_for  # noqa: E402
from paperfacts.models import (  # noqa: E402
    BACKENDS,
    DocumentInput,
    NormalizedBBox,
    PageGeometry,
    ParsedArtifact,
    RawParseOutput,
)
from paperfacts.readings import read_document_figures  # noqa: E402
from paperfacts.storage import DataLayout  # noqa: E402
from support.factories import make_blank_pdf, make_block  # noqa: E402
from support.llm import FakeLlmClient  # noqa: E402
from support.vision import FakeVisionClient, chart_answer  # noqa: E402

OUT = Path(__file__).resolve().parent
# One sample more on the PaddleOCR-VL side than generate_b0 lists, so the report has an unmatched sample.
EXTRA_SAMPLE = ("SnO2:Ta reference", "commercial reference film", {})


def _grounded(system: str, user: str) -> str:
    """generate_b0's answers, but the paper-level value cites the block that quotes it, so it is grounded and
    reaches the dataset's paper-level cells rather than only the audit."""
    answer = generate_b0.respond(system, user)
    match = generate_b0._FIELD.match(user)
    if match is None or match.group(1) != "component":
        return answer
    value = json.loads(answer)["values"][0]
    excerpts = user.split("Excerpts (Markdown with provenance markers):", 1)[1]
    parts = generate_b0._SOURCE.split(excerpts)[1:]
    quoting = [sid for sid, text in zip(parts[::2], parts[1::2], strict=True) if value["value_raw"] in text]
    value["source_ids"] = quoting[:1]
    return json.dumps({"values": [value]})


def _with_unmatched(system: str, user: str) -> str:
    answer = _grounded(system, user)
    if user.startswith("Paper excerpts") and generate_b0._lane(user) == "paddleocr_vl":
        inventory = json.loads(answer)
        cited = inventory["samples"][0]["source_ids"]
        sid, label, conditions = EXTRA_SAMPLE
        inventory["samples"].append({"sample_id": sid, "label": label, "conditions": conditions, "source_ids": cited})
        return json.dumps(inventory)
    return answer


def _no_film(system: str, user: str) -> str:
    if user.startswith("Paper excerpts"):
        return json.dumps({"samples": [], "no_tco_film": True})
    return _grounded(system, user)


def _run(work_dir: Path, respond) -> tuple[DataLayout, str, str, str]:
    """``run_document`` once over the recorded parses: the layout, the document id and both keys."""
    settings = Settings.from_env(
        {
            "PAPERFACTS_DATA_ROOT": str(work_dir / "data"),
            "PAPERFACTS_LLM_API_KEY": "sk-b1",
            "PAPERFACTS_FIGURES_ENABLED": "false",
        }
    )
    pages = RawParseOutput.load(generate_b0.FIXTURES["mineru"], "mineru").meta.pages
    pdf = make_blank_pdf(work_dir / "sample.pdf", [(page.width_pt, page.height_pt) for page in pages])
    document = DocumentInput.from_path(pdf)
    layout = DataLayout(settings.data_root)
    for backend in BACKENDS:
        shutil.copytree(generate_b0.FIXTURES[backend], layout.raw_dir(document.document_id, backend))
    real = workflow.build_llm_client(settings)
    real.close()
    fake = FakeLlmClient(respond, model=real.model, temperature=real.temperature, max_tokens=real.max_tokens)
    profile = workflow.load_run_profile(settings)
    with mock.patch.object(workflow, "build_llm_client", lambda _settings: fake):
        workflow.run_document(document, settings, profile)
    return (
        layout,
        document.document_id,
        extractor_key_for(settings, profile),
        comparison_key_for(settings, profile),
    )


def _figures(work_dir: Path) -> Path:
    settings = Settings.from_env(
        {
            "PAPERFACTS_DATA_ROOT": str(work_dir / "data"),
            "PAPERFACTS_LLM_API_KEY": "sk-b1",
            "PAPERFACTS_FIGURES_ENABLED": "true",
        }
    )
    document = DocumentInput.from_path(make_blank_pdf(work_dir / "figure.pdf", [(595, 842)]))
    box = NormalizedBBox(x1=0.1, y1=0.1, x2=0.6, y2=0.5)
    blocks = (
        make_block(page=0, order=0, type="figure", content="a.jpg", bbox=box, document_id=document.document_id),
        make_block(
            page=0,
            order=1,
            type="caption",
            content="Fig. 2 Sheet resistance of the films versus O2 flow.",
            bbox=box,
            document_id=document.document_id,
        ),
    )
    layout = DataLayout(settings.data_root)
    ParsedArtifact(
        document_id=document.document_id,
        backend="mineru",
        backend_version="x",
        pages=(PageGeometry(index=0, width_pt=595, height_pt=842),),
        blocks=blocks,
    ).write(layout.artifact_path(document.document_id, "mineru"))
    profile = workflow.load_run_profile(settings)
    read_document_figures(document, settings, profile, FakeVisionClient(chart_answer()))
    return layout.figures_path(document.document_id, figure_key_for(settings, profile), profile.name)


def generate(work_dir: Path) -> dict[str, Path]:
    """Every fixture's source file, by fixture name."""
    layout, doc, ext, cmp = _run(work_dir / "matched", _with_unmatched)
    empty_layout, empty_doc, empty_ext, _ = _run(work_dir / "no_film", _no_film)
    return {
        "lane.json": layout.extraction_path(doc, "mineru", ext),
        "lane_paddleocr_vl.json": layout.extraction_path(doc, "paddleocr_vl", ext),
        "lane_no_samples.json": empty_layout.extraction_path(empty_doc, "mineru", empty_ext),
        "report.json": layout.comparison_path(doc, ext, cmp),
        "dataset.json": layout.dataset_json_path(doc, ext, cmp),
        "figures.json": _figures(work_dir / "figures"),
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        for name, source in generate(Path(tmp)).items():
            shutil.copyfile(source, OUT / name)
            print(f"wrote {OUT / name}")
