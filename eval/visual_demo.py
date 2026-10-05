"""One-command synthetic lifecycle/export demo. No network, parser or model API is used."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import pypdfium2 as pdfium
from visual_evidence import export_experiment, freeze, run, status

from paperfacts.compare import compare_lanes
from paperfacts.errors import LlmError
from paperfacts.keys import ComparisonOptions, profile_extraction_fingerprint
from paperfacts.llm import LlmResult
from paperfacts.matching import SampleMatch, SampleMatching
from paperfacts.models import DocumentInput, NormalizedBBox, ParsedArtifact, SourceBlock
from paperfacts.pdf import _PDFIUM_LOCK, read_geometry
from paperfacts.profile_loader import load_profile
from paperfacts.records import FieldValue, LaneExtraction, SampleRecord
from paperfacts.storage import write_text_atomic
from paperfacts.visual_adoption import AdoptionPolicy
from paperfacts.visual_evidence import VisualEvidenceReport, visual_prompt
from paperfacts.visual_snapshot import ExperimentSnapshot, build_snapshot


class SyntheticClient:
    offline = True
    model = "synthetic-visual"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def complete_vision(self, *, system: str, user: str, image_png: bytes, refresh: bool = False) -> LlmResult:
        self.calls += 1
        if self.fail and self.calls % 2:
            raise LlmError("synthetic failure")
        observations = [
            dict(
                scope="sample",
                entity="sample",
                sample_raw=sample,
                field="thickness",
                value_raw=value,
                unit_raw="nm",
                condition=None,
                condition_status="not_stated",
                evidence_type="printed",
                basis=f"Synthetic row {sample}, thickness (nm) column",
            )
            for sample, value in (("S1", "10"), ("S2", "10"), ("S3", "~10"), ("S4", "5"))
        ]
        return LlmResult(
            text=json.dumps(dict(outcome="observed", reason="", observations=observations)), usage={}, cached=False
        )


def _pdf(path: Path) -> None:
    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument.new()
        try:
            for number in range(2):
                page = pdf.new_page(480, 360)
                try:
                    lines = [
                        "SYNTHETIC DEMO - NOT SCIENTIFIC EVIDENCE",
                        f"Table {number + 1}: coating thickness (nm)",
                        "S1  10",
                        "S2  10",
                        "S3  approximately 10",
                        "S4  5",
                    ]
                    for index, line in enumerate(lines):
                        obj = pdfium.PdfTextObj(pdfium.raw.FPDFPageObj_NewTextObj(pdf, b"Helvetica", 12), pdf=pdf)
                        encoded = ctypes.create_string_buffer(line.encode("utf-16-le") + b"\0\0")
                        if not pdfium.raw.FPDFText_SetText(obj, ctypes.cast(encoded, ctypes.POINTER(ctypes.c_ushort))):
                            raise ValueError("could not create synthetic PDF text")
                        obj.set_matrix(pdfium.PdfMatrix(e=20, f=325 - index * 40))
                        page.insert_obj(obj)
                    page.gen_content()
                finally:
                    page.close()
            pdf.save(str(path))
        finally:
            pdf.close()


def demo(root: Path) -> dict:
    root = root.resolve()
    if root.exists():
        raise FileExistsError("demo output directory already exists; choose a new name")
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    profile_path = inputs / "demo.json"
    profile_json = dict(
        format=1,
        name="demo",
        title_zh="合成演示",
        maturity="example",
        groups=[dict(name="coating", level="sample", label_zh="涂层")],
        prompt=dict(
            domain_subject="synthetic coatings",
            sample_definition="A sample is one labelled synthetic row.",
            field_scope="Only the synthetic coating.",
        ),
        retrieval=dict(condition_keywords=["coating"], condition_unit_pattern=r"\d\s*nm"),
        fields=[
            dict(
                name="thickness",
                group="coating",
                kind="numeric",
                description="Synthetic coating thickness",
                keywords=["thickness"],
                canonical_unit="nm",
            )
        ],
    )
    write_text_atomic(profile_path, json.dumps(profile_json))
    profile = load_profile(profile_path)
    pdf_path = inputs / "synthetic.pdf"
    _pdf(pdf_path)
    document = DocumentInput.from_path(pdf_path)
    artifacts, lanes = [], []
    box = NormalizedBBox(x1=0, y1=0, x2=1, y2=1)
    for backend, value in (("mineru", "9"), ("paddleocr_vl", "20")):
        blocks = tuple(
            SourceBlock(
                source_id=f"{backend}_p{page}_b0",
                document_id=document.document_id,
                backend=backend,
                page=page,
                order=0,
                bbox=box,
                type="table",
                content="Table: thickness (nm). S1 10; S2 10; S3 approximately 10; S4 5",
            )
            for page in range(2)
        )
        artifact = ParsedArtifact(
            document_id=document.document_id,
            backend=backend,
            backend_version="synthetic",
            pages=read_geometry(pdf_path).pages,
            blocks=blocks,
        )
        artifact.write(inputs / f"{backend}.artifact.json")
        samples = tuple(
            SampleRecord(
                sample_id=sample,
                fields=(
                    FieldValue(
                        field="thickness",
                        value_raw=value if sample == "S2" else "5",
                        unit_raw="nm",
                        source_ids=(blocks[0].source_id,),
                    ),
                )
                if sample in ("S2", "S4")
                else (),
            )
            for sample in ("S1", "S2", "S3", "S4")
        )
        lane = LaneExtraction(
            document_id=document.document_id,
            backend=backend,
            extractor_key="synthetic",
            model="synthetic",
            profile_fingerprint=profile_extraction_fingerprint(profile),
            artifact_sha256=artifact.content_hash(),
            samples=samples,
        )
        lane.write(inputs / f"{backend}.lane.json")
        artifacts.append(artifact)
        lanes.append(lane)
    matching = SampleMatching(
        pairs=tuple(
            SampleMatch(a_id=sample, b_id=sample, confidence=1, method="exact", justification="synthetic identity")
            for sample in ("S1", "S2", "S3", "S4")
        )
    )
    comparison = compare_lanes(*lanes, matching, ComparisonOptions(profile=profile, ambiguous_match_confidence=0.6))
    comparison.write(inputs / "comparison.json")
    config = dict(
        profile_path=str(profile_path),
        data_root=str(root / "cache"),
        documents=[
            dict(
                id=document.document_id,
                pdf_path=str(pdf_path),
                artifact_paths={a.backend: str(inputs / f"{a.backend}.artifact.json") for a in artifacts},
                lane_paths={lane.backend: str(inputs / f"{lane.backend}.lane.json") for lane in lanes},
                comparison_path=str(inputs / "comparison.json"),
            )
        ],
        model=SyntheticClient.model,
        base_url="https://example.invalid/v1",
        render=dict(dpi=72, max_pixels=200000),
        limit=4,
        metadata=dict(
            evaluation_set="synthetic", gold_revision="synthetic-demo-v1", scientific_acceptance="not_established"
        ),
    )
    config_path = inputs / "experiment.json"
    write_text_atomic(config_path, json.dumps(config))
    policy = AdoptionPolicy(revision="synthetic-v1", fields=("thickness",))
    policy_path = inputs / "policy.json"
    write_text_atomic(policy_path, policy.model_dump_json())
    run_root = root / "run"
    freeze(config_path, run_root)
    failing = SyntheticClient(fail=True)
    run(run_root, client=failing)
    first_states = [s.model_dump(mode="json") for s in status(run_root)]
    first_files = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run_root.glob("*.json")}
    repeated = SyntheticClient()
    run(run_root, client=repeated)
    retry = SyntheticClient()
    reports = run(run_root, client=retry, retry=True)
    if not all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in first_files.items()):
        raise AssertionError("retry modified history")
    exported = export_experiment(run_root, policy_path, run_root / "export-current")
    snapshot = ExperimentSnapshot.model_validate_json(exported[0].read_bytes())
    evidence = VisualEvidenceReport.model_validate_json(next(p for p in reports if ".balanced." in p.name).read_bytes())
    stale = build_snapshot(
        document=document,
        lanes=lanes,
        comparison=comparison,
        profile=profile,
        policy=policy,
        pdf_sha256=document.sha256,
        evidence=evidence,
        expected_model=SyntheticClient.model,
        expected_prompt_sha256=hashlib.sha256((visual_prompt(profile, None) + " changed").encode()).hexdigest(),
        expected_strategy="balanced",
    )
    stale.write(root / "stale.snapshot.json")
    summary = dict(
        synthetic_only=True,
        scientific_acceptance="not_established",
        first_states=first_states,
        repeat_requests=repeated.calls,
        retry_requests=retry.calls,
        adopted=sum(c.adopted for c in snapshot.audit.cells),
        refused=sum(not c.adopted for c in snapshot.audit.cells),
        stale_adopted=sum(c.adopted for c in stale.audit.cells) if stale.audit else 0,
        exports=[str(p) for p in exported],
    )
    write_text_atomic(root / "summary.json", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = demo(args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
