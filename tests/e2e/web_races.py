"""Browser end-to-end checks for the web frontend: navigation races, polling, layout and keyboard use.

There is no JS test runner (no build step), so the frontend's async behaviour is checked here, in a real
headless Chromium against a real server on a seeded temporary library. Slow responses are simulated by
delaying chosen requests with Playwright route interception. Nothing calls a parser or a model: the job
body is a stub that walks through the pipeline's stages.

Not collected by pytest itself (the file name has no ``test_`` prefix); ``test_e2e.py`` runs it under
``pytest -m e2e``. Run it directly with::

    uv run --with playwright python -m playwright install chromium   # once
    PYTHONPATH=src uv run --with playwright python tests/e2e/web_races.py

It prints one PASS/FAIL line per check and exits non-zero when any check fails. ``--headed`` shows the
browser; ``--only NAME`` runs the checks whose name contains NAME.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import json
import re
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import uvicorn
from playwright.async_api import Page, Route, async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/, for the shared seeding helpers

from paperfacts.compare import ComparisonCounts, ComparisonReport, FieldComparison
from paperfacts.config import Settings
from paperfacts.figures import FigureReading
from paperfacts.matching import SampleMatch, SampleMatching
from paperfacts.models import BACKENDS, NormalizedBBox, PageGeometry, ParsedArtifact
from paperfacts.readings import StoredReadings
from paperfacts.storage import document_key
from paperfacts.web.app import create_app
from paperfacts.web.documents import Library
from paperfacts.web.jobs import Job, JobManager
from paperfacts.workflow import FIGURES_NOT_REQUESTED, stage_names
from support.extraction import make_field, make_lane, make_sample
from support.factories import make_blank_pdf, make_block
from support.profiles import make_one_entity_profile, make_reference_profile, shipped_profile

STAGE_SECONDS = 0.3  # the stub job takes len(stage_names()) * this
# As long as the model's condition prose gets on real papers: the text that pushed the second lane off screen.
LONG_CONDITION = (
    "substrate temperature 300 °C; O2/(Ar+O2) flow ratio 2.5 %; working pressure 0.5 Pa; RF power 120 W;"
    " target-to-substrate distance 7 cm; post-deposition anneal 400 °C for 1 h in forming gas (5 % H2 in N2)"
)
FIELDS = [
    {"name": "thickness", "label": "厚度", "unit": "nm", "scope": "sample", "description": "膜厚"},
    {"name": "sheet_resistance", "label": "方阻", "unit": "Ω/sq", "scope": "sample", "description": "方块电阻"},
    {"name": "transmittance", "label": "透过率", "unit": "%", "scope": "sample", "description": "可见光平均透过率"},
    {"name": "target_purity", "label": "靶材纯度", "unit": "%", "scope": "paper", "description": "靶材纯度"},
]


# ---- a seeded library and a live server --------------------------------------------------------------------


def stub_runner(job: Job, mark: Callable[[str, str, str], None]) -> None:
    for stage in stage_names():
        if stage == "figures" and not job.figures:  # as run_document marks charts nobody asked for
            mark(stage, "skipped", FIGURES_NOT_REQUESTED)
            continue
        mark(stage, "running", "")
        time.sleep(STAGE_SECONDS)
        mark(stage, "done", "stub")


def seed_document(
    library: Library,
    root: Path,
    index: int,
    name: str,
    *,
    samples: int,
    comparisons: int,
    article_type: str | None = None,
    conflicts: int = 0,
    finished: bool = True,
) -> str:
    """A paper with both lanes, a report and (unless ``finished`` is off) a dataset under the library's keys.
    ``article_type`` is what both lanes were told ("review" tags the paper); ``conflicts`` of the ``comparisons`` are
    counted as conflicts in the report's tally, the rest as agreements. An unfinished paper has everything but the
    dataset: the export is the last stage, so the server lists it as not finished under this profile."""
    pdf = make_blank_pdf(root / f"{index}.pdf", [(400.0 + index, 600.0), (400.0 + index, 600.0)])
    document = library.register_upload(name, pdf.read_bytes())
    sha = document.sha256
    for backend in BACKENDS:
        blocks = tuple(
            make_block(page=page, order=order, backend=backend, document_id=sha, content=f"block {page}/{order}")
            for page in (0, 1)
            for order in (0, 1)
        )
        ParsedArtifact(
            document_id=sha,
            backend=backend,
            backend_version="stub",
            pages=tuple(PageGeometry(index=page, width_pt=400.0 + index, height_pt=600.0) for page in (0, 1)),
            blocks=blocks,
        ).write(library.layout.artifact_path(sha, backend))
        lane_samples = [
            make_sample(f"S{n}", [make_field("thickness", f"{100 * n}", unit_raw="nm", value=100.0 * n, unit="nm")])
            for n in range(1, samples + 1)
        ]
        lane = make_lane(backend=backend, samples=lane_samples, document_id=sha, extractor_key=library.extractor_key)
        # A paper with no samples is one that deposits no film of its own, the case with its own message.
        lane = lane.model_copy(update={"no_samples": not samples, "article_type": article_type})
        lane.write(library.layout.extraction_path(sha, backend, library.extractor_key))
    rows = tuple(
        FieldComparison(
            scope=f"sample:S{n % max(samples, 1) + 1}|S{n % max(samples, 1) + 1}",
            field="sheet_resistance" if n % 2 else "thickness",
            condition=LONG_CONDITION if n % 3 == 0 else None,
            status=("agree", "conflict", "missing", "ambiguous")[n % 4],
            missing_in="paddleocr_vl" if n % 4 == 2 else None,
            a=make_field("thickness", "12.5", unit_raw="Ω/sq", value=12.5, unit="ohm/sq", source_ids=["mineru_p1_b1"]),
            b=None
            if n % 4 == 2
            else make_field(
                "thickness", "12.7", unit_raw="Ω/sq", value=12.7, unit="ohm/sq", source_ids=["paddleocr_vl_p1_b0"]
            ),
            detail="relative difference 1.6% exceeds the 1% tolerance for this field" if n % 4 == 1 else "",
        )
        for n in range(comparisons)
    )
    ComparisonReport(
        document_id=sha,
        extractor_key=library.extractor_key,
        comparison_key=library.comparison_key,
        backend_a="mineru",
        backend_b="paddleocr_vl",
        matchings={"sample": SampleMatching()},
        counts=ComparisonCounts(total=comparisons, agree=comparisons - conflicts, conflict=conflicts),
        comparisons=rows,
    ).write(library.layout.comparison_path(sha, library.extractor_key, library.comparison_key))
    if not finished:
        return document_key(sha)
    sample_rows = [
        {
            "sample_id": f"S{n}",
            "sample_label": f"sample {n}",
            "conditions": LONG_CONDITION,
            "available_fields": 3,
            "agree_fields": 2,
            "thickness": 100.0 * n,
            "sheet_resistance": 12.5,
            "transmittance": 88.0,
        }
        for n in range(1, samples + 1)
    ]
    paper_row = {**sample_rows[0], "target_purity": 99.99} if sample_rows else {}
    path = library.layout.dataset_json_path(sha, library.extractor_key, library.comparison_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "document_id": sha,
                "filename": name,
                "fields": FIELDS,
                "paper_row": paper_row,
                "sample_rows": sample_rows,
                "quality_rows": [
                    {
                        "sample_id": row["sample_id"],
                        "field": "thickness",
                        "decision": "agree",
                        "source_ids": "mineru_p1_b1",
                    }
                    for row in sample_rows
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return document_key(sha)


def seed_entity_document(library: Library, root: Path, *, pdf: Path | None = None, name: str = "E 两种实体.pdf") -> str:
    """A paper of a two-entity profile (coatings, the primary one, and the wear tests run on them): each entity
    has a sample named S1, its own rows, records, comparisons and matching; the wear test names the coating it ran
    on (a reference field). Given the ``pdf`` of a paper already seeded under another profile of the same data root,
    it adds this profile's results to that paper and leaves its parse alone."""
    shared = pdf is not None
    pdf = pdf or make_blank_pdf(root / "entities.pdf", [(400.0, 600.0)])
    sha = library.register_upload(name, pdf.read_bytes()).sha256
    for backend in BACKENDS:
        blocks = tuple(
            make_block(page=0, order=order, backend=backend, document_id=sha, content=f"block {order}")
            for order in (0, 1)
        )
        if not shared:
            ParsedArtifact(
                document_id=sha,
                backend=backend,
                backend_version="stub",
                pages=(PageGeometry(index=0, width_pt=400.0, height_pt=600.0),),
                blocks=blocks,
            ).write(library.layout.artifact_path(sha, backend))
        source = [f"{backend}_p0_b1"]
        samples = [
            make_sample("S1", [make_field("coating_thickness", "100", unit_raw="nm", value=100.0, unit="nm")]),
            make_sample("S2", [make_field("coating_thickness", "200", unit_raw="nm", value=200.0, unit="nm")]),
            make_sample(
                "S1",
                [
                    make_field("test_temperature", "300", unit_raw="℃", value=300.0, unit="℃", source_ids=source),
                    make_field("wear_mode", "sliding" if backend == "mineru" else "rolling", source_ids=source),
                    make_field("tested_coating", "S1", source_ids=source),
                ],
            ).model_copy(update={"entity": "wear_test"}),
        ]
        samples[:2] = [sample.model_copy(update={"entity": "coating"}) for sample in samples[:2]]
        make_lane(backend=backend, samples=samples, document_id=sha, extractor_key=library.extractor_key).write(
            library.layout.extraction_path(sha, backend, library.extractor_key)
        )

    def matched(*ids: str) -> SampleMatching:
        return SampleMatching(
            pairs=tuple(SampleMatch(a_id=i, b_id=i, confidence=1.0, justification="", method="exact") for i in ids)
        )

    field = make_field("wear_mode", "sliding", source_ids=["mineru_p0_b1"])
    ComparisonReport(
        document_id=sha,
        extractor_key=library.extractor_key,
        comparison_key=library.comparison_key,
        backend_a="mineru",
        backend_b="paddleocr_vl",
        matchings={"coating": matched("S1", "S2"), "wear_test": matched("S1")},
        counts=ComparisonCounts(total=2, agree=1, conflict=1),
        comparisons=(
            FieldComparison(scope="coating:S1|S1", field="coating_thickness", status="agree", a=field, b=field),
            FieldComparison(scope="wear_test:S1|S1", field="wear_mode", status="conflict", a=field, b=field),
        ),
    ).write(library.layout.comparison_path(sha, library.extractor_key, library.comparison_key))
    identity = {"document_id": sha, "filename": "E 两种实体.pdf", "available_fields": 2, "agree_fields": 1}
    coatings = [
        {
            **identity,
            "entity": "coating",
            "sample_id": f"S{n}",
            "precursor_purity": 99.9,
            "coating_thickness": 100.0 * n,
        }
        for n in (1, 2)
    ]
    test = {
        **identity,
        "entity": "wear_test",
        "sample_id": "S1",
        "precursor_purity": 99.9,
        "test_temperature": 300.0,
        "tested_coating": "S1",
    }
    path = library.layout.dataset_json_path(sha, library.extractor_key, library.comparison_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    quality = [
        {"entity": "wear_test", "sample_id": "S1", "field": "test_temperature", "decision": "agree"},
        {"entity": "wear_test", "sample_id": "S1", "field": "tested_coating", "decision": "agree"},
        {"entity": "wear_test", "sample_id": "S1", "field": "wear_mode", "decision": "conflict", "detail": "两路冲突"},
    ]
    payload = {"document_id": sha, "filename": "E", "paper_row": coatings[0], "sample_rows": [*coatings, test]}
    path.write_text(json.dumps({**payload, "quality_rows": quality}, ensure_ascii=False), encoding="utf-8")
    return document_key(sha)


def seed_one_entity_document(library: Library, root: Path) -> str:
    """A paper of a profile declaring one entity type (coatings): its lane records name the entity, its dataset rows
    were written before rows named a lone declared entity, so they name none."""
    pdf = make_blank_pdf(root / "one-entity.pdf", [(400.0, 600.0)])
    sha = library.register_upload("O 一种实体.pdf", pdf.read_bytes()).sha256
    for backend in BACKENDS:
        blocks = tuple(
            make_block(page=0, order=order, backend=backend, document_id=sha, content=f"block {order}")
            for order in (0, 1)
        )
        ParsedArtifact(
            document_id=sha,
            backend=backend,
            backend_version="stub",
            pages=(PageGeometry(index=0, width_pt=400.0, height_pt=600.0),),
            blocks=blocks,
        ).write(library.layout.artifact_path(sha, backend))
        solvent = make_field("solvent", "water" if backend == "mineru" else "ethanol", source_ids=[f"{backend}_p0_b1"])
        sample = make_sample("S1", [solvent]).model_copy(update={"entity": "coating"})
        make_lane(backend=backend, samples=[sample], document_id=sha, extractor_key=library.extractor_key).write(
            library.layout.extraction_path(sha, backend, library.extractor_key)
        )
    field = make_field("solvent", "water", source_ids=["mineru_p0_b1"])
    ComparisonReport(
        document_id=sha,
        extractor_key=library.extractor_key,
        comparison_key=library.comparison_key,
        backend_a="mineru",
        backend_b="paddleocr_vl",
        matchings={
            "coating": SampleMatching(
                pairs=(SampleMatch(a_id="S1", b_id="S1", confidence=1.0, justification="", method="exact"),)
            )
        },
        counts=ComparisonCounts(total=1, conflict=1),
        comparisons=(FieldComparison(scope="coating:S1|S1", field="solvent", status="conflict", a=field, b=field),),
    ).write(library.layout.comparison_path(sha, library.extractor_key, library.comparison_key))
    row = {
        "document_id": sha,
        "filename": "O 一种实体.pdf",
        "sample_id": "S1",
        "solvent": None,
        "precursor_purity": None,
    }
    quality = [{"sample_id": "S1", "field": "solvent", "decision": "conflict", "detail": "两路冲突"}]
    path = library.layout.dataset_json_path(sha, library.extractor_key, library.comparison_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"document_id": sha, "filename": "O", "paper_row": row, "sample_rows": [row], "quality_rows": quality}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return document_key(sha)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def serve(root: Path) -> Iterator[tuple[str, dict[str, str], Path]]:
    settings = Settings(data_root=root / "data", repo_root=root, llm_api_key="sk-test", llm_model="fake-model")
    profile = shipped_profile()
    library = Library(settings, profile)
    docs = {
        "A": seed_document(
            library,
            root,
            0,
            "A Room-temperature-magnetron-sputtered indium tin oxide (ITO) films with Nb co-doping 氧化铟锡薄膜.pdf",
            samples=3,
            comparisons=14,
        ),
        "B": seed_document(library, root, 1, "B 掺铝氧化锌的透明导电性.pdf", samples=2, comparisons=6, conflicts=2),
        "C": seed_document(library, root, 2, "C 钙钛矿电池（买来的 ITO 玻璃）.pdf", samples=0, comparisons=0),
    }
    for index in range(3, 26):  # a long library, as on the real server: 30 papers in all
        seed_document(library, root, index, f"filler paper {index}.pdf", samples=1, comparisons=1)
    # The rail's chips and search have one paper each to find: a review, a paper not finished under this profile
    # (its lanes and report are stored, its dataset is not), and a second one with conflicts; "wu" is in two names,
    # in either case, and in no other.
    docs["V"] = seed_document(
        library, root, 26, "V 透明导电氧化物综述 WU.pdf", samples=1, comparisons=1, article_type="review"
    )
    docs["U"] = seed_document(library, root, 27, "U 还没处理完的论文.pdf", samples=1, comparisons=1, finished=False)
    docs["W"] = seed_document(library, root, 28, "W Wu 氧化锌薄膜的电学性质.pdf", samples=1, comparisons=2, conflicts=1)
    # A very tall results table, for the sticky tab-bar pin check: deep scrolling must survive a tab switch.
    # Its charts are read too, so the figures panel has its own (shorter) body: tall, but shorter than results.
    docs["T"] = seed_document(library, root, 30, "T 很长的结果表.pdf", samples=8, comparisons=250)
    StoredReadings(
        document_id=library.identity(docs["T"]).sha256,
        figure_key=library.figure_key,
        model="stub",
        profile=profile.name,
        readings=tuple(
            FigureReading(
                source_id="mineru_p1_b1",
                page=1,
                bbox=NormalizedBBox(x1=0.1, y1=0.1, x2=0.9, y2=0.9),
                figure="Fig. 1",
                caption="thickness against growth conditions",
                panel=1,
                field="thickness",
                note=LONG_CONDITION,
                y_raw=100.0 + n,
                precision=0.1,
            )
            for n in range(24)
        ),
    ).write(library.layout.figures_path(docs["T"], library.figure_key, profile.name))
    # A paper whose charts were read under the current settings (and gave nothing): its button offers a re-read.
    docs["R"] = seed_document(library, root, 29, "R 已识图的论文.pdf", samples=1, comparisons=1)
    StoredReadings(
        document_id=library.identity(docs["R"]).sha256,
        figure_key=library.figure_key,
        model="stub",
        profile=profile.name,
    ).write(library.layout.figures_path(docs["R"], library.figure_key, profile.name))
    app = create_app(settings, profile=profile, jobs=JobManager(stub_runner, stage_names(), workers=2))
    # A second server under a profile with two entity types, one seeded paper: the page groups by entity there. Its
    # address travels in `docs` beside that paper's id, so every check keeps the one signature.
    entity_settings = Settings(data_root=root / "entities", repo_root=root, llm_api_key="sk-test", llm_model="fake")
    # Its paper-level group's label holds markup, which the profile page must show as text.
    entity_profile = make_reference_profile({"groups.0.label_zh": MARKUP})
    entity_library = Library(entity_settings, entity_profile)
    docs["E"] = seed_entity_document(entity_library, root)
    entity_app = create_app(
        entity_settings, profile=entity_profile, jobs=JobManager(stub_runner, stage_names(), workers=1)
    )
    # A third under a profile declaring a single entity type: the page groups nothing, the records name the entity.
    one_settings = Settings(data_root=root / "one-entity", repo_root=root, llm_api_key="sk-test", llm_model="fake")
    one_profile = make_one_entity_profile()
    docs["O"] = seed_one_entity_document(Library(one_settings, one_profile), root)
    one_app = create_app(one_settings, profile=one_profile, jobs=JobManager(stub_runner, stage_names(), workers=1))
    # A fourth serving two profiles over one data root, the shipped one the default: paper M has results under both,
    # paper N under the default only.
    multi_settings = Settings(data_root=root / "multi", repo_root=root, llm_api_key="sk-test", llm_model="fake")
    demo = make_reference_profile()
    multi_library = Library(multi_settings, profile)
    docs["M"] = seed_document(multi_library, root, 40, "M 两个领域都有结果.pdf", samples=2, comparisons=6)
    seed_entity_document(Library(multi_settings, demo), root, pdf=root / "40.pdf", name="M 两个领域都有结果.pdf")
    docs["N"] = seed_document(multi_library, root, 41, "N 只有默认领域的结果.pdf", samples=1, comparisons=1)
    multi_jobs = JobManager(stub_runner, stage_names(), workers=2)
    multi_app = create_app(multi_settings, profile=profile, profiles=(profile, demo), jobs=multi_jobs)
    with (
        running(app) as base,
        running(entity_app) as entity_base,
        running(one_app) as one_base,
        running(multi_app) as multi_base,
    ):
        docs["entities"] = entity_base
        docs["one-entity"] = one_base
        docs["multi"] = multi_base
        yield base, docs, root / "0.pdf"


@contextlib.contextmanager
def running(app: object) -> Iterator[str]:
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))  # type: ignore[arg-type]
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# ---- checks -------------------------------------------------------------------------------------------------

Check = Callable[[Page, str, dict[str, str], Path], Awaitable[None]]
CHECKS: list[tuple[str, Check, dict[str, int]]] = []


def check(name: str, width: int = 1440, height: int = 900) -> Callable[[Check], Check]:
    def register(function: Check) -> Check:
        CHECKS.append((name, function, {"width": width, "height": height}))
        return function

    return register


def delayed(seconds: float) -> Callable[[Route], Awaitable[None]]:
    async def handler(route: Route) -> None:
        await asyncio.sleep(seconds)
        with contextlib.suppress(Exception):  # the page may have gone on without it
            await route.continue_()

    return handler


async def open_doc(page: Page, base: str, doc: str, fact: int | None = None) -> None:
    await page.goto(f"{base}/#/doc/{doc}" + (f"/fact/{fact}" if fact is not None else ""))
    await page.wait_for_selector("#document-view:not(.hidden) h1")


async def title(page: Page) -> str:
    return (await page.text_content("#document-view h1")) or ""


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@check("a late response for the document left behind never paints over the next one")
async def race_switch(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await page.goto(f"{base}/")
    await page.wait_for_selector("#doc-list .doc-item")
    await page.route(f"**/api/documents/{docs['A']}/report", delayed(1.5))
    await page.evaluate(f"location.hash = '#/doc/{docs['A']}'")
    await page.wait_for_timeout(200)
    await page.evaluate(f"location.hash = '#/doc/{docs['B']}'")
    await page.wait_for_timeout(2500)
    expect((await title(page)).startswith("B "), f"shows {await title(page)!r} under #/doc/B")
    active = await page.text_content(".doc-item.active .name")
    expect(bool(active) and active.startswith("B "), f"active library item is {active!r}")


@check("the home table never reappears over a document opened while it was loading")
async def race_home(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.route("**/api/dataset", delayed(1.5))
    await page.evaluate("location.hash = '#/'")
    await page.wait_for_timeout(200)
    await page.evaluate(f"location.hash = '#/doc/{docs['B']}'")
    await page.wait_for_timeout(2500)
    expect(await page.is_hidden("#corpus-view"), "the corpus table is visible over the document")
    expect(await page.is_hidden("#empty-state"), "the home intro is visible over the document")
    expect((await title(page)).startswith("B "), f"shows {await title(page)!r}")


@check("a new document opens at its top")
async def scroll_top(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    await page.evaluate(f"location.hash = '#/doc/{docs['B']}'")
    await page.wait_for_function("document.querySelector('#document-view h1')?.textContent.startsWith('B ')")
    expect(await page.evaluate("window.scrollY") == 0, "the new document kept the old scroll offset")


@check("switching to a short tab keeps the tab bar pinned", width=1440, height=700)
async def tab_bar_pin(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    async def bar_top() -> float:
        return await page.evaluate(
            "document.querySelector('#document-view [data-slot=tabs]').getBoundingClientRect().top"
        )

    await open_doc(page, base, docs["T"])
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    expect(await page.evaluate("window.scrollY") > 0, "there was room to scroll into the results table")
    expect(await bar_top() <= 57, f"the tab bar starts pinned (top {await bar_top()})")
    await page.click('#document-view .tab-bar [data-tab="figures"]')
    expect(await bar_top() <= 57, f"the tab bar un-pinned after switching to 图中读数 (top {await bar_top()})")
    await page.click('#document-view .tab-bar [data-tab="results"]')
    expect(await bar_top() <= 57, f"the tab bar un-pinned after switching back to 结果表 (top {await bar_top()})")
    # A genuinely short document (no results, no comparisons, no readings) has no scroll room of its
    # own; the middle column's minimum height must still leave enough to keep the bar pinned.
    await open_doc(page, base, docs["C"])
    # Coming from doc T this is a same-document hash navigation: the old content is still on the page
    # until the report lands, so wait for paper C itself before measuring anything.
    await page.wait_for_function("document.querySelector('#document-view h1')?.textContent.startsWith('C ')")
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    expect(await page.evaluate("window.scrollY") > 0, "a short document was still given room to scroll")
    await page.click('#document-view .tab-bar [data-tab="figures"]')
    expect(await bar_top() <= 57, f"the tab bar un-pinned on a short document's 图中读数 (top {await bar_top()})")
    await page.click('#document-view .tab-bar [data-tab="results"]')
    expect(await bar_top() <= 57, f"the tab bar un-pinned on a short document's 结果表 (top {await bar_top()})")
    await page.click('#document-view .tab-bar [data-tab="facts"]')
    expect(await bar_top() <= 57, f"the tab bar un-pinned on a short document's 事实对照 (top {await bar_top()})")


@check("a job finishing after the reader left does not redraw the page they left")
async def finish_elsewhere(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.click('#document-view [data-action="run"]')
    await page.wait_for_selector("#document-view .stage.running")
    await page.evaluate(f"location.hash = '#/doc/{docs['B']}'")
    await page.wait_for_timeout(len(stage_names()) * STAGE_SECONDS * 1000 + 2500)
    expect((await title(page)).startswith("B "), f"shows {await title(page)!r} under #/doc/B")
    toasts = await page.locator(".toast").all_text_contents()
    expect("处理完成" not in toasts, "the finished job of A toasted over B")


@check("polling survives a failed request and runs once")
async def polling(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    failed = {"armed": False, "n": 0}
    polls: list[float] = []

    async def flaky(route: Route) -> None:
        polls.append(time.monotonic())
        if failed["armed"] and failed["n"] == 0:
            failed["n"] += 1
            await route.abort()
        else:
            await route.continue_()

    await page.route("**/api/jobs/*", flaky)
    await page.click('#document-view [data-action="run"]')
    await page.wait_for_timeout(300)
    # leave and come back while the job runs: the first loop must end, not run beside the new one
    await page.evaluate("location.hash = '#/'")
    await page.wait_for_timeout(100)
    await page.evaluate(f"location.hash = '#/doc/{docs['A']}'")
    await page.wait_for_selector("#document-view .stage.running")
    polls.clear()
    failed["armed"] = True  # the next poll is dropped on the floor
    await page.wait_for_selector("text=处理完成", timeout=len(stage_names()) * STAGE_SECONDS * 1000 + 15000)
    run = page.locator('#document-view [data-action="run"]')
    await page.wait_for_function("!document.querySelector('#document-view [data-action=run]').disabled")
    expect(not await run.is_disabled(), "the run button stayed disabled after the job finished")
    gaps = [b - a for a, b in itertools.pairwise(polls)]
    expect(min(gaps, default=1.0) > 0.5, f"two loops polled side by side (gaps {gaps})")
    expect(failed["n"] == 1, "the failed request was never retried")


@check("re-uploading the open document follows its new job")
async def reupload(page: Page, base: str, docs: dict[str, str], pdf: Path) -> None:
    await open_doc(page, base, docs["A"], fact=2)
    polled = asyncio.Event()
    page.on("request", lambda request: polled.set() if "/api/jobs/" in request.url else None)
    await page.set_input_files("#file-input", str(pdf))
    await asyncio.wait_for(polled.wait(), timeout=5)
    await page.wait_for_selector("#document-view .stage.running", timeout=5000)


@check("an upload does not pull the reader back after they moved to another document")
async def upload_elsewhere(page: Page, base: str, docs: dict[str, str], pdf: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.route("**/api/documents?force=*", delayed(1.5))
    await page.set_input_files("#file-input", str(pdf))
    await page.wait_for_timeout(200)
    await page.evaluate(f"location.hash = '#/doc/{docs['B']}'")
    await page.wait_for_timeout(2500)
    expect((await page.evaluate("location.hash")).startswith(f"#/doc/{docs['B']}"), "the upload navigated away")
    expect((await title(page)).startswith("B "), f"shows {await title(page)!r} under #/doc/B")


@check("a failed library refresh keeps the rail refreshing")
async def library_retry(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await page.goto(f"{base}/")
    await page.wait_for_selector("#doc-list .doc-item")
    lists: list[float] = []

    async def flaky(route: Route) -> None:
        lists.append(time.monotonic())
        if len(lists) == 1:
            await route.abort()
        else:
            await route.continue_()

    await page.route("**/api/documents", flaky)
    # Queued from outside this page, so nothing here but the rail's own refresh can notice it.
    await page.evaluate(f"fetch('/api/documents/{docs['B']}/run', {{method: 'POST'}})")
    await page.click("#refresh-library")  # dropped: the next refresh must still come by itself
    await page.wait_for_timeout(7000)
    expect(len(lists) >= 2, "the rail stopped refreshing after one failed request")
    expect(await page.locator(".doc-item .queued").count() == 0, "the rail still marks a finished job as running")


@check("an older library list never paints over a newer one")
async def library_order(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await page.goto(f"{base}/")
    await page.wait_for_selector("#doc-list .doc-item")
    first = {"pending": True}

    async def stale_then_fresh(route: Route) -> None:
        if first["pending"]:
            first["pending"] = False
            await asyncio.sleep(1.5)
            with contextlib.suppress(Exception):
                await route.fulfill(status=200, content_type="application/json", body="[]")
        else:
            await route.continue_()

    await page.route("**/api/documents", stale_then_fresh)
    await page.click("#refresh-library")  # answered late, with an empty (older) list
    await page.wait_for_timeout(100)
    await page.click("#refresh-library")  # answered at once
    await page.wait_for_timeout(2500)
    expect(await page.locator("#doc-list .doc-item").count() > 0, "the late, older list replaced the newer one")


@check("the rail shows its skeleton on a cold boot until the first list answers")
async def rail_skeleton_cold_boot(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await page.route("**/api/documents", delayed(1.2))
    await page.goto(f"{base}/")
    await page.wait_for_selector("#doc-list .skeleton-row", state="visible")
    expect(await page.get_attribute("#doc-list", "aria-busy") == "true", "the skeleton did not mark the list busy")
    await page.wait_for_timeout(300)
    expect(
        await page.is_visible("#doc-list .skeleton-row"),
        "the skeleton gave way to the empty state before the list answered",
    )
    await page.wait_for_selector("#doc-list .doc-item", timeout=5000)
    expect(await page.locator("#doc-list .skeleton-row").count() == 0, "the skeleton stayed behind the items")
    expect(await page.get_attribute("#doc-list", "aria-busy") is None, "the list stayed busy after the list answered")


@check("重新处理 pressed as the view reloads still follows the new job")
async def rerun_during_reload(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["B"])
    await page.route(f"**/api/documents/{docs['B']}/run?*", delayed(1.0))
    await page.click('#document-view [data-action="run"]')
    # What the previous job's finish handler does while this request is out: reload the view.
    await page.evaluate("import('/router.js').then((router) => router.reloadView())")
    await page.wait_for_selector("#document-view .stage.running", timeout=5000)
    await page.wait_for_selector("text=处理完成", timeout=len(stage_names()) * STAGE_SECONDS * 1000 + 10000)


@check("识图 reads the charts of a paper that never had them read, and the section says what is going on")
async def read_charts(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["B"])
    await page.click('#document-view .tab-bar [data-tab="figures"]')
    button = page.locator('#document-view [data-action="figures"]')
    empty = page.locator('#document-view [data-slot="figures-empty"]')
    expect((await button.text_content()) == "识图", f"the button reads {await button.text_content()!r}")
    expect(await button.is_enabled(), "识图 is disabled on an idle paper with a PDF")
    expect("尚未识图" in (await empty.text_content() or ""), f"the empty section says {await empty.text_content()!r}")
    expect(await page.is_hidden('#document-view [data-slot="figures-body"]'), "an empty readings table is shown")
    stage = page.locator("#document-view .stage.skipped", has_text="识图")
    expect(await stage.count() == 1, "the never-requested figures stage is not shown as skipped")
    async with page.expect_request(lambda request: f"/api/documents/{docs['B']}/run?" in request.url) as sent:
        await button.click()
    url = (await sent.value).url
    expect("figures=true" in url and "force_figures=false" in url and "force=false" in url, f"asked {url}")
    await page.wait_for_selector("#document-view .stage.running", timeout=5000)
    expect(await button.is_disabled(), "识图 stays enabled while its job runs")
    expect("正在识图" in (await empty.text_content() or ""), f"while reading it says {await empty.text_content()!r}")
    await page.wait_for_selector("text=处理完成", timeout=len(stage_names()) * STAGE_SECONDS * 1000 + 10000)
    await page.wait_for_function(
        "!document.querySelector('#document-view [data-action=\"figures\"]').disabled", timeout=5000
    )


@check("重新识图 on a paper whose charts were read asks first, then re-asks every chart")
async def reread_charts(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["R"])
    await page.click('#document-view .tab-bar [data-tab="figures"]')
    button = page.locator('#document-view [data-action="figures"]')
    expect((await button.text_content()) == "重新识图", f"the button reads {await button.text_content()!r}")
    empty = await page.text_content('#document-view [data-slot="figures-empty"]') or ""
    expect("已识图" in empty, f"a read paper with no readings says {empty!r}")
    asked: list[str] = []
    page.on("request", lambda request: asked.append(request.url) if "/run?" in request.url else None)
    page.once("dialog", lambda dialog: asyncio.ensure_future(dialog.dismiss()))
    await button.click()
    await page.wait_for_timeout(300)
    expect(not asked, f"a dismissed confirmation still queued {asked}")
    page.once("dialog", lambda dialog: asyncio.ensure_future(dialog.accept()))
    async with page.expect_request(lambda request: "/run?" in request.url) as sent:
        await button.click()
    expect("force_figures=true" in (await sent.value).url, f"asked {(await sent.value).url}")
    await page.wait_for_selector("text=处理完成", timeout=len(stage_names()) * STAGE_SECONDS * 1000 + 10000)


@check("上传后识图 queues the upload's job with chart reading, and only when it is ticked")
async def upload_with_charts(page: Page, base: str, docs: dict[str, str], pdf: Path) -> None:
    await page.goto(f"{base}/")
    await page.wait_for_selector("#doc-list .doc-item")
    # The option is the upload dialog's; the hidden #file-input uploads at once with the options as the dialog has
    # them, so it is ticked there and the dialog closed before the file goes in.
    box = page.locator("#upload-figures")
    for tick in (False, True):
        await page.click("#upload-open")
        await page.wait_for_selector("#upload-dialog[open]")
        if not tick:
            expect(not await box.is_checked(), "上传后识图 starts ticked")
        await box.set_checked(tick)
        await page.keyboard.press("Escape")
        await page.wait_for_function("!document.getElementById('upload-dialog').open")
        async with page.expect_response(lambda response: "/api/documents?" in response.url) as answered:
            await page.set_input_files("#file-input", str(pdf))
        job = (await (await answered.value).json())["job"]
        expect(job["figures"] is tick, f"ticked={tick} queued a job with figures={job['figures']}")
        await page.wait_for_selector("text=处理完成", timeout=len(stage_names()) * STAGE_SECONDS * 1000 + 10000)


# ---- the shell: the rail's collapse, its search and chips, the upload dialog and the bulk run's confirmation ----

REVIEW = "V 透明导电氧化物综述 WU.pdf"
UNFINISHED = "U 还没处理完的论文.pdf"
WU = "W Wu 氧化锌薄膜的电学性质.pdf"
CONFLICTS = {"B 掺铝氧化锌的透明导电性.pdf", WU}
LIBRARY_SIZE = 30  # as seeded; earlier checks may have uploaded more, so a check reads the count it finds
CONTENT_WIDTH = "document.querySelector('.content').getBoundingClientRect().width"
VIEWPORT_WIDTH = "document.documentElement.clientWidth"
COLLAPSED = "document.querySelector('.shell').dataset.rail === 'collapsed'"
DIALOG_OPEN = "document.getElementById('upload-dialog').open"


async def home(page: Page, base: str) -> None:
    await page.goto(f"{base}/#/")
    # Attached, not visible: on a phone the list is folded away, and with the rail collapsed it is hidden.
    await page.wait_for_selector("#doc-list .doc-item", state="attached")


async def listed_names(page: Page) -> list[str]:
    return await page.locator("#doc-list .doc-item .name").all_text_contents()


@check("the rail collapses from the toggle and `[`, the content takes the whole width, and a reload keeps it")
async def rail_collapse(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await home(page, base)
    expect(await page.evaluate(CONTENT_WIDTH) < await page.evaluate(VIEWPORT_WIDTH), "the rail has no width to start")
    await page.click("#rail-toggle")
    expect(await page.evaluate(COLLAPSED), "the toggle did not collapse the rail")
    expect(
        await page.evaluate(CONTENT_WIDTH) == await page.evaluate(VIEWPORT_WIDTH),
        f"collapsed, the content is {await page.evaluate(CONTENT_WIDTH)}px of {await page.evaluate(VIEWPORT_WIDTH)}px",
    )
    expect(await page.get_attribute("#rail-toggle", "aria-expanded") == "false", "aria-expanded did not follow")
    expect(await page.get_attribute("#rail-toggle", "title") == "展开侧栏", "the tooltip did not follow")
    expect(await page.evaluate("localStorage.getItem('paperfacts.rail-collapsed')") == "1", "not remembered")
    await page.reload()
    await page.wait_for_selector("#doc-list .doc-item", state="attached")
    expect(await page.evaluate(COLLAPSED), "a reload forgot the collapsed rail")
    expect(await page.evaluate(CONTENT_WIDTH) == await page.evaluate(VIEWPORT_WIDTH), "the reloaded rail took width")
    await page.keyboard.press("[")  # focus is on the page itself
    expect(not await page.evaluate(COLLAPSED), "`[` did not expand the rail")
    expect(await page.get_attribute("#rail-toggle", "aria-expanded") == "true", "aria-expanded did not follow `[`")
    expect(
        await page.evaluate("localStorage.getItem('paperfacts.rail-collapsed')") == "0",
        "the expansion is not remembered",
    )
    expect(await page.evaluate(CONTENT_WIDTH) < await page.evaluate(VIEWPORT_WIDTH), "the expanded rail has no width")


RAIL_WIDTH = "document.querySelector('.rail').getBoundingClientRect().width"


@check("dragging the rail seam resizes the rail and remembers it")
async def rail_drag(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await home(page, base)
    box = await page.locator(".rail-resizer").bounding_box()
    before = await page.evaluate(RAIL_WIDTH)
    expect(before == 300, f"the rail does not start at its 300px default: {before}")
    await page.mouse.move(box["x"] + box["width"] / 2, 450)
    await page.mouse.down()
    await page.mouse.move(
        box["x"] + box["width"] / 2 + 120, 450, steps=5
    )  # no viewer pane on home: the live max is the 600 cap, so +120 (→420) does not clamp
    live = await page.evaluate(RAIL_WIDTH)
    expect(live > before, f"the rail did not widen while dragging: {before} -> {live}")
    expect(await page.evaluate("window.getSelection().toString()") == "", "dragging selected text")
    await page.mouse.up()
    after = await page.evaluate(RAIL_WIDTH)
    stored = await page.evaluate("localStorage.getItem('paperfacts.rail-width')")
    expect(stored == str(round(after)), f"the dragged width was not remembered: {after} vs {stored}")
    await page.reload()
    await page.wait_for_selector("#rail-search")
    kept = await page.evaluate(RAIL_WIDTH)
    expect(abs(kept - after) <= 1, f"reload dropped the rail width: {after} -> {kept}")
    await page.evaluate("localStorage.removeItem('paperfacts.rail-width')")  # leave defaults for the later checks


@check("the rail drag clamps at its bounds", width=1920, height=1080)
async def rail_drag_clamp(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await home(page, base)

    async def seam() -> tuple[float, float]:
        box = await page.locator(".rail-resizer").bounding_box()
        return box["x"] + box["width"] / 2, 450

    mid = await seam()
    await page.mouse.move(*mid)
    await page.mouse.down()
    await page.mouse.move(mid[0] - 400, 450, steps=5)
    await page.mouse.up()
    expect(
        await page.evaluate(RAIL_WIDTH) == 280,
        f"the rail went below its 280px floor: {await page.evaluate(RAIL_WIDTH)}",
    )
    mid = await seam()  # the seam moved with the rail: re-aim before dragging the other way
    await page.mouse.move(*mid)
    await page.mouse.down()
    await page.mouse.move(mid[0] + 900, 450, steps=5)
    await page.mouse.up()
    # No viewer pane on home, so the live max is the 600px cap itself.
    expect(await page.evaluate(RAIL_WIDTH) == 600, f"the rail passed its 600px cap: {await page.evaluate(RAIL_WIDTH)}")
    await page.evaluate("localStorage.removeItem('paperfacts.rail-width')")


@check("the rail seam is hidden when the rail is collapsed or the shell is stacked")
async def rail_drag_hidden(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await home(page, base)
    await page.click("#rail-toggle")
    expect(await page.is_hidden(".rail-resizer"), "the seam handle shows on a collapsed rail")
    await page.keyboard.press("[")
    expect(not await page.is_hidden(".rail-resizer"), "the seam handle did not come back with the rail")
    await page.set_viewport_size({"width": 390, "height": 844})
    expect(await page.is_hidden(".rail-resizer"), "the seam handle shows on a stacked shell")
    await page.set_viewport_size({"width": 1440, "height": 900})


@check("the rail seam resizes from the keyboard and resets on double-click")
async def rail_resize_keyboard(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await home(page, base)
    await page.focus(".rail-resizer")
    expect(await page.get_attribute(".rail-resizer", "role") == "separator", "the handle is no separator")
    expect(await page.get_attribute(".rail-resizer", "aria-orientation") == "vertical", "no vertical orientation")
    await page.keyboard.press("ArrowRight")
    await page.keyboard.press("ArrowRight")
    expect(
        await page.evaluate(RAIL_WIDTH) == 332,
        f"two ArrowRight did not widen by 32px: {await page.evaluate(RAIL_WIDTH)}",
    )
    await page.keyboard.press("Home")
    expect(await page.evaluate(RAIL_WIDTH) == 280, "Home did not reach the 280px floor")
    await page.keyboard.press("End")
    expect(await page.evaluate(RAIL_WIDTH) == 600, "End did not reach the 600px cap")
    await page.dblclick(".rail-resizer")
    expect(await page.evaluate(RAIL_WIDTH) == 300, "double-click did not reset to 300px")
    expect(
        await page.evaluate("localStorage.getItem('paperfacts.rail-width')") is None, "the reset kept the stored width"
    )


@check("shrinking the viewport re-clamps a stored rail width and never overflows")
async def rail_viewport_shrink(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.evaluate("localStorage.setItem('paperfacts.rail-width', '600')")
    await page.reload()
    await page.wait_for_selector("#document-view:not(.hidden) h1")
    # With the viewer pane live at 480, the boot clamp already caps the rail at 1440-56-570-480 = 334.
    await page.wait_for_function(f"Math.round({RAIL_WIDTH}) === 334")  # wait out the rAF-debounced re-clamp
    expect(
        await page.evaluate(RAIL_WIDTH) == 334,
        f"the stored 600px was not clamped at boot: {await page.evaluate(RAIL_WIDTH)}",
    )
    await page.set_viewport_size({"width": 1280, "height": 900})
    await page.wait_for_timeout(100)  # one rAF tick for the debounced re-clamp
    expect(
        await page.evaluate(RAIL_WIDTH) <= 314,
        f"the rail was not re-clamped after the shrink: {await page.evaluate(RAIL_WIDTH)}",
    )
    overflow = await page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    expect(overflow <= 1, f"the page overflows {overflow}px sideways after the shrink")
    await page.evaluate("localStorage.removeItem('paperfacts.rail-width')")


@check("the rail seam works on a document page in the stacked-pane band (961–1279px)", width=1100, height=900)
async def rail_stacked_band_doc(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    # Below 1280 the viewer pane is a stacked full-width band, not a side column: it must not count against
    # the rail's live max, or the clamp floors every width at 280 and drags move the rail backwards.
    await open_doc(page, base, docs["A"])
    await page.evaluate("localStorage.setItem('paperfacts.rail-width', '400')")
    await page.reload()
    await page.wait_for_selector("#document-view:not(.hidden) h1")
    await page.wait_for_function(f"Math.round({RAIL_WIDTH}) === 400")  # stored width survives, not snapped to 280
    expect(await page.is_hidden("#viewer-toggle"), "the viewer toggle shows below 1280px")
    expect(
        await page.evaluate(RAIL_WIDTH) == 400,
        f"the stored 400px did not survive load: {await page.evaluate(RAIL_WIDTH)}",
    )
    await page.focus(".rail-resizer")
    await page.keyboard.press("ArrowRight")
    expect(
        await page.evaluate(RAIL_WIDTH) == 416,
        f"ArrowRight did not widen the rail: {await page.evaluate(RAIL_WIDTH)}",
    )
    box = await page.locator(".rail-resizer").bounding_box()
    await page.mouse.move(box["x"] + box["width"] / 2, 450)
    await page.mouse.down()
    await page.mouse.move(box["x"] + box["width"] / 2 + 120, 450, steps=5)
    await page.mouse.up()
    # From 416, +120 clamps at the true stacked-band max 1100-56-570 = 474 (still forward, never 280).
    expect(
        await page.evaluate(RAIL_WIDTH) == 474,
        f"the drag did not move forward to the 474px max: {await page.evaluate(RAIL_WIDTH)}",
    )
    expect(
        await page.evaluate("localStorage.getItem('paperfacts.rail-width')") == "474",
        "the dragged width was not remembered",
    )
    await page.evaluate("localStorage.removeItem('paperfacts.rail-width')")


@check("the topbar's 总表 link is on every page, marks only home, and returns from a document to the home table")
async def topbar_home(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    nav = page.locator('.topnav [data-nav="home"]')
    await home(page, base)
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    expect(await nav.is_visible(), "总表 is not on the home view")
    expect(await nav.get_attribute("aria-current") == "page", "the home view does not mark 总表 current")
    await open_doc(page, base, docs["A"])
    expect(await nav.is_visible(), "总表 is not on the document view")
    expect(await nav.get_attribute("aria-current") is None, "the document view marks 总表 current")
    await nav.click()
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    expect(await nav.get_attribute("aria-current") == "page", "returning home did not mark 总表 current")
    await open_profile_page(page, f"{base}/#/profile")
    expect(await nav.is_visible(), "总表 is not on the profile page")
    expect(await nav.get_attribute("aria-current") is None, "the profile page marks 总表 current")


@check("the theme toggle cycles 跟随系统 → 浅色 → 深色, and an explicit choice survives a reload")
async def theme_cycle(page: Page, base: str, _: dict[str, str], __: Path) -> None:
    await home(page, base)
    root = page.locator("html")
    button = page.locator("#theme-toggle")
    expect(await root.get_attribute("data-theme") is None, "a fresh visit starts on 跟随系统")
    await button.click()
    expect(await root.get_attribute("data-theme") == "light", "one click did not pick 浅色")
    expect(await button.get_attribute("aria-label") == "主题：浅色", "the label does not say 浅色")
    await button.click()
    expect(await root.get_attribute("data-theme") == "dark", "two clicks did not pick 深色")
    expect(await button.get_attribute("aria-label") == "主题：深色", "the label does not say 深色")
    await page.reload()
    await page.wait_for_selector("#doc-list .doc-item", state="attached")
    expect(await root.get_attribute("data-theme") == "dark", "a reload forgot the explicit 深色")
    expect(await button.get_attribute("aria-label") == "主题：深色", "a reload reset the label")
    await page.locator("#theme-toggle").click()
    expect(await root.get_attribute("data-theme") is None, "the third click did not return to 跟随系统")
    expect(await page.evaluate("localStorage.getItem('pf-theme')") is None, "auto did not clear the stored choice")


@check("`[` typed into the rail search, or pressed in an open dialog, is not the collapse")
async def rail_bracket_guard(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await home(page, base)
    await page.focus("#rail-search")
    await page.keyboard.type("[")
    expect(await page.input_value("#rail-search") == "[", "the search box did not take the `[`")
    expect(not await page.evaluate(COLLAPSED), "`[` in the search box collapsed the rail")
    await page.fill("#rail-search", "")
    await page.click("#upload-open")
    await page.wait_for_selector("#upload-dialog[open]")
    await page.keyboard.press("[")
    expect(not await page.evaluate(COLLAPSED), "`[` in the open dialog collapsed the rail")
    await page.keyboard.press("Escape")
    await page.wait_for_function(f"!{DIALOG_OPEN}")


@check("the rail search finds names by substring whatever the case, keeps its filter and focus across a refresh")
async def rail_search(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await home(page, base)
    total = len(await listed_names(page))
    expect(total >= LIBRARY_SIZE, f"the library lists {total} papers")
    await page.fill("#rail-search", "wu")
    names = await listed_names(page)
    expect(sorted(names) == sorted([REVIEW, WU]), f"search 'wu' lists {names}")
    expect(
        await page.text_content("#doc-count") == f"（2/{total}）",
        f"the count reads {await page.text_content('#doc-count')!r}",
    )
    await page.focus("#rail-search")
    await page.evaluate("import('/library.js').then((library) => library.loadLibrary())")
    await page.wait_for_timeout(400)
    expect(sorted(await listed_names(page)) == sorted(names), "the refresh dropped the filter")
    expect(await page.input_value("#rail-search") == "wu", "the refresh cleared the search box")
    expect(await page.evaluate("document.activeElement.id") == "rail-search", "the refresh took the focus")
    await page.fill("#rail-search", "nothing like this")
    empty = page.locator("#doc-list .doc-list-empty")
    expect(await empty.count() == 1 and "没有匹配的文档" in (await empty.text_content() or ""), "no empty state")
    await empty.locator("button").click()
    expect(await page.input_value("#rail-search") == "", "清除筛选 left the query")
    expect(len(await listed_names(page)) == total, "清除筛选 did not restore the list")


async def press_chip(page: Page, chip: str) -> None:
    await page.click(f'#rail-chips [data-chip="{chip}"]')


async def chip_count(page: Page, chip: str) -> str:
    return await page.text_content(f'#rail-chips [data-chip="{chip}"] .n') or ""


@check(
    "the chips 综述, 有冲突 and 未完成 list exactly the seeded papers; 未完成 follows the profile, the default by name"
)
async def rail_chips(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await home(page, base)
    total = len(await listed_names(page))
    for chip, expected in (("review", {REVIEW}), ("conflict", CONFLICTS), ("unfinished", {UNFINISHED})):
        expect(
            await chip_count(page, chip) == str(len(expected)), f"chip {chip} counts {await chip_count(page, chip)!r}"
        )
        await press_chip(page, chip)
        expect(await page.get_attribute(f'#rail-chips [data-chip="{chip}"]', "aria-pressed") == "true", "not pressed")
        names = await listed_names(page)
        expect(set(names) == expected and len(names) == len(expected), f"chip {chip} lists {names}")
        await press_chip(page, chip)
    expect(len(await listed_names(page)) == total, "releasing the chips did not restore the list")
    await press_chip(page, "conflict")
    await page.fill("#rail-search", "wu")
    expect(await listed_names(page) == [WU], "a chip and the search do not combine")
    # Under two profiles: M is finished under both, N under the default only. Routed as the default (null in the
    # page, resolved to its name from /api/profiles) nothing is unfinished; under the demo profile, N is.
    await page.goto(f"{docs['multi']}/#/")
    await page.wait_for_selector("#doc-list .doc-item")
    expect(await chip_count(page, "unfinished") == "0", f"the default counts {await chip_count(page, 'unfinished')!r}")
    await press_chip(page, "unfinished")
    expect(await listed_names(page) == [], f"the default lists {await listed_names(page)} as unfinished")
    await press_chip(page, "unfinished")  # the profile switch below is a hash change: the chip would stay pressed
    await page.goto(f"{docs['multi']}/#/p/{DEMO}")
    await page.wait_for_selector("#doc-list .doc-item")
    expect(
        await chip_count(page, "unfinished") == "1", f"the demo profile counts {await chip_count(page, 'unfinished')!r}"
    )
    await press_chip(page, "unfinished")
    expect(
        await listed_names(page) == ["N 只有默认领域的结果.pdf"], f"the demo profile lists {await listed_names(page)}"
    )


DROP = """(kind) => {
  const transfer = new DataTransfer();
  if (kind === "text") transfer.items.add("some words", "text/plain");
  else transfer.items.add(new File(["%PDF-1.4"], "dropped.pdf", { type: "application/pdf" }));
  window.dispatchEvent(new DragEvent("drop", { dataTransfer: transfer, bubbles: true, cancelable: true }));
}"""


@check("a drop of text does not open the upload dialog; a drop of a PDF opens it with the file listed")
async def window_drop(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await home(page, base)
    await page.evaluate(DROP, "text")
    expect(not await page.evaluate(DIALOG_OPEN), "a drop of text opened the dialog")
    await page.evaluate(DROP, "file")
    expect(await page.evaluate(DIALOG_OPEN), "a drop of a PDF did not open the dialog")
    listed = await page.locator("#upload-files .upload-file .fname").evaluate_all(
        "rows => rows.map((row) => row.title)"
    )
    expect(listed == ["dropped.pdf"], f"the dialog lists {listed}")
    expect(await page.is_enabled("#upload-start"), "开始上传 is disabled with a file listed")
    await page.click("#upload-files .upload-file .remove")
    expect(await page.locator("#upload-files .upload-file").count() == 0, "移除 left the row")
    expect(not await page.is_enabled("#upload-start"), "开始上传 is enabled with nothing listed")
    await page.keyboard.press("Escape")


@check(
    "two files chosen in the upload dialog go up as two requests carrying the ticked options, each with its progress"
)
async def dialog_upload(page: Page, base: str, docs: dict[str, str], pdf: Path) -> None:
    await home(page, base)
    files = [make_blank_pdf(pdf.parent / f"dialog-{n}.pdf", [(430.0 + n, 630.0)]) for n in (1, 2)]
    await page.click("#upload-open")
    await page.wait_for_selector("#upload-dialog[open]")
    await page.set_input_files("#upload-pick", [str(file) for file in files])
    expect(await page.locator("#upload-files .upload-file").count() == 2, "the chosen files are not listed")
    expect(await page.text_content("#upload-start") == "开始上传（2 个）", "the button does not count the files")
    expect(await page.text_content("#upload-status") == "2 个待上传", "the status line does not count the files")
    await page.check("#upload-figures")
    await page.check("#upload-force")
    posts = []
    page.on(
        "request",
        lambda request: (
            posts.append(request) if request.method == "POST" and "/api/documents?" in request.url else None
        ),
    )
    await page.route("**/api/documents?force=*", delayed(0.8))
    await page.click("#upload-start")
    await page.wait_for_timeout(300)
    first = page.locator("#upload-files .upload-file").first
    expect(await first.locator("progress").count() == 1, "the row being uploaded shows no progress bar")
    expect(await first.locator(".state").text_content() == "上传中…", "the row being uploaded does not say so")
    expect(not await page.is_enabled("#upload-start"), "开始上传 stays enabled while uploading")
    await page.wait_for_function(f"!{DIALOG_OPEN}", timeout=10000)
    expect(len(posts) == 2, f"{len(posts)} uploads were posted")
    for request, file in zip(posts, files, strict=True):
        expect("force=true" in request.url, f"posted without force: {request.url}")
        body = request.post_data_buffer or b""
        expect(f'filename="{file.name}"'.encode() in body, f"the request does not carry {file.name}")
        expect(b'name="figures"' in body, "the request does not carry the figures option")
    await page.wait_for_function("location.hash.startsWith('#/doc/')")
    expect(await page.locator("#upload-files .upload-file").count() == 0, "uploaded rows stay listed after the close")
    await jobs_idle(page)


SI_READ_NOTE = "SI 中的表格、图注会被读取；SI 正文段落暂不读取"


async def si_names(page: Page) -> list[str]:
    return await page.locator("#upload-files .upload-file .si-name").evaluate_all(
        "rows => rows.map((row) => row.title)"
    )


async def opened_tag_titles(page: Page) -> tuple[str, str]:
    """The 「含 SI」 tag's title in the document header and on the rail's card of the open paper."""
    header = page.locator(".doc-head h1 .si-tag")
    await header.wait_for()
    expect(await header.text_content() == "含 SI", "the header's tag does not read 含 SI")
    card = page.locator("#doc-list .doc-item.active .si-tag")
    await card.wait_for(state="attached")
    return (await header.get_attribute("title") or "", await card.get_attribute("title") or "")


@check("a paper uploaded with an SI through the dialog is one request, and shows 含 SI with the page ranges")
async def dialog_upload_si(page: Page, base: str, docs: dict[str, str], pdf: Path) -> None:
    await home(page, base)
    main = make_blank_pdf(pdf.parent / "si-main.pdf", [(441.0, 641.0), (441.0, 641.0)])
    si = make_blank_pdf(pdf.parent / "si-only.pdf", [(442.0, 642.0)])
    await page.click("#upload-open")
    await page.wait_for_selector("#upload-dialog[open]")
    await page.set_input_files("#upload-pick", str(main))
    row = page.locator("#upload-files .upload-file").first
    expect(await row.locator(".si-add").text_content() == "添加 SI", "the row has no 添加 SI")
    await row.locator(".si-pick").set_input_files(str(si))
    expect(await si_names(page) == ["si-only.pdf"], f"the row lists SI {await si_names(page)}")
    expect(await page.text_content("#upload-start") == "开始上传", "an SI is counted as a paper of its own")
    posts = []
    page.on(
        "request",
        lambda request: (
            posts.append(request) if request.method == "POST" and "/api/documents?" in request.url else None
        ),
    )
    # Intercepted, so that the request carries its multipart body to the check.
    await page.route("**/api/documents?force=*", delayed(0.1))
    await page.click("#upload-start")
    await page.wait_for_function(f"!{DIALOG_OPEN}", timeout=10000)
    expect(len(posts) == 1, f"{len(posts)} requests were posted for one paper with its SI")
    body = posts[0].post_data_buffer or b""
    expect(b'name="file"; filename="si-main.pdf"' in body, f"the main PDF is not the request's file: {body[:300]!r}")
    expect(b'name="si"; filename="si-only.pdf"' in body, "the SI is not the request's si part")
    await page.wait_for_function("location.hash.startsWith('#/doc/')")
    header, card = await opened_tag_titles(page)
    for title, where in ((header, "header"), (card, "rail card")):
        lines = title.split("\n")
        expected = ["正文：第 1–2 页（si-main.pdf）", "SI 1：第 3 页（si-only.pdf）", SI_READ_NOTE]
        expect(lines == expected, f"the {where}'s 含 SI title reads {lines}")
    await page.wait_for_selector("#document-view .doc-facts .part")
    parts = await page.locator("#document-view .doc-facts .part").all_text_contents()
    expect(parts == expected[:2], f"the summary panel's 组成 reads {parts}")
    await jobs_idle(page)


@check("SI files are listed in the order added, can be moved and removed, and the page ranges follow that order")
async def dialog_si_order(page: Page, base: str, docs: dict[str, str], pdf: Path) -> None:
    await home(page, base)
    main = make_blank_pdf(pdf.parent / "order-main.pdf", [(451.0, 651.0)])
    parts = {
        name: make_blank_pdf(pdf.parent / f"{name}.pdf", sizes)
        for name, sizes in (
            ("si-a", [(452.0, 652.0)]),
            ("si-b", [(453.0, 653.0), (453.0, 653.0)]),
            ("si-c", [(454.0, 654.0)]),
        )
    }
    await page.click("#upload-open")
    await page.wait_for_selector("#upload-dialog[open]")
    await page.set_input_files("#upload-pick", str(main))
    row = page.locator("#upload-files .upload-file").first
    await row.locator(".si-pick").set_input_files([str(parts["si-a"]), str(parts["si-b"])])
    await row.locator(".si-pick").set_input_files(str(parts["si-c"]))
    expect(await si_names(page) == ["si-a.pdf", "si-b.pdf", "si-c.pdf"], f"listed {await si_names(page)}")
    expect(not await row.locator('[aria-label="上移：si-a.pdf"]').is_enabled(), "the first SI can move up")
    expect(not await row.locator('[aria-label="下移：si-c.pdf"]').is_enabled(), "the last SI can move down")
    await row.locator('[aria-label="移除 SI：si-c.pdf"]').click()
    await row.locator('[aria-label="下移：si-a.pdf"]').click()
    expect(
        await si_names(page) == ["si-b.pdf", "si-a.pdf"], f"after 移除 and 下移 the row lists {await si_names(page)}"
    )
    # Four at most: a fifth is refused on the row, in words.
    extra = [make_blank_pdf(pdf.parent / f"si-x{n}.pdf", [(460.0 + n, 660.0)]) for n in range(3)]
    await row.locator(".si-pick").set_input_files([str(path) for path in extra])
    expect(len(await si_names(page)) == 4, f"the row holds {len(await si_names(page))} SI files")
    expect("最多 4 个" in (await row.locator(".si-note").text_content() or ""), "the fifth SI is not refused in words")
    expect(not await row.locator(".si-add").is_enabled(), "添加 SI stays enabled with four SI files")
    for name in ("si-x0", "si-x1"):
        await row.locator(f'[aria-label="移除 SI：{name}.pdf"]').click()
    await page.click("#upload-start")
    await page.wait_for_function(f"!{DIALOG_OPEN}", timeout=10000)
    await page.wait_for_function("location.hash.startsWith('#/doc/')")
    header, _ = await opened_tag_titles(page)
    lines = header.split("\n")
    expected = [
        "正文：第 1 页（order-main.pdf）",
        "SI 1：第 2–3 页（si-b.pdf）",
        "SI 2：第 4 页（si-a.pdf）",
        SI_READ_NOTE,
    ]
    expect(lines == expected, f"the 含 SI title reads {lines}")
    await jobs_idle(page)


@check("处理全部未完成 asks first, and its 忽略缓存 option posts force=true")
async def run_all_confirm(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await home(page, base)
    total = len(await listed_names(page))
    sent: list[str] = []

    async def answer(route: Route) -> None:
        sent.append(route.request.url)
        await route.fulfill(status=202, content_type="application/json", body='{"submitted": [], "skipped": []}')

    await page.route("**/api/documents/run-all*", answer)
    await page.click("#run-all")
    await page.wait_for_selector("#run-all-dialog[open]")
    expect(not sent, "run-all posted before the reader confirmed")
    message = await page.text_content("#run-all-message") or ""
    expect(f"共 {total} 篇" in message, f"the confirmation reads {message!r}")
    await page.click('#run-all-dialog button:has-text("取消")')
    await page.wait_for_function("!document.getElementById('run-all-dialog').open")
    expect(not sent, "取消 posted the run")
    await page.click("#run-all")
    await page.wait_for_selector("#run-all-dialog[open]")
    expect(not await page.is_checked("#run-all-force"), "忽略缓存 starts ticked")
    await page.check("#run-all-force")
    await page.click("#run-all-confirm")
    await page.wait_for_function("!document.getElementById('run-all-dialog').open")
    await page.wait_for_timeout(300)
    expect(len(sent) == 1 and "force=true" in sent[0], f"the confirmed run posted {sent}")
    await page.click("#run-all")
    await page.wait_for_selector("#run-all-dialog[open]")
    expect(not await page.is_checked("#run-all-force"), "忽略缓存 stayed ticked for the next run")
    await page.click("#run-all-confirm")
    await page.wait_for_timeout(300)
    expect(len(sent) == 2 and "force=false" in sent[1], f"the second run posted {sent}")


@check("on a phone the rail stacks without a toggle, and the upload dialog fits the screen", width=390, height=844)
async def narrow_shell(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await home(page, base)
    expect(await page.is_hidden("#rail-toggle"), "the collapse toggle shows on a phone")
    expect(not await page.evaluate("document.getElementById('doc-list-wrap').open"), "the list is not folded away")
    await page.evaluate("localStorage.setItem('paperfacts.rail-collapsed', '1')")
    await page.reload()
    await page.wait_for_selector("#rail-search")
    expect(await page.is_visible("#rail-search"), "a remembered collapse hides the stacked rail")
    overflow = await page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    expect(overflow <= 0, f"the home page scrolls {overflow}px sideways")
    await page.click("#upload-open")
    await page.wait_for_selector("#upload-dialog[open]")
    box = await page.evaluate("document.getElementById('upload-dialog').getBoundingClientRect().toJSON()")
    expect(box["left"] >= 0 and box["right"] <= 390, f"the dialog spans {box['left']}–{box['right']}px")
    await page.keyboard.press("Escape")


@check("a link to no document says so in Chinese")
async def missing(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    for bad in ("0000000000000000", "zz"):
        await open_doc(page, base, docs["A"])
        await page.evaluate(f"location.hash = '#/doc/{bad}'")
        await page.wait_for_selector("#missing-view:not(.hidden)")
        text = await page.text_content("#missing-view")
        expect("找不到这篇文档" in (text or "") and bad in (text or ""), f"missing view says {text!r}")
        expect(await page.is_hidden("#document-view"), "the previous document stayed on screen")


@check("an unknown fact index clears the selection and the URL")
async def bad_fact(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"], fact=3)
    expect(await page.locator("tr.selected").count() == 1, "the deep-linked fact is not selected")
    await page.evaluate(f"location.hash = '#/doc/{docs['A']}/fact/999'")
    await page.wait_for_timeout(300)
    expect(await page.locator("tr.selected").count() == 0, "a stale row stays selected")
    expect(await page.evaluate("location.hash") == f"#/doc/{docs['A']}", "the URL still names fact 999")


@check("a filter keeps the selected fact, and a deep link clears a filter that hides it")
async def filter_selection(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"], fact=1)  # a conflict
    await page.click('[data-slot="filters"] [data-focus="filter:agree"]')
    await page.click('[data-slot="filters"] [data-focus="filter:all"]')
    expect(await page.locator('tr.selected[data-index="1"]').count() == 1, "the selection was lost to the filter")
    await page.click('[data-slot="filters"] [data-focus="filter:agree"]')
    await page.evaluate(f"location.hash = '#/doc/{docs['A']}/fact/2'")  # a missing
    await page.wait_for_selector('tr.selected[data-index="2"]')


@check("a paper without samples says why")
async def no_samples(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["C"])
    await page.click('#document-view .tab-bar [data-tab="facts"]')
    text = await page.text_content('[data-slot="rows-empty"]')
    expect("没有自己沉积的 TCO 膜" in (text or ""), f"facts empty state says {text!r}")
    expect(await page.is_hidden('[data-slot="dataset-copy"]'), "copy is offered for an empty table")


@check("a failed /api/profile leaves generic labels, not blanks, and the profile's own once a retry lands")
async def profile_retry(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    failing = True

    async def flaky(route: Route) -> None:
        if failing:
            await route.fulfill(status=500, content_type="application/json", body='{"detail": "profile down"}')
        else:
            await route.continue_()

    await page.route("**/api/profile", flaky)
    await open_doc(page, base, docs["C"])
    await page.click('#document-view .tab-bar [data-tab="facts"]')
    health = await page.text_content("#health")
    expect((health or "").startswith("model "), f"a profile failure marked the backend down: {health!r}")
    text = await page.text_content('[data-slot="rows-empty"]')
    expect("该论文没有范围内的样品" in (text or ""), f"generic no-samples message missing: {text!r}")
    header = await page.text_content("#document-view section.facts thead")
    expect("样品" in (header or ""), f"the facts header lost its entity label: {header!r}")
    failing = False
    # The first retry waits POLL_MS * 2; the view is then redrawn in the profile's words.
    await page.wait_for_function(
        "document.querySelector('[data-slot=\"rows-empty\"]')?.textContent.includes('TCO 膜')", timeout=10000
    )
    expect(bool(await page.text_content("#profile-title")), "the header never named the profile")


@check("opening a document logs no failed request")
async def quiet_console(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    errors: list[str] = []
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    await open_doc(page, base, docs["B"])
    await page.wait_for_timeout(500)
    expect(not errors, f"console errors: {errors}")


@check("a list column is joined by its column, on the page and in the clipboard copy")
async def list_cell(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    async def as_list(route: Route) -> None:
        # The shipped profile has no list field, so one sample column is turned into one on the wire.
        response = await route.fetch()
        data = await response.json()
        field = next(field for field in data["fields"] if field["name"] == "thickness")
        field["cardinality"] = "many"
        for row in data["sample_rows"]:
            row["thickness"] = ["LiOH", "NiSO4"]
        await route.fulfill(response=response, json=data)

    await page.route(f"**/api/documents/{docs['B']}/dataset", as_list)
    await open_doc(page, base, docs["B"])
    await page.wait_for_selector('[data-slot="results-rows"] tr')
    shown = await page.text_content('[data-slot="results-rows"]') or ""
    expect("LiOH；NiSO4" in shown, f"the list cell reads {shown!r}")
    copied = await page.evaluate(
        "import('/tsv.js').then((tsv) => tsv.fieldText(['LiOH', 'NiSO4'], {cardinality: 'many'}))"
    )
    expect(copied == "LiOH; NiSO4", f"the clipboard copy of a list reads {copied!r}")


@check("both lanes of 事实对照 fit side by side at 1440 px", width=1440)
async def facts_1440(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.click('#document-view .tab-bar [data-tab="facts"]')
    await lanes_visible(page)


@check("both lanes of 事实对照 fit side by side at 1280 px", width=1280)
async def facts_1280(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.click('#document-view .tab-bar [data-tab="facts"]')
    await lanes_visible(page)


@check("both lanes of 事实对照 fit side by side at 1920 px", width=1920, height=1080)
async def facts_1920(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.click('#document-view .tab-bar [data-tab="facts"]')
    await lanes_visible(page)


@check(
    "both lanes of 事实对照 fit side by side at 1600 px, with the summary above and the viewer pane beside",
    width=1600,
    height=1000,
)
async def facts_1600(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.click('#document-view .tab-bar [data-tab="facts"]')
    await lanes_visible(page)
    boxes = await page.evaluate(
        """() => ['.doc-summary', '.doc-main'].map((s) => document.querySelector(s).getBoundingClientRect().toJSON())"""
    )
    summary, main = boxes
    expect(summary["bottom"] <= main["top"], f"the summary panel is not in flow above the content: {boxes}")


PANE = """() => {
  const pane = document.querySelector('.viewer-pane').getBoundingClientRect();
  const viewer = document.querySelector('[data-slot="viewer"]');
  return { right: pane.right, width: window.innerWidth, inside: paneCompare(viewer.closest('.viewer-pane')) };
  function paneCompare(el) { return Boolean(el && document.querySelector('.viewer-pane').contains(el)); }
}"""


SEAM_ALIGN = """() => {
  const h = document.querySelector('.pane-resizer').getBoundingClientRect();
  const p = document.querySelector('.viewer-pane').getBoundingClientRect();
  return h.x + h.width / 2 - p.x;
}"""


@check("the viewer pane stands beside the content at 1280 px", width=1280)
async def pane_1280(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    pane = await page.evaluate(PANE)
    expect(0 < pane["right"] <= pane["width"], f"the viewer pane is not inside the viewport: {pane}")
    expect(pane["inside"], "the viewer does not live in the viewer pane")
    delta = await page.evaluate(SEAM_ALIGN)
    expect(abs(delta) <= 1, f"the drag handle is off the real seam by {delta}px (content padding drift)")


@check("the viewer pane stands beside the content at 1440 px, and a fact click scrolls no page", width=1440)
async def pane_1440(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.click('#document-view .tab-bar [data-tab="facts"]')
    pane = await page.evaluate(PANE)
    expect(0 < pane["right"] <= pane["width"], f"the viewer pane is not inside the viewport: {pane}")
    expect(pane["inside"], "the viewer does not live in the viewer pane")
    delta = await page.evaluate(SEAM_ALIGN)
    expect(abs(delta) <= 1, f"the drag handle is off the real seam by {delta}px (content padding drift)")
    before = await page.evaluate("window.scrollY")
    # focus without scrolling (a click would auto-scroll to the row): the keyboard path activates the row too
    await page.evaluate("document.querySelector('.facts tbody tr[data-index=\"0\"]').focus({preventScroll: true})")
    await page.keyboard.press("Enter")
    await page.wait_for_timeout(400)
    expect(await page.evaluate("window.scrollY") == before, "clicking a fact scrolled the page")
    box = await page.evaluate("document.querySelector('.viewer-pane').getBoundingClientRect().toJSON()")
    expect(box["top"] >= 56 and box["top"] < 900 and box["right"] <= 1440, f"the viewer pane is not on screen: {box}")
    # Stickiness: scrolling the content pins the pane under the topbar, wholly inside the viewport.
    await page.evaluate("window.scrollTo(0, 400)")
    await page.wait_for_timeout(200)
    box = await page.evaluate("document.querySelector('.viewer-pane').getBoundingClientRect().toJSON()")
    expect(box["top"] == 56 and box["bottom"] <= 901, f"the scrolled pane did not pin under the topbar: {box}")


@check("the viewer pane stands beside the content at 1920 px", width=1920, height=1080)
async def pane_1920(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    pane = await page.evaluate(PANE)
    expect(0 < pane["right"] <= pane["width"], f"the viewer pane is not inside the viewport: {pane}")
    expect(pane["inside"], "the viewer does not live in the viewer pane")


@check("the topbar's viewer toggle collapses and restores the pane, survives a reload, and `]` works")
async def viewer_collapse(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await page.goto(base)
    await page.wait_for_selector("#corpus-view:not(.hidden)")
    expect(await page.is_hidden("#viewer-toggle"), "the viewer toggle shows outside a document page")
    await open_doc(page, base, docs["A"])
    expect(await page.is_visible("#viewer-toggle"), "the document page does not show the viewer toggle")
    expect(await page.get_attribute("#viewer-toggle", "aria-expanded") == "true", "not marked expanded")
    await page.click("#viewer-toggle")
    expect(
        await page.get_attribute("#document-view", "data-viewer") == "collapsed",
        "the collapse toggle did not collapse the pane",
    )
    expect(await page.get_attribute("#viewer-toggle", "aria-expanded") == "false", "aria-expanded did not follow")
    expect(await page.get_attribute("#viewer-toggle", "title") == "展开预览", "the tooltip did not follow")
    expect(await page.evaluate("localStorage.getItem('paperfacts.viewer-collapsed')") == "1", "not remembered")
    expect(await page.is_hidden(".viewer-pane"), "the collapsed pane still takes width")
    await page.reload()
    await page.wait_for_selector(
        '[data-slot="viewer"] .page, [data-slot="viewer"] .viewer-empty, [data-slot="viewer"] .viewer-bar',
        state="attached",
    )
    expect(
        await page.get_attribute("#document-view", "data-viewer") == "collapsed",
        "a reload forgot the collapsed pane",
    )
    await page.keyboard.press("]")
    expect(
        await page.get_attribute("#document-view", "data-viewer") == "",
        "the `]` shortcut did not restore the pane",
    )
    expect(await page.is_visible(".viewer-pane"), "the expanded pane is not on screen")
    expect(
        await page.evaluate("localStorage.getItem('paperfacts.viewer-collapsed')") == "0",
        "the expansion is not remembered",
    )
    await page.click("#viewer-toggle")
    await page.keyboard.press("]")  # leave the state off for later checks
    expect(await page.evaluate("localStorage.getItem('paperfacts.viewer-collapsed')") == "0", "left collapsed")


PANE_WIDTH = "document.querySelector('.viewer-pane').getBoundingClientRect().width"


@check("dragging the viewer seam resizes the pane, keeps the lanes fitting, and remembers it")
async def pane_drag(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    box = await page.locator(".pane-resizer").bounding_box()
    before = await page.evaluate(PANE_WIDTH)
    expect(before == 480, f"the pane does not start at its 480px default: {before}")
    await page.mouse.move(box["x"] + box["width"] / 2, 450)
    await page.mouse.down()
    await page.mouse.move(box["x"] + box["width"] / 2 - 120, 450, steps=5)  # dragging left widens (right-anchored)
    live = await page.evaluate(PANE_WIDTH)
    expect(live > before, f"the pane did not widen while dragging: {before} -> {live}")
    expect(await page.evaluate("window.getSelection().toString()") == "", "dragging selected text")
    await page.mouse.up()
    after = await page.evaluate(PANE_WIDTH)
    stored = await page.evaluate("localStorage.getItem('paperfacts.viewer-pane-width')")
    expect(stored == str(round(after)), f"the dragged width was not remembered: {after} vs {stored}")
    await page.reload()
    await page.wait_for_selector("#document-view:not(.hidden) h1")
    kept = await page.evaluate(PANE_WIDTH)
    expect(abs(kept - after) <= 1, f"reload dropped the pane width: {after} -> {kept}")
    await page.click('#document-view .tab-bar [data-tab="facts"]')
    await lanes_visible(page)
    await page.evaluate("localStorage.removeItem('paperfacts.viewer-pane-width')")


@check("the viewer drag clamps at its bounds", width=1920, height=1080)
async def pane_drag_clamp(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])

    async def seam() -> tuple[float, float]:
        box = await page.locator(".pane-resizer").bounding_box()
        return box["x"] + box["width"] / 2, 450

    mid = await seam()
    await page.mouse.move(*mid)
    await page.mouse.down()
    await page.mouse.move(mid[0] - 900, 450, steps=5)
    await page.mouse.up()
    # With the rail at its 300px default the live max is 1920-56-570-300 = 994.
    expect(
        await page.evaluate(PANE_WIDTH) == 994,
        f"the pane passed its live max: {await page.evaluate(PANE_WIDTH)}",
    )
    mid = await seam()  # the seam moved with the pane: re-aim before dragging the other way
    await page.mouse.move(*mid)
    await page.mouse.down()
    await page.mouse.move(mid[0] + 900, 450, steps=5)  # dragging right narrows: 994 far overshoots the floor
    await page.mouse.up()
    width = await page.evaluate(PANE_WIDTH)
    expect(width == 300, f"the pane went below its 300px floor: {width}")
    await page.evaluate("localStorage.removeItem('paperfacts.viewer-pane-width')")


@check("the viewer seam is hidden when the pane is collapsed, and expanding restores the dragged width")
async def pane_drag_collapsed(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    box = await page.locator(".pane-resizer").bounding_box()
    await page.mouse.move(box["x"] + box["width"] / 2, 450)
    await page.mouse.down()
    await page.mouse.move(box["x"] + box["width"] / 2 - 80, 450, steps=5)
    await page.mouse.up()
    dragged = await page.evaluate(PANE_WIDTH)
    await page.click("#viewer-toggle")
    expect(await page.is_hidden(".pane-resizer"), "the seam handle shows on a collapsed pane")
    await page.click("#viewer-toggle")
    expect(await page.is_visible(".viewer-pane"), "the expanded pane is not on screen")
    expect(
        abs(await page.evaluate(PANE_WIDTH) - dragged) <= 1,
        f"expanding dropped the dragged width: {dragged} -> {await page.evaluate(PANE_WIDTH)}",
    )
    await page.evaluate("localStorage.removeItem('paperfacts.viewer-pane-width')")


@check("the viewer seam resizes from the keyboard (inverted arrows) and resets on double-click")
async def pane_resize_keyboard(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.focus(".pane-resizer")
    expect(await page.get_attribute(".pane-resizer", "role") == "separator", "the handle is no separator")
    expect(await page.get_attribute(".pane-resizer", "aria-orientation") == "vertical", "no vertical orientation")
    await page.keyboard.press("ArrowLeft")  # left widens: the pane is right-anchored
    width = await page.evaluate(PANE_WIDTH)
    expect(width == 496, f"ArrowLeft did not widen by 16px: {width}")
    await page.keyboard.press("ArrowRight")
    width = await page.evaluate(PANE_WIDTH)
    expect(width == 480, f"ArrowRight did not narrow by 16px: {width}")
    await page.keyboard.press("End")
    width = await page.evaluate(PANE_WIDTH)
    expect(width == 514, f"End did not reach the 1440px live max: {width}")
    await page.keyboard.press("Home")
    expect(await page.evaluate(PANE_WIDTH) == 300, "Home did not reach the 300px floor")
    await page.dblclick(".pane-resizer")
    expect(await page.evaluate(PANE_WIDTH) == 480, "double-click did not reset to 480px")
    expect(
        await page.evaluate("localStorage.getItem('paperfacts.viewer-pane-width')") is None,
        "the reset kept the stored width",
    )


@check("the pane's band defaults hold at 1280 and 1440 with no stored key, and follow a resize")
async def pane_band_defaults(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    expect(
        await page.evaluate(PANE_WIDTH) == 480,
        f"the 1440 band default is not 480px: {await page.evaluate(PANE_WIDTH)}",
    )
    await page.set_viewport_size({"width": 1280, "height": 900})
    await page.wait_for_timeout(100)  # one rAF tick for the debounced re-clamp / re-default
    expect(
        await page.evaluate(PANE_WIDTH) == 340,
        f"the 1280 band default is not 340px: {await page.evaluate(PANE_WIDTH)}",
    )
    await page.set_viewport_size({"width": 1440, "height": 900})
    await page.wait_for_timeout(100)
    expect(
        await page.evaluate(PANE_WIDTH) == 480,
        f"the band default did not return to 480px: {await page.evaluate(PANE_WIDTH)}",
    )


@check("a seeded pane width is clamped to the live max on load, and the lanes still fit")
async def pane_seeded_reload(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.evaluate("localStorage.setItem('paperfacts.viewer-pane-width', '700')")
    await page.reload()
    await page.wait_for_selector("#document-view:not(.hidden) h1")
    # The boot clamp caps the stored 700 at the 1440 live max, 1440-56-570-300 = 514.
    expect(
        await page.evaluate(PANE_WIDTH) == 514,
        f"the stored 700px was not clamped at boot: {await page.evaluate(PANE_WIDTH)}",
    )
    await page.click('#document-view .tab-bar [data-tab="facts"]')
    await lanes_visible(page)
    await page.set_viewport_size({"width": 1280, "height": 900})
    await page.wait_for_timeout(100)
    expect(
        await page.evaluate(PANE_WIDTH) == 354,
        f"the pane was not re-clamped at 1280: {await page.evaluate(PANE_WIDTH)}",
    )
    await page.evaluate("localStorage.removeItem('paperfacts.viewer-pane-width')")
    await page.set_viewport_size({"width": 1440, "height": 900})


@check("shrinking the viewport re-clamps a stored pane width and never overflows")
async def pane_viewport_shrink(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.evaluate("localStorage.setItem('paperfacts.viewer-pane-width', '514')")
    await page.reload()
    await page.wait_for_selector("#document-view:not(.hidden) h1")
    expect(await page.evaluate(PANE_WIDTH) == 514, f"the stored 514px did not load: {await page.evaluate(PANE_WIDTH)}")
    await page.set_viewport_size({"width": 1280, "height": 900})
    await page.wait_for_timeout(100)  # one rAF tick for the debounced re-clamp
    expect(
        await page.evaluate(PANE_WIDTH) == 354,
        f"the pane was not re-clamped after the shrink: {await page.evaluate(PANE_WIDTH)}",
    )
    overflow = await page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    expect(overflow <= 1, f"the page overflows {overflow}px sideways after the shrink")
    await page.evaluate("localStorage.removeItem('paperfacts.viewer-pane-width')")
    await page.set_viewport_size({"width": 1440, "height": 900})


@check("dragging the pane seam leaves the zoom state alone")
async def pane_zoom_smoke(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.wait_for_selector('[data-slot="viewer"] .page')
    level = page.locator(".viewer-bar .zoom-level")
    expect(await level.text_content() == "100%", "the zoom level does not start at 100%")
    page_width = await page.evaluate(
        "document.querySelector('[data-slot=\"viewer\"] .page').getBoundingClientRect().width"
    )
    box = await page.locator(".pane-resizer").bounding_box()
    await page.mouse.move(box["x"] + box["width"] / 2, 450)
    await page.mouse.down()
    await page.mouse.move(box["x"] + box["width"] / 2 - 60, 450, steps=5)
    await page.mouse.up()
    expect(await level.text_content() == "100%", "the pane drag disturbed the zoom level")
    after_drag = await page.evaluate(
        "document.querySelector('[data-slot=\"viewer\"] .page').getBoundingClientRect().width"
    )
    expect(after_drag > page_width, f"the page did not follow the wider pane: {page_width} -> {after_drag}")
    await page.click(".viewer-bar .zoomer button:last-child")
    expect(await level.text_content() == "125%", "the zoom ladder broke after a pane drag")
    await page.click(".viewer-bar .zoom-level")  # the percentage button resets to 100%
    await page.evaluate("localStorage.removeItem('paperfacts.viewer-pane-width')")


@check("the viewer's zoom controls step the ladder, disable at the ends and survive a reload")
async def zoom_controls(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"], fact=2)
    await page.wait_for_selector('[data-slot="viewer"] .page')
    level = page.locator(".viewer-bar .zoom-level")
    expect(await level.text_content() == "100%", f"the zoom level does not start at 100%: {await level.text_content()}")
    expect(await page.is_disabled(".viewer-bar .zoomer button:first-child"), "− is not disabled at 100%")

    async def page_width() -> float:
        return await page.evaluate(
            "document.querySelector('[data-slot=\"viewer\"] .page').getBoundingClientRect().width"
        )

    before = await page_width()
    await page.click(".viewer-bar .zoomer button:last-child")
    expect(await level.text_content() == "125%", "one + did not step to 125%")
    expect(await page_width() > before, "the page did not get wider")
    await page.wait_for_function("document.querySelectorAll('[data-slot=\"viewer\"] .hl').length > 0")
    # Step to the top: + disables there.
    for _ in range(6):
        await page.click(".viewer-bar .zoomer button:last-child")
    expect(await level.text_content() == "400%", "+ did not reach 400%")
    expect(await page.is_disabled(".viewer-bar .zoomer button:last-child"), "+ is not disabled at 400%")
    # The percentage button resets to 100%.
    await page.click(".viewer-bar .zoom-level")
    expect(await level.text_content() == "100%", "the percentage button did not reset to 100%")
    # Set an intermediate level, reload, it survives (localStorage).
    await page.click(".viewer-bar .zoomer button:last-child")
    expect(await level.text_content() == "125%", "one + did not step to 125%")
    await page.reload()
    await page.wait_for_selector('[data-slot="viewer"] .page')
    expect(await page.locator(".viewer-bar .zoom-level").text_content() == "125%", "a reload forgot the zoom level")


@check("ctrl/cmd+wheel over the page zooms; plain wheel scrolls instead")
async def zoom_wheel(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.wait_for_selector('[data-slot="viewer"] .page')
    await page.evaluate("document.querySelector('.page-viewport').scrollIntoView({block: 'center'})")
    center = await page.evaluate(
        "() => { const b = document.querySelector('.page-viewport').getBoundingClientRect();"
        " return [b.left + b.width / 2, b.top + b.height / 2]; }"
    )
    await page.mouse.move(*center)
    await page.keyboard.down("Control")
    await page.mouse.wheel(0, -240)
    await page.keyboard.up("Control")
    expect(await page.locator(".viewer-bar .zoom-level").text_content() == "125%", "ctrl+wheel up did not zoom in")
    await page.keyboard.down("Control")
    await page.mouse.wheel(0, 240)
    await page.keyboard.up("Control")
    expect(await page.locator(".viewer-bar .zoom-level").text_content() == "100%", "ctrl+wheel down did not zoom out")
    expect(await page.locator(".viewer-bar .zoom-level").text_content() == "100%", "plain wheel changed the zoom")
    # Slack for the scroll assertion: the document can already sit at max scroll here (the pane-mode
    # layout is barely taller than the viewport after the image loads), so reset to the top first —
    # otherwise the plain wheel has nowhere left to scroll and the check order-dependently fails.
    await page.evaluate("window.scrollTo(0, 0)")
    await page.wait_for_function("document.querySelector('[data-slot=\"viewer\"] .page img')?.naturalWidth > 0")
    before = await page.evaluate("window.scrollY")
    await page.mouse.wheel(0, 400)
    await page.wait_for_timeout(200)
    expect(await page.locator(".viewer-bar .zoom-level").text_content() == "100%", "plain wheel changed the zoom")
    expect(await page.evaluate("window.scrollY") > before, "plain wheel did not scroll the page")


@check("a failed hi-dpi render falls back to CSS zoom without errors or retry loops")
async def zoom_fallback(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.wait_for_selector('[data-slot="viewer"] .page')
    hi: list[str] = []
    errors: list[str] = []

    async def abort_220(route: Route) -> None:
        await route.abort()

    await page.route("**dpi=220*", abort_220)
    page.on("request", lambda request: hi.append(request.url) if "dpi=220" in request.url else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    await page.click(".viewer-bar .zoomer button:last-child")  # 125
    await page.click(".viewer-bar .zoomer button:last-child")  # 150
    await page.click(".viewer-bar .zoomer button:last-child")  # 175
    await page.click(".viewer-bar .zoomer button:last-child")  # 200
    expect(await page.locator(".viewer-bar .zoom-level").text_content() == "200%", "did not reach 200%")
    await page.click(".viewer-bar .zoomer button:last-child")  # 250 -> dpi 220 -> abort -> fallback
    expect(await page.locator(".viewer-bar .zoom-level").text_content() == "250%", "the zoom did not stay at 250%")
    await page.wait_for_function(
        "(() => { const img = document.querySelector('[data-slot=\"viewer\"] .page img');"
        " return img && img.src.includes('dpi=110') && img.naturalWidth > 0; })()"
    )
    # More zooming must not re-request the failed dpi.
    n = len(hi)
    await page.click(".viewer-bar .zoomer button:last-child")  # 300
    await page.wait_for_timeout(300)
    expect(len(hi) == n, f"the failed hi-dpi render was retried: {hi[n:]}")
    expect(not errors, f"page errors: {errors}")


async def zoom_ready(page: Page, clicks: int, level: str) -> None:
    """Zoom by `clicks` steps and wait for the render to settle (label + loaded image)."""
    for _ in range(clicks):
        await page.dispatch_event(".viewer-bar .zoomer button:last-child", "click")
    await page.wait_for_function(f"document.querySelector('.viewer-bar .zoom-level')?.textContent === '{level}'")
    await page.wait_for_function("document.querySelector('[data-slot=\"viewer\"] .page img')?.naturalWidth > 0")


@check("zooming keeps the reader's vertical place in the pane layout")
async def zoom_keeps_place_pane(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.wait_for_function("document.querySelector('[data-slot=\"viewer\"] .page img')?.naturalWidth > 0")
    # At 100% the page barely outgrows the pane, so climb to 250% first; the probed step (250 -> 300%)
    # stays on the same dpi, so the page's height scales exactly with its width.
    await zoom_ready(page, clicks=5, level="250%")
    await page.evaluate("document.querySelector('.viewer-pane').scrollTop = 600")
    before, max_before = await page.evaluate(
        "() => { const p = document.querySelector('.viewer-pane');"
        " return [p.scrollTop, p.scrollHeight - p.clientHeight]; }"
    )
    expect(before > 0, "the pane could not be scrolled for the vertical-preservation check")
    # dispatch_event: a real click would scroll the button into view and reset the pane's scroll.
    await page.dispatch_event(".viewer-bar .zoomer button:last-child", "click")
    await zoom_ready(page, clicks=0, level="300%")
    after, max_after = await page.evaluate(
        "() => { const p = document.querySelector('.viewer-pane');"
        " return [p.scrollTop, p.scrollHeight - p.clientHeight]; }"
    )
    expect(after > 0, f"zooming clamped the pane scroll to {after}")
    drift = abs(after / max_after - before / max_before)
    expect(drift <= 0.02, f"zooming moved the pane's scroll fraction: {before}/{max_before} -> {after}/{max_after}")


@check("zooming keeps the reader's vertical place in the stacked layout", width=1000, height=700)
async def zoom_keeps_place_stacked(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.wait_for_function("document.querySelector('[data-slot=\"viewer\"] .page img')?.naturalWidth > 0")
    await zoom_ready(page, clicks=5, level="250%")
    await page.evaluate("window.scrollTo(0, 800)")
    before, max_before = await page.evaluate(
        "() => { const s = document.scrollingElement; return [s.scrollTop, s.scrollHeight - s.clientHeight]; }"
    )
    expect(before > 0, "the window could not be scrolled for the vertical-preservation check")
    await page.dispatch_event(".viewer-bar .zoomer button:last-child", "click")
    await zoom_ready(page, clicks=0, level="300%")
    after, max_after = await page.evaluate(
        "() => { const s = document.scrollingElement; return [s.scrollTop, s.scrollHeight - s.clientHeight]; }"
    )
    expect(after > 0, f"zooming clamped the window scroll to {after}")
    drift = abs(after / max_after - before / max_before)
    expect(drift <= 0.02, f"zooming moved the window's scroll fraction: {before}/{max_before} -> {after}/{max_after}")


async def lanes_visible(page: Page) -> None:
    box = await page.evaluate(
        """() => {
          const wrap = document.querySelector('.facts .table-wrap').getBoundingClientRect();
          const b = document.querySelector('.facts th.lane-b').getBoundingClientRect();
          const w = document.querySelector('.facts .table-wrap');
          return { wrapRight: wrap.right, laneRight: b.right, scroll: w.scrollWidth, client: w.clientWidth };
        }"""
    )
    expect(box["laneRight"] <= box["wrapRight"] + 1, f"PaddleOCR-VL column is cut off: {box}")


async def summary_facts(page: Page) -> dict[str, str]:
    return await page.evaluate(
        """() => Object.fromEntries([...document.querySelectorAll('#document-view .doc-fact')].map(
          (row) => [row.querySelector('dt').textContent, row.querySelector('dd').textContent]))"""
    )


@check("the summary panel shows the paper's parts, sample count, tally, stages, finish time and actions")
async def summary_panel(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["B"])
    facts = await summary_facts(page)
    expect(facts.get("组成") == "单个 PDF", f"组成 reads {facts.get('组成')!r}")
    expect(facts.get("样品") == "2 个", f"the sample count reads {facts.get('样品')!r}")
    expect(facts.get("比较结果") == "4一致2冲突", f"the tally reads {facts.get('比较结果')!r}")
    expect(re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d", facts.get("最近完成", "")), f"finished: {facts}")
    stages = await page.locator("#document-view .doc-summary .stages li").count()
    expect(stages == len(stage_names()), f"the panel lists {stages} stages")
    for action in ('[data-action="run"]', '[data-action="figures"]', '[data-slot="summary-download"]'):
        expect(await page.locator(f"#document-view .doc-summary {action}").count() == 1, f"no {action} in the panel")
    download = page.locator('#document-view [data-slot="summary-download"]')
    expect(await download.is_visible(), "the panel's 下载 Excel is hidden")
    same = await page.get_attribute('#document-view [data-slot="dataset-download"]', "href")
    expect(await download.get_attribute("href") == same, "the panel's 下载 Excel is another address")
    await open_doc(page, base, docs["U"])  # lanes and report stored, no dataset: not finished under this profile
    await page.wait_for_function("document.querySelector('#document-view h1')?.textContent.startsWith('U ')")
    facts = await summary_facts(page)
    expect(facts.get("最近完成") == "尚未完成", f"an unfinished paper's finish time reads {facts.get('最近完成')!r}")
    expect(await download.is_hidden(), "下载 Excel is offered without a table")


async def tab_state(page: Page) -> dict:
    return await page.evaluate(
        """() => {
          const tabs = [...document.querySelectorAll('#document-view .tab-bar [role="tab"]')];
          const on = tabs.filter((b) => b.getAttribute('aria-selected') === 'true').map((b) => b.dataset.tab);
          const panels = Object.fromEntries([...document.querySelectorAll('#document-view [data-panel]')].map(
            (s) => [s.dataset.panel, !s.hidden]));
          const bar = document.querySelector('#document-view .tab-bar').getBoundingClientRect();
          const labels = tabs.map((b) => b.textContent);
          return { on, panels, barTop: bar.top, barBottom: bar.bottom, labels, hash: location.hash };
        }"""
    )


@check("the tab bar offers the four tabs, results first; a click shows only its panel, without touching the URL")
async def tab_bar(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    state = await tab_state(page)
    expect(state["labels"] == ["结果表", "图中读数", "事实对照", "样品记录"], f"the bar reads {state['labels']}")
    expect(state["on"] == ["results"], f"on open the bar marks {state['on']}")
    expect(
        state["panels"] == {"results": True, "figures": False, "facts": False, "samples": False},
        f"the panels read {state['panels']}",
    )
    hash_before = state["hash"]
    # A hidden panel shortens the page; the bar pins only where the page can scroll past its resting place.
    for key in ("figures", "facts", "samples", "results"):
        await page.click(f'#document-view .tab-bar [data-tab="{key}"]')
        await page.wait_for_timeout(100)
        # From the top, where the bar is not yet pinned, read its natural place; a panel that leaves the
        # page too short to scroll past it cannot pin the bar, and there the check does not apply.
        await page.evaluate("scrollTo(0, 0)")
        await page.wait_for_timeout(100)
        natural = await page.evaluate(
            "document.querySelector('#document-view .tab-bar').getBoundingClientRect().top + scrollY"
        )
        await page.evaluate(f"scrollTo(0, {natural} + 60)")
        await page.wait_for_timeout(100)
        state = await tab_state(page)
        expect(state["on"] == [key], f"after a click the bar marks {state['on']}")
        shown = {name: visible for name, visible in state["panels"].items() if visible}
        expect(shown == {key: True}, f"after clicking {key} the visible panels are {shown}")
        if await page.evaluate(f"scrollY >= {natural} - 56"):
            expect(abs(state["barTop"] - 56) <= 1, f"the bar is not stuck under the topbar: {state['barTop']} at {key}")
        expect(state["hash"] == hash_before, f"a tab click changed the URL to {state['hash']}")


@check("a fact deep link lands on the 事实对照 tab, its row selected and on screen under the tab bar")
async def tab_fact_link(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"], fact=2)
    await page.wait_for_selector('tr.selected[data-index="2"]')
    state = await tab_state(page)
    expect(state["on"] == ["facts"], f"the deep link left the bar on {state['on']}")
    expect(await page.evaluate("location.hash") == f"#/doc/{docs['A']}/fact/2", "the deep link lost its fact")
    box = await page.evaluate("document.querySelector('tr.selected').getBoundingClientRect().toJSON()")
    expect(state["barBottom"] <= box["top"] and box["bottom"] <= 901, f"the selected fact is at {box}")
    highlighted = await page.locator('[data-slot="viewer"] .hl').count()
    expect(highlighted > 0, "the viewer highlights none of the fact's blocks")
    await page.click('#document-view .tab-bar [data-tab="results"]')
    await page.wait_for_timeout(300)
    expect(await page.evaluate("location.hash") == f"#/doc/{docs['A']}/fact/2", "a tab click dropped the fact")
    expect(await page.locator('tr.selected[data-index="2"]').count() == 1, "a tab click unselected the fact")


@check("the processing log stays below the tab panels and appears once a job runs")
async def job_log(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["W"])
    expect(
        await page.evaluate("document.querySelector('#document-view details.joblog').classList.contains('hidden')"),
        "the log is offered with no job",
    )
    await page.click('#document-view [data-action="run"]')
    await page.wait_for_selector("#document-view details.joblog:not(.hidden)", timeout=5000)
    await jobs_idle(page)


@check("the open tab survives a job-finish reload, and a document switch returns to the results table")
async def tab_persistence(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["W"])
    await page.click('#document-view .tab-bar [data-tab="figures"]')
    # Mark the bar: a finish re-clones the template, so the mark vanishing proves the re-render happened.
    await page.evaluate("document.querySelector('#document-view .tab-bar').dataset.marked = '1'")
    await page.click('#document-view [data-action="run"]')
    await page.wait_for_selector("text=处理完成", timeout=len(stage_names()) * STAGE_SECONDS * 1000 + 10000)
    await page.wait_for_function("!document.querySelector('#document-view .tab-bar')?.dataset.marked", timeout=5000)
    # The finish reloaded the whole view (a fresh template clone): the tab must survive it.
    expect((await tab_state(page))["on"] == ["figures"], "the job-finish reload lost the tab")
    expect(await page.is_visible('[data-panel="figures"]'), "the figures panel did not survive the reload")
    # Another document: a fresh view starts on the results table again. The h1 the helper waits for belongs
    # to the old view until the swap, so the document id is what says the new view is on screen.
    await open_doc(page, base, docs["A"])
    await page.wait_for_function(
        f"document.querySelector('#document-view [data-slot=id]')?.textContent.startsWith('{docs['A'][:4]}')",
        timeout=5000,
    )
    state = await tab_state(page)
    expect(state["on"] == ["results"], f"a document switch left the bar on {state['on']}")
    expect(state["panels"]["results"], "the results panel is not shown on a fresh document")


@check("the tab bar exposes tablist/tab/tabpanel roles, keeps aria-selected in sync and arrows between tabs")
async def tabs_a11y(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    expect(await page.get_attribute("#document-view .tab-bar", "role") == "tablist", "the bar is not a tablist")
    roles = await page.evaluate(
        "[...document.querySelectorAll('#document-view .tab-bar > button')].map((b) => b.getAttribute('role'))"
    )
    expect(roles == ["tab"] * 4, f"the bar's buttons read {roles}")
    panels = await page.evaluate(
        "[...document.querySelectorAll('#document-view [data-panel]')].map((s) => s.getAttribute('role'))"
    )
    expect(panels == ["tabpanel"] * 4, f"the panels read {panels}")
    # Every aria-controls id must resolve: a dangling reference is invisible to keyboard users.
    dangling = await page.evaluate(
        "[...document.querySelectorAll('#document-view .tab-bar > button')]"
        ".map((b) => b.getAttribute('aria-controls'))"
        ".filter((id) => !document.getElementById(id))"
    )
    expect(not dangling, f"tabs point at ids that do not exist: {dangling}")
    # Roving tabindex plus arrow keys: focus and selection move together.
    await page.focus('#document-view .tab-bar [data-tab="results"]')
    expect(
        await page.evaluate("document.activeElement.getAttribute('tabindex')") == "0",
        "the selected tab does not own the roving tabindex",
    )
    for key, target in (("ArrowRight", "figures"), ("End", "samples"), ("ArrowLeft", "facts"), ("Home", "results")):
        await page.keyboard.press(key)
        expect(
            await page.evaluate("document.activeElement.dataset.tab") == target,
            f"{key} focused {await page.evaluate('document.activeElement.dataset.tab')!r}, not {target}",
        )
        expect((await tab_state(page))["on"] == [target], f"{key} did not select {target}")


async def narrow_doc(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    overflow = await page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    expect(overflow <= 0, f"the page scrolls {overflow}px sideways")
    right = await page.evaluate(
        "document.querySelector('#document-view [data-action=run]').getBoundingClientRect().right"
    )
    expect(right <= 390, f"重新处理 ends at {right}px")


@check("on a phone the home table starts near the top", width=390, height=844)
async def narrow_home(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await page.goto(f"{base}/#/")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    top = await page.evaluate("document.querySelector('#corpus-view').getBoundingClientRect().top + window.scrollY")
    expect(top < 900, f"the home table starts {top}px down the page")
    overflow = await page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    expect(overflow <= 0, f"the home page scrolls {overflow}px sideways")


@check("clicking a fact below the viewer brings the viewer into view", width=390, height=844)
async def reveal(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.click('#document-view .tab-bar [data-tab="facts"]')
    await page.locator('[data-slot="rows"] tr').first.click()
    await page.wait_for_timeout(800)
    top = await page.evaluate("document.querySelector('[data-slot=viewer]').getBoundingClientRect().top")
    expect(0 <= top < 844, f"the viewer is at {top}px, off screen")


@check("keyboard: toggles keep focus, rows and cells are reachable")
async def keyboard(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await page.goto(f"{base}/#/")
    await page.wait_for_selector("#corpus-view:not(.hidden) button.expand")
    await page.focus("#corpus-view button.expand")
    key = await page.evaluate("document.activeElement.dataset.focus")
    await page.keyboard.press("Enter")
    expect(await page.evaluate("document.activeElement.dataset.focus") == key, "▸ lost focus after Enter")
    await page.click('#corpus-view [data-focus="picker"]')
    await page.focus('#corpus-view [data-focus="field:thickness"]')
    await page.keyboard.press("Space")
    expect(
        await page.evaluate("document.activeElement.dataset.focus") == "field:thickness",
        "the field checkbox lost focus",
    )
    await page.keyboard.press("Space")  # a second press must still act
    expect(await page.is_checked('#corpus-view [data-focus="field:thickness"]'), "the second Space did nothing")
    await page.keyboard.press("Escape")

    await open_doc(page, base, docs["A"])
    await page.click('#document-view .tab-bar [data-tab="facts"]')
    await page.focus('[data-slot="rows"] tr[data-index="4"]')
    await page.keyboard.press("Enter")
    expect((await page.evaluate("location.hash")).endswith("/fact/4"), "Enter on a fact row did not select it")
    expect(await page.locator("td.cell[data-sources][tabindex='0']").count() > 0, "filled cells are not focusable")
    await page.goto("about:blank")  # a reload would resume Tab from the row focused last
    await open_doc(page, base, docs["A"])
    await page.keyboard.press("Tab")
    focused = await page.evaluate("document.activeElement.id || document.activeElement.outerHTML.slice(0, 80)")
    expect(focused == "skip-link", f"the first Tab lands on {focused!r}, not the skip link")
    await page.keyboard.press("Enter")
    expect(await page.evaluate("document.activeElement.id") == "content", "the skip link does not reach the content")
    expect((await page.evaluate("location.hash")).startswith(f"#/doc/{docs['A']}"), "the skip link navigated away")


@check("a paper of two entity types has one results table and one records group per entity")
async def entity_tables(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await open_doc(page, docs["entities"], docs["E"])
    await page.wait_for_selector('.entity-table[data-entity="wear_test"] tbody tr')
    first = await page.text_content('[data-slot="results-head"] th')
    expect(first == "涂层", f"the primary table's first column is {first!r}")
    expect(await page.text_content('[data-slot="results-entity"]') == "涂层", "the primary table is not named")
    main = await page.text_content('[data-slot="results-table"]') or ""
    expect("test_temperature" not in main, "the primary table shows another entity's field")
    expect(main.count("S1") == 1 and "S2" in main, f"the primary table rows read {main!r}")
    other = await page.text_content('.entity-table[data-entity="wear_test"]') or ""
    expect("磨损测试" in other and "test_temperature" in other, f"the wear test table reads {other!r}")
    expect("coating_thickness" not in other, "the wear test table shows the coatings' field")
    heads = await page.locator("#document-view .lane .lane-entity").all_text_contents()
    expect(heads == ["涂层 · 2", "磨损测试 · 1"] * 2, f"the lanes' entity groups read {heads}")
    scopes = await page.locator('[data-slot="rows"] td.mono').all_text_contents()
    expect(scopes == ["涂层 · S1", "磨损测试 · S1"], f"the comparison scopes read {scopes}")
    tiles = await page.locator(".kpi.samples .label").all_text_contents()
    expect(tiles == ["涂层配对", "磨损测试配对"], f"the matching tiles read {tiles}")
    # A reference names the coating's row by its id, labelled with the entity it is a row of.
    labelled = page.locator('.entity-table[data-entity="wear_test"] tbody td.cell', has=page.locator(".unit"))
    reference = await labelled.all_text_contents()
    expect(any(text.startswith("S1 涂层") for text in reference), f"the reference cell reads {reference!r}")


@check("an empty cell of the second entity marks that entity's records, not another's S1")
async def entity_evidence(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await open_doc(page, docs["entities"], docs["E"])
    await page.click('[data-slot="results-chips"] [data-focus="show-empty"]')  # wear_mode is empty on every row
    await page.wait_for_selector('.entity-table[data-entity="wear_test"] td.cell.empty[data-field="wear_mode"]')
    empty = page.locator('.entity-table[data-entity="wear_test"] td.cell.empty[data-field="wear_mode"]')
    expect(await empty.get_attribute("aria-label") == "两路冲突", "the refused cell lost its quality row")
    await empty.click()
    expect((await tab_state(page))["on"] == ["samples"], "the empty cell did not open the records tab")
    expect(await page.is_visible('[data-panel="samples"]'), "the records panel stayed hidden")
    marked = await page.evaluate(
        "[...document.querySelectorAll('.field.evidence')].map((row) => row.closest('.sample').dataset.entity)"
    )
    expect(marked == ["wear_test", "wear_test"], f"the marked records belong to {marked}")


@check("an empty cell under a single declared entity marks both lanes' records")
async def one_entity_evidence(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await open_doc(page, docs["one-entity"], docs["O"])
    await page.wait_for_selector('[data-slot="results-rows"] tr')
    expect(await page.locator(".entity-table, .lane-entity").count() == 0, "an entity group is shown")
    await page.click('[data-slot="results-chips"] [data-focus="show-empty"]')
    empty = page.locator('td.cell.empty[data-field="solvent"]')
    await empty.first.wait_for()
    await empty.first.click()
    expect((await tab_state(page))["on"] == ["samples"], "the empty cell did not open the records tab")
    expect(await page.is_visible('[data-panel="samples"]'), "the records panel stayed hidden")
    marked = await page.evaluate(
        "[...document.querySelectorAll('.field.evidence')].map((row) => row.closest('.sample').dataset.entity)"
    )
    expect(marked == ["coating", "coating"], f"the marked records belong to {marked}")


@check("the home table of a two-entity library switches between its entity types, and copies what it shows")
async def entity_corpus(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await page.goto(f"{docs['entities']}/#/")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    chips = await page.locator("#corpus-view .entity-switch .chip").all_text_contents()
    expect(chips == ["涂层", "磨损测试"], f"the entity chips read {chips}")
    text = await page.text_content("#corpus-view") or ""
    expect("coating_thickness" in text and "test_temperature" not in text, f"the corpus table reads {text!r}")
    expect("只列出" not in text, "the corpus view still says it shows the primary entity only")
    expect("2 个涂层" in text, f"the sample count counts another entity's rows: {text!r}")
    download = await page.get_attribute("#corpus-view a.download", "href")
    # A keyboard switch keeps the focus on the chip it pressed.
    await page.focus('#corpus-view [data-focus="entity:wear_test"]')
    await page.keyboard.press("Enter")
    await page.wait_for_selector('#corpus-view [data-focus="entity:wear_test"][aria-pressed="true"]')
    focused = await page.evaluate("document.activeElement.dataset.focus")
    expect(focused == "entity:wear_test", f"the focus moved to {focused!r}")
    heads = await page.locator("#corpus-view thead th").all_text_contents()
    # A secondary entity's rows are flattened rows, as a search's are: the sample's id first, its paper second.
    expect(heads[:3] == ["磨损测试", "论文", "可用/一致"], f"the wear test table's heads read {heads}")
    expect(any("test_temperature" in head for head in heads), f"the wear test table lacks its field: {heads}")
    expect(not any("coating_thickness" in head or "precursor" in head for head in heads), f"other fields: {heads}")
    rows = page.locator("#corpus-view tbody tr")
    expect(await rows.count() == 1, f"the wear test table has {await rows.count()} rows")
    cells = await rows.first.locator("td").all_text_contents()
    expect(cells[0] == "S1" and any(cell.startswith("S1 涂层") for cell in cells[2:]), f"the row reads {cells}")
    # The entity shown is part of the home query (explorer.js), so a reload or a shared link shows the same table.
    hash_ = await page.evaluate("location.hash")
    expect(hash_ == "#/?e=wear_test", f"the entity choice is not in the address: {hash_!r}")
    expect(await page.get_attribute("#corpus-view a.download", "href") == download, "the Excel link changed")
    await page.evaluate("navigator.clipboard.writeText = async (text) => { window.__copied = text; }")
    await page.click("#corpus-view button.copy-table")
    copied = (await page.evaluate("window.__copied") or "").split("\n")
    expect(len(copied) == 2, f"the copy has {len(copied)} lines: {copied}")
    header, line = (row.split("\t") for row in copied)
    expect(header[:3] == heads[:3] and len(header) == len(heads), f"the copied header reads {header}")
    expect(line[0] == "S1" and line[1] == cells[1] and "S1" in line[3:], f"the copied row reads {line}")
    await page.click('#corpus-view [data-focus="entity:coating"]')
    await page.wait_for_selector('#corpus-view [data-focus="entity:coating"][aria-pressed="true"]')
    expect("2 个涂层" in (await page.text_content("#corpus-view") or ""), "the primary table did not come back")
    expect(await page.evaluate("location.hash") == "#/", "the primary entity left a query behind")


# ---- the home table as a data table: sort, density, frozen columns, toolbar -----------------------------------------

# The sorting corpus: papers P1-P3 with a numeric field t = 80 / 85 / blank and r = 5 / 2 / 9, served in the order
# P3, P2, P1 (so "no sort" is told apart from ascending), and sixteen more numeric columns to scroll sideways.
SORT_FILLERS = [f"x{n:02d}" for n in range(1, 17)]
SORT_T = {"P1": 80.0, "P2": 85.0, "P3": None}
SORT_R = {"P1": 5.0, "P2": 2.0, "P3": 9.0}
SORT_SERVED = ["P3", "P2", "P1"]


async def sort_corpus(page: Page) -> None:
    """Serve the sorting corpus as /api/dataset: the real answer, with these fields and rows in place of its own."""

    async def corpus(route: Route) -> None:
        response = await route.fetch()
        data = await response.json()
        numeric = {"label": "", "scope": "sample", "description": "", "kind": "numeric", "cardinality": "one"}
        data["fields"] = [
            {**numeric, "name": "t", "unit": "nm"},
            {**numeric, "name": "r", "unit": "Ω/sq"},
            *({**numeric, "name": name, "label": f"填充列{name}", "unit": "nm"} for name in SORT_FILLERS),
        ]
        rows = []
        for name in SORT_SERVED:
            sample = {
                "sample_id": f"S-{name}",
                "available_fields": 2,
                "agree_fields": 2,
                "t": SORT_T[name],
                "r": SORT_R[name],
                **{filler: 1000.0 + n for n, filler in enumerate(SORT_FILLERS)},
            }
            row = {"document_id": f"{name.lower():0<16}", "name": name, "paper_row": sample, "sample_count": 1}
            rows.append({**row, "sample_rows": [sample]})
        data["rows"] = rows
        await route.fulfill(response=response, json=data)

    await page.route("**/api/dataset", corpus)


async def open_sort_corpus(page: Page, base: str) -> None:
    await sort_corpus(page)
    await page.goto(f"{base}/#/")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")


async def first_column(page: Page) -> list[str]:
    return [cell.strip() for cell in await page.locator("#corpus-view tbody tr td:first-child").all_text_contents()]


@check("a header click sorts the home table ascending, descending, then back to the server's order; blanks last")
async def corpus_sort(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_sort_corpus(page, base)
    expect(await first_column(page) == SORT_SERVED, f"the served order reads {await first_column(page)}")
    header = '#corpus-view thead th:has([data-sort="t"])'
    expect(await page.get_attribute(header, "aria-sort") == "none", "an unsorted header claims a sort")
    await page.click('#corpus-view [data-sort="t"]')
    expect(await first_column(page) == ["P1", "P2", "P3"], f"t ascending reads {await first_column(page)}")
    expect(await page.get_attribute(header, "aria-sort") == "ascending", "aria-sort is not ascending")
    expect("升序" in (await page.text_content(header) or ""), "the header does not say it is sorted ascending")
    # From the keyboard: Enter on the focused header sorts again and keeps the focus there.
    await page.focus('#corpus-view [data-sort="t"]')
    await page.keyboard.press("Enter")
    expect(await first_column(page) == ["P2", "P1", "P3"], f"t descending reads {await first_column(page)}")
    expect(await page.get_attribute(header, "aria-sort") == "descending", "aria-sort is not descending")
    expect("降序" in (await page.text_content(header) or ""), "the header does not say it is sorted descending")
    focused = await page.evaluate("document.activeElement.dataset.focus")
    expect(focused == "sort:t", f"the focus moved to {focused!r}")
    await page.click('#corpus-view [data-sort="t"]')
    expect(await first_column(page) == SORT_SERVED, f"a third click leaves {await first_column(page)}")
    expect(await page.get_attribute(header, "aria-sort") == "none", "a third click leaves a sort on the header")
    await page.click('#corpus-view [data-sort="r"]')
    expect(await first_column(page) == ["P2", "P1", "P3"], f"r ascending reads {await first_column(page)}")
    sort = await page.evaluate("import('/corpus.js').then((corpus) => corpus.getSort())")
    expect(sort == {"key": "r", "dir": "asc"}, f"getSort() reads {sort}")
    count = (await page.text_content("#corpus-view .row-count") or "").strip()
    expect(count == "3 篇论文 · 3 个样品", f"the row count reads {count!r}")


@check("the clipboard copy of a sorted home table is its visible columns in the sorted order")
async def corpus_sorted_copy(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_sort_corpus(page, base)
    await page.click('#corpus-view [data-sort="t"]')
    await page.evaluate("navigator.clipboard.writeText = async (text) => { window.__copied = text; }")
    await page.click("#corpus-view button.copy-table")
    lines = [line.split("\t") for line in (await page.evaluate("window.__copied") or "").split("\n")]
    heads = await page.locator("#corpus-view thead th").count()
    expect(len(lines) == 4 and all(len(line) == heads for line in lines), f"the copy is {lines}")
    expect(lines[0][:4] == ["论文", "样品", "可用/一致", "t (nm)"], f"the copied header reads {lines[0][:4]}")
    expect([line[0] for line in lines[1:]] == ["P1", "P2", "P3"], f"the copied rows are {lines[1:]}")
    expect([line[1] for line in lines[1:]] == ["S-P1", "S-P2", "S-P3"], f"the copied ids are {lines[1:]}")
    expect([line[3] for line in lines[1:]] == ["80", "85", ""], f"the copied t column is {lines[1:]}")


@check("the density switch survives a reload and the document page's results table follows it")
async def density(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    padding = "getComputedStyle(document.querySelector('{} .results-table tbody td')).paddingTop"
    await page.goto(f"{base}/#/")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    standard = await page.evaluate(padding.format("#corpus-view"))
    expect(await page.get_attribute('[data-focus="density:standard"]', "aria-pressed") == "true", "标准 not pressed")
    await page.click('#corpus-view [data-focus="density:compact"]')
    compact = await page.evaluate(padding.format("#corpus-view"))
    expect(compact != standard, f"紧凑 left the cell padding at {compact}")
    await page.reload()
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    expect(await page.get_attribute(".shell", "data-density") == "compact", "紧凑 did not survive a reload")
    expect(await page.get_attribute('[data-focus="density:compact"]', "aria-pressed") == "true", "紧凑 not pressed")
    expect(await page.evaluate(padding.format("#corpus-view")) == compact, "the reloaded table is not compact")
    await open_doc(page, base, docs["A"])
    await page.wait_for_selector('[data-slot="results-rows"] tr')
    shown = await page.evaluate(padding.format("#document-view"))
    expect(shown == compact, f"the document's results table pads {shown}, the compact home table {compact}")
    await page.goto(f"{base}/#/")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    await page.click('#corpus-view [data-focus="density:standard"]')
    expect(await page.get_attribute(".shell", "data-density") is None, "标准 left the attribute on")
    expect(await page.evaluate(padding.format("#corpus-view")) == standard, "标准 did not restore the padding")


FROZEN = """() => {
  const box = (selector) => document.querySelector(selector).getBoundingClientRect();
  const first = box('#corpus-view thead th:nth-child(1)');
  const second = box('#corpus-view thead th:nth-child(2)');
  const cell = box('#corpus-view tbody tr:first-child td:nth-child(2)');
  return { firstRight: first.right, secondLeft: second.left, cellLeft: cell.left,
           scroll: document.querySelector('#corpus-view .table-wrap').scrollLeft };
}"""


async def frozen_columns(page: Page, base: str, density: str) -> None:
    await open_sort_corpus(page, base)
    await page.click(f'#corpus-view [data-focus="density:{density}"]')
    before = await page.evaluate(FROZEN)
    expect(abs(before["firstRight"] - before["secondLeft"]) <= 1, f"the frozen columns overlap or part: {before}")
    await page.evaluate("document.querySelector('#corpus-view .table-wrap').scrollLeft = 600")
    after = await page.evaluate(FROZEN)
    expect(after["scroll"] == 600, f"the table did not scroll 600px: {after}")
    for key in ("secondLeft", "cellLeft", "firstRight"):
        expect(abs(after[key] - before[key]) <= 0.5, f"{key} moved while scrolling ({density}): {before} -> {after}")


@check("the second frozen column stays put while the table scrolls 600px right, at 1440 px in both densities")
async def frozen_1440(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await frozen_columns(page, base, "standard")
    await frozen_columns(page, base, "compact")


@check("the second frozen column stays put while the table scrolls 600px right, at 1280 px", width=1280)
async def frozen_1280(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await frozen_columns(page, base, "standard")
    await frozen_columns(page, base, "compact")


@check("the home toolbar is one row: 筛选, 列, 显示空字段, 密度, 展开全部, the count, 复制, 下载")
async def toolbar(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await page.goto(f"{base}/#/")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    tops = await page.evaluate(
        """() => [...document.querySelectorAll(
                   '#corpus-view .table-toolbar :is(button, a, .row-count):not(.picker-pop *)')]
                 .map((element) => Math.round(element.getBoundingClientRect().top + element.offsetHeight / 2))"""
    )
    expect(len(tops) >= 7 and max(tops) - min(tops) <= 4, f"the toolbar wraps: centres at {tops}")
    picker = (await page.text_content('#corpus-view [data-focus="picker"]') or "").strip()
    expect(picker.startswith("列"), f"the field picker reads {picker!r}")
    slot = (await page.text_content('#corpus-view [data-slot="explore"]') or "").strip()
    expect(slot == "筛选", f"the explorer slot holds {slot!r}")
    count = (await page.text_content("#corpus-view .row-count") or "").strip()
    expect(re.fullmatch(r"\d+ 篇论文 · \d+ 个样品", count) is not None, f"the row count reads {count!r}")


# ---- the profile page -------------------------------------------------------------------------------------------

# A label a profile may carry: shown as text, never parsed as an element.
MARKUP = '<img src=x onerror="window.__xss=1">'


async def open_profile_page(page: Page, url: str) -> None:
    await page.goto(url)
    await page.wait_for_selector("#profile-view:not(.hidden) .profile-fields tbody tr")


@check("the profile page shows the default profile's fields and asks for a prompt only when it is opened")
async def profile_page(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    prompts: list[str] = []
    page.on("request", lambda request: prompts.append(request.url) if "/prompts" in request.url else None)
    await page.goto(f"{base}/#/")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    await page.click("#profile-link")
    await page.wait_for_selector("#profile-view:not(.hidden) .profile-fields tbody tr")
    expect(await page.evaluate("location.hash") == "#/profile", "the header link does not open the profile page")
    expect(await page.is_hidden("#corpus-view") and await page.is_hidden("#empty-state"), "home is still shown")
    attributes = set(await page.locator("#profile-view .profile-fields thead th small").all_text_contents())
    wanted = {"name", "kind", "level", "entity", "canonical_unit", "categories", "cardinality", "range_policy"}
    wanted |= {"references", "condition_rule", "valid_range"}
    expect(wanted <= attributes, f"the fields table lacks {sorted(wanted - attributes)}")
    definition = await page.evaluate("fetch('/api/profile').then((response) => response.json())")
    rows = await page.locator("#profile-view .profile-fields tbody tr").count()
    expect(rows == len(definition["fields"]), f"{rows} field rows for {len(definition['fields'])} fields")
    expect(not prompts, f"a prompt was asked before any preview was opened: {prompts}")
    system = page.locator("#profile-view .prompt-preview").first
    await system.locator("summary").click()
    await system.locator(".prompt-section pre").first.wait_for()
    expect(len(prompts) == 1, f"opening the preview asked {prompts}")
    titles = await system.locator(".prompt-section h3").all_text_contents()
    expect("inventory system prompt (passage mode)" in titles, f"the sections read {titles}")
    await system.locator("summary").click()
    await system.locator("summary").click()
    expect(len(prompts) == 1, "reopening the preview asked again")
    per_field = page.locator("#profile-view .prompt-preview").nth(1)
    await per_field.locator("summary").click()
    name = definition["fields"][0]["name"]
    await per_field.locator("select").select_option(name)
    await per_field.locator(".prompt-section pre").first.wait_for()
    expect(len(prompts) == 2 and f"field={name}" in prompts[1], f"the field question was asked as {prompts}")
    await page.click(".brand")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    expect(await page.is_hidden("#profile-view"), "the profile page stays up at home")


@check("a two-entity profile's page lists its entities and its reference field; the switcher keeps the page")
async def profile_page_entities(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await open_profile_page(page, f"{docs['multi']}/#/p/{DEMO}/profile")
    expect(await selected_profile(page) == DEMO, f"the switcher shows {await selected_profile(page)!r}")
    expect("示例领域" in (await page.text_content("#profile-view h1") or ""), "the page does not name the profile")
    expect(await page.get_attribute("#profile-link", "href") == f"#/p/{DEMO}/profile", "the header link is not DEMO's")
    entities = " ".join(await page.locator("#profile-view .profile-entities li").all_text_contents())
    expect("涂层" in entities and "磨损测试" in entities and "主实体" in entities, f"the entities read {entities!r}")
    reference = page.locator("#profile-view .profile-fields tbody tr", has_text="tested_coating")
    cells = await reference.locator("td").all_text_contents()
    expect("reference" in cells and "涂层 coating" in cells and "磨损测试 wear_test" in cells, f"it reads {cells}")
    link = page.locator(f'#profile-view .profile-documents a[href="#/p/{DEMO}/doc/{docs["M"]}"]')
    expect(await link.count() == 1, "the paper with results under the profile is not linked")
    await page.select_option("#profile-select", "tco")
    await page.wait_for_function("location.hash === '#/profile'")
    await page.wait_for_function("!document.querySelector('#profile-view h1')?.textContent.includes('示例领域')")
    expect(await page.locator("#profile-view .profile-entities").count() == 0, "the default's page lists entities")


@check("profile text holding markup is shown as text on the profile page, prompts included")
async def profile_page_markup(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    await open_profile_page(page, f"{docs['entities']}/#/profile")
    text = await page.text_content("#profile-view") or ""
    expect(MARKUP in text, "the markup label is not shown as text")
    expect(await page.locator("#profile-view img, #profile-view script").count() == 0, "the label became an element")
    per_field = page.locator("#profile-view .prompt-preview").nth(1)
    await per_field.locator("summary").click()
    await per_field.locator("select").select_option("tested_coating")
    await per_field.locator(".prompt-section pre").first.wait_for()
    question = " ".join(await per_field.locator(".prompt-section pre").all_text_contents())
    expect("Coatings" in question, "the reference field's question does not name the entity it refers to")
    # With no <img> or <script> anywhere on the page, prompts drawn, there is nothing left that could run it later.
    expect(await page.locator("#profile-view img, #profile-view script").count() == 0, "a prompt became an element")
    expect(await page.evaluate("window.__xss") is None and not errors, f"markup ran: {errors}")


@check("a profile page answer that lands after the reader left draws nothing")
async def profile_page_late(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await page.goto(f"{docs['multi']}/")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    await page.route(lambda url: f"/api/profiles/{DEMO}" in url, delayed(1.5))
    answer = settled(page, f"/api/profiles/{DEMO}")
    async with page.expect_request(lambda request: f"/api/profiles/{DEMO}" in request.url):
        await page.evaluate(f"location.hash = '#/p/{DEMO}/profile'")
    await page.evaluate(f"location.hash = '#/doc/{docs['M']}'")
    await page.wait_for_selector("#document-view:not(.hidden) h1")
    await answer.wait()
    # One more round trip, so the page has run whatever the late answer's handler would draw.
    await page.evaluate("fetch('/api/health').then((response) => response.json())")
    expect(await page.is_hidden("#profile-view"), "the late profile page drew over the document")
    expect(await page.is_visible("#document-view"), "the document view was hidden")


@check("a profile without entity types names no entity on the page")
async def implicit_entity(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await page.wait_for_selector('[data-slot="results-rows"] tr')
    expect(await page.locator(".entity-table, .lane-entity").count() == 0, "an entity group is shown")
    expect(await page.is_hidden('[data-slot="results-entity"]'), "the only table is named")
    expect(await page.locator(".kpi.samples").count() == 1, "more than one matching tile")
    scopes = await page.locator('[data-slot="rows"] td.mono').all_text_contents()
    expect(all("·" not in scope for scope in scopes), f"a scope names its entity: {scopes}")


# ---- several profiles on one server (docs["multi"]: the shipped profile, the default, and "demo") ----------------

DEMO = "demo"
# The routes whose answer is the same under every profile; every other /api/ request is asked under one.
PROFILE_FREE = re.compile(
    r"^/api/(?:health|profiles(?:/.*)?|jobs(?:/[^/]+)?|documents/[^/]+/(?:jobs|artifact/[^/]+|pages/[^/]+))$"
)


async def selected_profile(page: Page) -> str:
    return await page.eval_on_selector("#profile-select", "select => select.value")


def settled(page: Page, fragment: str) -> asyncio.Event:
    """Set once a request whose URL holds ``fragment`` has finished (or failed): its answer has reached the page."""
    event = asyncio.Event()

    def done(request: object) -> None:
        if fragment in request.url:  # type: ignore[attr-defined]
            event.set()

    page.on("requestfinished", done)
    page.on("requestfailed", done)
    return event


# Every job of the server is done: nothing is left to redraw a later check's page.
JOBS_IDLE = """() => fetch('/api/jobs').then((response) => response.json())
  .then((jobs) => jobs.every((job) => job.status !== 'queued' && job.status !== 'running'))"""
JOB_WAIT_S = len(stage_names()) * STAGE_SECONDS * 3 + 10


async def jobs_idle(page: Page) -> None:
    # Polled from here: wait_for_function takes a returned Promise as truthy rather than awaiting it.
    deadline = time.monotonic() + JOB_WAIT_S
    while not await page.evaluate(JOBS_IDLE):
        expect(time.monotonic() < deadline, "the server's jobs did not finish")
        await asyncio.sleep(0.2)


@check("a one-profile server shows no profile switcher")
async def single_profile_unchanged(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    expect(await page.is_hidden("#profile-switch"), "the switcher is shown on a one-profile server")
    expect(await page.is_hidden("#upload-profile"), "the upload names a profile on a one-profile server")
    expect(await page.is_visible("#health"), "the model line is hidden")


@check("an old link opens under the default profile, and the switcher says so")
async def old_links(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await open_doc(page, docs["multi"], docs["M"], fact=2)
    expect(await page.is_visible("#profile-switch"), "the switcher is hidden on a two-profile server")
    expect(await selected_profile(page) == "tco", f"the switcher shows {await selected_profile(page)!r}")
    expect(await page.locator('tr.selected[data-index="2"]').count() == 1, "the deep-linked fact is not selected")
    expect(await page.locator(".entity-table").count() == 0, "the default's page groups by entity")
    options = await page.locator("#profile-select option").all_text_contents()
    expect(any("（示例）" in option for option in options), f"the example profile is not marked: {options}")


@check("switching profile on a paper shows that profile's results, never a late answer of the old one")
async def profile_switch_race(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await page.goto(f"{docs['multi']}/")
    await page.wait_for_selector("#doc-list .doc-item")
    await page.route(f"**/api/documents/{docs['M']}/report", delayed(1.5))
    old_report = settled(page, f"/api/documents/{docs['M']}/report")
    async with page.expect_request(lambda request: f"/api/documents/{docs['M']}/report" in request.url):
        await page.evaluate(f"location.hash = '#/doc/{docs['M']}/fact/1'")
    await page.select_option("#profile-select", DEMO)
    await old_report.wait()
    await page.wait_for_selector('.entity-table[data-entity="wear_test"] tbody tr')
    hash_ = await page.evaluate("location.hash")
    expect(hash_ == f"#/p/{DEMO}/doc/{docs['M']}", f"the switch went to {hash_!r} (the fact must be dropped)")
    await page.wait_for_selector('.entity-table[data-entity="wear_test"] tbody tr')
    await page.click('#document-view .tab-bar [data-tab="facts"]')
    first = await page.text_content('[data-slot="results-head"] th')
    expect(first == "涂层", f"the primary table's first column is {first!r}")
    scopes = await page.locator('[data-slot="rows"] td.mono').all_text_contents()
    expect(scopes == ["涂层 · S1", "磨损测试 · S1"], f"the comparison scopes read {scopes}")
    expect(await page.locator("tr.selected").count() == 0, "a fact of the old profile's report stays selected")


@check("while another profile loads, the old profile's view takes no clicks: no fact of its report reaches the URL")
async def stale_view_inert(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await open_doc(page, docs["multi"], docs["M"])
    await page.click('#document-view .tab-bar [data-tab="facts"]')
    await page.route(lambda url: f"/report?profile={DEMO}" in url, delayed(1.5))
    report = settled(page, f"/report?profile={DEMO}")
    await page.select_option("#profile-select", DEMO)
    await page.wait_for_function("document.getElementById('document-view').inert")
    # A reader's click on a fact of the view still on screen (the default's): it must not select it.
    row = page.locator('[data-slot="rows"] tr[data-index="1"]')
    await row.evaluate("(node) => node.scrollIntoView({ block: 'center' })")
    box = await row.bounding_box()
    expect(box is not None, "the old view's facts are not on screen")
    await page.mouse.click(box["x"] + 20, box["y"] + box["height"] / 2)  # type: ignore[index]
    hash_ = await page.evaluate("location.hash")
    expect(hash_ == f"#/p/{DEMO}/doc/{docs['M']}", f"a click on the old view wrote {hash_!r}")
    await report.wait()
    await page.wait_for_selector('.entity-table[data-entity="wear_test"] tbody tr')
    expect(not await page.evaluate("document.getElementById('document-view').inert"), "the new view is inert")
    expect(await page.locator("tr.selected").count() == 0, "a fact is selected")
    await page.click('#document-view .tab-bar [data-tab="facts"]')
    await page.click('[data-slot="rows"] tr[data-index="1"]')
    hash_ = await page.evaluate("location.hash")
    expect(hash_ == f"#/p/{DEMO}/doc/{docs['M']}/fact/1", f"a fact of the new view wrote {hash_!r}")


@check("home drops a table drawn under other labels until its own profile's answer lands")
async def home_table_labels(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await page.goto(f"{docs['multi']}/")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    await page.evaluate(f"location.hash = '#/p/{DEMO}/doc/{docs['M']}'")
    await page.wait_for_selector('.entity-table[data-entity="wear_test"] tbody tr')
    # Back home under the default: its table is still the one last read, but the labels on screen are the demo's.
    await page.route(lambda url: url.endswith("/api/dataset"), delayed(1.5))
    corpus = settled(page, "/api/dataset")
    await page.evaluate("location.hash = '#/'")
    await page.wait_for_selector("#empty-state:not(.hidden)")
    expect(await page.is_hidden("#corpus-view"), "the old table is shown under the other profile's labels")
    await corpus.wait()
    await page.wait_for_selector("#corpus-view:not(.hidden) table")


@check("a profile list that failed at start is asked again, and the switcher appears once it lands")
async def profiles_retry(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    failed = []

    async def fail_once(route: Route) -> None:
        if not failed:
            failed.append(route.request.url)
            await route.fulfill(status=503, body="down")
        else:
            await route.continue_()

    await page.route(lambda url: url.endswith("/api/profiles"), fail_once)
    await page.goto(f"{docs['multi']}/")
    await page.wait_for_selector("#doc-list .doc-item")
    expect(await page.is_hidden("#profile-switch"), "a switcher without a list")
    await page.wait_for_selector("#profile-switch:not(.hidden)", timeout=10000)
    expect(await selected_profile(page) == "tco", f"the switcher shows {await selected_profile(page)!r}")


@check("arrowing through the switcher opens no profile until the reader settles")
async def switcher_keyboard(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await page.goto(f"{docs['multi']}/")
    await page.wait_for_selector("#profile-switch:not(.hidden)")
    label = await page.get_attribute("#profile-select", "aria-label")
    expect(label is None, f"the select is labelled twice ({label!r} beside its <label>)")
    # A closed, focused select moves to the next option on an arrow key and fires a change (Chromium on Linux and
    # Windows). Chromium on macOS opens the option list instead and changes nothing, headless too; there the same
    # keydown-then-change is dispatched by hand after closing the list.
    await page.focus("#profile-select")
    await page.keyboard.press("ArrowDown")
    if await selected_profile(page) != DEMO:
        await page.keyboard.press("Escape")
        await page.evaluate(f"""() => {{
          const select = document.getElementById("profile-select");
          select.dispatchEvent(new KeyboardEvent("keydown", {{ key: "ArrowDown", bubbles: true }}));
          select.value = "{DEMO}";
          select.dispatchEvent(new Event("change", {{ bubbles: true }}));
        }}""")
    expect(await selected_profile(page) == DEMO, f"the arrow key moved to {await selected_profile(page)!r}")
    expect(await page.evaluate("location.hash") in ("", "#/"), "one arrow key switched at once")
    await page.wait_for_function(f"location.hash === '#/p/{DEMO}'", timeout=3000)


@check("a profile's labels never draw another profile's data, however late they arrive")
async def profile_view_late(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await open_doc(page, docs["multi"], docs["M"])
    await page.evaluate(
        """() => {
          window.__scopes = [];
          const view = document.getElementById("document-view");
          window.__observer = new MutationObserver(() => {
            window.__scopes.push([...view.querySelectorAll('[data-slot="rows"] td.mono')].map((td) => td.textContent));
          });
          window.__observer.observe(view, { childList: true, subtree: true });
        }"""
    )
    await page.route(lambda url: f"/api/profile?profile={DEMO}" in url, delayed(1.5))
    report = settled(page, f"/report?profile={DEMO}")
    await page.select_option("#profile-select", DEMO)
    await report.wait()  # the new profile's data is in; its labels are still on their way
    title_ = await page.text_content("#profile-title") or ""
    expect("示例领域" not in title_, "the new profile's title came before its view was read")
    await page.wait_for_selector('.entity-table[data-entity="wear_test"] tbody tr', timeout=5000)
    scopes = await page.evaluate("() => { window.__observer.disconnect(); return window.__scopes; }")
    expect(["S1", "S1"] not in scopes, "the entity profile's report was drawn with the default's labels")
    expect("示例领域" in (await page.text_content("#profile-title") or ""), "the header does not name the new profile")


@check("a profile deep link survives a reload: profile, fact and switcher")
async def profile_reload(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    link = f"{docs['multi']}/#/p/{DEMO}/doc/{docs['M']}/fact/1"
    for _attempt in range(2):
        if _attempt:
            await page.reload()
        else:
            await page.goto(link)
        await page.wait_for_selector('tr.selected[data-index="1"]')
        expect(await selected_profile(page) == DEMO, f"the switcher shows {await selected_profile(page)!r}")
        expect(await page.locator(".entity-table").count() > 0, "the page is not the entity profile's")
        brand = await page.get_attribute(".brand", "href")
        expect(brand == f"#/p/{DEMO}", f"the brand link goes to {brand!r}")


@check("a link to an unknown or malformed profile says so")
async def unknown_profile(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    for hash_, name in (("#/p/nosuch", "nosuch"), (f"#/p/Bad/doc/{docs['M']}", "Bad")):
        await page.goto(f"{docs['multi']}/{hash_}")
        await page.wait_for_selector("#missing-view:not(.hidden)")
        text = await page.text_content("#missing-view") or ""
        expect(f"没有名为「{name}」的领域配置" in text, f"the missing view says {text!r}")
        expect(await page.get_attribute("#missing-view .missing-actions a", "href") == "#/", "no link home")
        expect(await page.is_hidden("#corpus-view"), "the default's home table is shown under a missing profile")


@check("a refused profile leaves the switcher and the link home on the last profile shown")
async def refused_profile_keeps_last(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await page.goto(f"{docs['multi']}/#/p/{DEMO}")
    await page.wait_for_selector("#doc-list .doc-item")
    # An option the server does not serve, as a list read before a profile was removed would still offer it.
    await page.evaluate("""() => {
      const option = document.createElement("option");
      option.value = option.textContent = "gone";
      document.getElementById("profile-select").append(option);
    }""")
    await page.select_option("#profile-select", "gone")
    await page.wait_for_selector("#missing-view:not(.hidden)")
    expect(await selected_profile(page) == DEMO, f"the switcher stays on {await selected_profile(page)!r}")
    href = await page.get_attribute("#missing-view .missing-actions a", "href")
    expect(href == f"#/p/{DEMO}", f"the link home goes to {href!r}")


@check("a document link under a profile keeps the profile on the missing view's link home")
async def missing_keeps_profile(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await page.goto(f"{docs['multi']}/#/p/{DEMO}/doc/0000000000000000")
    await page.wait_for_selector("#missing-view:not(.hidden)")
    href = await page.get_attribute("#missing-view .missing-actions a", "href")
    expect(href == f"#/p/{DEMO}", f"the link home goes to {href!r}")


# ---- the home query (`#/?q=…`): read by the router, written in place, never a new table load ----------------------

# The modules are singletons, so an import from the page gives the very state object the page draws from.
HOME_QUERY = "import('/state.js').then((module) => module.state.homeQuery)"


async def wait_home_query(page: Page, expected: object) -> None:
    """``state.homeQuery`` becomes ``expected`` (a hashchange is handled a task after the hash is set)."""
    deadline = time.monotonic() + 3
    while (got := await page.evaluate(HOME_QUERY)) != expected:
        expect(time.monotonic() < deadline, f"state.homeQuery reads {got!r}, not {expected!r}")
        await asyncio.sleep(0.05)


def dataset_requests(page: Page) -> list[str]:
    requests: list[str] = []
    page.on("request", lambda request: requests.append(request.url) if "/api/dataset" in request.url else None)
    return requests


@check("a home query in the URL is read into state.homeQuery, loads the table once, and rides on the brand link")
async def home_query_read(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    requests = dataset_requests(page)
    await page.goto(f"{base}/#/?q=abc")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    await wait_home_query(page, {"q": "abc"})
    expect(len(requests) == 1, f"the table was asked {len(requests)} times")
    expect(await page.get_attribute(".brand", "href") == "#/?q=abc", "the brand link dropped the query")
    # The query alone changing is not a new view: no table load, the state and the brand link follow.
    await page.evaluate("location.hash = '#/?q=def'")
    await wait_home_query(page, {"q": "def"})
    await asyncio.sleep(0.3)
    expect(len(requests) == 1, f"a query change re-asked the table ({len(requests)} requests)")
    expect(await page.get_attribute(".brand", "href") == "#/?q=def", "the brand link did not follow the query")
    expect(await page.is_visible("#corpus-view table"), "the table went away on a query change")
    # Dropping the query is a change too; an empty value counts as no key.
    await page.evaluate("location.hash = '#/?q='")
    await wait_home_query(page, {})
    expect(len(requests) == 1, f"dropping the query re-asked the table ({len(requests)} requests)")
    # On a document the query is nobody's; the way home still carries the last one seen.
    await page.evaluate("location.hash = '#/?q=ghi'")
    await wait_home_query(page, {"q": "ghi"})
    await page.evaluate(f"location.hash = '#/doc/{docs['A']}'")
    await page.wait_for_selector("#document-view:not(.hidden) h1")
    await wait_home_query(page, None)
    expect(await page.get_attribute(".brand", "href") == "#/?q=ghi", "the brand link forgot the home query")
    await page.click(".brand")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    await wait_home_query(page, {"q": "ghi"})
    expect(len(requests) == 2, f"coming home loaded the table {len(requests) - 1} times")


@check("setHomeQuery writes the query in place: no hashchange, no history entry, the router's own memory moves")
async def home_query_write(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    requests = dataset_requests(page)
    await page.route("**/api/dataset", delayed(1.0))
    await page.goto(f"{base}/#/?q=a")
    await page.wait_for_selector("#empty-state:not(.hidden)")
    history_length = await page.evaluate("history.length")
    # Written while the table is still on its way: what the view lands with is the written query.
    await page.evaluate("import('/router.js').then((router) => router.setHomeQuery({ q: 'b', extra: '' }))")
    expect(await page.evaluate("location.hash") == "#/?q=b", f"the write left {await page.evaluate('location.hash')!r}")
    expect(await page.evaluate("history.length") == history_length, "a replaceState write added a history entry")
    expect(await page.evaluate(HOME_QUERY) == {"q": "b"}, "the write did not reach state.homeQuery")
    expect(await page.get_attribute(".brand", "href") == "#/?q=b", "the brand link did not follow the write")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    expect(await page.evaluate(HOME_QUERY) == {"q": "b"}, "the landing table reset the query")
    expect(len(requests) == 1, f"the table was asked {len(requests)} times")
    # A hashchange after the write is judged against the written query: the same spelling is no change at all, a
    # different one is a query change and still no table load.
    await page.evaluate("location.hash = '#/?q=b'")
    await asyncio.sleep(0.3)
    expect(len(requests) == 1, f"re-setting the written hash re-asked the table ({len(requests)} requests)")
    await page.evaluate("location.hash = '#/?q=c'")
    await wait_home_query(page, {"q": "c"})
    await asyncio.sleep(0.3)
    expect(len(requests) == 1, f"a query change after a write re-asked the table ({len(requests)} requests)")
    # Clearing leaves a bare home address, and a write on a document is refused.
    await page.evaluate("import('/router.js').then((router) => router.setHomeQuery({ q: '' }))")
    expect(await page.evaluate("location.hash") == "#/", f"clearing left {await page.evaluate('location.hash')!r}")
    expect(await page.evaluate(HOME_QUERY) == {}, "clearing did not reach state.homeQuery")
    await page.evaluate(f"location.hash = '#/doc/{docs['A']}'")
    await page.wait_for_selector("#document-view:not(.hidden) h1")
    refused = await page.evaluate(
        "import('/router.js').then((router) => router.setHomeQuery({ q: 'x' }))"
        ".then(() => null, (error) => error.message)"
    )
    expect(refused is not None, "a write on a document view was accepted")
    expect(await page.evaluate("location.hash") == f"#/doc/{docs['A']}", "a refused write changed the address")


@check("a home query under a profile opens that profile's home with the query; switching profile drops it")
async def home_query_profile(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    requests = dataset_requests(page)
    await page.goto(f"{docs['multi']}/#/p/{DEMO}/?q=x")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    expect(await selected_profile(page) == DEMO, f"the switcher shows {await selected_profile(page)!r}")
    expect(await page.locator("#corpus-view .entity-switch").count() == 1, "the home table is not the entity profile's")
    await wait_home_query(page, {"q": "x"})
    expect(len(requests) == 1 and f"profile={DEMO}" in requests[0], f"the table requests read {requests}")
    expect(await page.get_attribute(".brand", "href") == f"#/p/{DEMO}/?q=x", "the brand link lost the profile or query")
    # A query right after the name (no slash) is tolerated as the same address, never read as a profile name.
    await page.goto(f"{docs['multi']}/#/p/{DEMO}?q=y")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    expect(await selected_profile(page) == DEMO, "the slash-less form was not routed to the profile")
    expect(await page.is_hidden("#missing-view"), "the slash-less form was read as a missing profile")
    await wait_home_query(page, {"q": "y"})
    # A profile change is a new view and keeps no query.
    default = await page.evaluate("import('/state.js').then((module) => module.state.defaultProfile)")
    await page.select_option("#profile-select", default)
    await page.wait_for_function("location.hash === '#/'")
    await wait_home_query(page, {})
    expect(await page.get_attribute(".brand", "href") == "#/", "the brand link kept another profile's query")


@check("the rail marks results under other profiles, and the page links to them")
async def library_markers(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await page.goto(f"{docs['multi']}/")
    await page.wait_for_selector("#doc-list .doc-item")
    item = page.locator(f'.doc-item[data-focus="doc:{docs["M"]}"] .other-profiles')
    expect(await item.text_content() == "另有 1 个领域的结果", f"M's marker reads {await item.text_content()!r}")
    expect("示例领域" in (await item.get_attribute("title") or ""), "the marker does not name the other profile")
    only = page.locator(f'.doc-item[data-focus="doc:{docs["N"]}"] .other-profiles')
    expect(await only.count() == 0, "a paper with results under one profile is marked")
    await open_doc(page, docs["multi"], docs["M"])
    links = page.locator('[data-slot="other-profiles"] a')
    expect(await links.all_text_contents() == ["示例领域"], "the page does not link the other profile")
    expect(await links.first.get_attribute("href") == f"#/p/{DEMO}/doc/{docs['M']}", "the link is not the paper's")
    await page.evaluate(f"location.hash = '#/p/{DEMO}'")
    await page.wait_for_function(
        f"""document.querySelector('.doc-item[data-focus="doc:{docs["M"]}"] .other-profiles')?.title.includes('透明')"""
    )


@check("重新处理 under a profile runs under it and follows its own job")
async def per_profile_run(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await page.goto(f"{docs['multi']}/#/p/{DEMO}/doc/{docs['M']}")
    await page.wait_for_selector("#document-view:not(.hidden) h1")
    async with page.expect_request(lambda request: request.method == "POST" and "/run?" in request.url) as sent:
        await page.click('#document-view [data-action="run"]')
    expect(f"profile={DEMO}" in (await sent.value).url, f"the run was asked as {(await sent.value).url}")
    await page.wait_for_selector("#document-view .stage.running", timeout=5000)
    await page.wait_for_selector("text=处理完成", timeout=len(stage_names()) * STAGE_SECONDS * 1000 + 10000)
    jobs = await page.evaluate("fetch('/api/jobs').then((response) => response.json())")
    mine = [job for job in jobs if job["document_id"] == docs["M"]]
    expect(bool(mine) and mine[-1]["profile"] == DEMO, f"the job ran under {mine[-1]['profile'] if mine else None!r}")


@check("a job of another profile on the paper is noted, never drawn as this profile's progress or reload")
async def other_profile_busy(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await page.goto(f"{docs['multi']}/#/p/{DEMO}")
    await page.wait_for_selector("#doc-list .doc-item")
    # This profile's own earlier job, finished: the log panel must keep showing it.
    await page.evaluate(f"fetch('/api/documents/{docs['M']}/run?profile={DEMO}', {{method: 'POST'}})")
    await jobs_idle(page)
    reports: list[str] = []
    page.on("request", lambda request: reports.append(request.url) if "/report" in request.url else None)
    await page.evaluate(f"fetch('/api/documents/{docs['M']}/run?profile=tco', {{method: 'POST'}})")
    await page.evaluate(f"location.hash = '#/p/{DEMO}/doc/{docs['M']}'")
    await page.wait_for_selector('#document-view [data-slot="other-job"]:not(.hidden)')
    note = await page.text_content('#document-view [data-slot="other-job"]') or ""
    expect("透明导电" in note and "排队" in note, f"the note reads {note!r}")
    expect(await page.locator("#document-view .stage.running").count() == 0, "the other job's progress is drawn")
    # The log panel is this profile's last job's, never the running one of the other.
    status = await page.text_content('#document-view [data-slot="job-status"]') or ""
    expect("处理中" not in status and "排队中" not in status, f"the other job's log is shown: {status!r}")
    expect(not await page.is_disabled('#document-view [data-action="run"]'), "the run button is locked by it")
    await page.wait_for_selector(f'.doc-item[data-focus="doc:{docs["M"]}"] .queued')
    queued = await page.get_attribute(f'.doc-item[data-focus="doc:{docs["M"]}"] .queued', "title") or ""
    expect("透明导电" in queued, f"the busy marker reads {queued!r}")
    loads = len(reports)
    await jobs_idle(page)
    # The note goes once the rail sees the other job done; by then a reload it wrongly caused would be out too.
    await page.wait_for_selector('#document-view [data-slot="other-job"]', state="hidden", timeout=15000)
    await page.wait_for_selector(f'.doc-item[data-focus="doc:{docs["M"]}"] .queued', state="detached")
    expect(len(reports) == loads, f"the other profile's job reloaded this view: {reports[loads:]}")
    toasts = await page.locator(".toast").all_text_contents()
    expect("处理完成" not in toasts, "the other profile's job toasted over this view")


@check("an upload under a profile is processed under it")
async def upload_under_profile(page: Page, _: str, docs: dict[str, str], pdf: Path) -> None:
    fresh = make_blank_pdf(pdf.parent / "upload-demo.pdf", [(410.0, 610.0)])
    await page.goto(f"{docs['multi']}/#/p/{DEMO}")
    await page.wait_for_selector("#doc-list .doc-item")
    hint = await page.text_content("#upload-profile") or ""
    expect("示例领域" in hint, f"the upload hint reads {hint!r}")
    async with page.expect_response(lambda response: "/api/documents?force=" in response.url) as answer:
        await page.set_input_files("#file-input", str(fresh))
    response = await answer.value
    expect(f"profile={DEMO}" in response.url, f"the upload was sent as {response.url}")
    job = (await response.json())["job"]
    expect(job["profile"] == DEMO, f"the upload's job runs under {job['profile']!r}")
    await page.wait_for_function(f"location.hash.startsWith('#/p/{DEMO}/doc/')")


@check("under a profile, every per-profile request and link names it")
async def api_profile_param(page: Page, _: str, docs: dict[str, str], pdf: Path) -> None:
    requests: list[tuple[str, str]] = []
    page.on("request", lambda request: requests.append((request.method, request.url)))
    await page.goto(f"{docs['multi']}/#/p/{DEMO}")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    hrefs = [await page.get_attribute("#corpus-view a.download", "href")]
    await page.click(f'#corpus-view a[href="#/p/{DEMO}/doc/{docs["M"]}"]')
    await page.wait_for_selector('.entity-table[data-entity="wear_test"] tbody tr')
    hrefs.append(await page.get_attribute('[data-slot="dataset-download"]', "href"))
    await page.click('#document-view [data-action="run"]')
    await page.wait_for_selector("#document-view .stage.running", timeout=5000)
    fresh = make_blank_pdf(pdf.parent / "upload-demo-2.pdf", [(420.0, 620.0)])
    async with page.expect_response(lambda response: "/api/documents?force=" in response.url):
        await page.set_input_files("#file-input", str(fresh))
    await page.wait_for_function(f"location.hash.startsWith('#/p/{DEMO}/doc/')")
    await jobs_idle(page)  # neither job is left to run into a later check
    unnamed = []
    for method, url in requests:
        parts = urlsplit(url)
        if not parts.path.startswith("/api/") or PROFILE_FREE.match(parts.path):
            continue
        if parse_qs(parts.query).get("profile") != [DEMO]:
            unnamed.append(f"{method} {parts.path}?{parts.query}")
    expect(not unnamed, f"asked without profile={DEMO}: {unnamed}")
    kinds = {urlsplit(url).path.rsplit("/", 1)[-1] for _, url in requests if "/api/" in url}
    expect({"profile", "dataset", "report", "run"} <= kinds, f"the log missed a route: {sorted(kinds)}")
    expect(all(href and f"profile={DEMO}" in href for href in hrefs), f"the Excel links read {hrefs}")


INJECTED = '<img src=x onerror="window.__injected=1">'


async def run_check(page: Page, text: str) -> None:
    await page.fill("#check-text", text)
    async with page.expect_response(lambda response: "/api/profile-check" in response.url):
        await page.click("#check-run")
    await page.wait_for_selector("#check-result .check-verdict")


@check("the check page lists a pasted profile's errors, or previews a valid one with its markup as text")
async def check_page(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    errors: list[str] = []
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    await page.goto(f"{base}/")
    await page.wait_for_selector("#doc-list .doc-item")
    await page.click("#check-link")
    await page.wait_for_selector("#check-view:not(.hidden)")
    expect(await page.is_hidden("#empty-state") and await page.is_hidden("#corpus-view"), "the home view stays up")
    requests: list[str] = []
    page.on("request", lambda request: requests.append(f"{request.method} {urlsplit(request.url).path}"))

    await run_check(page, '{"format": 1, "name": "draft", "fields": [}')
    verdict = await page.text_content("#check-result .check-verdict") or ""
    expect("问题" in verdict, f"an invalid profile reads {verdict!r}")
    expect(await page.locator("#check-result .check-errors li").count() >= 1, "no error line is listed")

    data = json.loads(shipped_profile().source.read_text(encoding="utf-8"))
    data["title_zh"] = INJECTED
    data["fields"][0]["label"] = INJECTED
    await run_check(page, json.dumps(data, ensure_ascii=False))
    verdict = await page.text_content("#check-result .check-verdict") or ""
    expect("有效" in verdict, f"the shipped profile reads {verdict!r}")
    # The preview is the profile page's own renderer.
    expect(
        await page.locator("#check-result .profile-fields tbody tr").count() == len(data["fields"]), "fields missing"
    )
    await page.click("#check-result details.prompt-preview >> nth=0 >> summary")
    await page.wait_for_selector("#check-result .prompt-section pre")
    expect(await page.locator("#check-result .prompt-section").count() >= 3, "the prompts are not previewed")
    same = await page.text_content("#check-result .check-same-name") or ""
    expect("内容哈希相同" in same, f"a display-only edit reads {same!r}")
    expect(await page.locator("#check-view img").count() == 0, "pasted markup became an element")
    expect(await page.evaluate("window.__injected") is None, "pasted markup ran")
    expect(INJECTED in (await page.text_content("#check-result") or ""), "pasted markup is not shown as text")
    expect(requests == ["POST /api/profile-check"] * 2, f"the page asked more than the check: {requests}")
    # A field's question is the checked text asked again for that field.
    field = data["fields"][0]["name"]
    await page.click("#check-result details.prompt-preview >> nth=1 >> summary")
    async with page.expect_response(lambda response: f"field={field}" in response.url):
        await page.select_option("#check-result select.prompt-field", field)
    await page.wait_for_selector(f"#check-result h3:has-text('({field}')")
    expect(not errors, f"console errors: {errors}")

    await page.click(".brand")
    await page.wait_for_selector("#check-view.hidden", state="attached")
    await page.go_back()
    await page.wait_for_selector("#check-view:not(.hidden)")
    kept = await page.input_value("#check-text")
    expect(json.loads(kept)["title_zh"] == INJECTED, "the pasted text did not survive navigation")


@check("a check that lands after a profile switch is drawn for its text, and a new check clears the old answer")
async def check_across_switch(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await page.goto(f"{docs['multi']}/#/check")
    await page.wait_for_selector("#check-view:not(.hidden)")
    await run_check(page, '{"format": 1, "name": "draft", "fields": [}')
    await page.route("**/api/profile-check", delayed(1.5))
    answer = settled(page, "/api/profile-check")
    await page.fill("#check-text", shipped_profile().source.read_text(encoding="utf-8"))
    async with page.expect_request(lambda request: "/api/profile-check" in request.url):
        await page.click("#check-run")
    expect(await page.locator("#check-result > *").count() == 0, "the previous answer stands beside the new text")
    await page.select_option("#profile-select", DEMO)
    await page.wait_for_function(f"location.hash === '#/p/{DEMO}/check'")
    await answer.wait()
    await page.wait_for_selector("#check-result .check-verdict")
    verdict = await page.text_content("#check-result .check-verdict") or ""
    expect("有效" in verdict, f"the answer for the text on screen reads {verdict!r}")


# ---- the explorer: the search box, the filter panel and the home query that holds them -------------------------------

# The explorer corpus, served as /api/dataset in place of the real one: papers P1-P3 with one sample each (t = 80 / 85 /
# blank, r = 5 / 2 / 9, mode written "RF" / "rf-magnetron sputtering" / "DC and RF", which the server reads as the
# categories RF / RF / DC+RF), a value spanning eight decades (rho), an interval (iv) and a list (lst); P2 is a review;
# Z has no sample at all. Each paper borrows a seeded document's id, so its link opens and the rail's status joins it:
# P1 is B (it has conflicts), P3 is U (not finished under this profile).
EXPLORE_PAPERS = {
    "P1": {
        "doc": "B",
        "t": 80.0,
        "r": 5.0,
        "mode": "RF",
        "cat": "RF",
        "rho": 1e-4,
        "iv": [10.0, 20.0],
        "lst": [1.0, 50.0],
    },
    "P2": {
        "doc": "A",
        "t": 85.0,
        "r": 2.0,
        "mode": "rf-magnetron sputtering",
        "cat": "RF",
        "rho": 10.0,
        "iv": [30.0, None],
        "lst": [5.0],
    },
    "P3": {"doc": "U", "t": None, "r": 9.0, "mode": "DC and RF", "cat": "DC+RF", "rho": 1e4, "iv": None, "lst": None},
}
EXPLORE_ZERO = "Z 无数据的论文"
EXPLORE_REVIEW = "P2"
MODES = ["DC", "RF", "pulsed DC", "DC+RF", "HiPIMS"]


async def explore_corpus(page: Page, docs: dict[str, str]) -> None:
    """Serve the explorer corpus as /api/dataset: the real answer, with these fields and rows in place of its own."""

    async def corpus(route: Route) -> None:
        response = await route.fetch()
        data = await response.json()
        base = {
            "label": "",
            "scope": "sample",
            "description": "",
            "cardinality": "one",
            "group": "film",
            "categories": [],
        }
        data["fields"] = [
            {**base, "name": "t", "kind": "numeric", "unit": "nm"},
            {**base, "name": "r", "kind": "numeric", "unit": "Ω/sq"},
            {**base, "name": "mode", "kind": "text", "unit": None, "group": "process", "categories": MODES},
            {**base, "name": "rho", "kind": "numeric", "unit": "Ω·cm"},
            {**base, "name": "iv", "kind": "interval", "unit": "nm"},
            {**base, "name": "lst", "kind": "numeric", "unit": "nm", "cardinality": "many"},
        ]
        rows = []
        for name, paper in EXPLORE_PAPERS.items():
            sample = {
                "sample_id": f"S-{name}",
                "sample_label": f"label {name}",
                "conditions": "anneal in Ar" if name == "P1" else "",
                "available_fields": 3,
                "agree_fields": 2,
                **{field: paper[field] for field in ("t", "r", "mode", "rho", "iv", "lst")},
            }
            rows.append(
                {
                    "document_id": docs[paper["doc"]],
                    "name": name,
                    "paper_row": sample,
                    "sample_count": 1,
                    "sample_rows": [sample],
                    "article_type": "review" if name == EXPLORE_REVIEW else None,
                    "paper_categories": {"mode": [paper["cat"]]},
                    "sample_categories": [{"mode": [paper["cat"]]}],
                }
            )
        rows.append(
            {
                "document_id": docs["C"],
                "name": EXPLORE_ZERO,
                "paper_row": {},
                "sample_count": 0,
                "sample_rows": [],
                "article_type": None,
                "paper_categories": {},
                "sample_categories": [],
            }
        )
        data["rows"] = rows
        await route.fulfill(response=response, json=data)

    await page.route("**/api/dataset", corpus)


async def open_explore(page: Page, base: str, docs: dict[str, str], query: str = "") -> None:
    await explore_corpus(page, docs)
    await page.goto(f"{base}/#/" + (f"?{query}" if query else ""))
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    await page.wait_for_selector("#doc-list .doc-item", state="attached")  # the status filters join the rail's list


# Each row's first cell without the label under it (a sample's id, a paper row's "无<entity>", or unfiltered a paper's
# name), then its second cell.
ROWS = """() => [...document.querySelectorAll('#corpus-view tbody tr')].map((tr) => {
  const first = tr.cells[0].cloneNode(true);
  const label = first.querySelector('small');
  if (label && label.textContent.trim() !== first.textContent.trim()) label.remove();
  return [first.textContent.trim(), tr.cells[1].textContent.trim()];
})"""


async def shown_rows(page: Page) -> list[list[str]]:
    return await page.evaluate(ROWS)


async def shown_ids(page: Page) -> list[str]:
    return [row[0] for row in await shown_rows(page)]


async def wait_ids(page: Page, expected: list[str]) -> None:
    deadline = time.monotonic() + 3
    while (got := await shown_ids(page)) != expected:
        expect(time.monotonic() < deadline, f"the rows read {got}, not {expected}")
        await asyncio.sleep(0.05)


async def set_range(page: Page, field: str, low: str = "", high: str = "") -> None:
    for end, value in (("min", low), ("max", high)):
        box = f'#corpus-view [data-focus="{end}:{field}"]'
        await page.fill(box, value)
        await page.press(box, "Enter")


def range_box(field: str) -> str:
    return f'#corpus-view .range-filter[data-field="{field}"]'


@check("a range filter keeps exactly the values inside it; a blank never passes and the panel counts it")
async def explore_range(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_explore(page, base, docs)
    expect(await shown_ids(page) != ["S-P2"], "the table starts filtered")
    note = (await page.text_content(f"{range_box('t')} .filter-note") or "").strip()
    expect(note == "1 个无确定值，不参与筛选", f"t's panel note reads {note!r}")
    await set_range(page, "t", "85")
    await wait_ids(page, ["S-P2"])
    expect(
        "f.t=85%7E" in await page.evaluate("location.hash"), f"the address reads {await page.evaluate('location.hash')}"
    )
    chips = await page.locator("#corpus-view .active-filters .chip").all_text_contents()
    expect([chip.rstrip("×").strip() for chip in chips] == ["t ≥ 85 nm"], f"the active filters read {chips}")
    count = (await page.text_content("#corpus-view .row-count") or "").strip()
    expect(count == "1 篇论文 · 1 个样品（共 4 篇）", f"the row count reads {count!r}")
    heads = await page.locator("#corpus-view thead th").all_text_contents()
    expect(heads[:3] == ["样品", "论文", "可用/一致"], f"flattened rows put the id first: {heads[:3]}")
    # The two ends of the slider show the range, and the bars inside it are marked.
    expect(await page.locator(f"{range_box('t')} .histogram .bar.in").count() > 0, "no bar is marked inside the range")
    # The slider sets the same filter from the keyboard: the low end moved to the top leaves t ≥ 85 (the largest t).
    await set_range(page, "t")
    await wait_ids(page, ["P1", "P2", "P3", EXPLORE_ZERO])
    await page.focus('#corpus-view [data-focus="range-lo:t"]')
    await page.keyboard.press("End")
    await wait_ids(page, ["S-P2"])
    expect(await page.input_value('#corpus-view [data-focus="min:t"]') == "85", "the slider did not fill the box")
    # An interval passes when it overlaps the range; a list when any element is inside; a decade-spanning field is
    # drawn on a log scale, the others not.
    await set_range(page, "t")
    await set_range(page, "iv", "15", "25")
    await wait_ids(page, ["S-P1"])
    await set_range(page, "iv", "25")
    await wait_ids(page, ["S-P2"])
    await set_range(page, "iv")
    await set_range(page, "lst", "40")
    await wait_ids(page, ["S-P1"])
    expect("对数刻度" in (await page.text_content(f"{range_box('rho')} legend") or ""), "rho is not on a log scale")
    expect("对数刻度" not in (await page.text_content(f"{range_box('t')} legend") or ""), "t is on a log scale")
    # Removing the chip removes the filter.
    await page.click('#corpus-view .active-filters [data-focus="remove:f:lst"]')
    await wait_ids(page, ["P1", "P2", "P3", EXPLORE_ZERO])
    expect(await page.locator("#corpus-view .active-filters").count() == 0, "the chip row outlived its last filter")
    expect(await page.evaluate("location.hash") == "#/", f"the address kept {await page.evaluate('location.hash')!r}")


@check("category checkboxes filter on the server's canonical category, statuses join the rail, types the article")
async def explore_categories(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_explore(page, base, docs)
    counts = {
        label: (await page.text_content(f'#corpus-view .filter-check:has([data-focus="pick:mode:{label}"]) .n') or "")
        for label in ("RF", "DC+RF", "DC")
    }
    expect(counts == {"RF": "2", "DC+RF": "1", "DC": "0"}, f"the mode counts read {counts}")
    await page.check('#corpus-view [data-focus="pick:mode:RF"]')
    await wait_ids(page, ["S-P1", "S-P2"])
    expect(await page.is_checked('#corpus-view [data-focus="pick:mode:RF"]'), "the box did not stay ticked")
    focused = await page.evaluate("document.activeElement.dataset.focus")
    expect(focused == "pick:mode:RF", f"the focus moved to {focused!r}")
    await page.uncheck('#corpus-view [data-focus="pick:mode:RF"]')
    await page.check('#corpus-view [data-focus="pick:mode:DC+RF"]')
    await wait_ids(page, ["S-P3"])
    await page.uncheck('#corpus-view [data-focus="pick:mode:DC+RF"]')
    # 未完成 under the default profile: the rail's list says U (P3) is not finished under it.
    await page.check('#corpus-view [data-focus="status:unfinished"]')
    await wait_ids(page, ["S-P3"])
    await page.uncheck('#corpus-view [data-focus="status:unfinished"]')
    await page.check('#corpus-view [data-focus="status:conflict"]')
    await wait_ids(page, ["S-P1"])
    await page.uncheck('#corpus-view [data-focus="status:conflict"]')
    await page.check('#corpus-view [data-focus="type:review"]')
    await wait_ids(page, ["S-P2"])
    # Two filters are both applied: a review that is unfinished is nothing here.
    await page.check('#corpus-view [data-focus="status:unfinished"]')
    await wait_ids(page, [])
    expect(await page.is_visible("#corpus-view .explore-empty"), "an empty result does not say so")
    await page.click("#corpus-view .explore-empty .clear-all")
    # Unfiltered, the table is one row per paper again.
    await wait_ids(page, ["P1", "P2", "P3", EXPLORE_ZERO])
    expect(await page.evaluate("location.hash") == "#/", f"清除全部 left {await page.evaluate('location.hash')!r}")
    # The research papers include the one without samples: a paper-level filter lists it as a paper row.
    await page.check('#corpus-view [data-focus="type:research"]')
    await wait_ids(page, ["S-P1", "S-P3", "无样品"])


@check("the search finds a paper's samples, a sample-less paper by name, conditions only when asked; typing is local")
async def explore_search(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    requests = dataset_requests(page)
    await open_explore(page, base, docs)
    expect(len(requests) == 1, f"the table was asked {len(requests)} times")
    await page.click("#corpus-search")
    await page.keyboard.type("ｐ2", delay=40)  # full width: NFKC folds it to "p2"
    await wait_ids(page, ["S-P2"])
    rows = await shown_rows(page)
    expect(rows == [["S-P2", "P2综述"]], f"the P2 search reads {rows}")
    marks = await page.locator("#corpus-view tbody mark").all_text_contents()
    expect("P2" in marks, f"the match is not marked: {marks}")
    expect(
        await page.evaluate("location.hash") == "#/?q=%EF%BD%902",
        f"the address reads {await page.evaluate('location.hash')}",
    )
    # Five more characters typed: no table load, the focus never left the box.
    await page.fill("#corpus-search", "")
    await page.keyboard.type("无数据的论", delay=60)
    await wait_ids(page, ["无样品"])
    expect(len(requests) == 1, f"typing asked for the table again ({len(requests)} requests)")
    expect(await page.evaluate("document.activeElement.id") == "corpus-search", "typing lost the focus")
    rows = await shown_rows(page)
    expect(rows == [["无样品", EXPLORE_ZERO]], f"the sample-less paper reads {rows}")
    # The conditions text is searched only with 含条件描述.
    await page.fill("#corpus-search", "anneal")
    await wait_ids(page, [])
    await page.check("#corpus-view .search-cond input")
    await wait_ids(page, ["S-P1"])
    heads = await page.locator("#corpus-view thead th").all_text_contents()
    expect("条件" in heads, f"the conditions column is not shown: {heads}")
    # The copy is the visible rows, the id first.
    await page.evaluate("navigator.clipboard.writeText = async (text) => { window.__copied = text; }")
    await page.click("#corpus-view button.copy-table")
    lines = [line.split("\t") for line in (await page.evaluate("window.__copied") or "").split("\n")]
    expect(len(lines) == 2 and lines[1][:2] == ["S-P1", "P1"], f"the copy reads {lines}")
    # The clear button empties the box and the address, and keeps the focus in the box.
    await page.click("#corpus-view .search-clear")
    await page.wait_for_function("location.hash === '#/?cond=1'")
    expect(await page.input_value("#corpus-search") == "", "the box was not cleared")
    expect(await page.evaluate("document.activeElement.id") == "corpus-search", "清除 lost the focus")
    expect(len(requests) == 1, f"the search asked for the table again ({len(requests)} requests)")


@check("a reload restores the search, the filters and the sort; another profile clears them")
async def explore_round_trip(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_explore(page, base, docs)
    await page.fill("#corpus-search", "S-P")
    await wait_ids(page, ["S-P1", "S-P2", "S-P3"])
    await page.check('#corpus-view [data-focus="pick:mode:RF"]')
    await wait_ids(page, ["S-P1", "S-P2"])
    await page.click('#corpus-view [data-sort="t"]')
    await page.click('#corpus-view [data-sort="t"]')
    await wait_ids(page, ["S-P2", "S-P1"])
    hash_ = await page.evaluate("location.hash")
    query = parse_qs(urlsplit(hash_[1:]).query)
    expect(query == {"q": ["S-P"], "f.mode": ["RF"], "sort": ["-t"]}, f"the address holds {query}")
    await page.reload()
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    await wait_ids(page, ["S-P2", "S-P1"])
    expect(await page.input_value("#corpus-search") == "S-P", "the search box is empty after a reload")
    expect(await page.is_checked('#corpus-view [data-focus="pick:mode:RF"]'), "the RF box is unticked after a reload")
    sort = await page.get_attribute('#corpus-view thead th:has([data-sort="t"])', "aria-sort")
    expect(sort == "descending", f"the sort after a reload is {sort!r}")
    # Back to the bare address: the table and every control follow, with no reload.
    await page.evaluate("location.hash = '#/'")
    await page.wait_for_function("document.querySelectorAll('#corpus-view tbody tr').length === 4")
    expect(await page.input_value("#corpus-search") == "", "the search box kept a query the address dropped")


@check("switching profile clears the search and filters; flattened rows follow the entity switch")
async def explore_profiles(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await page.goto(f"{docs['multi']}/#/?q=S1&f.thickness=50~")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    expect(await page.input_value("#corpus-search") == "S1", "the search box did not read the address")
    await page.select_option("#profile-select", DEMO)
    await page.wait_for_function(f"location.hash === '#/p/{DEMO}'")
    await page.wait_for_selector('#corpus-view [data-focus="entity:coating"]')
    expect(await page.input_value("#corpus-search") == "", "another profile kept the search")
    expect(await page.locator("#corpus-view .active-filters").count() == 0, "another profile kept the filters")
    # A two-entity library: a search gives one row per sample of the entity shown, and follows the entity switch.
    await page.goto(f"{docs['entities']}/#/?q=S1")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    await wait_ids(page, ["S1"])
    heads = await page.locator("#corpus-view thead th").all_text_contents()
    expect(heads[:2] == ["涂层", "论文"] and any("coating_thickness" in head for head in heads), f"heads: {heads}")
    await page.click('#corpus-view [data-focus="entity:wear_test"]')
    await page.wait_for_function("location.hash === '#/?q=S1&e=wear_test'")
    await wait_ids(page, ["S1"])
    heads = await page.locator("#corpus-view thead th").all_text_contents()
    expect(heads[:2] == ["磨损测试", "论文"] and any("test_temperature" in head for head in heads), f"heads: {heads}")
    await page.fill("#corpus-search", "S2")
    await wait_ids(page, [])
    await page.click('#corpus-view [data-focus="entity:coating"]')
    await wait_ids(page, ["S2"])


@check("the filter panel opens and closes from 筛选, and a reload keeps the choice")
async def explore_panel(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_explore(page, base, docs)
    expect(await page.is_visible("#filter-panel"), "the panel is closed on a wide screen")
    await page.click('#corpus-view [data-focus="filter-toggle"]')
    expect(await page.locator("#filter-panel").count() == 0, "筛选 did not close the panel")
    expect(await page.get_attribute('#corpus-view [data-focus="filter-toggle"]', "aria-expanded") == "false", "aria")
    expect(await page.evaluate("document.activeElement.dataset.focus") == "filter-toggle", "筛选 lost the focus")
    await page.reload()
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    expect(await page.locator("#filter-panel").count() == 0, "a reload reopened the panel")
    # With the panel closed, an active filter is still shown over the table and counted on the button.
    await page.goto(f"{base}/#/?f.t=85~")
    await page.wait_for_selector("#corpus-view .active-filters")
    label = (await page.text_content('#corpus-view [data-focus="filter-toggle"]') or "").strip()
    expect(label == "筛选1", f"the button reads {label!r}")


async def refresh_rail(page: Page) -> None:
    """One refresh of the rail's list, as the 5 s timer makes while a job runs; resolves once it has painted."""
    await page.evaluate("import('/library.js').then((library) => library.loadLibrary())")


async def status_count(page: Page, key: str) -> str:
    return (await page.text_content(f'#corpus-view .filter-check:has([data-focus="status:{key}"]) .n') or "").strip()


@check("a rail refresh that moves no status leaves a range box being typed into alone; one that does redraws")
async def explore_rail_refresh(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_explore(page, base, docs)
    await page.check('#corpus-view [data-focus="status:unfinished"]')
    await wait_ids(page, ["S-P3"])
    expect(await status_count(page, "conflict") == "1", f"有冲突 counts {await status_count(page, 'conflict')!r}")
    box = '#corpus-view [data-focus="min:t"]'
    await page.fill(box, "8")  # typed, not committed (no Enter, no change)
    await refresh_rail(page)
    await refresh_rail(page)
    expect(await page.input_value(box) == "8", f"a refresh reset the typed range to {await page.input_value(box)!r}")
    expect(await page.evaluate("document.activeElement.dataset.focus") == "min:t", "a refresh moved the focus")
    expect("f.t=" not in await page.evaluate("location.hash"), "the typed range was committed")

    # B (P1) is now unfinished and has no conflict: the table and both counts follow the next refresh. The redraw takes
    # the focused box away, which commits what was typed in it (t ≥ 8) as leaving the box would, once and whole.
    async def moved(route: Route) -> None:
        response = await route.fetch()
        listed = await response.json()
        for doc in listed:
            if doc["document_id"] == docs["B"]:
                doc["profiles_done"] = []
                doc["counts"] = {**(doc.get("counts") or {}), "conflict": 0}
        await route.fulfill(response=response, json=listed)

    await page.route(re.compile(r"/api/documents(\?|$)"), moved)
    await refresh_rail(page)
    await wait_ids(page, ["S-P1"])
    counts = (await status_count(page, "conflict"), await status_count(page, "unfinished"))
    expect(counts == ("0", "2"), f"after the refresh 有冲突 / 未完成 count {counts}")
    hash_ = await page.evaluate("location.hash")
    expect("f.t=8%7E" in hash_, f"the address reads {hash_}")
    expect(await page.input_value(box) == "8", f"the committed range reads {await page.input_value(box)!r}")
    tables = await page.locator("#corpus-view table").count()
    expect(tables == 1, f"the home view holds {tables} tables")
    await page.uncheck('#corpus-view [data-focus="status:unfinished"]')
    await wait_ids(page, ["S-P1", "S-P2"])


@check("a header click sorts the home table with one redraw")
async def explore_sort_once(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_explore(page, base, docs)
    await page.evaluate(
        """() => {
          window.__draws = 0;
          new MutationObserver((changes) => {
            for (const change of changes)
              for (const node of change.addedNodes) if (node.classList?.contains('explorer')) window.__draws += 1;
          }).observe(document.getElementById('corpus-view'), { childList: true });
        }"""
    )
    await page.click('#corpus-view [data-sort="t"]')
    await page.wait_for_function("location.hash.includes('sort=t')")
    await page.wait_for_timeout(200)
    draws = await page.evaluate("window.__draws")
    expect(draws == 1, f"one sort click drew the table {draws} times")


@check("a search typed just before a profile switch is not written into the other profile's address")
async def explore_search_switch(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await page.goto(f"{docs['multi']}/#/")
    await page.wait_for_selector("#corpus-view:not(.hidden) #corpus-search")
    # The other profile's table is slow to come, so nothing redraws the search box before the pause is over.
    await page.route(re.compile(r"/api/dataset\?.*profile="), delayed(1.0))
    await page.click("#corpus-search")
    await page.keyboard.type("abc")
    await page.select_option("#profile-select", DEMO)  # well within the search's 120 ms pause
    await page.wait_for_function(f"location.hash === '#/p/{DEMO}'")
    await page.wait_for_timeout(400)
    hash_ = await page.evaluate("location.hash")
    expect(hash_ == f"#/p/{DEMO}", f"the other profile's address became {hash_!r}")


@check(
    "on a phone the search box and the filter panel stack over the table without sideways scrolling",
    width=390,
    height=844,
)
async def explore_narrow(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_explore(page, base, docs, "f.t=85~")
    expect(await page.locator("#filter-panel").count() == 0, "the panel opens by default on a phone")
    await page.click('#corpus-view [data-focus="filter-toggle"]')
    await page.wait_for_selector("#filter-panel")
    panel = await page.evaluate("document.querySelector('#filter-panel').getBoundingClientRect().toJSON()")
    table = await page.evaluate("document.querySelector('#corpus-view .table-wrap').getBoundingClientRect().toJSON()")
    expect(panel["bottom"] <= table["top"], f"the panel does not sit over the table: {panel} / {table}")
    overflow = await page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    expect(overflow <= 0, f"the page scrolls {overflow}px sideways")
    await wait_ids(page, ["S-P2"])


@check("the rail's delete asks first, cancels without asking the server, then deletes its own upload")
async def delete_document(page: Page, base: str, docs: dict[str, str], pdf: Path) -> None:
    await home(page, base)
    own = make_blank_pdf(pdf.parent / "delete-me.pdf", [(465.0, 665.0)])  # a size no seed uses
    async with page.expect_response(lambda response: "/api/documents?" in response.url):
        await page.set_input_files("#file-input", str(own))
    await jobs_idle(page)  # the delete must not race the upload's job: a busy paper is refused (409)
    listing = await page.evaluate("fetch('/api/documents').then((response) => response.json())")
    mine = [doc for doc in listing if doc["name"] == "delete-me.pdf"]
    expect(len(mine) == 1, f"the upload is not in the list once: {[d['name'] for d in listing]}")
    doc_id = mine[0]["document_id"]
    row = page.locator(f'#doc-list li:has(.doc-item[data-focus="doc:{doc_id}"])')

    deletes: list[str] = []
    page.on("request", lambda request: deletes.append(request.url) if request.method == "DELETE" else None)

    # Cancel first: the dialog opens and names the paper, but no request may leave.
    await row.locator(".doc-delete").click()
    await page.wait_for_selector("#delete-dialog[open]")
    message = await page.text_content("#delete-message") or ""
    expect("delete-me.pdf" in message and "不可恢复" in message, f"the dialog says {message!r}")
    await page.click('#delete-dialog [data-action="close"]:not(.dialog-close)')
    await page.wait_for_function("!document.getElementById('delete-dialog').open")
    expect(deletes == [], f"cancelling sent {deletes}")
    expect("delete-me.pdf" in await listed_names_join(page), "the row vanished on cancel")

    # Confirm: the open paper's row goes, the toast names the paper, and the rail still counts.
    # The paper is deleted while it is the open one, so the page must land back on home (corpus view),
    # not flash the missing view for a document that was just deleted under the route.
    await page.evaluate(f"location.hash = '#/doc/{doc_id}'")
    await page.wait_for_selector("#document-view:not(.hidden)")
    await row.locator(".doc-delete").click()
    await page.wait_for_selector("#delete-dialog[open]")
    await page.click("#delete-confirm")
    await page.wait_for_selector("#delete-dialog[open]", state="detached")
    await page.wait_for_selector("#corpus-view:not(.hidden)")
    for _ in range(50):
        if "delete-me.pdf" not in await listed_names_join(page):
            break
        await page.wait_for_timeout(200)
    expect("delete-me.pdf" not in await listed_names_join(page), "the row stayed after the delete")
    toasts = await page.locator(".toast").all_text_contents()
    expect(any("已删除「delete-me.pdf」" in toast for toast in toasts), f"no toast: {toasts}")
    count = await page.text_content("#doc-count")
    expect(count is not None and "delete-me" not in count, f"the count reads {count!r}")

    # A stale link to the deleted paper shows the missing view.
    await page.evaluate(f"location.hash = '#/doc/{doc_id}'")
    await page.wait_for_selector("text=找不到这篇文档")
    listing = await page.evaluate("fetch('/api/documents').then((response) => response.json())")
    expect(all(doc["document_id"] != doc_id for doc in listing), "the server still lists the deleted id")
    expect(deletes and len(deletes) == 1, f"exactly one DELETE should have left: {deletes}")


async def listed_names_join(page: Page) -> str:
    return "|".join(await listed_names(page))


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    failures = 0
    with tempfile.TemporaryDirectory() as directory, serve(Path(directory)) as (base, docs, pdf):
        async with async_playwright() as playwright:
            # A system proxy on macOS intercepts localhost; the page must talk to the server directly.
            browser = await playwright.chromium.launch(headless=not args.headed, args=["--no-proxy-server"])
            for name, function, viewport in CHECKS:
                if args.only not in name:
                    continue
                context = await browser.new_context(viewport=viewport)
                page = await context.new_page()
                try:
                    await function(page, base, docs, pdf)
                    print(f"PASS  {name}")
                except Exception as exc:  # report every check, not just the first failure
                    failures += 1
                    print(f"FAIL  {name}: {exc}")
                finally:
                    await context.close()
            await browser.close()
    print(f"{failures} failed" if failures else "all passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
