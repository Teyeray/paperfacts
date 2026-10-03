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
from paperfacts.matching import SampleMatch, SampleMatching
from paperfacts.models import BACKENDS, PageGeometry, ParsedArtifact
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
    await lanes_visible(page)


@check("both lanes of 事实对照 fit side by side at 1280 px", width=1280)
async def facts_1280(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await lanes_visible(page)


@check("both lanes of 事实对照 fit side by side at 1920 px", width=1920, height=1080)
async def facts_1920(page: Page, base: str, docs: dict[str, str], _: Path) -> None:
    await open_doc(page, base, docs["A"])
    await lanes_visible(page)


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


@check("a phone-width page does not scroll sideways and keeps 重新处理 on screen", width=390, height=844)
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
    expect(heads[:3] == ["论文", "磨损测试", "可用/一致"], f"the wear test table's heads read {heads}")
    expect(any("test_temperature" in head for head in heads), f"the wear test table lacks its field: {heads}")
    expect(not any("coating_thickness" in head or "precursor" in head for head in heads), f"other fields: {heads}")
    rows = page.locator("#corpus-view tbody tr")
    expect(await rows.count() == 1, f"the wear test table has {await rows.count()} rows")
    cells = await rows.first.locator("td").all_text_contents()
    expect(cells[1] == "S1" and any(cell.startswith("S1 涂层") for cell in cells[2:]), f"the row reads {cells}")
    expect(await page.evaluate("location.hash") == "#/", "the entity choice changed the address")
    expect(await page.get_attribute("#corpus-view a.download", "href") == download, "the Excel link changed")
    await page.evaluate("navigator.clipboard.writeText = async (text) => { window.__copied = text; }")
    await page.click("#corpus-view button.copy-table")
    copied = (await page.evaluate("window.__copied") or "").split("\n")
    expect(len(copied) == 2, f"the copy has {len(copied)} lines: {copied}")
    header, line = (row.split("\t") for row in copied)
    expect(header[:3] == heads[:3] and len(header) == len(heads), f"the copied header reads {header}")
    expect(line[0] == cells[0] and line[1] == "S1" and "S1" in line[3:], f"the copied row reads {line}")
    await page.click('#corpus-view [data-focus="entity:coating"]')
    await page.wait_for_selector('#corpus-view [data-focus="entity:coating"][aria-pressed="true"]')
    expect("2 个涂层" in (await page.text_content("#corpus-view") or ""), "the primary table did not come back")


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


@check("the home toolbar is one row: 列, 显示空字段, 密度, 展开全部, the count, 复制, 下载; the explorer slot is empty")
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
    slot = await page.locator('#corpus-view [data-slot="explore"]').inner_html()
    expect(slot == "", f"the explorer slot holds {slot!r}")
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
    first = await page.text_content('[data-slot="results-head"] th')
    expect(first == "涂层", f"the primary table's first column is {first!r}")
    scopes = await page.locator('[data-slot="rows"] td.mono').all_text_contents()
    expect(scopes == ["涂层 · S1", "磨损测试 · S1"], f"the comparison scopes read {scopes}")
    expect(await page.locator("tr.selected").count() == 0, "a fact of the old profile's report stays selected")


@check("while another profile loads, the old profile's view takes no clicks: no fact of its report reaches the URL")
async def stale_view_inert(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await open_doc(page, docs["multi"], docs["M"])
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
