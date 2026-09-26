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
from paperfacts.storage import document_key
from paperfacts.web.app import create_app
from paperfacts.web.documents import Library
from paperfacts.web.jobs import Job, JobManager
from paperfacts.workflow import stage_names
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
        mark(stage, "running", "")
        time.sleep(STAGE_SECONDS)
        mark(stage, "done", "stub")


def seed_document(library: Library, root: Path, index: int, name: str, *, samples: int, comparisons: int) -> str:
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
        lane = lane.model_copy(update={"no_samples": not samples})
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
        counts=ComparisonCounts(total=comparisons, agree=comparisons),
        comparisons=rows,
    ).write(library.layout.comparison_path(sha, library.extractor_key, library.comparison_key))
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
        "B": seed_document(library, root, 1, "B 掺铝氧化锌的透明导电性.pdf", samples=2, comparisons=6),
        "C": seed_document(library, root, 2, "C 钙钛矿电池（买来的 ITO 玻璃）.pdf", samples=0, comparisons=0),
    }
    for index in range(3, 28):  # a long library, as on the real server
        seed_document(library, root, index, f"filler paper {index}.pdf", samples=1, comparisons=1)
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
    await page.wait_for_timeout(300)
    expect(await page.evaluate("window.__xss") is None and not errors, f"markup ran: {errors}")


@check("a profile page answer that lands after the reader left draws nothing")
async def profile_page_late(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await page.goto(f"{docs['multi']}/")
    await page.wait_for_selector("#corpus-view:not(.hidden) table")
    await page.route(lambda url: f"/api/profiles/{DEMO}" in url, delayed(1.5))
    await page.evaluate(f"location.hash = '#/p/{DEMO}/profile'")
    await page.wait_for_timeout(200)
    await page.evaluate(f"location.hash = '#/doc/{docs['M']}'")
    await page.wait_for_selector("#document-view:not(.hidden) h1")
    await page.wait_for_timeout(2000)
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
    await page.evaluate(f"location.hash = '#/doc/{docs['M']}/fact/1'")
    await page.wait_for_timeout(200)
    await page.select_option("#profile-select", DEMO)
    await page.wait_for_timeout(2500)
    hash_ = await page.evaluate("location.hash")
    expect(hash_ == f"#/p/{DEMO}/doc/{docs['M']}", f"the switch went to {hash_!r} (the fact must be dropped)")
    await page.wait_for_selector('.entity-table[data-entity="wear_test"] tbody tr')
    first = await page.text_content('[data-slot="results-head"] th')
    expect(first == "涂层", f"the primary table's first column is {first!r}")
    scopes = await page.locator('[data-slot="rows"] td.mono').all_text_contents()
    expect(scopes == ["涂层 · S1", "磨损测试 · S1"], f"the comparison scopes read {scopes}")
    expect(await page.locator("tr.selected").count() == 0, "a fact of the old profile's report stays selected")


@check("a profile's labels never draw another profile's data, however late they arrive")
async def profile_view_late(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await open_doc(page, docs["multi"], docs["M"])
    await page.evaluate(
        """() => {
          window.__scopes = [];
          const view = document.getElementById("document-view");
          new MutationObserver(() => {
            window.__scopes.push([...view.querySelectorAll('[data-slot="rows"] td.mono')].map((td) => td.textContent));
          }).observe(view, { childList: true, subtree: true });
        }"""
    )
    await page.route(lambda url: f"/api/profile?profile={DEMO}" in url, delayed(1.5))
    await page.select_option("#profile-select", DEMO)
    await page.wait_for_timeout(800)
    title_ = await page.text_content("#profile-title") or ""
    expect("示例领域" not in title_, "the new profile's title came before its view was read")
    await page.wait_for_selector('.entity-table[data-entity="wear_test"] tbody tr', timeout=5000)
    scopes = await page.evaluate("window.__scopes")
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


@check("a document link under a profile keeps the profile on the missing view's link home")
async def missing_keeps_profile(page: Page, _: str, docs: dict[str, str], __: Path) -> None:
    await page.goto(f"{docs['multi']}/#/p/{DEMO}/doc/0000000000000000")
    await page.wait_for_selector("#missing-view:not(.hidden)")
    href = await page.get_attribute("#missing-view .missing-actions a", "href")
    expect(href == f"#/p/{DEMO}", f"the link home goes to {href!r}")


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
    reports: list[str] = []
    page.on("request", lambda request: reports.append(request.url) if "/report" in request.url else None)
    await page.evaluate(f"fetch('/api/documents/{docs['M']}/run?profile=tco', {{method: 'POST'}})")
    await page.evaluate(f"location.hash = '#/p/{DEMO}/doc/{docs['M']}'")
    await page.wait_for_selector('#document-view [data-slot="other-job"]:not(.hidden)')
    note = await page.text_content('#document-view [data-slot="other-job"]') or ""
    expect("透明导电" in note and "排队" in note, f"the note reads {note!r}")
    expect(await page.locator("#document-view .stage.running").count() == 0, "the other job's progress is drawn")
    # The log panel is this profile's last job's (an earlier check ran one), never the running one of the other.
    status = await page.text_content('#document-view [data-slot="job-status"]') or ""
    expect("处理中" not in status and "排队中" not in status, f"the other job's log is shown: {status!r}")
    expect(not await page.is_disabled('#document-view [data-action="run"]'), "the run button is locked by it")
    await page.wait_for_selector(f'.doc-item[data-focus="doc:{docs["M"]}"] .queued')
    queued = await page.get_attribute(f'.doc-item[data-focus="doc:{docs["M"]}"] .queued', "title") or ""
    expect("透明导电" in queued, f"the busy marker reads {queued!r}")
    loads = len(reports)
    await page.wait_for_timeout(len(stage_names()) * STAGE_SECONDS * 1000 + 2500)
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
    await page.wait_for_timeout(500)
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
