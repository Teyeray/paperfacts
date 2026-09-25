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
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import uvicorn
from playwright.async_api import Page, Route, async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/, for the shared seeding helpers

from paperfacts.compare import ComparisonCounts, ComparisonReport, FieldComparison
from paperfacts.config import Settings
from paperfacts.matching import SampleMatching
from paperfacts.models import BACKENDS, PageGeometry, ParsedArtifact
from paperfacts.storage import document_key
from paperfacts.web.app import create_app
from paperfacts.web.documents import Library
from paperfacts.web.jobs import Job, JobManager
from paperfacts.workflow import stage_names
from support.extraction import make_field, make_lane, make_sample
from support.factories import make_blank_pdf, make_block
from support.profiles import shipped_profile

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
    {"name": "target_purity", "label": "靶材纯度", "unit": "%", "scope": "target", "description": "靶材纯度"},
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
        lane = lane.model_copy(update={"no_tco_film": not samples})
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
        matching=SampleMatching(),
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
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}", docs, root / "0.pdf"
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
