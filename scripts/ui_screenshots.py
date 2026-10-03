"""Screenshots of the web pages against a seeded server, at desktop (1440x900) and phone (390x844) width.

The server, its library and the stub job are ``tests/e2e/web_races.py``'s: nothing calls a parser or a model, and
the pages hold the same seeded papers every run, so two checkouts give comparable pictures. Each page is shot once per
viewport, full length, into ``--out`` (default ``.omc/artifacts/ui-screenshots``, which git ignores: PNGs are attached
to a review, never committed). Run it as the e2e checks are run::

    uv run --with playwright python -m playwright install chromium   # once
    PYTHONPATH=src uv run --with playwright python scripts/ui_screenshots.py [--out DIR] [--only NAME]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

from playwright.async_api import Page, async_playwright

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests" / "e2e"))  # the seeded server

from web_races import serve  # noqa: E402

from support.factories import make_blank_pdf  # noqa: E402  (tests/ is on the path once web_races is imported)

VIEWPORTS = {"1440x900": (1440, 900), "390x844": (390, 844)}
# name -> (hash, the page is ready once this selector shows). ``{A}`` is the seeded paper A's id. A page whose name is
# in PREPARE is arranged by that step before the shot.
PAGES = {
    "home": ("#/", "#corpus-view:not(.hidden) table"),
    "home-rail-collapsed": ("#/", "#corpus-view:not(.hidden) table"),
    "home-rail-filtered": ("#/", "#corpus-view:not(.hidden) table"),
    "home-query": ("#/?q=abc", "#corpus-view:not(.hidden) table"),
    "home-compact": ("#/", "#corpus-view:not(.hidden) table"),
    "home-sorted": ("#/", "#corpus-view:not(.hidden) table"),
    "home-scrolled": ("#/", "#corpus-view:not(.hidden) table"),
    "home-dark": ("#/", "#corpus-view:not(.hidden) table"),
    # The explorer: a search reading the conditions too, and a range filter with the table sorted by that field.
    "home-search": ("#/?q=sample%202&cond=1", "#corpus-view:not(.hidden) table"),
    "home-filtered": ("#/?f.thickness=150~&sort=-thickness", "#corpus-view:not(.hidden) table"),
    "home-explore-dark": ("#/?q=sample%202&f.thickness=150~", "#corpus-view:not(.hidden) table"),
    "upload-dialog": ("#/", "#corpus-view:not(.hidden) table"),
    "upload-si-dialog": ("#/", "#corpus-view:not(.hidden) table"),
    "document-si": ("#/", "#corpus-view:not(.hidden) table"),
    "run-all-dialog": ("#/", "#corpus-view:not(.hidden) table"),
    "document": ("#/doc/{A}", "#document-view:not(.hidden) .results-table tbody tr"),
    "document-compact": ("#/doc/{A}", "#document-view:not(.hidden) .results-table tbody tr"),
}
# The density is an attribute on .shell, set by the home toolbar's 紧凑 and remembered by the browser; the document
# page has no switch of its own, so its picture sets the attribute directly.
COMPACT = "document.querySelector('.shell').dataset.density = 'compact'"


async def sort_home(page: Page, _: Path) -> None:
    # By paper name, with the papers expanded: the sample rows stay under their paper.
    await page.click('#corpus-view [data-sort="paper"]')
    await page.click('#corpus-view [data-focus="expand-all"]')


async def scroll_home(page: Page, _: Path) -> None:
    # Every field shown, so the table is wider than the page: the two frozen columns stay while the values move.
    await page.click('#corpus-view [data-focus="show-empty"]')
    await page.evaluate("document.querySelector('#corpus-view .table-wrap').scrollLeft = 600")


async def open_filters(page: Page, _: Path) -> None:
    # The panel starts closed on a phone; the picture shows it open on both widths.
    if not await page.locator("#filter-panel").count():
        await page.click('#corpus-view [data-focus="filter-toggle"]')
        await page.wait_for_selector("#filter-panel")


async def collapse_rail(page: Page, _: Path) -> None:
    await page.click("#rail-toggle")


async def filter_rail(page: Page, _: Path) -> None:
    await page.fill("#rail-search", "wu")
    await page.click('#rail-chips [data-chip="conflict"]')


async def open_upload_dialog(page: Page, pdf: Path) -> None:
    await page.click("#upload-open")
    await page.wait_for_selector("#upload-dialog[open]")
    await page.set_input_files("#upload-pick", [str(pdf), str(pdf.with_name("1.pdf"))])
    await page.check("#upload-figures")


def si_files(pdf: Path) -> tuple[Path, list[Path]]:
    """A three-page main text and two SI files, made beside the seeded PDF."""
    main = make_blank_pdf(pdf.with_name("Main text.pdf"), [(612.0, 792.0)] * 3)
    si = [
        make_blank_pdf(pdf.with_name("Supporting Information.pdf"), [(612.0, 792.0)] * 2),
        make_blank_pdf(pdf.with_name("SI tables.pdf"), [(612.0, 792.0)]),
    ]
    return main, si


async def open_upload_dialog_with_si(page: Page, pdf: Path) -> None:
    main, si = si_files(pdf)
    await page.click("#upload-open")
    await page.wait_for_selector("#upload-dialog[open]")
    await page.set_input_files("#upload-pick", [str(main), str(pdf)])
    await page.locator("#upload-files .upload-file").first.locator(".si-pick").set_input_files([str(p) for p in si])


async def upload_with_si(page: Page, pdf: Path) -> None:
    # Uploaded through the dialog, as a reader would: the stub job walks the stages, the header shows 含 SI.
    main, si = si_files(pdf)
    await page.click("#upload-open")
    await page.wait_for_selector("#upload-dialog[open]")
    await page.set_input_files("#upload-pick", str(main))
    await page.locator("#upload-files .upload-file").first.locator(".si-pick").set_input_files([str(p) for p in si])
    await page.click("#upload-start")
    await page.wait_for_selector(".doc-head h1 .si-tag")
    await page.locator(".doc-head h1 .si-tag").hover()


async def open_run_all_dialog(page: Page, _: Path) -> None:
    await page.click("#run-all")
    await page.wait_for_selector("#run-all-dialog[open]")


# The collapse exists on a wide screen only (a phone stacks the rail).
DESKTOP_ONLY = {"home-rail-collapsed", "home-scrolled"}
# Shot with the browser asking for a dark page.
DARK = {"home-dark", "home-explore-dark"}
PREPARE = {
    "home-rail-collapsed": collapse_rail,
    "home-rail-filtered": filter_rail,
    "home-compact": lambda page, _: page.click('#corpus-view [data-focus="density:compact"]'),
    "home-sorted": sort_home,
    "home-scrolled": scroll_home,
    "home-filtered": open_filters,
    "home-explore-dark": open_filters,
    "document-compact": lambda page, _: page.evaluate(COMPACT),
    "upload-dialog": open_upload_dialog,
    "upload-si-dialog": open_upload_dialog_with_si,
    "document-si": upload_with_si,
    "run-all-dialog": open_run_all_dialog,
}


async def shoot(page: Page, base: str, docs: dict[str, str], pdf: Path, name: str, out: Path, viewport: str) -> Path:
    hash_, ready = PAGES[name]
    await page.goto(f"{base}/{hash_.format(**docs)}")
    await page.wait_for_selector(ready)
    if name in PREPARE:
        await PREPARE[name](page, pdf)
    await page.wait_for_timeout(300)  # fonts and the sticky header settle
    path = out / f"{name}-{viewport}.png"
    # A full-length page would put a modal dialog's backdrop over only the first screen; a dialog is shot as seen.
    await page.screenshot(path=str(path), full_page=not name.endswith("-dialog"))
    return path


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / ".omc" / "artifacts" / "ui-screenshots")
    parser.add_argument("--only", default="", help="shoot only the pages whose name contains this")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory, serve(Path(directory)) as (base, docs, pdf):
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(args=["--no-proxy-server"])
            for viewport, (width, height) in VIEWPORTS.items():
                for name in PAGES:
                    if args.only not in name or (name in DESKTOP_ONLY and width <= 960):
                        continue
                    # A context per shot: what one page remembers (the collapsed rail) must not shape the next.
                    context = await browser.new_context(
                        viewport={"width": width, "height": height},
                        color_scheme="dark" if name in DARK else "light",
                    )
                    page = await context.new_page()
                    print(await shoot(page, base, docs, pdf, name, args.out, viewport))
                    await context.close()
            await browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
