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

VIEWPORTS = {"1440x900": (1440, 900), "390x844": (390, 844)}
# name -> (hash, the page is ready once this selector shows). ``{A}`` is the seeded paper A's id.
PAGES = {
    "home": ("#/", "#corpus-view:not(.hidden) table"),
    "home-query": ("#/?q=abc", "#corpus-view:not(.hidden) table"),
    "home-compact": ("#/", "#corpus-view:not(.hidden) table"),
    "document": ("#/doc/{A}", "#document-view:not(.hidden) .results-table tbody tr"),
    "document-compact": ("#/doc/{A}", "#document-view:not(.hidden) .results-table tbody tr"),
}
# The density hook is an attribute on .shell; nothing in the page sets it yet, so the picture does.
COMPACT = "document.querySelector('.shell').dataset.density = 'compact'"


async def shoot(page: Page, base: str, docs: dict[str, str], name: str, out: Path, viewport: str) -> Path:
    hash_, ready = PAGES[name]
    await page.goto(f"{base}/{hash_.format(**docs)}")
    await page.wait_for_selector(ready)
    if name.endswith("-compact"):
        await page.evaluate(COMPACT)
    await page.wait_for_timeout(300)  # fonts and the sticky header settle
    path = out / f"{name}-{viewport}.png"
    await page.screenshot(path=str(path), full_page=True)
    return path


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / ".omc" / "artifacts" / "ui-screenshots")
    parser.add_argument("--only", default="", help="shoot only the pages whose name contains this")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory, serve(Path(directory)) as (base, docs, _pdf):
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(args=["--no-proxy-server"])
            for viewport, (width, height) in VIEWPORTS.items():
                context = await browser.new_context(viewport={"width": width, "height": height})
                page = await context.new_page()
                for name in PAGES:
                    if args.only in name:
                        print(await shoot(page, base, docs, name, args.out, viewport))
                await context.close()
            await browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
