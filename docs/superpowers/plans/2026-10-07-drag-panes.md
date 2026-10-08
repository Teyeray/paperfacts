# Plan — mouse-draggable three-pane resizing (rail | middle | viewer pane)

**Date:** 2026-10-07 · **Design:** `.pi/plans/2026-10-06-drag-panes/plan.md` (user-approved) · **Base:** `c87ed07`

Two PRs, each independently green. Defaults stay byte-for-byte today's (rail 300 / pane 480 ≥1440),
so every existing check and screenshot keeps passing.

## PR1 — rail seam + shared controller

- [x] `static/panes.js`: clamp constants (`RAIL_MIN 280`, `RAIL_MAX 600`, `PANE_MIN 300`,
      `MIDDLE_FLOOR 570`, `CONTENT_PAD 56`), `paneDefault()` band default, whitelisted `readWidth()`
      storage, shared `wireSeparator()` (pointer capture, keyboard ±16/Home/End direction-aware,
      double-click reset, aria-valuenow/min/max sync, document-level pointerup safety).
- [x] `setupRailResizer()` (boot, static `.shell`; inline `--rail` only when a stored key exists)
      and `installReclamp()` (rAF-debounced `resize` + `#document-view` MutationObserver, because
      the viewer pane — which caps the rail's live max — renders after boot).
- [x] Rail handle in static `.shell` markup (`role="separator"`, `aria-orientation="vertical"`,
      `tabindex="0"`, 调整侧栏宽度); resizer CSS: fixed out-of-flow strips (no grid-track change,
      no transitions), hover/focus/drag accent tint (the 10th and last allowed `transparent`),
      repo focus ring, gates for collapsed rail / ≤960px.
- [x] Module lists: CLAUDE.md + `tests/test_web_app.py` served-module parametrize gain `panes`.
- [x] e2e in `tests/e2e/web_races.py`: `rail_drag` (live resize, no text selection, persist +
      reload), `rail_drag_clamp` (exact 280/600 bounds at 1920×1080), `rail_drag_hidden`
      (collapsed + 390px), `rail_resize_keyboard` (arrows/Home/End/dblclick + stored key removed),
      `rail_viewport_shrink` (600px stored → 334 at boot on a document page → 314 at 1280, no
      horizontal overflow).
- [x] Gates: `uv run pytest`, ruff check/format, `tests/e2e/web_races.py` all green;
      `scripts/ui_screenshots.py` — home/dialog PNGs byte-identical, document PNGs differ only in
      the seed's own nondeterminism (document hash, timestamps, list order — reproduced
      main-vs-main).

## PR2 — viewer seam + the 1280–1439.98 rung

- [x] `setupViewerPaneResizer(view, node)` wired per render inside `setupViewerPane`; band default
      (340 below 1440, else 480) via `paneDefault()`; rung deletion in `app.css`; pane side of
      `reclamp()`; e2e pane checks; final ISC walk-through and screenshot run.
- [x] Viewer seam handle in `#tpl-document` (between the expand button and `aside.viewer-pane`,
      调整预览宽度), wired per render — the pane element is captured at wire time because the
      template fragment empties once appended. Inline `--viewer-pane` on `#document-view` always
      applied (band default or stored width, clamped against the rail's live width); Arrow keys
      inverted (left widens), Home/End to bounds, double-click resets to the band default and
      removes the key.
- [x] `app.css` 1280–1439.98 rung deleted: the single ≥1280 rule reads `var(--viewer-pane)`; JS
      owns the band default (accepted risk: no-JS 1280–1439.98 would show 480 — the UI is
      JS-rendered anyway).
- [x] `reclamp()` extended: the pane re-clamps against the new viewport (stored width or a fresh
      band default when crossing 1440), and a `readWidth` fix — `Number(null)` is 0, so a missing
      key short-circuits to the fallback before the clamp.
- [x] e2e in `tests/e2e/web_races.py`: `pane_drag` (live + persist + reload + lanes relation),
      `pane_drag_clamp` (994 max / 300 floor at 1920×1080), `pane_drag_collapsed` (handle hidden,
      expand restores the dragged width), `pane_resize_keyboard` (inverted arrows, End 514, Home
      300, dblclick 480), `pane_band_defaults` (480@1440 / 340@1280 with no key, following a
      resize), `pane_seeded_reload` (700→514 at boot, →354 at 1280, lanes fit),
      `pane_viewport_shrink` (514→354, no overflow), `pane_zoom_smoke` (a drag leaves the zoom
      level and ladder alone).
- [x] Gates: `uv run pytest`, ruff check/format, `tests/e2e/web_races.py` all green (all checks,
      old and new); `scripts/ui_screenshots.py` — no new pixel diffs: branch-vs-main matches
      main-vs-baseline diff set (the seed's own viewer-image nondeterminism plus a 1×2 px
      antialiasing flake in home-query); pane geometry identical at rest.
