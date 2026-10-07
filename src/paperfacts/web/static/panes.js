// The three-pane seams' drag controller (plan: .pi/plans/2026-10-06-drag-panes/plan.md). Widths live as
// inline custom properties shadowing the :root tokens (--rail on .shell, --viewer-pane on #document-view);
// collapse state and width stay independent keys, so expanding a collapsed pane restores the last dragged width.
import { readStored, writeStored, removeStored } from "./html.js";

const RAIL_KEY = "paperfacts.rail-width";
const PANE_KEY = "paperfacts.viewer-pane-width";
const RAIL_MIN = 280, RAIL_MAX = 600, PANE_MIN = 300;
const MIDDLE_FLOOR = 570; // the facts table's two lane columns need ~570 px (app.css @1280 comment)
const CONTENT_PAD = 56; // .content's left+right padding (22px 28px 60px)
const KEY_STEP = 16;

const railMax = (vw, pane) => Math.min(RAIL_MAX, vw - CONTENT_PAD - MIDDLE_FLOOR - pane);
const paneMax = (vw, rail) => vw - CONTENT_PAD - MIDDLE_FLOOR - rail;
const clampPx = (px, min, max) => Math.round(Math.max(min, Math.min(px, max)));
// The old CSS rung (app.css had 1280–1439.98 = 340), now owned by JS — the panelOpen() default-by-breakpoint
// precedent (explorer.js).
export const paneDefault = (vw) => (vw >= 1440 ? 480 : 340);

// Width storage: whitelisted on read like viewer.js's readZoomIndex — anything odd falls back to the default.
// A missing key must short-circuit before Number(): Number(null) is 0, which would clamp to the minimum.
function readWidth(key, min, max, fallback) {
  const raw = readStored(key);
  if (raw === null) return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? clampPx(n, min, max) : fallback;
}

// One seam: pointer drag with capture, keyboard parity, and a double-click reset. `invert` flips the delta
// (the pane is right-anchored: dragging left widens it). `limit()` supplies the live max (computed against the
// other pane's current width, so both maxima can never squeeze the middle below MIDDLE_FLOOR); `apply()` sets
// the inline custom property; `reset()` is the double-click default. Document-level cleanup means a template
// re-render mid-drag cannot leak body.pane-dragging.
export function wireSeparator(handle, { key, min, limit, current, apply, reset, invert = false, step = KEY_STEP }) {
  const syncAria = (px) => {
    handle.setAttribute("aria-valuenow", String(px));
    handle.setAttribute("aria-valuemin", String(min));
    handle.setAttribute("aria-valuemax", String(limit()));
  };
  const move = (px, persist) => {
    const v = clampPx(px, min, limit());
    apply(v);
    syncAria(v);
    if (persist) writeStored(key, String(v));
    return v;
  };
  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    handle.setPointerCapture(event.pointerId);
    handle.classList.add("is-active");
    document.body.classList.add("pane-dragging"); // the upload overlay's body.drag precedent (upload.js)
    const startX = event.clientX;
    const startWidth = current();
    const onMove = (e) => move(startWidth + (invert ? startX - e.clientX : e.clientX - startX), false);
    const cleanup = () => {
      handle.classList.remove("is-active");
      document.body.classList.remove("pane-dragging");
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", onUp);
      handle.removeEventListener("pointercancel", onCancel);
    };
    const onUp = () => { move(current(), true); cleanup(); };
    const onCancel = cleanup;
    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onUp);
    handle.addEventListener("pointercancel", onCancel);
    // Orphaned-capture safety: if the handle is re-rendered away mid-drag, a document-level pointerup still
    // persists and cleans up.
    document.addEventListener("pointerup", onUp, { once: true });
  });
  // Keyboard parity with tabs.js's roving-arrow discipline: arrows resize by step (direction-aware), Home/End
  // jump to the bounds, modifiers pass through.
  handle.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const delta = { ArrowLeft: invert ? step : -step, ArrowRight: invert ? -step : step }[event.key];
    let handled = true;
    if (delta !== undefined) move(current() + delta, true);
    else if (event.key === "Home") move(min, true);
    else if (event.key === "End") move(limit(), true);
    else handled = false;
    if (handled) event.preventDefault();
  });
  handle.addEventListener("dblclick", () => {
    removeStored(key);
    apply(reset());
    syncAria(reset());
  });
  syncAria(current());
}

// The rail seam, wired once at boot (the .shell is static, never re-rendered). Only a stored width sets the
// inline --rail: with no key the :root default 300px must stay byte-for-byte.
export function setupRailResizer() {
  const shell = document.querySelector(".shell");
  const handle = document.querySelector(".rail-resizer");
  const railEl = document.querySelector(".rail");
  if (!shell || !handle || !railEl) return;
  const current = () => Math.round(railEl.getBoundingClientRect().width);
  // The viewer pane only takes a column at ≥1280 expanded; otherwise the rail max is just geometry/RAIL_MAX.
  const paneLiveWidth = () => {
    const pane = document.querySelector(".viewer-pane");
    if (!pane || getComputedStyle(pane).display === "none") return 0;
    return Math.round(pane.getBoundingClientRect().width);
  };
  const limit = () => railMax(window.innerWidth, paneLiveWidth());
  const apply = (px) => shell.style.setProperty("--rail", `${px}px`);
  if (readStored(RAIL_KEY) !== null) apply(readWidth(RAIL_KEY, RAIL_MIN, limit(), 300));
  wireSeparator(handle, { key: RAIL_KEY, min: RAIL_MIN, limit, current, apply, reset: () => 300 });
}

// The viewer-pane seam, wired per render inside setupViewerPane (the template clones fresh handles). Unlike
// the rail, the inline --viewer-pane is always applied: with no stored key it is the JS band default
// (340 below 1440, else 480 — the deleted CSS rung), so the value shadows :root at every width.
export function setupViewerPaneResizer(view, node) {
  const handle = node.querySelector(".pane-resizer");
  if (!handle) return;
  // The pane element is captured now, not re-queried later: `node` is a DocumentFragment, and once it is
  // appended to the view the fragment empties — the element lives on, the fragment's query would not find it.
  const paneEl = node.querySelector(".viewer-pane");
  // The rail's live width caps the pane: the middle column keeps its 570px floor however wide both get.
  const railLiveWidth = () => {
    const rail = document.querySelector(".rail");
    return rail && getComputedStyle(rail).display !== "none" ? Math.round(rail.getBoundingClientRect().width) : 300;
  };
  const fallback = () => paneDefault(window.innerWidth);
  const limit = () => paneMax(window.innerWidth, railLiveWidth());
  // The template fragment is not laid out at wire time, so the applied width — not the pane's rect — is the
  // source of truth until the first drag (by then the pane is live and the rect takes over).
  let applied = readWidth(PANE_KEY, PANE_MIN, limit(), fallback());
  const apply = (px) => { applied = px; view.style.setProperty("--viewer-pane", `${px}px`); };
  const current = () => Math.round(paneEl.getBoundingClientRect().width) || applied;
  apply(applied);
  wireSeparator(handle, { key: PANE_KEY, min: PANE_MIN, limit, current, apply, reset: fallback, invert: true });
}

// The viewport listeners (the app has none today), rAF-debounced: stored widths are re-clamped against the
// new viewport so a shrunk window never overflows horizontally. Besides window resize, the static
// #document-view is observed: the document (and its viewer pane, which caps the rail's live max) renders
// after boot, and a collapse toggle changes the pane's column without a resize.
export function installReclamp() {
  let tick = 0;
  const reclamp = () => {
    const shell = document.querySelector(".shell");
    const railEl = document.querySelector(".rail");
    const pane = document.querySelector(".viewer-pane");
    const paneWidth = pane && getComputedStyle(pane).display !== "none" ? Math.round(pane.getBoundingClientRect().width) : 0;
    if (shell && railEl && readStored(RAIL_KEY) !== null) {
      shell.style.setProperty("--rail", `${clampPx(Number(readStored(RAIL_KEY)) || 300, RAIL_MIN, railMax(window.innerWidth, paneWidth))}px`);
    }
    // The pane re-clamps too — a stored width against the shrunk viewport, or a fresh band default when the
    // window crossed 1440 since the last render. Inert below 1280/collapsed, where the media rules win anyway.
    const view = document.getElementById("document-view");
    if (view && pane && getComputedStyle(pane).display !== "none") {
      const railWidth = Math.round(railEl.getBoundingClientRect().width);
      view.style.setProperty("--viewer-pane", `${readWidth(PANE_KEY, PANE_MIN, paneMax(window.innerWidth, railWidth), paneDefault(window.innerWidth))}px`);
    }
  };
  const schedule = () => {
    cancelAnimationFrame(tick);
    tick = requestAnimationFrame(reclamp);
  };
  window.addEventListener("resize", schedule);
  const view = document.getElementById("document-view");
  if (view) new MutationObserver(schedule).observe(view, { childList: true, attributes: true, attributeFilter: ["data-viewer"] });
}
