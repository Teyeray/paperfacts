// PaperFacts frontend entry point: wire up the upload dialog, profile switcher, router, and document library, then
// open whatever document the URL points at.
// No build step; module breakdown: state (state & shared constants), api, html (small utilities),
// router (routes, the profile prefix and the view generation), profiles (the header switcher and each profile's
// view), library (left rail: the list, its search and chips, the collapse, the bulk run), upload (the upload dialog
// and the page-wide drop), document (document view, home view, profile page and the missing-document /
// missing-profile states), profile (the read-only profile page's renderers), table (the results table and the
// column model, sorting and the density switch), fieldpicker (which field columns are shown), tsv (the clipboard copy), corpus (the home view's
// library-wide results table), explorer (its search box, the home query and the flattened rows), filters (its filter
// panel), facts (fact comparison), figures (chart readings), samples (sample records), job
// (job progress), viewer (page-level provenance), panes (the drag seams' width controller), theme (the manual
// light/dark override), check (the page that checks a pasted profile).

import { api } from "./api.js";
import { setupCheck, showCheck } from "./check.js";
import { onSortChange, refreshCorpus } from "./corpus.js";
import { showDocument, showEmpty, showMissing, showMissingProfile, showProfilePage } from "./document.js";
import { toast, readStored, writeStored } from "./html.js";
import { sortParam, updateQuery } from "./explorer.js";
import {
  loadLibrary,
  onDocsChange,
  setupLibraryDisclosure,
  setupLibraryFilter,
  setupRailToggle,
  setupDelete,
  setupRunAll,
} from "./library.js";
import { installReclamp, setupRailResizer } from "./panes.js";
import { loadProfiles, setupSwitcher, syncSwitcher } from "./profiles.js";
import { installRouter, reloadView, route } from "./router.js";
import { applyUiCopy, state } from "./state.js";
import { applyStoredDensity } from "./table.js";
import { initTheme, cycleTheme } from "./theme.js";
import { setupUpload } from "./upload.js";

// Another profile routed to: the rail lists its documents, the old profile's home table is hidden until the new one
// lands, and the switcher and header link follow. The old profile's document stays on screen while the new one loads,
// but inert: a fact picked or a rerun pressed there would act under the old profile beneath the new one's address.
function onProfile() {
  document.getElementById("corpus-view").classList.add("hidden");
  document.getElementById("document-view").inert = true;
  syncSwitcher();
  loadLibrary();
}

// The skip link cannot be a plain #content link: every hash here is a route, and that one would go home.
function setupSkipLink() {
  document.getElementById("skip-link").addEventListener("click", (event) => {
    event.preventDefault();
    document.getElementById("content").focus();
  });
}

// The theme button cycles 自动 → 浅色 → 深色 (auto follows the OS preference); the label tells where it is,
// including on a fresh load that restores a stored choice.
const THEME_LABELS = { auto: "主题：跟随系统", light: "主题：浅色", dark: "主题：深色" };
function setupThemeToggle() {
  const button = document.getElementById("theme-toggle");
  const label = () => {
    button.setAttribute("aria-label", THEME_LABELS[document.documentElement.dataset.theme ?? "auto"]);
  };
  label();
  button.addEventListener("click", () => {
    cycleTheme();
    label();
  });
}

// The viewer pane's one collapse control, in the topbar like the rail's: a click or `]` flips the document's
// data-viewer attribute and remembers it (setupViewerPane applies the stored value per render). Visible only
// while the document view is on screen at ≥1280px — the pane itself is a wide-screen layout.
const VIEWER_KEY = "paperfacts.viewer-collapsed";
function setupViewerToggle() {
  const view = document.getElementById("document-view");
  const toggle = document.getElementById("viewer-toggle");
  const wide = matchMedia("(min-width: 1280px)");
  const apply = (collapsed) => {
    if (collapsed) view.dataset.viewer = "collapsed";
    else view.dataset.viewer = "";
    toggle.setAttribute("aria-expanded", String(!collapsed));
    const word = collapsed ? "展开预览" : "收起预览";
    toggle.title = word;
    toggle.setAttribute("aria-label", word);
  };
  const flip = () => {
    const collapsed = view.dataset.viewer !== "collapsed";
    apply(collapsed);
    writeStored(VIEWER_KEY, collapsed ? "1" : "0");
  };
  const sync = () => {
    apply(readStored(VIEWER_KEY) === "1");
    toggle.hidden = !wide.matches;
  };
  toggle.hidden = true;
  toggle.addEventListener("click", flip);
  // `]` anywhere on the page, except where the key is text or the toggle is not offered, mirroring the rail's `[`.
  document.addEventListener("keydown", (event) => {
    if (event.key !== "]" || event.ctrlKey || event.metaKey || event.altKey) return;
    if (toggle.hidden) return;
    const target = event.target instanceof Element ? event.target : null;
    if (target?.closest("input, textarea, select, [contenteditable], dialog[open]")) return;
    if (document.querySelector("dialog[open]")) return;
    event.preventDefault();
    flip();
  });
  wide.addEventListener("change", sync);
  // Route hook: showViews toggles .hidden on the views; the document view's class is the one truth for
  // "a document is on screen", so follow it wherever it changes (document, home, missing, check, …).
  new MutationObserver(sync).observe(view, { attributes: true, attributeFilter: ["class"] });
}

document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  setupThemeToggle();
  setupViewerToggle();
  setupSkipLink();
  setupRailToggle();
  setupRailResizer();
  installReclamp();
  applyStoredDensity();
  setupUpload();
  setupRunAll();
  setupDelete();
  setupLibraryFilter();
  setupLibraryDisclosure();
  setupSwitcher();
  setupCheck();
  document.getElementById("refresh-library").addEventListener("click", loadLibrary);
  // The home table's sort is the home query's; its status filters and counts follow the rail's list when a status moves.
  onSortChange((sort) => updateQuery(sortParam(sort)));
  onDocsChange(() => refreshCorpus({ docsChanged: true }));
  document.querySelector('#missing-view [data-action="retry"]').addEventListener("click", reloadView);
  applyUiCopy(document);
  // Settled apart: a profile list that fails does not make the backend unavailable (the page then runs on the
  // default, without a switcher), and the health line does not wait on it. The router starts once both answered:
  // it needs the list to tell a served profile from a missing one. Each view loads its own profile's view.
  const [health] = await Promise.allSettled([api("/api/health"), loadProfiles()]);
  if (health.status === "fulfilled") {
    document.getElementById("health").textContent = `model ${health.value.model}`;
  } else {
    document.getElementById("health").textContent = "后端不可用";
    toast(health.reason.message, true);
  }
  installRouter({
    onDocument: showDocument,
    onEmpty: showEmpty,
    // The home query changed under the same home view (typed, or arrived at): the table is not re-fetched; the brand
    // link follows it and the table on screen is redrawn from state.homeQuery.
    onHomeQuery: () => {
      syncSwitcher();
      refreshCorpus();
    },
    onMissing: showMissing,
    onMissingProfile: showMissingProfile,
    onProfile,
    onProfilePage: showProfilePage,
    onCheck: showCheck,
  });
  route();
  // The router loads the rail itself when the URL names a profile; for the default it is loaded here.
  if (state.profileName === null) {
    syncSwitcher();
    loadLibrary();
  }
});
