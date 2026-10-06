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
// (job progress), viewer (page-level provenance), theme (the manual light/dark override), check (the page that checks a pasted profile).

import { api } from "./api.js";
import { setupCheck, showCheck } from "./check.js";
import { onSortChange, refreshCorpus } from "./corpus.js";
import { showDocument, showEmpty, showMissing, showMissingProfile, showProfilePage } from "./document.js";
import { toast } from "./html.js";
import { sortParam, updateQuery } from "./explorer.js";
import {
  loadLibrary,
  onDocsChange,
  setupLibraryDisclosure,
  setupLibraryFilter,
  setupRailToggle,
  setupRunAll,
} from "./library.js";
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

document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  setupThemeToggle();
  setupSkipLink();
  setupRailToggle();
  applyStoredDensity();
  setupUpload();
  setupRunAll();
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
