// PaperFacts frontend entry point: wire up the upload zone, profile switcher, router, and document library, then
// open whatever document the URL points at.
// No build step; module breakdown: state (state & shared constants), api, html (small utilities),
// router (routes, the profile prefix and the view generation), profiles (the header switcher and each profile's
// view), library (left rail), document (document view, home view and the missing-document / missing-profile
// states), table (the results table and the column model), fieldpicker (which field columns are shown), tsv (the
// clipboard copy), corpus (the home view's library-wide results table), facts (fact comparison), figures (chart
// readings), samples (sample records), job (job progress), viewer (page-level provenance).

import { api } from "./api.js";
import { showDocument, showEmpty, showMissing, showMissingProfile } from "./document.js";
import { toast } from "./html.js";
import { loadLibrary, setupLibraryDisclosure, setupRunAll, setupUpload } from "./library.js";
import { loadProfiles, setupSwitcher, syncSwitcher } from "./profiles.js";
import { installRouter, reloadView, route } from "./router.js";
import { applyUiCopy, state } from "./state.js";

// Another profile routed to: the rail lists its documents, the old profile's home table is hidden until the new one
// lands, and the switcher and header link follow.
function onProfile() {
  document.getElementById("corpus-view").classList.add("hidden");
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

document.addEventListener("DOMContentLoaded", async () => {
  setupSkipLink();
  setupUpload();
  setupRunAll();
  setupLibraryDisclosure();
  setupSwitcher();
  document.getElementById("refresh-library").addEventListener("click", loadLibrary);
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
  installRouter({ onDocument: showDocument, onEmpty: showEmpty, onMissing: showMissing, onMissingProfile: showMissingProfile, onProfile });
  route();
  // The router loads the rail itself when the URL names a profile; for the default it is loaded here.
  if (state.profileName === null) {
    syncSwitcher();
    loadLibrary();
  }
});
