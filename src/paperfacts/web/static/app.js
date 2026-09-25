// PaperFacts frontend entry point: wire up the upload zone, router, and document library, then
// open whatever document the URL points at.
// No build step; module breakdown: state (state & shared constants), api, html (small utilities),
// router (routes and the view generation), library (left rail), document (document view, home view and
// the missing-document state), table (the results table and the column model), fieldpicker (which field
// columns are shown), tsv (the clipboard copy), corpus (the home view's library-wide results table), facts
// (fact comparison), figures (chart readings), samples (sample records), job (job progress), viewer
// (page-level provenance).

import { api } from "./api.js";
import { showDocument, showEmpty, showMissing } from "./document.js";
import { toast } from "./html.js";
import { loadLibrary, setupLibraryDisclosure, setupRunAll, setupUpload } from "./library.js";
import { installRouter, reloadView, route } from "./router.js";

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
  document.getElementById("refresh-library").addEventListener("click", loadLibrary);
  document.querySelector('#missing-view [data-action="retry"]').addEventListener("click", reloadView);
  installRouter({ onDocument: showDocument, onEmpty: showEmpty, onMissing: showMissing });
  try {
    const health = await api("/api/health");
    document.getElementById("health").textContent = `model ${health.model}`;
  } catch (error) {
    document.getElementById("health").textContent = "后端不可用";
    toast(error.message, true);
  }
  await loadLibrary();
  route();
});
