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
import { state } from "./state.js";

// The header names the domain this server runs, and says so when its profile is only an example.
function showProfile(profile) {
  const title = document.getElementById("profile-title");
  title.textContent = profile.title_zh;
  title.title = profile.description_zh ?? "";
  if (profile.maturity === "example") {
    const badge = document.createElement("span");
    badge.className = "profile-badge";
    badge.textContent = "示例配置";
    title.append(badge);
  }
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
  document.getElementById("refresh-library").addEventListener("click", loadLibrary);
  document.querySelector('#missing-view [data-action="retry"]').addEventListener("click", reloadView);
  installRouter({ onDocument: showDocument, onEmpty: showEmpty, onMissing: showMissing });
  try {
    // The profile is loaded before the first view draws: every paper-level and sample label comes from it.
    const [health, profile] = await Promise.all([api("/api/health"), api("/api/profile")]);
    document.getElementById("health").textContent = `model ${health.model}`;
    state.profile = profile;
    showProfile(profile);
  } catch (error) {
    document.getElementById("health").textContent = "后端不可用";
    toast(error.message, true);
  }
  await loadLibrary();
  route();
});
