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
import { MAX_BACKOFF_MS, MAX_RETRIES, POLL_MS } from "./job.js";
import { loadLibrary, setupLibraryDisclosure, setupRunAll, setupUpload } from "./library.js";
import { installRouter, reloadView, route } from "./router.js";
import { applyUiCopy, state } from "./state.js";

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

function applyProfile(profile) {
  state.profile = profile;
  showProfile(profile);
  applyUiCopy(document);
}

// A profile that failed to load leaves the generic copy on screen (uiCopy's defaults) and is asked for again
// with the job poll's backoff; when it lands, the view on screen is redrawn in its words.
function retryProfile(failures = 1) {
  setTimeout(async () => {
    try {
      applyProfile(await api("/api/profile"));
      reloadView();
    } catch (error) {
      if (failures < MAX_RETRIES) retryProfile(failures + 1);
      else toast(`读取领域配置失败，页面使用通用名称：${error.message}`, true);
    }
  }, Math.min(POLL_MS * 2 ** failures, MAX_BACKOFF_MS));
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
  applyUiCopy(document);
  // Settled apart: a profile that fails does not make the backend unavailable, and the health line does not
  // wait on it. The first view draws, and the router starts listening, only once the profile has answered.
  const [health, profile] = await Promise.allSettled([api("/api/health"), api("/api/profile")]);
  if (health.status === "fulfilled") {
    document.getElementById("health").textContent = `model ${health.value.model}`;
  } else {
    document.getElementById("health").textContent = "后端不可用";
    toast(health.reason.message, true);
  }
  if (profile.status === "fulfilled") applyProfile(profile.value);
  await loadLibrary();
  installRouter({ onDocument: showDocument, onEmpty: showEmpty, onMissing: showMissing });
  route();
  if (profile.status === "rejected") retryProfile();
});
