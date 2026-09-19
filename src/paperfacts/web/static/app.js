// PaperFacts frontend entry point: wire up the upload zone, router, and document library, then
// open whatever document the URL points at.
// No build step; module breakdown: state (state & shared constants), api, html (small utilities),
// router, library (left rail), document (document view), facts (fact comparison), samples
// (sample records), job (job progress), viewer (page-level provenance), corpus (the home view's
// library-wide results table).

import { api } from "./api.js";
import { showDocument, showEmpty } from "./document.js";
import { toast } from "./html.js";
import { loadLibrary, setupUpload } from "./library.js";
import { installRouter, route } from "./router.js";

document.addEventListener("DOMContentLoaded", async () => {
  setupUpload();
  document.getElementById("refresh-library").addEventListener("click", loadLibrary);
  installRouter({ onDocument: showDocument, onEmpty: showEmpty });
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
