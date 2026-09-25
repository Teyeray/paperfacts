// Left rail: the document library list and upload. On a successful upload, navigate to that
// document; the document view takes over showing progress from there.

import { api } from "./api.js";
import { escapeHtml, keepFocus, toast } from "./html.js";
import { documentHash, navigate, reloadView } from "./router.js";
import { STAGE_LABEL, STAGE_STATUS, STATUS, STATUS_ORDER, isActive, isCurrent, state } from "./state.js";

// While anything is queued or running, the rail refreshes itself: a bulk run's progress would otherwise
// stay frozen until the reader pressed ↻. A failed refresh is retried with a growing pause, like job
// polling, so one dropped request cannot end the auto-refresh.
const REFRESH_MS = 5000;
const MAX_REFRESH_BACKOFF_MS = 60000;
let refreshTimer = null;
let refreshFailures = 0;
// Refreshes overlap (the timer, ↻, a finished job, run-all, an upload); each takes a token and only the
// newest one paints, so an older list that arrives late can never overwrite a newer one.
let refreshToken = 0;
// Stacked layout: the rail sits above the content, so its list is folded away rather than pushing the
// home table and every document ~3000 px down the page.
const WIDE = window.matchMedia("(min-width: 961px)");

export async function loadLibrary() {
  const token = ++refreshToken;
  clearTimeout(refreshTimer);
  refreshTimer = null;
  let docs;
  try {
    docs = await api("/api/documents");
  } catch (error) {
    if (token !== refreshToken) return;
    refreshFailures += 1;
    // Said once per outage, not on every retry; the last list read stays on screen meanwhile.
    if (refreshFailures === 1) toast(`读取文档库失败：${error.message}`, true);
    refreshTimer = setTimeout(loadLibrary, Math.min(REFRESH_MS * 2 ** (refreshFailures - 1), MAX_REFRESH_BACKOFF_MS));
    return;
  }
  const activeDocs = await loadActiveDocs();
  if (token !== refreshToken) return;
  refreshFailures = 0;
  state.docs = docs;
  state.activeDocs = activeDocs;
  renderLibrary();
  refreshTimer = state.activeDocs.size ? setTimeout(loadLibrary, REFRESH_MS) : null;
}

// Which documents are busy right now. DocumentSummary knows nothing about jobs, so this is one
// extra request for the whole list — never one per row, and without the jobs' logs.
async function loadActiveDocs() {
  try {
    const jobs = await api("/api/jobs");
    return new Set(jobs.filter(isActive).map((job) => job.document_id));
  } catch (error) {
    // The marker is a nicety; a failure here must not hide the library — but it must not vanish
    // without trace either, or a broken /api/jobs looks like "nothing is running".
    console.warn("读取任务列表失败：", error);
    return new Set();
  }
}

export function renderLibrary() {
  const list = document.getElementById("doc-list");
  document.getElementById("doc-count").textContent = `（${state.docs.length}）`;
  keepFocus(list, () => {
    list.innerHTML = "";
    if (!state.docs.length) {
      list.innerHTML = `<li class="doc-list-empty muted">还没有文档</li>`;
      return;
    }
    for (const doc of state.docs) list.append(libraryItem(doc));
  });
}

// A button holds phrasing content only, so every line of the entry is a <span> set as a block.
function libraryItem(doc) {
  const li = document.createElement("li");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "doc-item" + (doc.document_id === state.current ? " active" : "");
  button.dataset.focus = `doc:${doc.document_id}`;
  if (doc.document_id === state.current) button.setAttribute("aria-current", "page");
  button.innerHTML = `
    <span class="name" title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</span>
    <span class="sub">${progressDots(doc)}${queuedMark(doc)}<code>${escapeHtml(doc.document_id.slice(0, 8))}</code></span>
    ${miniCounts(doc)}`;
  button.addEventListener("click", () => {
    if (!WIDE.matches) document.getElementById("doc-list-wrap").open = false;
    navigate(documentHash(doc.document_id));
  });
  li.append(button);
  return li;
}

// One dot per pipeline stage, from the server's reading of the disk; a word count beside them (and the full
// list as the accessible name) so progress never rests on the dots' fill alone.
function progressDots(doc) {
  const stages = doc.stages ?? [];
  const applicable = stages.filter((stage) => stage.status !== "skipped");
  const done = applicable.filter((stage) => stage.status === "done").length;
  const words = stages
    .map((stage) => `${STAGE_LABEL[stage.name] ?? stage.name}：${STAGE_STATUS[stage.status]?.label ?? stage.status}`)
    .join("，");
  const dots = stages.map((stage) => `<span class="dot ${escapeHtml(stage.status)}"></span>`).join("");
  return (
    `<span class="dots" role="img" title="${escapeHtml(words)}" aria-label="进度 ${done}/${applicable.length}：${escapeHtml(words)}">${dots}</span>` +
    `<span class="dots-text" aria-hidden="true">${done}/${applicable.length}</span>`
  );
}

function queuedMark(doc) {
  return state.activeDocs.has(doc.document_id)
    ? `<span class="queued" role="img" title="排队或处理中" aria-label="排队或处理中">⏳</span>`
    : "";
}

// The comparison tally as four small badges. Each keeps its status colour but always carries the
// count and a word, so the row stays readable without relying on colour.
function miniCounts(doc) {
  const c = doc.counts;
  if (!c) return "";
  const badges = STATUS_ORDER
    .map((kind) => [kind, STATUS[kind].label, c[kind]])
    .map(([kind, label, n]) => `<span class="tally-item ${kind}${n ? "" : " zero"}" title="${label}：${n}"><i></i>${n}<em>${label}</em></span>`)
    .join("");
  return `<span class="tally">${badges}</span>`;
}

export function setupLibraryDisclosure() {
  const wrap = document.getElementById("doc-list-wrap");
  const sync = () => { wrap.open = WIDE.matches; };
  sync();
  WIDE.addEventListener("change", sync);
}

export function setupUpload() {
  const zone = document.getElementById("dropzone");
  const input = document.getElementById("file-input");
  document.getElementById("pick-file").addEventListener("click", () => input.click());
  input.addEventListener("change", () => { uploadAll([...input.files]); input.value = ""; });
  for (const type of ["dragenter", "dragover"]) zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.add("drag"); });
  for (const type of ["dragleave", "drop"]) zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.remove("drag"); });
  zone.addEventListener("drop", (e) => uploadAll([...e.dataTransfer.files]));
}

// One request per file, in order (the server takes one PDF per upload); the last one that went in is opened,
// unless the reader moved to another view while the files went up. `reload`, because it may be the document
// already on screen, whose new job the view must start following.
async function uploadAll(files) {
  if (!files.length) return;
  const generation = state.generation;
  const zone = document.getElementById("dropzone");
  const force = document.getElementById("upload-force").checked;
  let last = null;
  zone.classList.add("busy");
  try {
    for (const file of files) {
      const form = new FormData();
      form.append("file", file, file.name);
      try {
        const result = await api(`/api/documents?force=${force}`, { method: "POST", body: form });
        last = result.document.document_id;
        toast(`已上传 ${file.name}，开始处理`);
      } catch (error) {
        toast(`上传失败（${file.name}）：${error.message}`, true);
      }
    }
  } finally {
    zone.classList.remove("busy");
  }
  await loadLibrary();
  if (last && isCurrent(generation)) navigate(documentHash(last), { reload: true });
}

// Queue everything that is not finished yet. The backend decides what counts as unfinished; here we
// only report how many went in and how many were left out.
export function setupRunAll() {
  const button = document.getElementById("run-all");
  button.addEventListener("click", async () => {
    const force = document.getElementById("upload-force").checked;
    // A forced bulk run re-parses every PDF as well, which on a laptop without the MLX service is hours per
    // paper on a single worker: worth one question before it is queued.
    if (force && !window.confirm(`将忽略缓存、强制重跑全部 ${state.docs.length} 篇（含重新解析 PDF），确定？`)) return;
    button.disabled = true;
    try {
      const result = await api(`/api/documents/run-all?force=${force}`, { method: "POST" });
      toast(`已排队 ${result.submitted.length} 篇，跳过 ${result.skipped.length} 篇`);
      await loadLibrary();
      // The open document may be one of them: re-read it so its view follows the new job.
      if (state.current && result.submitted.some((job) => job.document_id === state.current)) reloadView();
    } catch (error) {
      toast(`批量处理失败：${error.message}`, true);
    } finally {
      button.disabled = false;
    }
  });
}
