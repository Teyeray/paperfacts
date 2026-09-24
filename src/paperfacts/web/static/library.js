// Left rail: the document library list and upload. On a successful upload, navigate to that
// document; the document view takes over showing progress from there.

import { api } from "./api.js";
import { escapeHtml, toast } from "./html.js";
import { documentHash, navigate } from "./router.js";
import { LANES, LANE_LABEL, isActive, state } from "./state.js";

export async function loadLibrary() {
  try {
    state.docs = await api("/api/documents");
  } catch (error) {
    toast(`读取文档库失败：${error.message}`, true);
    return;
  }
  await loadActiveDocs();
  renderLibrary();
}

// Which documents are busy right now. DocumentSummary knows nothing about jobs, so this is one
// extra request for the whole list — never one per row.
async function loadActiveDocs() {
  try {
    const jobs = await api("/api/jobs");
    state.activeDocs = new Set(jobs.filter(isActive).map((job) => job.document_id));
  } catch (error) {
    // The marker is a nicety; a failure here must not hide the library — but it must not vanish
    // without trace either, or a broken /api/jobs looks like "nothing is running".
    console.warn("读取任务列表失败：", error);
    state.activeDocs = new Set();
  }
}

export function renderLibrary() {
  const list = document.getElementById("doc-list");
  list.innerHTML = "";
  if (!state.docs.length) {
    list.innerHTML = `<li class="doc-list-empty muted">还没有文档</li>`;
    return;
  }
  for (const doc of state.docs) {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "doc-item" + (doc.document_id === state.current ? " active" : "");
    button.innerHTML = `
      <div class="name" title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</div>
      <div class="sub"><span class="dots">${progressDots(doc)}</span>${queuedMark(doc)}<code>${escapeHtml(doc.document_id.slice(0, 8))}</code></div>
      ${miniCounts(doc)}`;
    button.addEventListener("click", () => navigate(documentHash(doc.document_id)));
    li.append(button);
    list.append(li);
  }
}

// Five dots = five stages; filled = done, hollow = not done (shape + color, not color alone)
function progressDots(doc) {
  const steps = [
    ...LANES.map((l) => [`解析 ${LANE_LABEL[l]}`, doc.parsed[l]]),
    ...LANES.map((l) => [`抽取 ${LANE_LABEL[l]}`, doc.extracted[l]]),
    ["比较", doc.compared],
  ];
  return steps
    .map(([label, done]) => `<span class="dot${done ? " on" : ""}" role="img" title="${label}：${done ? "已完成" : "未完成"}" aria-label="${label}：${done ? "已完成" : "未完成"}"></span>`)
    .join("");
}

function queuedMark(doc) {
  return state.activeDocs.has(doc.document_id)
    ? `<span class="queued" role="img" title="排队或处理中" aria-label="排队或处理中">\u23f3</span>`
    : "";
}

// The comparison tally as four small badges. Each keeps its status colour but always carries the
// count and a word, so the row stays readable without relying on colour.
function miniCounts(doc) {
  const c = doc.counts;
  if (!c) return "";
  const items = [
    ["agree", "一致", c.agree],
    ["conflict", "冲突", c.conflict],
    ["ambiguous", "不确定", c.ambiguous],
    ["missing", "缺失", c.missing],
  ];
  const badges = items
    .map(([kind, label, n]) => `<span class="tally-item ${kind}${n ? "" : " zero"}" title="${label}：${n}"><i></i>${n}<em>${label}</em></span>`)
    .join("");
  return `<div class="tally">${badges}</div>`;
}

export function setupUpload() {
  const zone = document.getElementById("dropzone");
  const input = document.getElementById("file-input");
  document.getElementById("pick-file").addEventListener("click", () => input.click());
  input.addEventListener("change", () => { if (input.files[0]) upload(input.files[0]); input.value = ""; });
  for (const type of ["dragenter", "dragover"]) zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.add("drag"); });
  for (const type of ["dragleave", "drop"]) zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.remove("drag"); });
  zone.addEventListener("drop", (e) => { const file = e.dataTransfer.files[0]; if (file) upload(file); });
}

async function upload(file) {
  const zone = document.getElementById("dropzone");
  const force = document.getElementById("upload-force").checked;
  const form = new FormData();
  form.append("file", file, file.name);
  zone.classList.add("busy");
  try {
    const result = await api(`/api/documents?force=${force}`, { method: "POST", body: form });
    toast(`已上传 ${file.name}，开始处理`);
    await loadLibrary();
    navigate(documentHash(result.document.document_id));
  } catch (error) {
    toast(`上传失败：${error.message}`, true);
  } finally {
    zone.classList.remove("busy");
  }
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
    } catch (error) {
      toast(`批量处理失败：${error.message}`, true);
    } finally {
      button.disabled = false;
    }
  });
}
