// Left rail: the document library list and upload. On a successful upload, navigate to that
// document; the document view takes over showing progress from there.

import { api } from "./api.js";
import { escapeHtml, toast } from "./html.js";
import { documentHash, navigate } from "./router.js";
import { LANES, LANE_LABEL, state } from "./state.js";

export async function loadLibrary() {
  try {
    state.docs = await api("/api/documents");
  } catch (error) {
    toast(`读取文档库失败：${error.message}`, true);
    return;
  }
  renderLibrary();
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
      <div class="sub"><span class="dots">${progressDots(doc)}</span>${miniCounts(doc)}<code>${doc.document_id.slice(0, 8)}</code></div>`;
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

function miniCounts(doc) {
  const c = doc.counts;
  return c ? `<span class="mini-counts" title="agree / conflict / ambiguous / missing">${c.agree}✓ ${c.conflict}✗ ${c.ambiguous}? ${c.missing}–</span>` : "";
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
