// Uploading: the 「上传」 dialog (files listed one paper each, each with its SI files, the options, a progress bar per
// paper, errors inline), a drop of files anywhere on the page, and the programmatic path through the hidden
// #file-input, which uploads at once with the dialog's options. On a successful upload, navigate to that document; the document view takes over
// showing progress from there. Every upload goes under the profile on screen.

import { profileUpload } from "./api.js";
import { el, keepFocus, toast } from "./html.js";
import { loadLibrary } from "./library.js";
import { documentHash, navigate } from "./router.js";
import { isCurrent, state } from "./state.js";

// The rows of the dialog: { id, file, si: [{ id, file }], row, status: "pending" | "uploading" | "done" | "error" }.
// Kept between openings, so a row that failed can be sent again; a row that went up is dropped once the dialog closes.
const pending = [];
let uploading = false;
let nextId = 0;
// The server merges at most this many SI files after a paper's main text (web/app.py's MAX_SI_PARTS).
const MAX_SI = 4;

const dialog = () => document.getElementById("upload-dialog");
const options = () => ({
  force: document.getElementById("upload-force").checked,
  figures: document.getElementById("upload-figures").checked,
});

export function setupUpload() {
  const box = dialog();
  const input = document.getElementById("file-input");
  const pick = document.getElementById("upload-pick");
  document.getElementById("upload-open").addEventListener("click", () => openUploadDialog());
  document.getElementById("pick-file").addEventListener("click", () => pick.click());
  pick.addEventListener("change", () => { addFiles([...pick.files]); pick.value = ""; });
  document.getElementById("upload-start").addEventListener("click", () => startUploads());
  for (const close of box.querySelectorAll('[data-action="close"]')) close.addEventListener("click", () => box.close());
  box.addEventListener("close", () => {
    // What went up is done with; what failed stays listed for another try.
    for (const entry of [...pending]) if (entry.status === "done") removeEntry(entry);
    renderStatus();
  });
  // The programmatic path: no dialog, the options as the dialog has them (both off until a reader ticks them).
  input.addEventListener("change", () => {
    uploadAll([...input.files].map((file) => ({ file, si: [] })), options());
    input.value = "";
  });
  setupWindowDrop();
}

// A drop of files anywhere opens the dialog with them listed. Only files: dragging selected text over the page must
// do nothing, so the types are read, not the files (which the browser hides until the drop).
function setupWindowDrop() {
  const carriesFiles = (event) => [...(event.dataTransfer?.types ?? [])].includes("Files");
  let depth = 0;
  window.addEventListener("dragenter", (event) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    depth += 1;
    document.body.classList.add("drag");
  });
  window.addEventListener("dragover", (event) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  });
  window.addEventListener("dragleave", (event) => {
    if (!carriesFiles(event)) return;
    depth = Math.max(0, depth - 1);
    if (!depth) document.body.classList.remove("drag");
  });
  window.addEventListener("drop", (event) => {
    depth = 0;
    document.body.classList.remove("drag");
    if (!carriesFiles(event)) return;
    event.preventDefault();
    openUploadDialog([...event.dataTransfer.files]);
  });
}

function openUploadDialog(files = []) {
  addFiles(files);
  const box = dialog();
  if (!box.open) box.showModal();
}

const isPdf = (file) => file.type === "application/pdf" || /\.pdf$/i.test(file.name);

function addFiles(files) {
  for (const file of files) {
    const entry = { id: nextId++, file, si: [], status: "pending", row: null };
    entry.row = fileRow(entry);
    renderSi(entry);
    if (!isPdf(file)) setState(entry, "error", "不是 PDF，不会上传");
    pending.push(entry);
    document.getElementById("upload-files").append(entry.row);
  }
  renderStatus();
}

// One row per paper: the name and size, a remove button, and under them the state line, the paper's SI files with
// 「添加 SI」, and the progress bar.
function fileRow(entry) {
  const name = el("span", { className: "fname", title: entry.file.name }, entry.file.name);
  name.append(el("small", { text: formatSize(entry.file.size) }));
  const remove = el("button", { className: "ghost remove", title: "移除" }, "✕");
  remove.type = "button";
  remove.setAttribute("aria-label", `移除 ${entry.file.name}`);
  remove.addEventListener("click", () => { removeEntry(entry); renderStatus(); });
  const row = el("li", { className: "upload-file" }, name, remove, el("span", { className: "state" }));
  row.querySelector(".state").dataset.slot = "state";
  const si = el("div", { className: "upload-si" });
  si.dataset.slot = "si";
  row.append(si);
  return row;
}

// The SI files of a paper, in the order they will follow its main text: the order they were added, changed with
// ↑/↓. They go up in the same request as the paper and become one document with it.
function renderSi(entry) {
  const slot = entry.row.querySelector('[data-slot="si"]');
  const locked = entry.status === "uploading" || entry.status === "done";
  keepFocus(slot, () => {
    slot.replaceChildren();
    if (entry.si.length) {
      const list = el("ol", { className: "si-list" });
      list.setAttribute("aria-label", `${entry.file.name} 的 SI 文件（按合并顺序）`);
      entry.si.forEach((part, index) => {
        const move = (to, label, symbol, key) => {
          const button = el("button", { className: "ghost si-move", title: label }, symbol);
          button.type = "button";
          button.dataset.focus = `si:${entry.id}:${part.id}:${key}`;
          button.setAttribute("aria-label", `${label}：${part.file.name}`);
          button.disabled = locked || to < 0 || to >= entry.si.length;
          button.addEventListener("click", () => {
            entry.si.splice(index, 1);
            entry.si.splice(to, 0, part);
            renderSi(entry);
          });
          return button;
        };
        const remove = el("button", { className: "ghost si-remove", title: "移除这个 SI 文件" }, "✕");
        remove.type = "button";
        remove.dataset.focus = `si:${entry.id}:${part.id}:remove`;
        remove.setAttribute("aria-label", `移除 SI：${part.file.name}`);
        remove.disabled = locked;
        remove.addEventListener("click", () => {
          entry.si.splice(entry.si.indexOf(part), 1);
          renderSi(entry);
        });
        const name = el("span", { className: "si-name", title: part.file.name }, `SI ${index + 1}：${part.file.name}`);
        name.append(el("small", { text: formatSize(part.file.size) }));
        list.append(el("li", { className: "si-item" }, name, move(index - 1, "上移", "↑", "up"), move(index + 1, "下移", "↓", "down"), remove));
      });
      slot.append(list);
    }
    const pick = el("input", { className: "si-pick" });
    pick.type = "file";
    pick.accept = "application/pdf,.pdf";
    pick.multiple = true;
    pick.hidden = true;
    pick.addEventListener("change", () => { addSi(entry, [...pick.files]); pick.value = ""; });
    const add = el("button", { className: "linklike si-add" }, "添加 SI");
    add.type = "button";
    add.dataset.focus = `si:${entry.id}:add`;
    add.disabled = locked || !isPdf(entry.file) || entry.si.length >= MAX_SI;
    add.title = entry.si.length >= MAX_SI ? `每篇最多 ${MAX_SI} 个 SI 文件` : "附上补充材料（SI）PDF：合并在正文之后，作为同一篇论文处理";
    add.addEventListener("click", () => pick.click());
    const note = el("span", { className: "si-note" }, entry.siNote ?? "");
    slot.append(add, pick, note);
  });
}

function addSi(entry, files) {
  const notes = [];
  for (const file of files) {
    if (!isPdf(file)) notes.push(`${file.name} 不是 PDF，没有加入`);
    else if (entry.si.length >= MAX_SI) notes.push(`每篇最多 ${MAX_SI} 个 SI 文件，${file.name} 没有加入`);
    else entry.si.push({ id: nextId++, file });
  }
  entry.siNote = notes.join("；");
  renderSi(entry);
}

function setState(entry, status, text) {
  entry.status = status;
  const line = entry.row.querySelector('[data-slot="state"]');
  line.textContent = text;
  line.className = `state ${status === "error" ? "error" : status === "done" ? "done" : ""}`.trim();
  entry.row.querySelector(".remove").disabled = status === "uploading";
  renderSi(entry);
}

function removeEntry(entry) {
  if (entry.status === "uploading") return;
  entry.row.remove();
  pending.splice(pending.indexOf(entry), 1);
}

function renderStatus() {
  const ready = pending.filter((entry) => entry.status === "pending");
  const start = document.getElementById("upload-start");
  start.disabled = uploading || !ready.length;
  start.textContent = ready.length > 1 ? `开始上传（${ready.length} 个）` : "开始上传";
  const status = document.getElementById("upload-status");
  if (uploading) status.textContent = "上传中…";
  else if (!pending.length) status.textContent = "";
  else status.textContent = ready.length ? `${ready.length} 个待上传` : "";
}

const formatSize = (bytes) => (bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`);

// The dialog's rows go up with the options as ticked when 开始上传 is pressed; each row shows its own progress and
// its own error, and the ones that succeeded leave the list when the box closes.
async function startUploads() {
  if (uploading) return;
  const ready = pending.filter((entry) => entry.status === "pending");
  if (!ready.length) return;
  const chosen = options();
  uploading = true;
  renderStatus();
  try {
    await uploadAll(ready, chosen, (index, progress) => reportRow(ready[index], progress), true);
    // Every row went up: the box has nothing left to say. A failed row keeps it open, its error on the row.
    if (ready.every((entry) => entry.status === "done")) dialog().close();
  } finally {
    uploading = false;
    renderStatus();
  }
}

function reportRow(entry, progress) {
  if (progress.error) {
    setState(entry, "error", `上传失败：${progress.error}`);
    entry.row.querySelector("progress")?.remove();
    return;
  }
  if (progress.done) {
    setState(entry, "done", progress.duplicate ? `已上传，开始处理。${progress.duplicate}` : "已上传，开始处理");
    entry.row.querySelector("progress")?.remove();
    return;
  }
  if (entry.status !== "uploading") setState(entry, "uploading", "上传中…");
  let bar = entry.row.querySelector("progress");
  if (!bar) {
    bar = document.createElement("progress");
    bar.max = 1;
    bar.setAttribute("aria-label", `${entry.file.name} 的上传进度`);
    entry.row.append(bar);
  }
  bar.value = progress.fraction;
}

// One request per paper, in order (the server takes one paper per upload, its SI files in the same request as `si`
// parts, in their order); the last one that went in is opened, unless the reader moved to another view while the
// files went up. `reload`, because it may be the document already on screen, whose new job the view must start
// following. `report(index, {fraction} | {done, duplicate} | {error})` follows each paper; without it (the
// programmatic path) a toast says what happened. A main text the library already holds with or without SI is said
// either way. Resolves to the last uploaded id, or null.
async function uploadAll(papers, { force, figures }, report = null, quiet = false) {
  if (!papers.length) return null;
  const generation = state.generation;
  const profile = state.profileName;
  let last = null;
  for (const [index, { file, si }] of papers.entries()) {
    const form = new FormData();
    form.append("file", file, file.name);
    for (const part of si) form.append("si", part.file, part.file.name);
    if (figures) form.append("figures", "true");
    try {
      report?.(index, { fraction: 0 });
      const result = await profileUpload(profile, `/api/documents?force=${force}`, form, (fraction) =>
        report?.(index, { fraction }),
      );
      last = result.document.document_id;
      const duplicate = duplicateNote(result.duplicate_of);
      report?.(index, { done: true, duplicate });
      if (duplicate) toast(`${file.name}：${duplicate}`);
      if (!quiet) toast(`已上传 ${file.name}，开始处理${figures ? "（含识图）" : ""}`);
    } catch (error) {
      report?.(index, { error: error.message });
      if (!quiet) toast(`上传失败（${file.name}）：${error.message}`, true);
    }
  }
  await loadLibrary();
  // The generation covers the profile too: a switch is a new view, so an upload made under the old one opens nothing.
  if (last && isCurrent(generation)) navigate(documentHash(last), { reload: true });
  return last;
}

// The library already holds this paper's main text as another document: alone, when this upload carried SI; merged
// with its SI, when it did not. Both stay; the reader is only told.
function duplicateNote(duplicate) {
  if (!duplicate) return "";
  return `文库里已有这篇正文（${duplicate.has_si ? "含 SI" : "不含 SI"}）：${duplicate.name}`;
}
