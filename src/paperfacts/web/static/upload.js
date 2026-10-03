// Uploading: the 「上传」 dialog (files listed one paper each, the options, a progress bar per file, errors inline), a
// drop of files anywhere on the page, and the programmatic path through the hidden #file-input, which uploads at once
// with the dialog's options. On a successful upload, navigate to that document; the document view takes over
// showing progress from there. Every upload goes under the profile on screen.

import { profileUpload } from "./api.js";
import { el, toast } from "./html.js";
import { loadLibrary } from "./library.js";
import { documentHash, navigate } from "./router.js";
import { isCurrent, state } from "./state.js";

// The rows of the dialog: { file, row, status: "pending" | "uploading" | "done" | "error" }. Kept between openings, so a
// row that failed can be sent again; a row that went up is dropped once the dialog closes.
const pending = [];
let uploading = false;

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
  input.addEventListener("change", () => { uploadAll([...input.files], options()); input.value = ""; });
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

export function openUploadDialog(files = []) {
  addFiles(files);
  const box = dialog();
  if (!box.open) box.showModal();
}

const isPdf = (file) => file.type === "application/pdf" || /\.pdf$/i.test(file.name);

function addFiles(files) {
  for (const file of files) {
    const entry = { file, status: "pending", row: null };
    entry.row = fileRow(entry);
    if (!isPdf(file)) setState(entry, "error", "不是 PDF，不会上传");
    pending.push(entry);
    document.getElementById("upload-files").append(entry.row);
  }
  renderStatus();
}

// One row per file: the name and size, a remove button, and under them the state line and the progress bar. The
// empty `si` slot is where 「添加 SI」 goes once supplementary files can be attached; until then it holds nothing.
function fileRow(entry) {
  const name = el("span", { className: "fname", title: entry.file.name }, entry.file.name);
  name.append(el("small", { text: formatSize(entry.file.size) }));
  const remove = el("button", { className: "ghost remove", title: "移除" }, "✕");
  remove.type = "button";
  remove.setAttribute("aria-label", `移除 ${entry.file.name}`);
  remove.addEventListener("click", () => { removeEntry(entry); renderStatus(); });
  const row = el("li", { className: "upload-file" }, name, remove, el("span", { className: "state" }));
  row.querySelector(".state").dataset.slot = "state";
  const si = el("span");
  si.dataset.slot = "si";
  row.append(si);
  return row;
}

function setState(entry, status, text) {
  entry.status = status;
  const line = entry.row.querySelector('[data-slot="state"]');
  line.textContent = text;
  line.className = `state ${status === "error" ? "error" : status === "done" ? "done" : ""}`.trim();
  entry.row.querySelector(".remove").disabled = status === "uploading";
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
    await uploadAll(ready.map((entry) => entry.file), chosen, (index, progress) => reportRow(ready[index], progress), true);
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
    setState(entry, "done", "已上传，开始处理");
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

// One request per file, in order (the server takes one PDF per upload); the last one that went in is opened, unless
// the reader moved to another view while the files went up. `reload`, because it may be the document already on
// screen, whose new job the view must start following. `report(index, {fraction} | {done} | {error})` follows each
// file; without it (the programmatic path) a toast says what happened. Resolves to the last uploaded id, or null.
async function uploadAll(files, { force, figures }, report = null, quiet = false) {
  if (!files.length) return null;
  const generation = state.generation;
  const profile = state.profileName;
  let last = null;
  for (const [index, file] of files.entries()) {
    const form = new FormData();
    form.append("file", file, file.name);
    if (figures) form.append("figures", "true");
    try {
      report?.(index, { fraction: 0 });
      const result = await profileUpload(profile, `/api/documents?force=${force}`, form, (fraction) =>
        report?.(index, { fraction }),
      );
      last = result.document.document_id;
      report?.(index, { done: true });
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
