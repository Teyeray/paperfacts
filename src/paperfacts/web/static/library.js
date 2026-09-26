// Left rail: the document library list and upload. On a successful upload, navigate to that
// document; the document view takes over showing progress from there. The list, its progress and tallies, the
// upload and the bulk run are all under the profile on screen; a document with results under other profiles says so.

import { api, profileApi } from "./api.js";
import { escapeHtml, keepFocus, toast } from "./html.js";
import { profileTitle, servedProfile } from "./profiles.js";
import { documentHash, navigate, reloadView } from "./router.js";
import { STAGE_LABEL, STAGE_STATUS, STATUS, STATUS_ORDER, isActive, isCurrent, slot, state, viewShows } from "./state.js";

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

// A profile switch reloads the list too (app.js): the refresh token, not the view generation, owns the rail, and a
// newer refresh under the new profile retires any older one still on its way.
export async function loadLibrary() {
  const token = ++refreshToken;
  const profile = state.profileName;
  clearTimeout(refreshTimer);
  refreshTimer = null;
  let docs;
  let activeDocs;
  try {
    // Both lists are part of one refresh: a failed /api/jobs would otherwise read as "nothing is running",
    // clear the busy markers and stop the timer mid-run.
    [docs, activeDocs] = await Promise.all([profileApi(profile, "/api/documents"), loadActiveDocs()]);
  } catch (error) {
    if (token !== refreshToken) return;
    refreshFailures += 1;
    // Said once per outage, not on every retry; the last list read stays on screen meanwhile.
    if (refreshFailures === 1) toast(`读取文档库失败：${error.message}`, true);
    refreshTimer = setTimeout(loadLibrary, Math.min(REFRESH_MS * 2 ** (refreshFailures - 1), MAX_REFRESH_BACKOFF_MS));
    return;
  }
  if (token !== refreshToken) return;
  refreshFailures = 0;
  state.docs = docs;
  state.activeDocs = activeDocs;
  // The open paper's note on another profile's job lasts as long as the rail still sees that job active.
  const other = state.otherJob;
  if (other && !(activeDocs.get(other.document_id) ?? []).includes(other.profile)) {
    state.otherJob = null;
    slot("other-job")?.classList.add("hidden");
  }
  renderLibrary();
  refreshTimer = state.activeDocs.size ? setTimeout(loadLibrary, REFRESH_MS) : null;
}

// Which documents are busy right now, and under which profiles: document id -> the profiles of its active jobs. Busy
// is per document whatever the profile (one document never has two running jobs), and so is the rail's marker.
// DocumentSummary knows nothing about jobs, so this is one extra request for the whole list — never one per row, and
// without the jobs' logs.
async function loadActiveDocs() {
  const jobs = await api("/api/jobs");
  const active = new Map();
  for (const job of jobs.filter(isActive)) active.set(job.document_id, [...(active.get(job.document_id) ?? []), job.profile]);
  return active;
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
    ${miniCounts(doc)}${otherProfilesMark(doc)}`;
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
  const profiles = state.activeDocs.get(doc.document_id);
  if (!profiles) return "";
  const named = manyProfiles() ? `（按${profiles.map((name) => `「${profileTitle(name)}」`).join("、")}）` : "";
  const words = escapeHtml(`排队或处理中${named}`);
  return `<span class="queued" role="img" title="${words}" aria-label="${words}">⏳</span>`;
}

const manyProfiles = () => (state.profiles?.profiles?.length ?? 0) > 1;

// Results under the other served profiles, as a count in words with their titles beside it. Unknown while the
// profile list has not loaded (the page cannot tell which name is its own).
function otherProfilesMark(doc) {
  const own = servedProfile(state.profileName)?.name;
  if (!own) return "";
  const others = (doc.profiles_done ?? []).filter((name) => name !== own);
  if (!others.length) return "";
  const titles = escapeHtml(others.map(profileTitle).join("、"));
  return `<span class="other-profiles" title="${titles}">另有 ${others.length} 个领域的结果</span>`;
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
  const profile = state.profileName;
  const zone = document.getElementById("dropzone");
  const force = document.getElementById("upload-force").checked;
  let last = null;
  zone.classList.add("busy");
  try {
    for (const file of files) {
      const form = new FormData();
      form.append("file", file, file.name);
      try {
        const result = await profileApi(profile, `/api/documents?force=${force}`, { method: "POST", body: form });
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
  // The generation covers the profile too: a switch is a new view, so an upload made under the old one opens nothing.
  if (last && isCurrent(generation)) navigate(documentHash(last), { reload: true });
}

// Queue everything that is not finished yet. The backend decides what counts as unfinished; here we
// only report how many went in and how many were left out.
export function setupRunAll() {
  const button = document.getElementById("run-all");
  button.addEventListener("click", async () => {
    const force = document.getElementById("upload-force").checked;
    const profile = state.profileName;
    const named = manyProfiles() ? `按「${profileTitle(profile)}」` : "";
    // A forced bulk run re-parses every PDF as well, which on a laptop without the MLX service is hours per
    // paper on a single worker: worth one question before it is queued. So is any bulk run under a profile other
    // than the default, which spends tokens on every paper not finished under it (an example profile, by accident).
    if (force && !window.confirm(`将${named}忽略缓存、强制重跑全部 ${state.docs.length} 篇（含重新解析 PDF），确定？`)) return;
    if (!force && profile !== null && !window.confirm(`将${named}处理文档库里全部未完成的文档（共 ${state.docs.length} 篇），确定？`)) return;
    button.disabled = true;
    try {
      const result = await profileApi(profile, `/api/documents/run-all?force=${force}`, { method: "POST" });
      toast(`已${named}排队 ${result.submitted.length} 篇，跳过 ${result.skipped.length} 篇`);
      await loadLibrary();
      // The open document may be one of them: re-read it so its view follows the new job -- only while it is still
      // shown under the profile the jobs run under.
      if (result.submitted.some((job) => viewShows(job.document_id, profile))) reloadView();
    } catch (error) {
      toast(`批量处理失败：${error.message}`, true);
    } finally {
      button.disabled = false;
    }
  });
}
