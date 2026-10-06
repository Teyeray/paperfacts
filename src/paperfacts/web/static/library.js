// Left rail: the document library list, its search and chips, the rail's collapse and the bulk run. The list, its
// progress and tallies and the bulk run are all under the profile on screen; a document with results under other
// profiles says so. Uploading is upload.js.

import { api, profileApi } from "./api.js";
import { articleTag, escapeHtml, keepFocus, readStored, siTag, toast, writeStored } from "./html.js";
import { profileTitle, servedProfile } from "./profiles.js";
import { documentHash, navigate, reloadView } from "./router.js";
import { STAGE_LABEL, STAGE_STATUS, STATUS, STATUS_ORDER, isActive, slot, state, viewShows } from "./state.js";

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
// False until the first list has answered: the one time the rail shows skeleton rows instead of a
// stale (or empty) list. Later refreshes swap the rows in place, so the shimmer never replays.
let libraryLoaded = false;
// Told when a refresh replaced state.docs: the home table's status filters are joined with it.
let docsListener = null;
export const onDocsChange = (listener) => {
  docsListener = listener;
};
// Stacked layout: the rail sits above the content, so its list is folded away rather than pushing the
// home table and every document ~3000 px down the page.
const WIDE = window.matchMedia("(min-width: 961px)");

// A profile switch reloads the list too (app.js): the refresh token, not the view generation, owns the rail, and a
// newer refresh under the new profile retires any older one still on its way.
export async function loadLibrary() {
  const token = ++refreshToken;
  const profile = state.profileName;
  if (!libraryLoaded) showDocListSkeleton();
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
  docsListener?.();
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

// First-load placeholder: shimmer rows where the list will be. renderLibrary (or the error path, which
// keeps the skeletons up) owns the swap; the refresh token decides whose rows land.
function showDocListSkeleton() {
  const list = document.getElementById("doc-list");
  list.setAttribute("aria-busy", "true");
  list.replaceChildren(
    ...Array.from({ length: 6 }, () => {
      const li = document.createElement("li");
      li.className = "skeleton-row";
      const name = document.createElement("div");
      name.className = "skeleton";
      const sub = document.createElement("div");
      sub.className = "skeleton";
      sub.style.width = "55%";
      li.append(name, sub);
      return li;
    }),
  );
}

// ---------- search and chips ----------
//
// The filter is the rail's own, never the URL's: a refresh redraws the list through it, and the controls are static
// markup, so what was typed and what is pressed survive every redraw. Each chip is one predicate over a summary.
const CHIPS = {
  review: (doc) => doc.article_type === "review",
  conflict: (doc) => (doc.counts?.conflict ?? 0) > 0,
  unfinished: (doc) => !finishedHere(doc),
};
const filter = { query: "", chips: new Set() };

// Finished under the profile on screen: the server's `profiles_done` names every served profile the paper is
// exported under, the one asked about included, so this is the same "finished" run-all skips. The default is routed
// as null and compared by its name from /api/profiles; until that list has answered, any finished profile counts.
// The home table's 未完成 filter asks the same question.
export function finishedHere(doc) {
  const done = doc.profiles_done ?? [];
  const name = state.profileName ?? state.defaultProfile;
  return name == null ? done.length > 0 : done.includes(name);
}

const matches = (doc) =>
  doc.name.toLowerCase().includes(filter.query) && [...filter.chips].every((chip) => CHIPS[chip](doc));

export function setupLibraryFilter() {
  const search = document.getElementById("rail-search");
  search.addEventListener("input", () => {
    filter.query = search.value.trim().toLowerCase();
    renderLibrary();
  });
  for (const chip of document.querySelectorAll("#rail-chips .chip")) {
    chip.addEventListener("click", () => {
      const name = chip.dataset.chip;
      if (filter.chips.has(name)) filter.chips.delete(name);
      else filter.chips.add(name);
      renderLibrary();
    });
  }
}

function clearFilter() {
  filter.query = "";
  filter.chips.clear();
  document.getElementById("rail-search").value = "";
  renderLibrary();
}

export function renderLibrary() {
  const list = document.getElementById("doc-list");
  list.removeAttribute("aria-busy");
  libraryLoaded = true;
  const shown = state.docs.filter(matches);
  const filtered = Boolean(filter.query) || filter.chips.size > 0;
  document.getElementById("doc-count").textContent = filtered
    ? `（${shown.length}/${state.docs.length}）`
    : `（${state.docs.length}）`;
  for (const chip of document.querySelectorAll("#rail-chips .chip")) {
    const name = chip.dataset.chip;
    chip.setAttribute("aria-pressed", String(filter.chips.has(name)));
    chip.querySelector(".n").textContent = String(state.docs.filter(CHIPS[name]).length);
  }
  keepFocus(list, () => {
    list.innerHTML = "";
    if (!state.docs.length) {
      list.innerHTML = `<li class="doc-list-empty muted">还没有文档</li>`;
      return;
    }
    if (!shown.length) {
      const empty = document.createElement("li");
      empty.className = "doc-list-empty muted";
      empty.append("没有匹配的文档。");
      const clear = document.createElement("button");
      clear.type = "button";
      clear.className = "linklike";
      clear.textContent = "清除筛选";
      clear.addEventListener("click", clearFilter);
      empty.append(clear);
      list.append(empty);
      return;
    }
    for (const doc of shown) list.append(libraryItem(doc));
  });
}

// Two lines: the name, then everything else. A button holds phrasing content only, so each line is a <span>
// set as a block.
function libraryItem(doc) {
  const li = document.createElement("li");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "doc-item" + (doc.document_id === state.current ? " active" : "");
  button.dataset.focus = `doc:${doc.document_id}`;
  if (doc.document_id === state.current) button.setAttribute("aria-current", "page");
  button.innerHTML = `
    <span class="name" title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</span>
    <span class="sub">${articleTag(doc)}${siTag(doc)}${progressDots(doc)}${queuedMark(doc)}${miniCounts(doc)}${otherProfilesMark(doc)}</span>`;
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

// The comparison tally on one line (the rail's cards and the document page's summary), "26 一致 · 3 冲突 · 4 缺失": a zero count is left out, and each count keeps its
// status colour but always carries the number and the word. Nothing at all when every count is zero.
export function miniCounts(doc) {
  const c = doc.counts;
  if (!c) return "";
  const items = STATUS_ORDER.filter((kind) => c[kind]).map(
    (kind) => `<span class="tally-item ${kind}"><i></i>${c[kind]}<em>${STATUS[kind].label}</em></span>`
  );
  if (!items.length) return "";
  const words = escapeHtml(STATUS_ORDER.map((kind) => `${STATUS[kind].label} ${c[kind]}`).join("，"));
  return `<span class="tally" title="${words}">${items.join("")}</span>`;
}

export function setupLibraryDisclosure() {
  const wrap = document.getElementById("doc-list-wrap");
  const sync = () => { wrap.open = WIDE.matches; };
  sync();
  WIDE.addEventListener("change", sync);
}

// ---------- the rail's collapse ----------
//
// One attribute on .shell is the whole state (app.css closes the first grid track on it); remembered per browser so
// a reader who put the rail away finds it away; without storage it starts expanded.
const RAIL_KEY = "paperfacts.rail-collapsed";

export function setupRailToggle() {
  const shell = document.querySelector(".shell");
  const toggle = document.getElementById("rail-toggle");
  const apply = (collapsed) => {
    if (collapsed) shell.dataset.rail = "collapsed";
    else delete shell.dataset.rail;
    toggle.setAttribute("aria-expanded", String(!collapsed));
    const word = collapsed ? "展开侧栏" : "收起侧栏";
    toggle.title = word;
    toggle.setAttribute("aria-label", word);
  };
  apply(readStored(RAIL_KEY) === "1");
  const flip = () => {
    const collapsed = shell.dataset.rail !== "collapsed";
    apply(collapsed);
    writeStored(RAIL_KEY, collapsed ? "1" : "0");
  };
  toggle.addEventListener("click", flip);
  // `[` anywhere on the page, except where the key is text: a field, a select, editable content, an open dialog.
  document.addEventListener("keydown", (event) => {
    if (event.key !== "[" || event.ctrlKey || event.metaKey || event.altKey) return;
    const target = event.target instanceof Element ? event.target : null;
    if (target?.closest("input, textarea, select, [contenteditable], dialog[open]")) return;
    if (document.querySelector("dialog[open]")) return;
    event.preventDefault();
    flip();
  });
}

// ---------- the bulk run ----------
//
// Queue everything that is not finished yet, after one confirmation that also offers 「忽略缓存，全部重跑」. The backend
// decides what counts as unfinished; here we only report how many went in and how many were left out.
export function setupRunAll() {
  const button = document.getElementById("run-all");
  const dialog = document.getElementById("run-all-dialog");
  const force = document.getElementById("run-all-force");
  const confirm = document.getElementById("run-all-confirm");
  const message = document.getElementById("run-all-message");
  for (const close of dialog.querySelectorAll('[data-action="close"]')) close.addEventListener("click", () => dialog.close());
  button.addEventListener("click", () => {
    const named = manyProfiles() ? `按「${profileTitle(state.profileName)}」` : "";
    message.textContent = `将${named}处理文档库里全部未完成的文档（共 ${state.docs.length} 篇），每篇一个任务，按文档库顺序排队。`;
    force.checked = false;
    dialog.showModal();
  });
  // A forced bulk run re-parses every PDF as well, which on a laptop without the MLX service is hours per paper on a
  // single worker; the box says so beside the option, and the confirmation names the count.
  confirm.addEventListener("click", async () => {
    const forced = force.checked;
    const profile = state.profileName;
    const named = manyProfiles() ? `按「${profileTitle(profile)}」` : "";
    dialog.close();
    button.disabled = true;
    try {
      const result = await profileApi(profile, `/api/documents/run-all?force=${forced}`, { method: "POST" });
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
