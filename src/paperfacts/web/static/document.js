// Document view: load every artifact for one document, render the template, and take over its background job.
// Also the other things the content area can show: the home view, the profile page and the "no such document /
// profile" states.
//
// Every view is loaded under the profile the URL routes to, read once before the first await, with that profile's
// view (profiles.js) fetched beside the data and adopted only together with it.

import { api, optional, profileApi } from "./api.js";
import { toast } from "./html.js";
import { renderDefinition } from "./profile.js";
import { adoptProfile, profileTitle, profileView, servedProfile, syncSwitcher } from "./profiles.js";
import { LANES, applyUiCopy, currentJob, isActive, isCurrent, jobInProfile, slot, state, uiCopy, viewShows } from "./state.js";
import { PageViewer } from "./viewer.js";
import { renderFilters, renderKpis, renderRows, selectRowByIndex } from "./facts.js";
import { renderLanes } from "./samples.js";
import { renderResults } from "./table.js";
import { renderFigures } from "./figures.js";
import { loadCorpus, renderCorpus } from "./corpus.js";
import { renderJobLog, renderStages, startPolling, stopPolling, submitRun } from "./job.js";
import { loadLibrary, renderLibrary } from "./library.js";
import { documentHash, factFromHash, hashFor, reloadView } from "./router.js";

const VIEWS = ["empty-state", "corpus-view", "document-view", "profile-view", "missing-view", "check-view"];

function showViews(...visible) {
  for (const id of VIEWS) document.getElementById(id).classList.toggle("hidden", !visible.includes(id));
}

// A page that is no document's (the profile check): the document being left stops drawing, and only `id` shows.
export function showPage(id) {
  leaveDocument();
  showViews(id);
  renderLibrary();
}

// Nothing of the document being left may keep drawing: its poller stops, and the next document starts clean.
function leaveDocument() {
  stopPolling();
  state.current = null;
  state.currentProfile = null;
  state.summary = null;
  state.job = null;
  state.otherJob = null;
  state.viewer = null;
  state.selectedFact = null;
}

// Router entry point: a new view loads everything; the open document only moves to the fact in the URL.
export function showDocument(id, factIndex, { reload }) {
  if (!reload && id === state.current && state.summary) selectRowByIndex(factIndex);
  else openDocument(id);
}

// The home view: the intro, and under it the whole library's mined table once anything has been mined.
export async function showEmpty() {
  const generation = state.generation;
  const profile = state.profileName;
  leaveDocument();
  const intro = document.getElementById("empty-state");
  // The table last drawn stays up while it is refreshed, unless it or the labels on screen are another profile's: a
  // chip or a column toggle on it would draw with the other profile's groups and store under its key.
  const shown = state.corpus?.rows?.length && state.corpusProfile === profile && state.shownProfile === profile;
  showViews("empty-state", ...(shown ? ["corpus-view"] : []));
  renderLibrary();
  const [corpus, view] = await Promise.all([
    loadCorpus(profile).then((data) => ({ data }), (error) => ({ error })),
    profileView(profile),
  ]);
  if (!isCurrent(generation)) return; // the reader opened a document (or another profile) while the table was on its way
  adoptProfile(profile, view);
  if (corpus.error) toast(`读取结果总表失败：${corpus.error.message}`, true);
  state.corpus = corpus.data ?? null;
  state.corpusProfile = profile;
  const root = document.getElementById("corpus-view");
  const hasRows = Boolean(state.corpus?.rows?.length);
  root.classList.toggle("hidden", !hasRows);
  intro.classList.toggle("with-corpus", hasRows);
  renderCorpus(root);
}

// The read-only page of the routed profile. Its definition is asked by name, so the default's name comes from the
// profile list, or from the profile's own view when that list did not load.
export async function showProfilePage() {
  const generation = state.generation;
  const profile = state.profileName;
  leaveDocument();
  const root = document.getElementById("profile-view");
  root.replaceChildren();
  showViews("profile-view");
  renderLibrary();
  let view;
  let definition;
  try {
    view = await profileView(profile);
    const name = profile ?? state.defaultProfile ?? view?.name;
    if (name == null) throw new Error("不知道默认领域配置的名字");
    definition = await api(`/api/profiles/${encodeURIComponent(name)}`);
  } catch (error) {
    if (!isCurrent(generation)) return;
    showMissing(null, error, { heading: "读取领域配置失败", message: `领域配置暂时读不出来：${error.message}` });
    return;
  }
  if (!isCurrent(generation)) return; // the reader went elsewhere while the definition was on its way
  adoptProfile(profile, view);
  const name = definition.name;
  renderDefinition(root, definition, {
    // Asked only when a preview is opened; an answer that lands after the reader left draws nothing.
    loadPrompts: async (field) => {
      const query = field == null ? "" : `?field=${encodeURIComponent(field)}`;
      const answer = await api(`/api/profiles/${encodeURIComponent(name)}/prompts${query}`);
      return isCurrent(generation) ? answer : null;
    },
    documentHref: (id) => documentHash(id),
    documentName: (id) => state.docs.find((doc) => doc.document_id === id)?.name ?? id,
  });
}

// A link that names no document (or one that could not be read) says so, instead of leaving the previous
// page on screen under the new address or falling back to the empty-library intro.
export function showMissing(id, error = null, { heading = null, message = null } = {}) {
  leaveDocument();
  showViews("missing-view");
  const view = document.getElementById("missing-view");
  const notFound = !error || error.status === 404;
  view.querySelector("h1").textContent = heading ?? (notFound ? "找不到这篇文档" : "读取文档失败");
  view.querySelector('[data-slot="missing-message"]').textContent =
    message ??
    (notFound
      ? `文档库里没有编号为「${id}」的文档：链接可能写错了，或者这篇文档已被删除。`
      : `文档 ${id} 暂时读不出来：${error.message}`);
  view.querySelector('[data-action="retry"]').classList.toggle("hidden", notFound && !heading);
  homeLink(view, hashFor({ profile: state.profileName }));
  renderLibrary();
}

// A URL naming a profile this server does not serve, one that did not load, or one it cannot run. The router has
// left state.profileName on the last profile it could show, so the switcher returns to it and the link home goes
// there.
export function showMissingProfile(name, refusal) {
  leaveDocument();
  showViews("missing-view");
  const view = document.getElementById("missing-view");
  const errors = (refusal.errors ?? []).filter(Boolean);
  const heading = {
    missing: "找不到这个领域配置",
    invalid: "领域配置没有加载成功",
    not_runnable: "这个领域配置在本服务器上不可运行",
  };
  view.querySelector("h1").textContent = heading[refusal.reason] ?? heading.missing;
  view.querySelector('[data-slot="missing-message"]').textContent =
    refusal.reason === "missing" ? `没有名为「${name}」的领域配置。` : `「${name}」：${errors.join("；")}`;
  view.querySelector('[data-action="retry"]').classList.add("hidden");
  homeLink(view, hashFor({ profile: state.profileName }));
  syncSwitcher();
  renderLibrary();
}

function homeLink(view, href) {
  view.querySelector(".missing-actions a").setAttribute("href", href);
}

async function openDocument(id) {
  const generation = state.generation;
  const profile = state.profileName;
  // The same paper under another profile is another view: another report, whose facts the old index and filter
  // do not number.
  const switching = id !== state.current || profile !== state.currentProfile;
  let data;
  let view;
  try {
    [data, view] = await Promise.all([loadDocumentData(id, profile), profileView(profile)]);
  } catch (error) {
    if (isCurrent(generation)) showMissing(id, error);
    return;
  }
  if (!isCurrent(generation)) return; // a later navigation owns the page now
  adoptProfile(profile, view);
  if (switching) {
    stopPolling();
    state.filter = null;
    state.viewer = null;
    state.selectedFact = null;
  }
  // A rerun queued while this load was on its way may be newer than the job list it read: keep following it
  // rather than let the list's finished job stop the poll (see rerun).
  const held = switching ? null : currentJob();
  if (isActive(held) && !isActive(data.job) && held.job_id !== data.job?.job_id) data.job = held;
  state.current = id;
  state.currentProfile = profile;
  Object.assign(state, data);
  // A job queued from elsewhere (another tab, another profile) that the rail has not seen yet: its busy marker.
  if ((isActive(state.job) || isActive(state.otherJob)) && !state.activeDocs.has(id)) loadLibrary();
  else renderLibrary();
  renderDocument();
  const factIndex = factFromHash();
  // A new document opens at its top, not at whatever depth the previous one was scrolled to.
  if (switching && factIndex == null) window.scrollTo(0, 0);
  selectRowByIndex(factIndex);
  if (isActive(state.job)) startPolling(state.job.job_id, pollCallbacks);
  else stopPolling();
}

// The summary first: an id with no document ends here with one 404, not nine. The parse artifacts and the job list
// are the document's under every profile; the rest is the profile's.
async function loadDocumentData(id, profile) {
  const summary = await profileApi(profile, `/api/documents/${id}`);
  const [report, dataset, figures, jobs, ...rest] = await Promise.all([
    optional(profileApi(profile, `/api/documents/${id}/report`)),
    optional(profileApi(profile, `/api/documents/${id}/dataset`)),
    optional(profileApi(profile, `/api/documents/${id}/figures`)),
    optional(api(`/api/documents/${id}/jobs`)),
    ...LANES.map((l) => optional(profileApi(profile, `/api/documents/${id}/extraction/${l}`))),
    ...LANES.map((l) => optional(api(`/api/documents/${id}/artifact/${l}`))),
  ]);
  // The backend orders by creation time ascending, so the last one is the most recent. A job under another profile
  // never draws this view's progress; while one is active it is only noted (this profile's queues behind it).
  const own = (jobs ?? []).filter((job) => jobInProfile(job, profile));
  const others = (jobs ?? []).filter((job) => !jobInProfile(job, profile) && isActive(job));
  return {
    summary,
    report,
    dataset,
    figures,
    job: own.at(-1) ?? null,
    otherJob: others.at(-1) ?? null,
    lanes: Object.fromEntries(LANES.map((l, i) => [l, rest[i]])),
    artifacts: Object.fromEntries(LANES.map((l, i) => [l, rest[LANES.length + i]])),
  };
}

function renderDocument() {
  const view = document.getElementById("document-view");
  showViews("document-view");
  view.inert = false; // made inert by a profile switch (app.js) until this profile's view is drawn
  const viewerState = state.viewer?.getState() ?? null; // re-rendering after a job finishes must not lose the page or highlight the viewer was on
  view.innerHTML = "";
  const node = document.getElementById("tpl-document").content.cloneNode(true);
  const s = (name) => slot(name, node);
  const { summary } = state;
  applyUiCopy(node);
  s("samples").setAttribute("aria-label", `${uiCopy("entity_label_zh")}记录`);

  s("name").textContent = summary.name;
  s("id").textContent = summary.document_id;
  s("uploaded").textContent = summary.uploaded_at ? `上传于 ${summary.uploaded_at.replace("T", " ").slice(0, 16)}` : "由命令行处理";
  const runButton = node.querySelector('[data-action="run"]');
  runButton.disabled = !summary.runnable || isActive(currentJob());
  if (!summary.runnable) runButton.title = "没有 PDF，也没有两路的解析缓存：请重新上传后再处理";
  // Held, not looked up at click time: once the template is appended its fragment is empty, and a slot()
  // lookup in it finds nothing -- the button used to throw on every click.
  const force = s("force");
  runButton.addEventListener("click", () => rerun(runButton, force.checked));

  renderOtherProfiles(s("other-profiles"));
  renderOtherJob(s("other-job"));
  renderStages(s("stages"));
  renderKpis(s("kpis"));
  renderResults(node.querySelector(".results"));
  renderFigures(s("figures"));
  renderFilters(s("filters"));
  renderRows(s("rows"), s("rows-empty"));
  renderLanes(s("lanes"));
  renderJobLog(s("joblog"), s("log"), s("job-status"));
  view.append(node);
  mountViewer(slot("viewer"), viewerState);
}

// The same paper under the other profiles it has results under, one link each.
function renderOtherProfiles(root) {
  const own = servedProfile(state.currentProfile)?.name;
  const others = own ? (state.summary.profiles_done ?? []).filter((name) => name !== own) : [];
  root.classList.toggle("hidden", !others.length);
  if (!others.length) return;
  root.append("在其他领域查看：");
  others.forEach((name, index) => {
    const link = document.createElement("a");
    link.href = hashFor({ profile: name, id: state.current });
    link.textContent = profileTitle(name);
    root.append(...(index ? ["、"] : []), link);
  });
}

function renderOtherJob(root) {
  const job = state.otherJob;
  root.classList.toggle("hidden", !isActive(job));
  if (isActive(job)) root.textContent = `该文档正在按「${profileTitle(job.profile)}」处理，本领域的任务会在其后排队。`;
}

function mountViewer(root, initial) {
  const { summary, artifacts } = state;
  const pageCount = artifacts.mineru?.pages?.length ?? artifacts.paddleocr_vl?.pages?.length ?? 0;
  if (!pageCount) {
    state.viewer = null;
    root.innerHTML = `<div class="viewer-empty">解析完成后这里会显示页面与来源块。</div>`;
    return;
  }
  state.viewer = new PageViewer(root, {
    documentId: summary.document_id,
    pageCount,
    pdfAvailable: summary.pdf_available,
    blocks: Object.fromEntries(LANES.map((l) => [l, artifacts[l]?.blocks ?? []])),
    initial,
  });
}

function renderJobPanels() {
  renderStages(slot("stages"));
  renderJobLog(slot("joblog"), slot("log"), slot("job-status"));
  const runButton = document.querySelector('#document-view [data-action="run"]');
  if (runButton) runButton.disabled = !state.summary?.runnable || isActive(currentJob());
}

// Only a loop still owned by the open document gets here (job.js). Finishing re-reads the whole view through
// the router, which also picks up a forced rerun that was queued behind this job.
const pollCallbacks = {
  onUpdate: renderJobPanels,
  onFinish: (job) => {
    toast(job.status === "done" ? "处理完成" : `处理失败：${job.error}`, job.status === "failed");
    loadLibrary();
    reloadView();
  },
  onLost: () => {
    state.job = null;
    reloadView();
  },
};

// Guarded by the document and profile rather than the view generation: pressed just as the previous job finishes,
// that job's finish handler reloads the view (a new generation) while this request is out, and the new job must
// still be followed. The reload in flight keeps it too (openDocument). A switch to another document or profile
// meanwhile drops it: that view has its own jobs.
async function rerun(button, force) {
  button.disabled = true; // lock the button while queued; the backend's resubmission for the same document is idempotent too
  const id = state.current;
  const profile = state.currentProfile;
  try {
    const job = await submitRun(id, profile, force);
    if (!viewShows(id, profile)) return;
    state.job = job;
    toast(force ? "已排队：全部重跑" : "已排队：按缓存增量处理");
    renderJobPanels();
    startPolling(job.job_id, pollCallbacks);
  } catch (error) {
    toast(`无法重新处理：${error.message}`, true);
    if (viewShows(id, profile)) button.disabled = false;
  }
}
