// Document view: load every artifact for one document, render the template, and take over its background job.
// Also the two other things the content area can show: the home view and the "no such document" state.

import { api, optional } from "./api.js";
import { toast } from "./html.js";
import { LANES, currentJob, isActive, isCurrent, slot, state } from "./state.js";
import { PageViewer } from "./viewer.js";
import { renderFilters, renderKpis, renderRows, selectRowByIndex } from "./facts.js";
import { renderLanes } from "./samples.js";
import { renderResults } from "./table.js";
import { renderFigures } from "./figures.js";
import { loadCorpus, renderCorpus } from "./corpus.js";
import { renderJobLog, renderStages, startPolling, stopPolling, submitRun } from "./job.js";
import { loadLibrary, renderLibrary } from "./library.js";
import { factFromHash, reloadView } from "./router.js";

const VIEWS = ["empty-state", "corpus-view", "document-view", "missing-view"];

function showViews(...visible) {
  for (const id of VIEWS) document.getElementById(id).classList.toggle("hidden", !visible.includes(id));
}

// Nothing of the document being left may keep drawing: its poller stops, and the next document starts clean.
function leaveDocument() {
  stopPolling();
  state.current = null;
  state.summary = null;
  state.job = null;
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
  leaveDocument();
  const intro = document.getElementById("empty-state");
  showViews("empty-state", ...(state.corpus?.rows?.length ? ["corpus-view"] : []));
  renderLibrary();
  await loadCorpus();
  if (!isCurrent(generation)) return; // the reader opened a document while the table was on its way
  const root = document.getElementById("corpus-view");
  const hasRows = Boolean(state.corpus?.rows?.length);
  root.classList.toggle("hidden", !hasRows);
  intro.classList.toggle("with-corpus", hasRows);
  renderCorpus(root);
}

// A link that names no document (or one that could not be read) says so, instead of leaving the previous
// page on screen under the new address or falling back to the empty-library intro.
export function showMissing(id, error = null) {
  leaveDocument();
  showViews("missing-view");
  const view = document.getElementById("missing-view");
  const notFound = !error || error.status === 404;
  view.querySelector("h1").textContent = notFound ? "找不到这篇文档" : "读取文档失败";
  view.querySelector('[data-slot="missing-message"]').textContent = notFound
    ? `文档库里没有编号为「${id}」的文档：链接可能写错了，或者这篇文档已被删除。`
    : `文档 ${id} 暂时读不出来：${error.message}`;
  view.querySelector('[data-action="retry"]').classList.toggle("hidden", notFound);
  renderLibrary();
}

async function openDocument(id) {
  const generation = state.generation;
  const switching = id !== state.current;
  let data;
  try {
    data = await loadDocumentData(id);
  } catch (error) {
    if (isCurrent(generation)) showMissing(id, error);
    return;
  }
  if (!isCurrent(generation)) return; // a later navigation owns the page now
  if (switching) {
    stopPolling();
    state.filter = null;
    state.viewer = null;
    state.selectedFact = null;
  }
  state.current = id;
  Object.assign(state, data);
  renderLibrary();
  renderDocument();
  const factIndex = factFromHash();
  // A new document opens at its top, not at whatever depth the previous one was scrolled to.
  if (switching && factIndex == null) window.scrollTo(0, 0);
  selectRowByIndex(factIndex);
  if (isActive(state.job)) startPolling(state.job.job_id, pollCallbacks);
  else stopPolling();
}

// The summary first: an id with no document ends here with one 404, not nine.
async function loadDocumentData(id) {
  const summary = await api(`/api/documents/${id}`);
  const [report, dataset, figures, jobs, ...rest] = await Promise.all([
    optional(api(`/api/documents/${id}/report`)),
    optional(api(`/api/documents/${id}/dataset`)),
    optional(api(`/api/documents/${id}/figures`)),
    optional(api(`/api/documents/${id}/jobs`)),
    ...LANES.map((l) => optional(api(`/api/documents/${id}/extraction/${l}`))),
    ...LANES.map((l) => optional(api(`/api/documents/${id}/artifact/${l}`))),
  ]);
  return {
    summary,
    report,
    dataset,
    figures,
    job: jobs?.at(-1) ?? null, // the backend orders by creation time ascending, so the last one is the most recent
    lanes: Object.fromEntries(LANES.map((l, i) => [l, rest[i]])),
    artifacts: Object.fromEntries(LANES.map((l, i) => [l, rest[LANES.length + i]])),
  };
}

function renderDocument() {
  const view = document.getElementById("document-view");
  showViews("document-view");
  const viewerState = state.viewer?.getState() ?? null; // re-rendering after a job finishes must not lose the page or highlight the viewer was on
  view.innerHTML = "";
  const node = document.getElementById("tpl-document").content.cloneNode(true);
  const s = (name) => slot(name, node);
  const { summary } = state;

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

async function rerun(button, force) {
  button.disabled = true; // lock the button while queued; the backend's resubmission for the same document is idempotent too
  const generation = state.generation;
  try {
    const job = await submitRun(state.current, force);
    if (!isCurrent(generation)) return;
    state.job = job;
    toast(force ? "已排队：全部重跑" : "已排队：按缓存增量处理");
    renderJobPanels();
    startPolling(job.job_id, pollCallbacks);
  } catch (error) {
    toast(`无法重新处理：${error.message}`, true);
    if (isCurrent(generation)) button.disabled = false;
  }
}
