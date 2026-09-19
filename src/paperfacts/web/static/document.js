// Document view: load every artifact for one document, render the template, and take over its background job.

import { api, optional } from "./api.js";
import { toast } from "./html.js";
import { LANES, currentJob, isActive, slot, state } from "./state.js";
import { PageViewer } from "./viewer.js";
import { renderFilters, renderKpis, renderRows, selectRowByIndex } from "./facts.js";
import { renderLanes } from "./samples.js";
import { renderResults } from "./table.js";
import { renderJobLog, renderStages, startPolling, stopPolling, submitRun } from "./job.js";
import { loadLibrary, renderLibrary } from "./library.js";

// Router entry point: if the document is already open, just jump to that fact; otherwise load the whole thing
export function showDocument(id, factIndex = null) {
  if (id === state.current && state.summary) selectRowByIndex(factIndex);
  else openDocument(id, factIndex);
}

export function showEmpty() {
  state.current = null;
  state.viewer = null;
  stopPolling();
  document.getElementById("empty-state").classList.remove("hidden");
  document.getElementById("document-view").classList.add("hidden");
  renderLibrary();
}

async function openDocument(id, factIndex) {
  try {
    const [data, jobs] = await Promise.all([loadDocumentData(id), optional(api(`/api/documents/${id}/jobs`))]);
    // only switch `current` once the data has actually arrived: if the read fails, the page stays
    // on the previous document instead of showing a half-old, half-new mix
    if (id !== state.current) { state.filter = null; state.viewer = null; }
    state.current = id;
    Object.assign(state, data);
    state.job = jobs?.at(-1) ?? null; // the backend orders by creation time ascending, so the last one is the most recent
  } catch (error) {
    toast(`读取文档失败：${error.message}`, true);
    return;
  }
  renderLibrary();
  renderDocument();
  selectRowByIndex(factIndex);
  if (isActive(state.job)) startPolling(state.job.job_id, pollCallbacks);
  else stopPolling();
}

async function loadDocumentData(id) {
  const [summary, report, dataset, ...rest] = await Promise.all([
    api(`/api/documents/${id}`),
    optional(api(`/api/documents/${id}/report`)),
    optional(api(`/api/documents/${id}/dataset`)),
    ...LANES.map((l) => optional(api(`/api/documents/${id}/extraction/${l}`))),
    ...LANES.map((l) => optional(api(`/api/documents/${id}/artifact/${l}`))),
  ]);
  return {
    summary,
    report,
    dataset,
    lanes: Object.fromEntries(LANES.map((l, i) => [l, rest[i]])),
    artifacts: Object.fromEntries(LANES.map((l, i) => [l, rest[LANES.length + i]])),
  };
}

function renderDocument() {
  const view = document.getElementById("document-view");
  document.getElementById("empty-state").classList.add("hidden");
  view.classList.remove("hidden");
  const viewerState = state.viewer?.getState() ?? null; // re-rendering after a job finishes must not lose the page or highlight the viewer was on
  view.innerHTML = "";
  const node = document.getElementById("tpl-document").content.cloneNode(true);
  const s = (name) => slot(name, node);
  const { summary } = state;

  s("name").textContent = summary.name;
  s("id").textContent = summary.document_id;
  s("uploaded").textContent = summary.uploaded_at ? `上传于 ${summary.uploaded_at.replace("T", " ").slice(0, 16)}` : "由命令行处理";
  const runButton = node.querySelector('[data-action="run"]');
  runButton.disabled = !summary.pdf_available || isActive(currentJob());
  runButton.addEventListener("click", () => rerun(runButton, s("force").checked));

  renderStages(s("stages"));
  renderKpis(s("kpis"));
  renderResults(node.querySelector(".results"));
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
}

const pollCallbacks = {
  onUpdate: renderJobPanels,
  onFinish: async (job) => {
    toast(job.status === "done" ? "处理完成" : `处理失败：${job.error}`, job.status === "failed");
    try {
      Object.assign(state, await loadDocumentData(job.document_id));
    } catch (error) {
      toast(`读取结果失败：${error.message}`, true);
      return;
    }
    renderDocument();
    await loadLibrary();
  },
};

async function rerun(button, force) {
  button.disabled = true; // lock the button while queued; the backend's resubmission for the same document is idempotent too
  try {
    state.job = await submitRun(state.current, force);
    toast(force ? "已排队：全部重跑" : "已排队：按缓存增量处理");
    renderJobPanels();
    startPolling(state.job.job_id, pollCallbacks);
  } catch (error) {
    toast(`无法重新处理：${error.message}`, true);
    button.disabled = false;
  }
}
