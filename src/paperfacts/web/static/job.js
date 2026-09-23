// Background jobs: the stage progress bar, processing log, queuing, and polling. Depends only on
// state / api; the document view's callbacks decide what happens once a job finishes.

import { api } from "./api.js";
import { escapeHtml, toast } from "./html.js";
import { LANES, STAGE_LABEL, currentJob, isActive, state } from "./state.js";

const POLL_MS = 1500;
let pollTimer = null;

// Draw from the job snapshot (with each step's detail) when one exists; otherwise draw from whatever artifacts are on disk
export function renderStages(list) {
  list.innerHTML = "";
  const stages = currentJob()?.stages ?? summaryStages(state.summary);
  for (const stage of stages) {
    const li = document.createElement("li");
    li.className = `stage ${stage.status}`;
    li.innerHTML = `<span class="st"></span>${escapeHtml(STAGE_LABEL[stage.name] ?? stage.name)}${stage.detail ? ` <span class="detail">${escapeHtml(stage.detail)}</span>` : ""}`;
    list.append(li);
  }
}

function summaryStages(summary) {
  const done = (ok) => (ok ? "done" : "pending");
  return [
    ...LANES.map((l) => ({ name: `parse:${l}`, status: done(summary.parsed[l]), detail: "" })),
    ...LANES.map((l) => ({ name: `extract:${l}`, status: done(summary.extracted[l]), detail: "" })),
    { name: "compare", status: done(summary.compared), detail: "" },
  ];
}

export function renderJobLog(details, pre, statusSpan) {
  const job = currentJob();
  if (!job) { details.classList.add("hidden"); return; }
  details.classList.remove("hidden");
  statusSpan.textContent = `${job.status}${job.error ? ` · ${job.error}` : ""}`;
  pre.textContent = job.log.join("\n");
  if (isActive(job)) details.open = true;
}

export const submitRun = (documentId, force) => api(`/api/documents/${documentId}/run?force=${force}`, { method: "POST" });

// Poll until the job ends: onUpdate(job) on every tick, onFinish(job) at the end. Switching documents automatically invalidates this polling loop.
export function startPolling(jobId, { onUpdate, onFinish }) {
  stopPolling();
  const tick = async () => {
    let job;
    try {
      job = await api(`/api/jobs/${jobId}`);
    } catch (error) {
      toast(`读取任务失败：${error.message}`, true);
      return;
    }
    if (job.document_id !== state.current) return;
    state.job = job;
    if (isActive(job)) {
      onUpdate(job);
      pollTimer = setTimeout(tick, POLL_MS);
    } else {
      await onFinish(job);
    }
  };
  tick();
}

export function stopPolling() {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = null;
}
