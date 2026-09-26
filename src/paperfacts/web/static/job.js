// Background jobs: the stage progress bar, processing log, queuing, and polling. Depends only on
// state / api; the document view's callbacks decide what happens once a job finishes.

import { api, profileApi } from "./api.js";
import { escapeHtml, toast } from "./html.js";
import { JOB_STATUS_LABEL, STAGE_LABEL, STAGE_STATUS, currentJob, isActive, isCurrent, jobInProfile, state } from "./state.js";

export const POLL_MS = 1500;
// A dropped request (a tunnel hiccup, a restarting server) is retried with a growing pause; only a run of
// failures this long gives up, so one bad response cannot freeze the panel at "running".
export const MAX_RETRIES = 5;
export const MAX_BACKOFF_MS = 15000;
// Each loop owns a token; stopping bumps it, so a tick already waiting on the network cannot schedule
// another one afterwards and two loops can never run side by side.
let pollToken = 0;
let pollTimer = null;

// The live job's stages (with each step's detail) when there is one, otherwise the server's reading of the
// files on disk: the same stages in the same order either way.
export function renderStages(list) {
  list.innerHTML = "";
  const stages = currentJob()?.stages ?? state.summary?.stages ?? [];
  for (const stage of stages) {
    const li = document.createElement("li");
    const status = STAGE_STATUS[stage.status] ?? { label: stage.status, glyph: "?" };
    const name = STAGE_LABEL[stage.name] ?? stage.name;
    li.className = `stage ${stage.status}`;
    li.title = `${name}：${status.label}${stage.detail ? ` · ${stage.detail}` : ""}`;
    li.innerHTML =
      `<span class="st" aria-hidden="true">${status.glyph}</span>${escapeHtml(name)}` +
      `<span class="visually-hidden">：${escapeHtml(status.label)}</span>` +
      (stage.detail ? ` <span class="detail">${escapeHtml(stage.detail)}</span>` : "");
    list.append(li);
  }
}

export function renderJobLog(details, pre, statusSpan) {
  const job = currentJob();
  if (!job) { details.classList.add("hidden"); return; }
  details.classList.remove("hidden");
  statusSpan.textContent = `${JOB_STATUS_LABEL[job.status] ?? job.status}${job.error ? ` · ${job.error}` : ""}`;
  pre.textContent = (job.log ?? []).join("\n");
  if (isActive(job)) details.open = true;
}

export const submitRun = (documentId, profile, force) =>
  profileApi(profile, `/api/documents/${documentId}/run?force=${force}`, { method: "POST" });

// Poll until the job ends: onUpdate(job) on every tick, onFinish(job) at the end, onLost() when the job can
// no longer be read (the server restarted and forgot it, or the network stayed down). A loop belongs to the
// view that started it: once the router moves on, it stops without drawing anything.
export function startPolling(jobId, { onUpdate, onFinish, onLost }) {
  stopPolling();
  const token = pollToken;
  const generation = state.generation;
  const live = () => token === pollToken && isCurrent(generation);
  let failures = 0;
  const schedule = (ms) => { pollTimer = setTimeout(tick, ms); };
  const tick = async () => {
    pollTimer = null;
    let job;
    try {
      job = await api(`/api/jobs/${jobId}`);
    } catch (error) {
      if (!live()) return;
      failures += 1;
      if (error.status === 404 || failures > MAX_RETRIES) {
        stopPolling();
        if (error.status !== 404) toast(`读取任务进度失败：${error.message}`, true);
        onLost();
        return;
      }
      schedule(Math.min(POLL_MS * 2 ** failures, MAX_BACKOFF_MS));
      return;
    }
    if (!live()) return;
    failures = 0;
    // A job of another document, or of this one under another profile, never draws this view's progress.
    if (job.document_id !== state.current || !jobInProfile(job, state.currentProfile)) { stopPolling(); return; }
    state.job = job;
    if (isActive(job)) {
      onUpdate(job);
      schedule(POLL_MS);
    } else {
      stopPolling();
      await onFinish(job);
    }
  };
  tick();
}

export function stopPolling() {
  pollToken += 1;
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = null;
}
