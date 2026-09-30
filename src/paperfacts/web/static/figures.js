// Chart readings: values the vision model read off property-vs-condition charts (the opt-in figures stage).
// They are approximate, never compared across lanes and never put in the results table, so they get a
// section of their own in a neutral colour; blue and orange belong to the two lanes, and a chart belongs to
// neither.

import { releaseFact } from "./facts.js";
import { escapeHtml, fmt, onActivate, toast } from "./html.js";
import { LANES, currentJob, isActive, slot, state } from "./state.js";
import { revealViewer } from "./viewer.js";

// The figures stage as the server reports it: the live job's when there is one, else the files on disk.
export function figuresStage() {
  const stages = currentJob()?.stages ?? state.summary?.stages ?? [];
  return stages.find((stage) => stage.name === "figures") ?? null;
}

// Whether the stored readings are current (read under today's settings): the button then offers a re-read.
export const figuresRead = () => state.summary?.stages?.find((stage) => stage.name === "figures")?.status === "done";

// With no readings the section still shows, and says why: charts are read only on request, so an empty section
// must read as "not asked yet", not as a broken page.
function emptyMessage() {
  const job = currentJob();
  if (isActive(job) && job.figures) return "正在识图：视觉模型逐张读图（每张约一分钟），完成后读数会显示在这里。";
  const stage = figuresStage();
  if (stage?.status === "failed") return `上次识图没有完成${stage.detail ? `（${stage.detail}）` : ""}，可以再次识图。`;
  if (figuresRead()) return "已识图：没有从图上读到可用的数值（本文可能没有可读的“性质—条件”图）。";
  if (!state.summary?.pdf_available) return "尚未识图。识图要从 PDF 上截取图片，这篇文档没有 PDF：请重新上传后再识图。";
  return "尚未识图。点上方「识图」用视觉模型读取图中数值（较慢，按图计费；解析和抽取沿用已有结果），或在上传时勾选「上传后识图」。";
}

export function renderFigures(root) {
  const rows = state.figures?.rows ?? [];
  const empty = slot("figures-empty", root);
  empty.textContent = rows.length ? "" : emptyMessage();
  empty.classList.toggle("hidden", Boolean(rows.length));
  slot("figures-body", root).classList.toggle("hidden", !rows.length);
  const warn = slot("figures-warning", root);
  const notes = [];
  if (state.figures?.stale) notes.push("这些读数是在旧设置下读出的（模型、提示或字段表已变），重新读图前仅供参考。");
  if (state.figures?.orphaned?.length) notes.push(`${state.figures.orphaned.length} 个图块在当前解析里已不存在，无法在页面上定位。`);
  warn.textContent = notes.join(" ");
  warn.classList.toggle("hidden", !notes.length);
  const body = slot("figure-rows", root);
  body.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    const value = row.value != null
      ? `${fmt(row.value)} <span class="unit">${escapeHtml(row.unit ?? "")}</span>`
      : `<span class="muted">无法换算</span>`;
    tr.innerHTML = `<td>${escapeHtml(row.figure ?? "图")}<small>第 ${escapeHtml(row.page)} 页 · 子图 ${escapeHtml(row.panel)}</small></td>`
      + `<td>${escapeHtml(row.field)}</td><td>${escapeHtml(row.series ?? "")}</td><td class="x">${escapeHtml(row.x ?? "")}</td>`
      + `<td class="val">${value}<span class="flag approx" title="从图上读出，不是论文写出的数">近似值 ${escapeHtml(row.precision)}</span></td>`
      + `<td class="mono">${escapeHtml(row.value_raw ?? "")}</td><td class="note">${escapeHtml(row.detail ?? "")}</td>`;
    tr.title = row.caption ?? "";
    tr.tabIndex = 0;
    onActivate(tr, () => locate(row));
    body.append(tr);
  }
}

// The row cites the figure block; its page and box come from the artifact the page already loaded. The row
// is an untyped dict, so a reading without a source_id finds no block rather than breaking the page.
function locate(row) {
  const sourceId = String(row.source_id ?? "");
  const lane = LANES.find((l) => sourceId.startsWith(`${l}_`));
  const block = state.artifacts[lane]?.blocks?.find((b) => b.source_id === sourceId);
  if (!block || !state.viewer) {
    toast("找不到这张图在页面上的位置", true);
    return;
  }
  releaseFact();
  state.viewer.showRegion({ page: block.page, bbox: block.bbox, label: row.figure ?? sourceId });
  revealViewer();
}
