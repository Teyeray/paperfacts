// Sample records: each lane's own LaneExtraction (raw text -> normalized value <- source block); clicking a source id highlights it in the viewer.

import { caveats, escapeHtml, fmt, toast } from "./html.js";
import { LANES, LANE_LABEL, state } from "./state.js";

// The sid the paper-level record is rendered under; the results table's target row points at it.
export const TARGET_SID = "靶材";

export function renderLanes(root) {
  root.innerHTML = "";
  for (const lane of LANES) root.append(laneNode(lane, state.lanes[lane]));
}

function laneNode(lane, data) {
  const box = document.createElement("div");
  box.className = "lane";
  const reasoning = data?.usage?.reasoning_tokens ? `（推理 ${data.usage.reasoning_tokens}）` : "";
  const meta = data ? `${data.samples.length} 样品 · ${data.usage?.total_tokens ?? "?"} tokens${reasoning} · key ${data.extractor_key}` : "";
  box.innerHTML = `<div class="lane-head ${lane === "mineru" ? "a" : "b"}"><span>${LANE_LABEL[lane]}</span><span class="meta">${escapeHtml(meta)}</span></div>`;
  if (!data) { box.append(note("还没有抽取结果。")); return box; }
  if (data.target) box.append(sampleNode({ sample_id: TARGET_SID, label: "论文级", conditions: {}, fields: data.target.fields }));
  if (!data.samples.length) box.append(note("模型没有识别出样品。"));
  for (const sample of data.samples) box.append(sampleNode(sample));
  // Values the model found but could not place on any sample. Shown apart because nothing compares them:
  // hiding them would make the lane look emptier than it was.
  if (data.unattributed?.length) {
    box.append(sampleNode({ sample_id: "未归属", label: "没能对应到任何样品", conditions: {}, fields: data.unattributed }));
  }
  if (data.invalid_source_ids?.length || data.dropped?.length) {
    box.append(note(`清洗记录：${data.invalid_source_ids.length} 个编造的 source_id 被剔除；${data.dropped.length} 个取值被丢弃`));
  }
  return box;
}

function note(text) {
  const div = document.createElement("div");
  div.className = "lane-empty";
  div.textContent = text;
  return div;
}

function sampleNode(sample) {
  const div = document.createElement("div");
  div.className = "sample";
  div.dataset.sample = sample.sample_id ?? "";
  const conditions = Object.entries(sample.conditions ?? {}).map(([k, v]) => `${k}: ${v}`).join(" · ");
  div.innerHTML = `<span class="sid">${escapeHtml(sample.sample_id)}</span><span class="label">${escapeHtml(sample.label ?? "")}</span>${conditions ? `<div class="cond">${escapeHtml(conditions)}</div>` : ""}`;
  for (const f of sample.fields) div.append(fieldNode(f));
  return div;
}

function fieldNode(f) {
  const row = document.createElement("div");
  row.className = "field";
  row.dataset.field = f.field ?? "";
  const cond = f.condition ? ` <small>@${escapeHtml(f.condition)}</small>` : "";
  const norm = f.value != null ? ` <small>= ${fmt(f.value)} ${escapeHtml(f.unit ?? "")}</small>` : (f.normalization_note ? ` <small>(${escapeHtml(f.normalization_note)})</small>` : "");
  row.innerHTML = `<span class="fname">${escapeHtml(f.field)}</span><span class="fval">${escapeHtml(f.value_raw)} ${escapeHtml(f.unit_raw ?? "")}${cond}${norm}${caveats(f)}</span><button type="button" class="src">${escapeHtml(f.source_ids.join(", ") || "无来源")}</button>`;
  row.querySelector(".src").addEventListener("click", () => state.viewer?.highlight(f.source_ids));
  return row;
}

// ---------- jumping from a refused cell to the records behind it ----------
//
// An empty cell in the results table is a refusal, and the reason for it is in what the two lanes
// actually recorded. Clicking the cell opens this section and marks every record that fed that
// (sample, field): the reader sees the raw disagreement instead of a blank.

// A matched row's sample_id is the two lanes' ids joined by `_scopes` in dataset.py; either half
// identifies the record in its own lane.
const ID_SEPARATOR = " | ";

// Marks go stale the moment anything else takes over the viewer, so every entry point clears them first.
export function clearEvidence(host) {
  for (const marked of host?.querySelectorAll(".field.evidence") ?? []) marked.classList.remove("evidence");
}

export function showEvidence(host, fieldName, rowSampleId) {
  if (!host) return;
  clearEvidence(host);
  const wanted = new Set(String(rowSampleId ?? "").split(ID_SEPARATOR).map((id) => id.trim()).filter(Boolean));
  const rows = [];
  for (const sample of host.querySelectorAll(".sample")) {
    if (!wanted.has(sample.dataset.sample ?? "")) continue;
    for (const row of sample.querySelectorAll(".field")) {
      if ((row.dataset.field ?? "") === fieldName) rows.push(row);
    }
  }
  host.open = true;
  if (!rows.length) {
    host.scrollIntoView({ behavior: "smooth", block: "start" });
    toast("两路都没有这个字段的记录");
    return;
  }
  for (const row of rows) row.classList.add("evidence");
  rows[0].scrollIntoView({ behavior: "smooth", block: "center" });
}
