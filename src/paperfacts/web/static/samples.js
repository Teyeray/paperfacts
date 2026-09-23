// Sample records: each lane's own LaneExtraction (raw text -> normalized value <- source block); clicking a source id highlights it in the viewer.

import { caveats, escapeHtml, fmt } from "./html.js";
import { LANES, LANE_LABEL, state } from "./state.js";

export function renderLanes(root) {
  root.innerHTML = "";
  for (const lane of LANES) root.append(laneNode(lane, state.lanes[lane]));
}

function laneNode(lane, data) {
  const box = document.createElement("div");
  box.className = "lane";
  const meta = data ? `${data.samples.length} 样品 · ${data.usage?.total_tokens ?? "?"} tokens · key ${data.extractor_key}` : "";
  box.innerHTML = `<div class="lane-head ${lane === "mineru" ? "a" : "b"}"><span>${LANE_LABEL[lane]}</span><span class="meta">${escapeHtml(meta)}</span></div>`;
  if (!data) { box.append(note("还没有抽取结果。")); return box; }
  if (data.target) box.append(sampleNode({ sample_id: "靶材", label: "论文级", conditions: {}, fields: data.target.fields }));
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
  const conditions = Object.entries(sample.conditions ?? {}).map(([k, v]) => `${k}: ${v}`).join(" · ");
  div.innerHTML = `<span class="sid">${escapeHtml(sample.sample_id)}</span><span class="label">${escapeHtml(sample.label ?? "")}</span>${conditions ? `<div class="cond">${escapeHtml(conditions)}</div>` : ""}`;
  for (const f of sample.fields) div.append(fieldNode(f));
  return div;
}

function fieldNode(f) {
  const row = document.createElement("div");
  row.className = "field";
  const cond = f.condition ? ` <small>@${escapeHtml(f.condition)}</small>` : "";
  const norm = f.value != null ? ` <small>= ${fmt(f.value)} ${escapeHtml(f.unit ?? "")}</small>` : (f.normalization_note ? ` <small>(${escapeHtml(f.normalization_note)})</small>` : "");
  row.innerHTML = `<span class="fname">${escapeHtml(f.field)}</span><span class="fval">${escapeHtml(f.value_raw)} ${escapeHtml(f.unit_raw ?? "")}${cond}${norm}${caveats(f)}</span><button type="button" class="src">${escapeHtml(f.source_ids.join(", ") || "无来源")}</button>`;
  row.querySelector(".src").addEventListener("click", () => state.viewer?.highlight(f.source_ids));
  return row;
}
