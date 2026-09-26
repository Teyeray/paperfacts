// Sample records: each lane's own LaneExtraction (raw text -> normalized value <- source block); clicking a source id highlights it in the viewer.

import { releaseFact } from "./facts.js";
import { caveats, escapeHtml, toast } from "./html.js";
import { LANES, LANE_LABEL, entityGroups, entityOf, inEntity, state, uiCopy } from "./state.js";
import { readingText } from "./tsv.js";
import { revealViewer } from "./viewer.js";

// The paper-level record (under the profile's short name for it) and the unplaced values are told apart from
// real samples by `data-kind`, never by the name: a model is free to call a sample either name too.
const UNATTRIBUTED_SID = "未归属";

export function renderLanes(root) {
  root.innerHTML = "";
  for (const lane of LANES) root.append(laneNode(lane, state.lanes[lane]));
}

function laneNode(lane, data) {
  const box = document.createElement("div");
  box.className = "lane";
  const reasoning = data?.usage?.reasoning_tokens ? `（推理 ${data.usage.reasoning_tokens}）` : "";
  const meta = data ? `${data.samples.length} ${uiCopy("entity_label_zh")} · ${data.usage?.total_tokens ?? "?"} tokens${reasoning} · key ${data.extractor_key}` : "";
  box.innerHTML = `<div class="lane-head ${lane === "mineru" ? "a" : "b"}"><span>${LANE_LABEL[lane]}</span><span class="meta">${escapeHtml(meta)}</span></div>`;
  if (!data) { box.append(note("还没有抽取结果。")); return box; }
  if (data.paper) box.append(sampleNode({ sample_id: uiCopy("paper_level_short_zh"), label: uiCopy("paper_level_label_zh"), conditions: {}, fields: data.paper.fields }, "paper"));
  if (!data.samples.length) box.append(note(`模型没有识别出${uiCopy("entity_label_zh")}。`));
  // With several entity types each entity's samples come under its label; with one, they are just listed.
  const groups = entityGroups();
  for (const group of groups) {
    const samples = data.samples.filter((sample) => inEntity(group, sample));
    if (groups.length > 1 && data.samples.length) box.append(entityHead(group, samples.length));
    for (const sample of samples) box.append(sampleNode(sample));
  }
  // Values the model found but could not place on any sample. Shown apart because nothing compares them:
  // hiding them would make the lane look emptier than it was.
  if (data.unattributed?.length) {
    box.append(sampleNode({ sample_id: UNATTRIBUTED_SID, label: `没能对应到任何${uiCopy("entity_label_zh")}`, conditions: {}, fields: data.unattributed }, "unattributed"));
  }
  if (data.failed_questions?.length) {
    const fields = data.failed_questions.map((q) => q.field).join("、");
    box.append(note(`${fields}：模型两次都没有给出有效回答，这些字段为空；下次运行会只重问这几个问题。`));
  }
  if (data.invalid_source_ids?.length || data.dropped?.length) {
    box.append(note(`清洗记录：${data.invalid_source_ids.length} 个编造的 source_id 被剔除；${data.dropped.length} 个取值被丢弃`));
  }
  return box;
}

function entityHead(group, count) {
  const head = document.createElement("div");
  head.className = "entity-head lane-entity";
  head.textContent = `${group.label} · ${count}`;
  return head;
}

function note(text) {
  const div = document.createElement("div");
  div.className = "lane-empty";
  div.textContent = text;
  return div;
}

function sampleNode(sample, kind = "sample") {
  const div = document.createElement("div");
  div.className = "sample";
  div.dataset.kind = kind;
  div.dataset.sample = sample.sample_id ?? "";
  div.dataset.entity = entityOf(sample);
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
  // Stated once for the whole series and written onto every sample: worth saying next to the number.
  const entity = escapeHtml(uiCopy("entity_label_zh"));
  const series = f.series ? `<span class="flag series" title="论文对整个${entity}系列只写了一次，这里是按系列写到每个${entity}上的">全系列</span>` : "";
  const reading = readingText(f);
  const norm = reading ? ` <small>${escapeHtml(reading)}</small>` : "";
  row.innerHTML = `<span class="fname">${escapeHtml(f.field)}</span><span class="fval">${escapeHtml(f.value_raw)} ${escapeHtml(f.unit_raw ?? "")}${cond}${norm}${series}${caveats(f)}</span><button type="button" class="src">${escapeHtml(f.source_ids.join(", ") || "无来源")}</button>`;
  row.querySelector(".src").addEventListener("click", () => {
    releaseFact();
    state.viewer?.highlight(f.source_ids);
    revealViewer();
  });
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

// `kind` is "paper" for the results table's paper-level row (it matches the lanes' paper-level records,
// whatever their name) and "sample" for a sample row (matched by id among real samples of its `entity` only: two
// entity types may each have an "S1").
export function showEvidence(host, fieldName, rowSampleId, kind = "sample", entity = null) {
  if (!host) return;
  clearEvidence(host);
  const wanted = new Set(String(rowSampleId ?? "").split(ID_SEPARATOR).map((id) => id.trim()).filter(Boolean));
  const rows = [];
  for (const sample of host.querySelectorAll(".sample")) {
    if (sample.dataset.kind !== kind) continue;
    if (kind === "sample" && !wanted.has(sample.dataset.sample ?? "")) continue;
    if (kind === "sample" && entity && sample.dataset.entity !== entity) continue;
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
