// Results table: the consolidated dataset (one row per sample, one column per field) that the pipeline
// exports. This is the deliverable; the comparison workbench below it explains how each cell got there.

import { chosenFields, fieldPicker, toggleChip, visibleFields } from "./fieldpicker.js";
import { escapeHtml, fmt } from "./html.js";
import { clearEvidence, showEvidence, TARGET_SID } from "./samples.js";
import { LANE_LABEL, state } from "./state.js";
import { copyTable, tsvHeader, tsvRow } from "./tsv.js";

const TARGET_ROW_ID = "target";
const TARGET_LABEL = "靶材（论文级）";
// The identity columns of the per-sample results table, shared by its header and its clipboard copy.
const LEADING = ["样品", "标签", "条件", "可用/一致"];
// A cell is worth showing only when the pipeline committed to a value. `agree` and `single_source` are the
// two-lane decisions that produce one; `vlm_resolved` (the VLM settled a conflict) and `vlm_filled` (the cell
// was blank and the VLM's table transcription supplied it) are the two the visual stage adds. Every other
// decision deliberately leaves the cell empty.
const CELL_CLASS = { agree: "ok", single_source: "warn", vlm_resolved: "warn", vlm_filled: "vlm" };
// A decided cell says how it was decided: 双路 when both lanes agreed, otherwise the name of the lane
// the value actually came from -- "单路" alone left the reader guessing which one. A cell the visual stage
// decided says so, because its value rests on a reading no parser made.
const LANE_CLASS = { mineru: "lane-a", paddleocr_vl: "lane-b" };
function cellBadge(decision) {
  const status = decision?.decision ?? "";
  if (status === "agree") return { text: "双路", cls: "" };
  if (status === "vlm_resolved") return { text: "视觉裁定", cls: "vlm" };
  if (status === "vlm_filled") return { text: "视觉补全", cls: "vlm" };
  if (status !== "single_source") return { text: "", cls: "" };
  const lanes = String(decision?.lanes ?? "").split(";").map((l) => l.trim()).filter(Boolean);
  if (!lanes.length) return { text: "单路", cls: "" };
  return {
    text: lanes.map((l) => LANE_LABEL[l] ?? l).join("+"),
    cls: lanes.length === 1 ? (LANE_CLASS[lanes[0]] ?? "") : "",
  };
}
// quality_rows are keyed by (sample_id, field); a sample_id may itself contain "|", so join on a
// character that cannot occur in either half.
const KEY_SEPARATOR = "\u0000";

// Which document the toggle below belongs to: opening another paper starts from the default view again.
let showAllFields = false;
let toggleOwner = null;

export function renderResults(root) {
  const data = state.dataset;
  const slot = (name) => root.querySelector(`[data-slot="${name}"]`);
  const download = slot("dataset-download");
  const copy = slot("dataset-copy");
  const hasData = Boolean(data && (data.sample_rows?.length || data.paper_row));
  download.classList.toggle("hidden", !hasData);
  copy.classList.toggle("hidden", !hasData);
  if (hasData) download.href = `/api/documents/${state.current}/dataset.xlsx`;
  if (toggleOwner !== state.current) {
    toggleOwner = state.current;
    showAllFields = false;
  }

  slot("results-chips").innerHTML = "";
  slot("results-head").innerHTML = "";
  slot("results-rows").innerHTML = "";
  slot("results-empty").classList.toggle("hidden", hasData);
  if (!hasData) return;

  const chosen = chosenFields(data.fields);
  const fields = visibleFields(chosen, [data.paper_row, ...(data.sample_rows ?? [])], showAllFields);
  slot("results-chips").append(
    toggleChip(chosen, showAllFields, () => { showAllFields = !showAllFields; renderResults(root); }),
    fieldPicker(data.fields, () => renderResults(root)),
  );
  slot("results-head").append(headRow(LEADING, fields));
  const quality = qualityIndex(data.quality_rows ?? []);
  const paperSampleId = data.paper_row?.sample_id ?? "";
  const rows = slot("results-rows");
  rows.append(targetRow(data, fields, quality));
  for (const row of data.sample_rows ?? []) rows.append(sampleRow(row, fields, quality, paperSampleId));

  // The clipboard copy is built from the same `fields` and rows the renderer just used, so what lands in
  // the spreadsheet is exactly what is on screen -- and never the badges or tooltips wrapped around it.
  const values = [targetValues(data, fields), ...(data.sample_rows ?? []).map((row) => sampleValues(row, fields))];
  copy.onclick = () => copyTable(tsvHeader(LEADING, fields), values);
}


function qualityIndex(rows) {
  const index = new Map();
  for (const row of rows) index.set(`${row.sample_id}${KEY_SEPARATOR}${row.field}`, row);
  return index;
}

// `leading` are the identity columns each table brings of its own; the field columns are identical.
export function headRow(leading, fields) {
  const tr = document.createElement("tr");
  const cells = leading.map((label) => `<th>${escapeHtml(label)}</th>`);
  for (const field of fields) {
    // With a Chinese label the header reads label over the id it exports under. The unit is not repeated
    // here: every decided cell carries it next to its value, and saying it twice only adds noise.
    const title = field.label || field.name;
    const second = field.label ? field.name : "";
    const sub = second ? `<small>${escapeHtml(second)}</small>` : "";
    cells.push(`<th class="fcol" title="${escapeHtml(field.description ?? "")}">${escapeHtml(title)}${sub}</th>`);
  }
  tr.innerHTML = cells.join("");
  return tr;
}

function targetRow(data, fields, quality) {
  const tr = document.createElement("tr");
  tr.className = "target-row";
  const paper = data.paper_row ?? {};
  const cells = fields.map((field) =>
    field.scope === "target" ? cell(paper[field.name], quality, TARGET_ROW_ID, field) : "<td></td>",
  );
  tr.innerHTML = `<td class="mono">${escapeHtml(TARGET_LABEL)}</td><td></td><td></td><td></td>${cells.join("")}`;
  bindCells(tr);
  return tr;
}

function sampleRow(row, fields, quality, paperSampleId) {
  const tr = document.createElement("tr");
  const isPaperRow = row.sample_id === paperSampleId;
  if (isPaperRow) tr.className = "paper-row";
  const marker = isPaperRow ? `<span class="paper-mark" title="被选作论文行的样品">★ 论文行</span>` : "";
  const conditions = String(row.conditions ?? "");
  // Target values are identical on every sample, so they stay on the target row alone.
  const cells = fields.map((field) =>
    field.scope === "target" ? "<td></td>" : cell(row[field.name], quality, row.sample_id, field),
  );
  tr.innerHTML =
    `<td class="mono">${escapeHtml(row.sample_id ?? "")}${marker}</td>` +
    `<td class="label" title="${escapeHtml(row.sample_label ?? "")}">${escapeHtml(row.sample_label ?? "")}</td>` +
    `<td class="muted cond" title="${escapeHtml(conditions)}">${escapeHtml(conditions)}</td>` +
    `<td class="mono">${escapeHtml(row.available_fields ?? 0)} / ${escapeHtml(row.agree_fields ?? 0)}</td>` +
    cells.join("");
  bindCells(tr);
  return tr;
}

// How a committed value is written out: numbers through `fmt`, everything else as its own text. The
// field is optional so the plain text is still available on its own (the clipboard copy wants it bare).
const shownValue = (value) => (typeof value === "number" ? fmt(value) : String(value));

// A value as the reader sees it in a cell: the number and the field's canonical unit, e.g. `125 nm`.
// Text fields have no unit, and a unitless number stays a bare number.
export function valueHtml(value, field) {
  const shown = escapeHtml(shownValue(value));
  const unit = typeof value === "number" && field?.unit ? field.unit : "";
  return unit ? `${shown} <span class="unit">${escapeHtml(unit)}</span>` : shown;
}

function cell(value, quality, sampleId, field) {
  const decision = quality.get(`${sampleId}${KEY_SEPARATOR}${field.name}`);
  const detail = decision?.detail ?? "";
  // An empty cell is a refusal with a reason, not a gap: the reason is one hover away.
  // Refused, not missing: the reason is the cell's accessible name (so it does not need a hover) and the
  // cell is focusable, because clicking it jumps to the two lanes' records for this (sample, field).
  if (value == null) {
    const reason = detail || "流水线没有给出取值";
    return (
      `<td class="cell empty" tabindex="0" title="${escapeHtml(detail)}" aria-label="${escapeHtml(reason)}"` +
      ` data-field="${escapeHtml(field.name)}" data-sample="${escapeHtml(sampleId === TARGET_ROW_ID ? TARGET_SID : sampleId)}">—</td>`
    );
  }
  const status = decision?.decision ?? "";
  const badge = cellBadge(decision);
  const sources = decision?.source_ids ?? "";
  // The accepted evidence may have been stated for the whole sample series rather than for this sample.
  // That belongs in the tooltip, not in a badge: next to a lane name it read as "MinerU 全系列".
  const hint = decision?.series ? `${detail}${detail ? "；" : ""}论文对整个系列只写了一次` : detail;
  return (
    `<td class="cell ${CELL_CLASS[status] ?? ""}" title="${escapeHtml(hint)}" data-sources="${escapeHtml(sources)}">` +
    `${valueHtml(value, field)}<small class="${badge.cls}">${escapeHtml(badge.text)}</small></td>`
  );
}

// Clicking a value shows the blocks it was merged from, the same gesture the comparison table uses.
function bindCells(tr) {
  for (const td of tr.querySelectorAll("td.cell[data-sources]")) {
    td.addEventListener("click", () => {
      for (const other of td.closest("tbody").querySelectorAll("td.selected")) other.classList.remove("selected");
      clearEvidence(samplesHost(td));
      td.classList.add("selected");
      state.viewer?.highlight(td.dataset.sources.split("; ").filter(Boolean));
    });
  }
  for (const td of tr.querySelectorAll("td.cell.empty[data-field]")) {
    const jump = () => showEvidence(samplesHost(td), td.dataset.field, td.dataset.sample);
    td.addEventListener("click", jump);
    td.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      jump();
    });
  }
}

// The records section of the document this table belongs to, not of whichever document rendered first.
const samplesHost = (td) =>
  td.closest(".document")?.querySelector("details.samples") ?? document.querySelector("details.samples");

// ---------- clipboard rows ----------
//
// The values behind the rendered rows, in the same column order: what a spreadsheet should receive once
// the badges, markers and tooltips are stripped away.

// Target values live on the target row alone, exactly as the rendered table places them.
function targetValues(data, fields) {
  const paper = data.paper_row ?? {};
  return tsvRow(LEADING, [TARGET_LABEL], fields, (field) => (field.scope === "target" ? paper[field.name] : null));
}

function sampleValues(row, fields) {
  const identity = [
    row.sample_id ?? "",
    row.sample_label ?? "",
    row.conditions ?? "",
    `${row.available_fields ?? 0} / ${row.agree_fields ?? 0}`,
  ];
  return tsvRow(LEADING, identity, fields, (field) => (field.scope === "target" ? null : row[field.name]));
}
