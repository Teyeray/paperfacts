// Results table: the consolidated dataset (one row per sample, one column per field) that the pipeline
// exports. This is the deliverable; the comparison workbench below it explains how each cell got there.

import { chosenFields, fieldPicker, toggleChip, visibleFields } from "./fieldpicker.js";
import { escapeHtml, fmt } from "./html.js";
import { clearEvidence, showEvidence, TARGET_SID } from "./samples.js";
import { state } from "./state.js";
import { copyTable, tsvHeader, tsvRow } from "./tsv.js";

const TARGET_ROW_ID = "target";
const TARGET_LABEL = "靶材（论文级）";
// The identity columns of the per-sample results table, shared by its header and its clipboard copy.
const LEADING = ["样品", "标签", "条件", "可用/一致"];
// A cell is worth showing only when the pipeline committed to a value. `agree` and `single_source` are the
// two decisions that produce one; every other decision deliberately leaves the cell empty.
const CELL_CLASS = { agree: "ok", single_source: "warn" };
const CELL_BADGE = { agree: "双路", single_source: "单路" };
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
    // With a Chinese label the header reads label over `name unit`, so the column is recognisable at a
    // glance while the id it exports under stays visible. Without one it is the id over the unit, as before.
    const title = field.label || field.name;
    const second = [field.label ? field.name : "", field.unit ?? ""].filter(Boolean).join(" ");
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

// How a committed value is written out: numbers through `fmt`, everything else as its own text.
export const shownValue = (value) => (typeof value === "number" ? fmt(value) : String(value));

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
  const shown = shownValue(value);
  const badge = CELL_BADGE[status] ?? "";
  const sources = decision?.source_ids ?? "";
  // The accepted evidence was stated for the whole sample series, not for this sample on its own.
  const series = decision?.series
    ? `<small title="论文对整个样品系列只写了一次，这里是按系列写到该样品上的">系列</small>`
    : "";
  return (
    `<td class="cell ${CELL_CLASS[status] ?? ""}" title="${escapeHtml(detail)}" data-sources="${escapeHtml(sources)}">` +
    `${escapeHtml(shown)}<small>${escapeHtml(badge)}</small>${series}</td>`
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
