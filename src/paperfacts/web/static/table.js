// Results table: the consolidated dataset (one row per sample, one column per field) that the pipeline
// exports. This is the deliverable; the comparison workbench below it explains how each cell got there.

import { escapeHtml, fmt } from "./html.js";
import { state } from "./state.js";

const TARGET_ROW_ID = "target";
const TARGET_LABEL = "靶材（论文级）";
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
  const hasData = Boolean(data && (data.sample_rows?.length || data.paper_row));
  download.classList.toggle("hidden", !hasData);
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

  const fields = visibleFields(data.fields, [data.paper_row, ...(data.sample_rows ?? [])], showAllFields);
  slot("results-chips").append(toggleChip(data.fields, showAllFields, () => { showAllFields = !showAllFields; renderResults(root); }));
  slot("results-head").append(headRow(["样品", "标签", "条件", "可用/一致"], fields));
  const quality = qualityIndex(data.quality_rows ?? []);
  const paperSampleId = data.paper_row?.sample_id ?? "";
  const rows = slot("results-rows");
  rows.append(targetRow(data, fields, quality));
  for (const row of data.sample_rows ?? []) rows.append(sampleRow(row, fields, quality, paperSampleId));
}

// A field earns a column when at least one row put a value in it; the toggle brings the rest back so the
// full table stays inspectable without making the default view mostly blank. Shared with the corpus table,
// which applies the same rule across papers instead of across samples.
export function visibleFields(fields, rows, showAll) {
  const all = fields ?? [];
  if (showAll) return all;
  const present = (rows ?? []).filter(Boolean);
  return all.filter((field) => present.some((row) => row[field.name] != null));
}

export function toggleChip(fields, showAll, onToggle) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "chip" + (showAll ? " on" : "");
  button.setAttribute("aria-pressed", String(showAll));
  button.innerHTML = `显示空字段<span class="n">${(fields ?? []).length}</span>`;
  button.addEventListener("click", onToggle);
  return button;
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
    const unit = field.unit ? `<small>${escapeHtml(field.unit)}</small>` : "";
    cells.push(`<th class="fcol">${escapeHtml(field.name)}${unit}</th>`);
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
  if (value == null) return `<td class="cell empty" title="${escapeHtml(detail)}">—</td>`;
  const status = decision?.decision ?? "";
  const shown = shownValue(value);
  const badge = CELL_BADGE[status] ?? "";
  const sources = decision?.source_ids ?? "";
  return (
    `<td class="cell ${CELL_CLASS[status] ?? ""}" title="${escapeHtml(detail)}" data-sources="${escapeHtml(sources)}">` +
    `${escapeHtml(shown)}<small>${escapeHtml(badge)}</small></td>`
  );
}

// Clicking a value shows the blocks it was merged from, the same gesture the comparison table uses.
function bindCells(tr) {
  for (const td of tr.querySelectorAll("td.cell[data-sources]")) {
    td.addEventListener("click", () => {
      for (const other of td.closest("tbody").querySelectorAll("td.selected")) other.classList.remove("selected");
      td.classList.add("selected");
      state.viewer?.highlight(td.dataset.sources.split("; ").filter(Boolean));
    });
  }
}
