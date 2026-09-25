// Results table: the consolidated dataset (one row per sample, one column per field) that the pipeline
// exports. This is the deliverable; the comparison workbench below it explains how each cell got there.
//
// Both results tables -- this one and the corpus table on the home view (corpus.js) -- are a list of
// columns. A column is `{ header, head, html(item), text(item) }`: its clipboard header, its <th>, its <td>
// for one row, and the raw value the clipboard gets for that row. The rendered table and the copy are both
// `columns.map(...)` over the same list, so they cannot disagree about which column holds what.

import { chosenFields, fieldPicker, toggleChip, visibleFields } from "./fieldpicker.js";
import { escapeHtml, fmt, keepFocus, onActivate } from "./html.js";
import { releaseFact } from "./facts.js";
import { clearEvidence, showEvidence } from "./samples.js";
import { LANE_LABEL, noSamplesReason, state } from "./state.js";
import { copyTable } from "./tsv.js";
import { revealViewer } from "./viewer.js";

const TARGET_LABEL = "靶材（论文级）";
// A cell is worth showing only when the pipeline committed to a value. `agree` and `single_source` are the
// two decisions that produce one; every other decision deliberately leaves the cell empty.
const CELL_CLASS = { agree: "ok", single_source: "warn" };
// A decided cell says how it was decided: 双路 when both lanes agreed, otherwise the name of the lane
// the value actually came from -- "单路" alone left the reader guessing which one.
const LANE_CLASS = { mineru: "lane-a", paddleocr_vl: "lane-b" };
function cellBadge(decision) {
  const status = decision?.decision ?? "";
  if (status === "agree") return { text: "双路", cls: "" };
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

// ---------- columns, shared with the corpus table ----------

// An identity column: a plain header, and its cell and clipboard value per row.
export const column = (header, html, text) => ({ header, head: `<th>${escapeHtml(header)}</th>`, html, text });

// A field column. `value(item)` is the committed value this row shows in it (or null), and `html(item, value)`
// wraps it in a <td>; the clipboard gets `value(item)` alone.
export function fieldColumn(field, value, html) {
  // With a Chinese label the header reads label over the id it exports under. The unit is not repeated on
  // screen: every decided cell carries it next to its value. The clipboard header does carry it, so the
  // numbers stay readable once they leave the page.
  const title = field.label || field.name;
  const sub = field.label ? `<small>${escapeHtml(field.name)}</small>` : "";
  return {
    header: field.unit ? `${title} (${field.unit})` : title,
    head: `<th class="fcol" title="${escapeHtml(field.description ?? "")}">${escapeHtml(title)}${sub}</th>`,
    html: (item) => html(item, value(item)),
    text: value,
  };
}

export function headRow(columns) {
  const tr = document.createElement("tr");
  tr.innerHTML = columns.map((c) => c.head).join("");
  return tr;
}

export function bodyRow(columns, item, className = "") {
  const tr = document.createElement("tr");
  if (className) tr.className = className;
  tr.innerHTML = columns.map((c) => c.html(item)).join("");
  return tr;
}

// How a committed value is written out: numbers through `fmt`, everything else as its own text.
const shownValue = (value) => (typeof value === "number" ? fmt(value) : String(value));

// A value as the reader sees it in a cell: the number and the field's canonical unit, e.g. `125 nm`.
// Text fields have no unit, and a unitless number stays a bare number.
export function valueHtml(value, field) {
  const shown = escapeHtml(shownValue(value));
  const unit = typeof value === "number" && field?.unit ? field.unit : "";
  return unit ? `${shown} <span class="unit">${escapeHtml(unit)}</span>` : shown;
}

// A cell with no provenance behind it (the corpus table has no quality rows): the value, or a dash.
export const plainCell = (value, field) =>
  value == null ? `<td class="cell empty">—</td>` : `<td class="cell">${valueHtml(value, field)}</td>`;

// ---------- the per-document table ----------

// Which document the toggle below belongs to: opening another paper starts from the default view again.
let showAllFields = false;
let toggleOwner = null;

export function renderResults(root) {
  const data = state.dataset;
  const slot = (name) => root.querySelector(`[data-slot="${name}"]`);
  const download = slot("dataset-download");
  const copy = slot("dataset-copy");
  const empty = slot("results-empty");
  if (toggleOwner !== state.current) {
    toggleOwner = state.current;
    showAllFields = false;
  }
  slot("results-chips").innerHTML = "";
  slot("results-head").innerHTML = "";
  slot("results-rows").innerHTML = "";

  const samples = data?.sample_rows ?? [];
  const paper = data?.paper_row ?? {};
  const targetHasValue = (data?.fields ?? []).some((field) => field.scope === "target" && paper[field.name] != null);
  const hasRows = Boolean(data) && (samples.length > 0 || targetHasValue);
  download.classList.toggle("hidden", !hasRows);
  copy.classList.toggle("hidden", !hasRows);
  slot("results-table").classList.toggle("hidden", !hasRows);
  // A processed paper without samples says why, beside whatever paper-level values it still has.
  empty.textContent = data ? noSamplesReason() : "还没有结果表，处理完成后会出现在这里。";
  empty.classList.toggle("hidden", Boolean(data) && samples.length > 0);
  if (!hasRows) return;
  download.href = `/api/documents/${state.current}/dataset.xlsx`;

  const rerender = () => keepFocus(root, () => renderResults(root));
  const chosen = chosenFields(data.fields);
  const fields = visibleFields(chosen, [paper, ...samples], showAllFields);
  slot("results-chips").append(
    toggleChip(chosen, showAllFields, () => { showAllFields = !showAllFields; rerender(); }),
    fieldPicker(data.fields, rerender),
  );
  const columns = documentColumns(fields, qualityIndex(data.quality_rows ?? []), paper.sample_id ?? "");
  const items = [{ kind: "target", row: paper }, ...samples.map((row) => ({ kind: "sample", row }))];
  slot("results-head").append(headRow(columns));
  const rows = slot("results-rows");
  for (const item of items) {
    const tr = bodyRow(columns, item, rowClass(item, paper.sample_id));
    bindCells(tr);
    rows.append(tr);
  }
  copy.onclick = () => copyTable(columns, items);
}

function rowClass(item, paperSampleId) {
  if (item.kind === "target") return "target-row";
  return item.row.sample_id === paperSampleId ? "paper-row" : "";
}

// Target values are the same for every sample, so they sit on the target row alone, and sample values on
// the sample rows alone: a field column is filled only on the rows of its own scope.
function documentColumns(fields, quality, paperSampleId) {
  const isTarget = (item) => item.kind === "target";
  // An identity column the target row leaves blank.
  const sampleColumn = (header, className, text) =>
    column(
      header,
      (item) => (isTarget(item) ? "<td></td>" : `<td class="${className}" title="${escapeHtml(text(item))}">${escapeHtml(text(item))}</td>`),
      (item) => (isTarget(item) ? "" : text(item)),
    );
  const paperMark = `<span class="paper-mark" title="被选作论文行的样品">★ 论文行</span>`;
  return [
    column(
      "样品",
      (item) => isTarget(item)
        ? `<td class="mono">${escapeHtml(TARGET_LABEL)}</td>`
        : `<td class="mono">${escapeHtml(item.row.sample_id ?? "")}${item.row.sample_id === paperSampleId ? paperMark : ""}</td>`,
      (item) => (isTarget(item) ? TARGET_LABEL : item.row.sample_id ?? ""),
    ),
    sampleColumn("标签", "label", (item) => String(item.row.sample_label ?? "")),
    sampleColumn("条件", "muted cond", (item) => String(item.row.conditions ?? "")),
    sampleColumn("可用/一致", "mono", (item) => `${item.row.available_fields ?? 0} / ${item.row.agree_fields ?? 0}`),
    ...fields.map((field) => {
      const ownScope = (item) => (field.scope === "target") === isTarget(item);
      return fieldColumn(
        field,
        (item) => (ownScope(item) ? item.row[field.name] ?? null : null),
        (item, value) => (ownScope(item) ? cell(value, quality, item, field) : "<td></td>"),
      );
    }),
  ];
}

function qualityIndex(rows) {
  const index = new Map();
  for (const row of rows) index.set(`${row.sample_id}${KEY_SEPARATOR}${row.field}`, row);
  return index;
}

function cell(value, quality, item, field) {
  const sampleId = item.kind === "target" ? "target" : item.row.sample_id;
  const decision = quality.get(`${sampleId}${KEY_SEPARATOR}${field.name}`);
  const detail = decision?.detail ?? "";
  // Refused, not missing: the reason is the cell's accessible name (so it does not need a hover) and the
  // cell is focusable, because clicking it jumps to the two lanes' records for this (sample, field).
  if (value == null) {
    const reason = detail || "流水线没有给出取值";
    return (
      `<td class="cell empty" tabindex="0" title="${escapeHtml(detail)}" aria-label="${escapeHtml(reason)}"` +
      ` data-field="${escapeHtml(field.name)}" data-kind="${item.kind}" data-sample="${escapeHtml(item.row.sample_id ?? "")}">—</td>`
    );
  }
  const status = decision?.decision ?? "";
  const badge = cellBadge(decision);
  const sources = decision?.source_ids ?? "";
  // The accepted evidence may have been stated for the whole sample series rather than for this sample.
  // That belongs in the tooltip, not in a badge: next to a lane name it read as "MinerU 全系列".
  const hint = decision?.series ? `${detail}${detail ? "；" : ""}论文对整个系列只写了一次` : detail;
  return (
    `<td class="cell ${CELL_CLASS[status] ?? ""}" tabindex="0" title="${escapeHtml(hint)}" data-sources="${escapeHtml(sources)}">` +
    `${valueHtml(value, field)}<small class="${badge.cls}">${escapeHtml(badge.text)}</small></td>`
  );
}

// Clicking (or pressing Enter on) a value shows the blocks it was merged from, the same gesture the
// comparison table uses; an empty cell jumps to the records that explain the refusal.
function bindCells(tr) {
  for (const td of tr.querySelectorAll("td.cell[data-sources]")) {
    onActivate(td, () => {
      for (const other of td.closest("tbody").querySelectorAll("td.selected")) other.classList.remove("selected");
      clearEvidence(samplesHost(td));
      td.classList.add("selected");
      releaseFact();
      state.viewer?.highlight(td.dataset.sources.split("; ").filter(Boolean));
      revealViewer();
    });
  }
  for (const td of tr.querySelectorAll("td.cell.empty[data-field]")) {
    onActivate(td, () => showEvidence(samplesHost(td), td.dataset.field, td.dataset.sample, td.dataset.kind));
  }
}

// The records section of the document this table belongs to, not of whichever document rendered first.
const samplesHost = (td) =>
  td.closest(".document")?.querySelector("details.samples") ?? document.querySelector("details.samples");
