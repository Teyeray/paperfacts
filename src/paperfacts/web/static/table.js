// Results table: the consolidated dataset (one row per sample, one column per field) that the pipeline
// exports. This is the deliverable; the comparison workbench below it explains how each cell got there.
//
// Both results tables -- this one and the corpus table on the home view (corpus.js) -- are a list of
// columns. A column is `{ header, head, html(item), text(item) }`: its clipboard header, its <th>, its <td>
// for one row, and the raw value the clipboard gets for that row. The rendered table and the copy are both
// `columns.map(...)` over the same list, so they cannot disagree about which column holds what. A column the
// home table can sort by also carries `key` (its stable name: a field's name, or "paper" / "entity" / "counts") and
// `sort(item)`, the value it is ordered by; sorting reorders the items before both the rows and the copy are built.

import { profileHref } from "./api.js";
import { chosenFields, fieldPicker, toggleChip, visibleFields } from "./fieldpicker.js";
import { escapeHtml, fmt, keepFocus, onActivate, readStored, writeStored } from "./html.js";
import { releaseFact } from "./facts.js";
import { clearEvidence, showEvidence } from "./samples.js";
import { LANE_LABEL, entityGroups, entityLabel, entityOf, inEntity, noSamplesReason, state, uiCopy } from "./state.js";
import { copyButton, copyTable, fieldText, intervalText } from "./tsv.js";
import { revealViewer } from "./viewer.js";

// A cell is worth showing only when the pipeline committed to a value. `agree`, `single_source` and
// `supervised` (one lane's value, the other's ruled out by the supervisor) are the decisions that produce one;
// every other decision deliberately leaves the cell empty.
const CELL_CLASS = { agree: "ok", single_source: "warn", supervised: "warn" };
// A decided cell says how it was decided: 双路 when both lanes agreed, otherwise the name of the lane
// the value actually came from -- "单路" alone left the reader guessing which one. A supervised cell says so
// beside the lane, so it is never read as a plain single-source one.
const LANE_CLASS = { mineru: "lane-a", paddleocr_vl: "lane-b" };
function cellBadge(decision) {
  const status = decision?.decision ?? "";
  if (status === "agree") return { text: "双路", cls: "" };
  if (status !== "single_source" && status !== "supervised") return { text: "", cls: "" };
  const suffix = status === "supervised" ? "·监督" : "";
  const lanes = String(decision?.lanes ?? "").split(";").map((l) => l.trim()).filter(Boolean);
  if (!lanes.length) return { text: `单路${suffix}`, cls: "" };
  return {
    text: lanes.map((l) => LANE_LABEL[l] ?? l).join("+") + suffix,
    cls: lanes.length === 1 ? (LANE_CLASS[lanes[0]] ?? "") : "",
  };
}
// quality_rows are keyed by (entity, sample_id, field); a sample_id may itself contain "|", so join on a
// character that cannot occur in any part. A paper-level quality row names no entity.
const KEY_SEPARATOR = "\u0000";

// ---------- columns, shared with the corpus table ----------

// An identity column: a plain header, and its cell and clipboard value per row.
export const column = (header, html, text) => ({ header, head: `<th>${escapeHtml(header)}</th>`, html, text });

// A field column. `value(item)` is the committed value this row shows in it (or null), and `html(item, value)`
// wraps it in a <td>; the clipboard gets `value(item)` alone, written out as the column's kind says.
export function fieldColumn(field, value, html) {
  // With a Chinese label the header reads label over the id it exports under. The unit is not repeated on
  // screen: every decided cell carries it next to its value. The clipboard header does carry it, so the
  // numbers stay readable once they leave the page.
  const title = field.label || field.name;
  const sub = field.label ? `<small>${escapeHtml(field.name)}</small>` : "";
  return {
    header: field.unit ? `${title} (${field.unit})` : title,
    head: `<th class="fcol${numClass(field)}" title="${escapeHtml(field.description ?? "")}">${escapeHtml(title)}${sub}</th>`,
    html: (item) => html(item, value(item)),
    text: (item) => fieldText(value(item), field),
    key: field.name,
    sort: (item) => sortValue(value(item)),
  };
}

// Numbers and intervals line up on their digits: the column, never the value's shape, decides (a numeric column's
// refusal dash sits where its numbers do).
const NUMERIC_KINDS = new Set(["numeric", "interval"]);
const numClass = (field) => (NUMERIC_KINDS.has(field?.kind) ? " num" : "");

// ---------- sorting ----------

// What a value is ordered by: a number as itself (an interval by its first end, a list by its first element), a
// yes/no as 0/1, anything else as its text; no value at all is null, which every order puts last.
export function sortValue(value) {
  if (value == null || value === "") return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "boolean") return Number(value);
  if (Array.isArray(value)) return sortValue(value.find((item) => item != null) ?? null);
  return String(value);
}

const collator = new Intl.Collator("zh-CN", { numeric: true, sensitivity: "base" });
// Numbers before text, numbers by size, text by locale (with runs of digits read as numbers, so S2 < S10).
function compareValues(a, b) {
  if (typeof a === "number" && typeof b === "number") return a - b;
  if (typeof a === "number") return -1;
  if (typeof b === "number") return 1;
  return collator.compare(a, b);
}

// `items` ordered by `column.sort` in `dir` ("asc" | "desc"). Blanks go last in both directions, and ties (blanks
// included) keep the order the items came in, so a sort undone restores exactly what the server sent.
export function sortItems(items, column, dir) {
  if (!column?.sort) return items;
  const sign = dir === "desc" ? -1 : 1;
  return items
    .map((item, index) => ({ item, index, value: column.sort(item) }))
    .sort((a, b) => {
      if (a.value === null || b.value === null) return (a.value === null) - (b.value === null) || a.index - b.index;
      return sign * compareValues(a.value, b.value) || a.index - b.index;
    })
    .map((entry) => entry.item);
}

const SORT_LABEL = { asc: "↑ 升序", desc: "↓ 降序" };
const NEXT_DIR = { asc: "desc", desc: null };

// The next sort after a click on the header of `key`: another column starts ascending, the same one goes
// ascending → descending → off.
export const nextSort = (sort, key) => {
  if (sort?.key !== key) return { key, dir: "asc" };
  const dir = NEXT_DIR[sort.dir];
  return dir ? { key, dir } : null;
};

// Turns the header cells of the sortable columns in `tr` into buttons: `aria-sort` on the cell, the direction in
// words beside the title (never a colour alone), and `onSort(key)` on a click or Enter. Each button has a
// `data-focus` key, so the re-render a sort causes keeps the keyboard on it.
export function sortableHeads(tr, columns, sort, onSort) {
  columns.forEach((column, index) => {
    const th = tr.cells[index];
    if (!th || !column.key || !column.sort) return;
    const dir = sort?.key === column.key ? sort.dir : null;
    th.setAttribute("aria-sort", dir === "asc" ? "ascending" : dir === "desc" ? "descending" : "none");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sort-button";
    button.dataset.focus = `sort:${column.key}`;
    button.dataset.sort = column.key;
    button.title = "排序：升序 → 降序 → 取消";
    button.append(...th.childNodes);
    if (dir) {
      const mark = document.createElement("span");
      mark.className = "sort-mark";
      mark.textContent = SORT_LABEL[dir];
      button.insertBefore(mark, button.querySelector("small"));
    }
    button.addEventListener("click", () => onSort(column.key));
    th.append(button);
  });
}

// ---------- density ----------
//
// 标准 or 紧凑, one choice per browser for every results table: the `data-density` attribute on .shell, which the
// table's padding and frozen-column widths read (app.css). Storage may be refused; the page then starts at 标准.
const DENSITY_KEY = "paperfacts.density";
const DENSITIES = [
  { name: "standard", label: "标准" },
  { name: "compact", label: "紧凑" },
];

function setDensity(name) {
  const shell = document.querySelector(".shell");
  if (!shell) return;
  if (name === "compact") shell.dataset.density = "compact";
  else delete shell.dataset.density;
}

export function applyStoredDensity() {
  setDensity(readStored(DENSITY_KEY));
}

const currentDensity = () => (document.querySelector(".shell")?.dataset.density === "compact" ? "compact" : "standard");

// Two pressed-or-not buttons. A switch only flips the attribute: nothing is re-rendered, so the focus stays put.
export function densitySwitch() {
  const group = document.createElement("div");
  group.className = "segmented density-switch";
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "密度");
  const buttons = DENSITIES.map(({ name, label }) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.density = name;
    button.dataset.focus = `density:${name}`;
    button.textContent = label;
    button.addEventListener("click", () => {
      setDensity(name);
      writeStored(DENSITY_KEY, name);
      for (const other of buttons) other.setAttribute("aria-pressed", String(other === button));
    });
    button.setAttribute("aria-pressed", String(name === currentDensity()));
    return button;
  });
  group.append(...buttons);
  return group;
}

function headRow(columns) {
  const tr = document.createElement("tr");
  tr.innerHTML = columns.map((c) => c.head).join("");
  return tr;
}

function bodyRow(columns, item, className = "") {
  const tr = document.createElement("tr");
  if (className) tr.className = className;
  tr.innerHTML = columns.map((c) => c.html(item)).join("");
  return tr;
}

// A whole <table> from `columns` over `items`: the header row, then one body row per item, classed by `rowClass`.
export function resultsTable(columns, items, { rowClass = () => "", className = "results-table" } = {}) {
  const table = document.createElement("table");
  table.className = `facts-table ${className}`.trim();
  const thead = document.createElement("thead");
  thead.append(headRow(columns));
  const tbody = document.createElement("tbody");
  for (const item of items) tbody.append(bodyRow(columns, item, rowClass(item)));
  table.append(thead, tbody);
  return table;
}

// How a committed value is written out, decided by its column: the values of a `many` column joined with "；",
// an interval as its ends ("2.8–4.3", "≥ 80"), a boolean as 是/否, numbers through `fmt`, everything else as its
// own text.
const shownValue = (value, field) => {
  if (field?.cardinality === "many" && Array.isArray(value)) return value.map((item) => (item == null ? "" : shownValue(item))).join("；");
  if (field?.kind === "interval" && Array.isArray(value)) return intervalText(value);
  if (field?.kind === "boolean" && typeof value === "boolean") return value ? "是" : "否";
  return typeof value === "number" ? fmt(value) : String(value);
};

// A value as the reader sees it in a cell: the number (or an interval's ends) and the field's canonical unit,
// e.g. `125 nm`. Text fields have no unit, and a unitless number stays a bare number. A reference is the id of a row
// of another entity's table, followed by that entity's label where the unit would be.
function valueHtml(value, field) {
  const shown = escapeHtml(shownValue(value, field));
  const numeric = typeof value === "number" || (field?.kind === "interval" && Array.isArray(value));
  const unit = field?.references ? entityLabel(field.references) : numeric && field?.unit ? field.unit : "";
  return unit ? `${shown} <span class="unit">${escapeHtml(unit)}</span>` : shown;
}

// A cell with no provenance behind it (the corpus table has no quality rows): the value, or a dash.
export const plainCell = (value, field) =>
  value == null
    ? `<td class="cell empty${numClass(field)}">—</td>`
    : `<td class="cell${numClass(field)}">${valueHtml(value, field)}</td>`;

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
  slot("results-entities").innerHTML = "";
  slot("results-entity").classList.add("hidden");

  const samples = data?.sample_rows ?? [];
  const paper = data?.paper_row ?? {};
  const paperHasValue = (data?.fields ?? []).some((field) => field.scope === "paper" && paper[field.name] != null);
  const hasRows = Boolean(data) && (samples.length > 0 || paperHasValue);
  download.classList.toggle("hidden", !hasRows);
  copy.classList.toggle("hidden", !hasRows);
  slot("results-table").classList.toggle("hidden", !hasRows);
  // A processed paper without samples says why, beside whatever paper-level values it still has.
  empty.textContent = data ? noSamplesReason() : "还没有结果表，处理完成后会出现在这里。";
  empty.classList.toggle("hidden", Boolean(data) && samples.length > 0);
  if (!hasRows) return;
  download.href = profileHref(state.currentProfile, `/api/documents/${state.current}/dataset.xlsx`);

  const rerender = () => keepFocus(root, () => renderResults(root));
  const chosen = chosenFields(data.fields);
  slot("results-chips").append(
    toggleChip(chosen, showAllFields, () => { showAllFields = !showAllFields; rerender(); }),
    fieldPicker(data.fields, rerender),
  );
  const quality = qualityIndex(data.quality_rows ?? []);
  // One table per entity type. The first (the primary entity's) sits in the template's own table with the
  // paper-level row, as the only table does for a profile without entity types; each other entity's table follows
  // under its label, with its own fields and its own copy button.
  const [primary, ...others] = entityGroups();
  const ownFields = (group) =>
    chosen.filter((field) => (field.scope === "sample" ? inEntity(group, field) : group === primary));
  const primaryRows = samples.filter((row) => inEntity(primary, row));
  const fields = visibleFields(ownFields(primary), [paper, ...primaryRows], showAllFields);
  const columns = documentColumns(fields, quality, paper.sample_id ?? "", primary);
  const items = [{ kind: "paper", row: paper }, ...primaryRows.map((row) => ({ kind: "sample", row }))];
  slot("results-head").append(headRow(columns));
  const rows = slot("results-rows");
  for (const item of items) {
    const tr = bodyRow(columns, item, rowClass(item, paper.sample_id));
    bindCells(tr);
    rows.append(tr);
  }
  copy.onclick = () => copyTable(columns, items);
  if (!others.length) return;
  const heading = slot("results-entity");
  heading.textContent = primary.label;
  heading.classList.remove("hidden");
  for (const group of others) slot("results-entities").append(entityTable(group, samples, ownFields(group), quality));
}

// A non-primary entity's table: its rows under its label, with only its own fields (the paper-level values are on
// the primary table's paper-level row).
function entityTable(group, samples, fields, quality) {
  const rows = samples.filter((row) => inEntity(group, row));
  const shown = visibleFields(fields, rows, showAllFields);
  const columns = documentColumns(shown, quality, "", group);
  const items = rows.map((row) => ({ kind: "sample", row }));
  const box = document.createElement("div");
  box.className = "entity-table";
  box.dataset.entity = group.name;
  const head = document.createElement("div");
  head.className = "results-head";
  const title = document.createElement("h3");
  title.className = "entity-head";
  title.textContent = group.label;
  head.append(title, copyButton(() => copyTable(columns, items)));
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = resultsTable(columns, items);
  for (const tr of table.tBodies[0].rows) bindCells(tr);
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "table-empty";
    empty.textContent = `没有${group.label}。`;
    wrap.append(empty);
  }
  wrap.append(table);
  box.append(head, wrap);
  return box;
}

function rowClass(item, paperSampleId) {
  if (item.kind === "paper") return "paper-level-row";
  return item.row.sample_id === paperSampleId ? "paper-row" : "";
}

// Paper-level values are the same for every sample, so they sit on the paper-level row alone, and sample values on
// the sample rows alone: a field column is filled only on the rows of its own scope.
function documentColumns(fields, quality, paperSampleId, group) {
  const isPaperLevel = (item) => item.kind === "paper";
  // An identity column the paper-level row leaves blank.
  const sampleColumn = (header, className, text) =>
    column(
      header,
      (item) => (isPaperLevel(item) ? "<td></td>" : `<td class="${className}" title="${escapeHtml(text(item))}">${escapeHtml(text(item))}</td>`),
      (item) => (isPaperLevel(item) ? "" : text(item)),
    );
  const paperMark = `<span class="paper-mark" title="被选作论文行的${escapeHtml(uiCopy("entity_label_zh"))}">★ 论文行</span>`;
  const paperLabel = uiCopy("paper_level_label_zh");
  return [
    column(
      group.label,
      (item) => isPaperLevel(item)
        ? `<td class="mono">${escapeHtml(paperLabel)}</td>`
        : `<td class="mono">${escapeHtml(item.row.sample_id ?? "")}${item.row.sample_id === paperSampleId ? paperMark : ""}</td>`,
      (item) => (isPaperLevel(item) ? paperLabel : item.row.sample_id ?? ""),
    ),
    sampleColumn("标签", "label", (item) => String(item.row.sample_label ?? "")),
    sampleColumn("条件", "muted cond", (item) => String(item.row.conditions ?? "")),
    sampleColumn("可用/一致", "mono", (item) => `${item.row.available_fields ?? 0} / ${item.row.agree_fields ?? 0}`),
    ...fields.map((field) => {
      const ownScope = (item) => (field.scope === "paper") === isPaperLevel(item);
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
  for (const row of rows) index.set(qualityKey(row.entity ?? "", row.sample_id, row.field), row);
  return index;
}

const qualityKey = (entity, sampleId, field) => [entity, sampleId, field].join(KEY_SEPARATOR);

function cell(value, quality, item, field) {
  const paperLevel = item.kind === "paper";
  const decision = quality.get(
    qualityKey(paperLevel ? "" : item.row.entity ?? "", paperLevel ? "paper" : item.row.sample_id, field.name),
  );
  const detail = decision?.detail ?? "";
  // Refused, not missing: the reason is the cell's accessible name (so it does not need a hover) and the
  // cell is focusable, because clicking it jumps to the two lanes' records for this (sample, field).
  if (value == null) {
    const reason = detail || "流水线没有给出取值";
    return (
      `<td class="cell empty${numClass(field)}" tabindex="0" title="${escapeHtml(detail)}" aria-label="${escapeHtml(reason)}"` +
      ` data-field="${escapeHtml(field.name)}" data-kind="${item.kind}" data-sample="${escapeHtml(item.row.sample_id ?? "")}"` +
      ` data-entity="${escapeHtml(entityOf(item.row))}">—</td>`
    );
  }
  const status = decision?.decision ?? "";
  const badge = cellBadge(decision);
  const sources = decision?.source_ids ?? "";
  // The accepted evidence may have been stated for the whole sample series rather than for this sample.
  // That belongs in the tooltip, not in a badge: next to a lane name it read as "MinerU 全系列".
  const hint = decision?.series ? `${detail}${detail ? "；" : ""}论文对整个系列只写了一次` : detail;
  return (
    `<td class="cell ${CELL_CLASS[status] ?? ""}${numClass(field)}" tabindex="0" title="${escapeHtml(hint)}" data-sources="${escapeHtml(sources)}">` +
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
  // The records are narrowed to the cell's entity only where the page groups by entity: with one entity type every
  // record is its, and a row naming none must not filter out the records that do.
  const byEntity = entityGroups().length > 1;
  for (const td of tr.querySelectorAll("td.cell.empty[data-field]")) {
    onActivate(td, () =>
      showEvidence(samplesHost(td), td.dataset.field, td.dataset.sample, td.dataset.kind, byEntity ? td.dataset.entity : null),
    );
  }
}

// The records section of the document this table belongs to, not of whichever document rendered first.
const samplesHost = (td) =>
  td.closest(".document")?.querySelector("details.samples") ?? document.querySelector("details.samples");
