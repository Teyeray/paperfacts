// Results table: the consolidated dataset (one row per sample, one column per field) that the pipeline
// exports. This is the deliverable; the comparison workbench below it explains how each cell got there.

import { escapeHtml, fmt, toast } from "./html.js";
import { clearEvidence, showEvidence, TARGET_SID } from "./samples.js";
import { state } from "./state.js";

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

// A field earns a column when at least one row put a value in it; the toggle brings the rest back so the
// full table stays inspectable without making the default view mostly blank. Shared with the corpus table,
// which applies the same rule across papers instead of across samples.
export function visibleFields(fields, rows, showAll) {
  const all = fields ?? [];
  if (showAll) return all;
  const present = (rows ?? []).filter(Boolean);
  return all.filter((field) => present.some((row) => row[field.name] != null));
}

// ---------- field column picker ----------
//
// Which field columns the reader wants at all, independent of whether they happen to be empty. The
// choice is one per browser and shared by both results tables, so a column hidden on the home table
// stays hidden inside a document. Storage may be unavailable (private mode, blocked site data); every
// failure degrades to "show every field", never to an empty table.

const PICKER_KEY = "paperfacts.chosen-fields";
// Fields carry `scope`, not the config's richer `group`, so the picker groups by the distinction the
// dataset actually exposes: what belongs to the target/paper and what belongs to a sample.
const SCOPE_LABEL = { target: "论文级", sample: "样品级" };
const SCOPE_ORDER = ["target", "sample"];

// null means "no choice stored" -- every field is shown, including ones added after the last choice.
function readChosen() {
  try {
    const raw = window.localStorage.getItem(PICKER_KEY);
    if (!raw) return null;
    const names = JSON.parse(raw);
    return Array.isArray(names) ? new Set(names.map(String)) : null;
  } catch {
    return null;
  }
}

function writeChosen(names) {
  try {
    if (names === null) window.localStorage.removeItem(PICKER_KEY);
    else window.localStorage.setItem(PICKER_KEY, JSON.stringify([...names]));
  } catch {
    // A browser that refuses storage still gets a working picker for this page view.
  }
}

// The chosen subset of the field list, in the dataset's own column order.
export function chosenFields(fields) {
  const all = fields ?? [];
  const chosen = readChosen();
  if (chosen === null) return all;
  const kept = all.filter((field) => chosen.has(field.name));
  // A stored choice that matches nothing is a choice made before the fields were renamed, not a request
  // for an empty table; that request is an empty set, which is honoured. Fall back to showing everything.
  if (!kept.length && chosen.size) {
    writeChosen(null);
    return all;
  }
  return kept;
}

// Re-rendering the table rebuilds the chip, so the open/closed state lives outside the element.
let pickerOpen = false;

export function fieldPicker(fields, onChange) {
  const all = fields ?? [];
  const chosen = readChosen();
  const isOn = (field) => chosen === null || chosen.has(field.name);

  const wrap = document.createElement("div");
  wrap.className = "field-picker";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "chip" + (chosen === null ? "" : " on");
  button.setAttribute("aria-expanded", "false");
  button.setAttribute("aria-haspopup", "dialog");
  button.innerHTML = `选择字段<span class="n">${all.filter(isOn).length}/${all.length}</span>`;

  // Hiding every column is a legitimate request, but a table with only its identity columns looks broken.
  // Say why it is empty next to the control that caused it, so the way back is one click away.
  const note = document.createElement("span");
  note.className = "picker-note";
  note.hidden = all.length === 0 || all.some(isOn);
  note.textContent = "已隐藏全部字段列";

  const pop = document.createElement("div");
  pop.className = "picker-pop";
  pop.setAttribute("role", "dialog");
  pop.setAttribute("aria-label", "选择要显示的字段");
  pop.hidden = true;

  // Both links write an explicit set: "全选" stores every current name rather than clearing the key, so
  // the choice stays what the reader saw. Only unchecking everything leaves an empty table by request.
  const actions = document.createElement("div");
  actions.className = "picker-actions";
  actions.append(
    linkButton("全选", () => commit(new Set(all.map((field) => field.name)))),
    linkButton("清空", () => commit(new Set())),
  );
  pop.append(actions);

  for (const scope of SCOPE_ORDER) {
    const group = all.filter((field) => (field.scope ?? "sample") === scope);
    if (!group.length) continue;
    const box = document.createElement("div");
    box.className = "picker-group";
    const title = document.createElement("h4");
    title.textContent = SCOPE_LABEL[scope];
    box.append(title);
    for (const field of group) box.append(checkbox(field, isOn(field)));
    pop.append(box);
  }

  function checkbox(field, on) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = on;
    input.value = field.name;
    input.addEventListener("change", () => {
      const boxes = [...pop.querySelectorAll("input[type=checkbox]")];
      commit(new Set(boxes.filter((box) => box.checked).map((box) => box.value)));
    });
    const text = document.createElement("span");
    text.textContent = field.unit ? `${field.name}（${field.unit}）` : field.name;
    label.append(input, text);
    return label;
  }

  function commit(names) {
    writeChosen(names);
    // onChange discards this element; its document-level listeners would otherwise outlive it.
    detach();
    onChange();
  }

  const onKey = (event) => {
    if (event.key === "Escape") {
      close();
      button.focus();
    }
  };
  const onOutside = (event) => {
    if (!wrap.contains(event.target)) close();
  };

  function open(focusFirst) {
    pop.hidden = false;
    pickerOpen = true;
    button.setAttribute("aria-expanded", "true");
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onOutside);
    if (focusFirst) pop.querySelector("input[type=checkbox]")?.focus();
  }

  function detach() {
    document.removeEventListener("keydown", onKey);
    document.removeEventListener("pointerdown", onOutside);
  }

  function close() {
    pop.hidden = true;
    pickerOpen = false;
    button.setAttribute("aria-expanded", "false");
    detach();
  }

  button.addEventListener("click", () => (pop.hidden ? open(true) : close()));
  wrap.append(button, note, pop);
  // A checkbox re-renders the table, which replaces this chip; the popover stays where the reader left it.
  if (pickerOpen) open(false);
  return wrap;
}

function linkButton(text, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "picker-link";
  button.textContent = text;
  button.addEventListener("click", onClick);
  return button;
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

// ---------- clipboard: the visible table as tab-separated text ----------
//
// Built from rows and fields, never from the DOM: the cells carry badges, markers and tooltips that must
// not reach a spreadsheet. A row is a flat array of raw values, identity columns first, in column order.

// Target values live on the target row alone, exactly as the rendered table places them.
function targetValues(data, fields) {
  const paper = data.paper_row ?? {};
  const cells = fields.map((field) => (field.scope === "target" ? paper[field.name] : null));
  return [TARGET_LABEL, "", "", ...cells];
}

function sampleValues(row, fields) {
  const cells = fields.map((field) => (field.scope === "target" ? null : row[field.name]));
  return [
    row.sample_id ?? "",
    row.sample_label ?? "",
    row.conditions ?? "",
    `${row.available_fields ?? 0} / ${row.agree_fields ?? 0}`,
    ...cells,
  ];
}

// A field column is labelled `name (unit)` when the field has a unit, so the numbers stay readable once
// they leave the page that showed the unit in the header's second line.
export function tsvHeader(leading, fields) {
  return [...leading, ...fields.map((field) => (field.unit ? `${field.name} (${field.unit})` : field.name))];
}

// Tabs and newlines inside a value would invent columns and rows, so they collapse to a space.
const tsvCell = (value) => {
  if (value == null) return "";
  const text = typeof value === "number" ? fmt(value) : String(value);
  return text.replace(/[\t\r\n]+/g, " ");
};

// execCommand is deprecated but is the only copy path left in an insecure context (plain http on a lab
// machine), which is exactly where this server usually runs.
function legacyCopy(text) {
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.cssText = "position:fixed;top:0;left:-9999px;opacity:0";
  document.body.append(area);
  area.select();
  try {
    return document.execCommand("copy");
  } catch {
    return false;
  } finally {
    area.remove();
  }
}

export async function copyTable(header, rows) {
  const text = [header, ...rows].map((row) => row.map(tsvCell).join("\t")).join("\n");
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    if (!legacyCopy(text)) {
      toast("复制失败，请手动选择表格复制", true);
      return;
    }
  }
  toast(`已复制 ${rows.length} 行`);
}

export function copyButton(onCopy) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "download copy-table";
  button.textContent = "复制表格";
  button.addEventListener("click", onCopy);
  return button;
}
