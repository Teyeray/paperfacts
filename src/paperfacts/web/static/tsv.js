// The clipboard copy of a results table, as tab-separated text.
//
// Built from the table's own column list (table.js), never from the DOM: the cells carry badges, markers
// and tooltips that must not reach a spreadsheet. Each column gives its header and the raw value of a row,
// so the header and every line come from the same list and cannot disagree about how many columns there are.

import { fmt, toast } from "./html.js";

// A pasted cell starting with one of these is read by a spreadsheet as a formula, and a paper's text is not
// ours to have evaluated. A number ("-3.5") is still pasted as a number: only other text gets the quote.
const FORMULA_START = /^[=+\-@]/;
const isNumber = (text) => text.trim() !== "" && Number.isFinite(Number(text));

// Tabs and newlines inside a value would invent columns and rows, so they collapse to a space.
const tsvCell = (value) => {
  if (value == null) return "";
  if (typeof value === "number") return fmt(value);
  const text = String(value).replace(/[\t\r\n]+/g, " ");
  return FORMULA_START.test(text) && !isNumber(text) ? `'${text}` : text;
};

// An interval cell, [low, high] with null for an open end: "2.8–4.3", "≥ 80", "≤ 5". Its ends are written in full,
// exactly as the workbook writes them (kinds.interval_text), never cut to a number of significant digits.
export const intervalText = ([low, high]) =>
  high == null ? `≥ ${String(low)}` : low == null ? `≤ ${String(high)}` : `${String(low)}–${String(high)}`;

// What the code read out of one lane value's quote, shown beside it in the evidence views: the canonical number, a
// yes/no, the ISO date, an interval's ends or the sample a reference names -- or why nothing was read.
export function readingText(field) {
  const unit = field.unit ?? "";
  if (field.value != null) return `= ${fmt(field.value)} ${unit}`.trim();
  if (field.holds != null) return `= ${field.holds ? "是" : "否"}`;
  if (field.iso_date != null) return `= ${field.iso_date}`;
  if (field.bounds != null) return `= ${intervalText(field.bounds)} ${unit}`.trim();
  if (field.ref_id != null) return `→ ${field.ref_id}`;
  return field.normalization_note ? `(${field.normalization_note})` : "";
}

// A field's value as the clipboard gets it, decided by its column (a dataset field: `kind`, `cardinality`): the
// values of a `many` column joined with "; ", an interval as its two ends, a boolean as true/false, every other
// value as it is.
export function fieldText(value, field) {
  if (field?.cardinality === "many" && Array.isArray(value)) {
    return value.map((item) => (typeof item === "number" ? fmt(item) : String(item ?? ""))).join("; ");
  }
  if (field?.kind === "interval" && Array.isArray(value)) return intervalText(value);
  if (typeof value === "boolean") return String(value);
  return value;
}

export function buildTsv(columns, items) {
  const lines = [columns.map((column) => column.header), ...items.map((item) => columns.map((column) => column.text(item)))];
  return lines.map((line) => line.map(tsvCell).join("\t")).join("\n");
}

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

export async function copyTable(columns, items) {
  const text = buildTsv(columns, items);
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    if (!legacyCopy(text)) {
      toast("复制失败，请手动选择表格复制", true);
      return;
    }
  }
  toast(`已复制 ${items.length} 行`);
}

export function copyButton(onCopy) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "download copy-table";
  button.textContent = "复制表格";
  button.addEventListener("click", onCopy);
  return button;
}
