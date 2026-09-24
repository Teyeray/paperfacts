// The clipboard copy of a results table, as tab-separated text.
//
// Built from rows and fields, never from the DOM: the cells carry badges, markers and tooltips that must
// not reach a spreadsheet. A row is a flat array of raw values, identity columns first, in column order.
//
// Header and rows are built by `tsvHeader` and `tsvRow` from the same `leading` and `fields`, kept side
// by side here so their lengths cannot drift apart -- a row one identity column short is exactly the bug
// this arrangement prevents, and `buildTsv` throws if one ever slips through anyway.

import { fmt, toast } from "./html.js";

// One row: the identity cells padded to exactly `leading.length`, then one cell per field. Callers give
// the identity cells they have; a table that grows an identity column does not have to grow the blanks.
export function tsvRow(leading, identity, fields, valueOf) {
  const cells = (identity ?? []).slice(0, leading.length);
  return [...cells, ...Array(leading.length - cells.length).fill(""), ...(fields ?? []).map(valueOf)];
}

// A field column is labelled `name (unit)` when the field has a unit, so the numbers stay readable once
// they leave the page that showed the unit in the header's second line. A field with a Chinese label is
// labelled with it instead, matching what the reader saw on screen.
export function tsvHeader(leading, fields) {
  return [
    ...leading,
    ...fields.map((field) => {
      const name = field.label || field.name;
      return field.unit ? `${name} (${field.unit})` : name;
    }),
  ];
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

// A mismatched row would silently shift every value after it into the wrong column, so it is a throw
// rather than a padded guess: the table that built it is wrong and a developer has to see that.
export function buildTsv(header, rows) {
  for (const [index, row] of rows.entries()) {
    if (row.length !== header.length) {
      throw new Error(`TSV row ${index} has ${row.length} cells, expected ${header.length}`);
    }
  }
  return [header, ...rows].map((row) => row.map(tsvCell).join("\t")).join("\n");
}

export async function copyTable(header, rows) {
  const text = buildTsv(header, rows);
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
