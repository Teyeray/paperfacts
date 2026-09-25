// Home view: the whole library as one table. Each row is a paper's selected sample row (the same
// `paper_row` the Excel export puts on the 论文数据 sheet), so the mined result for the corpus is
// visible without opening a single document. A paper with several samples can be expanded in place to
// show every sample row beneath it; the one the paper row was chosen from is marked.
//
// There are no quality rows here -- those are per-document -- so a cell is just its value, and an
// empty cell is "this paper has no committed value for this field".

import { api } from "./api.js";
import { escapeHtml, toast } from "./html.js";
import { documentHash } from "./router.js";
import { state } from "./state.js";
import { chosenFields, fieldPicker, toggleChip, visibleFields } from "./fieldpicker.js";
import { headRow, valueHtml } from "./table.js";
import { copyButton, copyTable, tsvHeader, tsvRow } from "./tsv.js";

let showAllFields = false;
// Which papers are expanded. Kept across re-renders (a field toggle, a refresh) for as long as the page
// lives; a paper that left the library simply stops matching.
const expanded = new Set();

export async function loadCorpus() {
  try {
    state.corpus = await api("/api/dataset");
  } catch (error) {
    state.corpus = null;
    toast(`读取结果总表失败：${error.message}`, true);
  }
}

export function renderCorpus(root) {
  root.innerHTML = "";
  const data = state.corpus;
  const rows = data?.rows ?? [];
  if (!rows.length) return;

  // Columns are decided over every sample, so a field that only an expanded sample has still gets one.
  const allRows = rows.flatMap((row) => [row.paper_row ?? {}, ...(row.sample_rows ?? [])]);
  const chosen = chosenFields(data.fields);
  const fields = visibleFields(chosen, allRows, showAllFields);
  const expandable = rows.filter((row) => (row.sample_rows ?? []).length > 1);
  const rerender = () => renderCorpus(root);

  // No heading of its own: on the home view the page title above the table already names it.
  const head = document.createElement("div");
  head.className = "results-head";
  const chips = document.createElement("div");
  chips.className = "chips";
  chips.append(
    toggleChip(chosen, showAllFields, () => { showAllFields = !showAllFields; rerender(); }),
    fieldPicker(data.fields, rerender),
  );
  if (expandable.length) chips.append(expandAllChip(expandable, rerender));
  head.append(chips);
  const leading = ["论文", "样品", "可用/一致"];
  // The copy is what the table shows: every paper row, plus the sample rows of the papers expanded.
  head.append(copyButton(() => copyTable(tsvHeader(leading, fields), rows.flatMap((row) => copiedRows(row, fields, leading)))));
  const download = document.createElement("a");
  download.className = "download";
  download.href = "/api/dataset.xlsx";
  download.setAttribute("download", "");
  download.textContent = "下载全部 Excel";
  head.append(download);

  const note = document.createElement("p");
  note.className = "results-note muted";
  note.textContent =
    `${rows.length} 篇论文各取一个完整样品行，点样品数可展开该论文的全部样品；` +
    "空白单元格是流水线拒绝猜测的取值，不是 0。";

  const table = document.createElement("table");
  table.className = "facts-table results-table";
  const thead = document.createElement("thead");
  thead.append(headRow(leading, fields));
  const tbody = document.createElement("tbody");
  for (const row of rows) {
    tbody.append(paperRow(row, fields));
    if (expanded.has(row.document_id)) tbody.append(...sampleRows(row, fields));
  }
  tbody.addEventListener("click", (event) => {
    const button = event.target.closest("button.expand");
    if (!button) return;
    const id = button.dataset.doc;
    if (expanded.has(id)) expanded.delete(id);
    else expanded.add(id);
    rerender();
  });
  table.append(thead, tbody);

  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.append(table);

  const section = document.createElement("section");
  section.className = "results corpus";
  section.setAttribute("aria-label", "结果总表");
  section.append(head, note, wrap);
  root.append(section);
}

function paperRow(row, fields) {
  const tr = document.createElement("tr");
  const paper = row.paper_row ?? {};
  const name = escapeHtml(row.name ?? row.document_id);
  const sampleId = escapeHtml(paper.sample_id ?? "");
  const cells = fields.map((field) => valueCell(paper[field.name], field));
  const count = (row.sample_rows ?? []).length || row.sample_count || 0;
  const open = expanded.has(row.document_id);
  // Only a paper with more than one sample has anything to expand into.
  const countHtml =
    count > 1
      ? `<button type="button" class="expand" data-doc="${escapeHtml(row.document_id)}" aria-expanded="${open}"` +
        ` title="${open ? "收起" : "展开"}全部样品">${open ? "▾" : "▸"} ${escapeHtml(count)} 个样品</button>`
      : `<small>${escapeHtml(count)} 个样品</small>`;
  if (open) tr.classList.add("expanded");
  tr.innerHTML =
    `<td class="label" title="${name}"><a href="${escapeHtml(documentHash(row.document_id))}">${name}</a></td>` +
    `<td class="mono">${sampleId}${countHtml}</td>` +
    `<td class="mono">${escapeHtml(paper.available_fields ?? 0)} / ${escapeHtml(paper.agree_fields ?? 0)}</td>` +
    cells.join("");
  return tr;
}

// Every sample of one paper, beneath its paper row. The sample the paper row was chosen from is marked,
// since it repeats the paper row's values.
function sampleRows(row, fields) {
  const chosenId = row.paper_row?.sample_id ?? "";
  return (row.sample_rows ?? []).map((sample) => {
    const tr = document.createElement("tr");
    tr.className = "sample-row";
    const id = String(sample.sample_id ?? "");
    const label = escapeHtml(sample.sample_label ?? "");
    const mark = id === chosenId ? `<small class="chosen">论文行取自此样品</small>` : "";
    tr.innerHTML =
      `<td class="label indent" title="${label}">${label || "&nbsp;"}</td>` +
      `<td class="mono">${escapeHtml(id)}${mark}</td>` +
      `<td class="mono">${escapeHtml(sample.available_fields ?? 0)} / ${escapeHtml(sample.agree_fields ?? 0)}</td>` +
      fields.map((field) => valueCell(sample[field.name], field)).join("");
    return tr;
  });
}

// One chip that expands every expandable paper, or collapses them all once they are all open.
function expandAllChip(expandable, rerender) {
  const allOpen = expandable.every((row) => expanded.has(row.document_id));
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = `chip${allOpen ? " on" : ""}`;
  chip.textContent = allOpen ? "收起全部样品" : "展开全部样品";
  chip.addEventListener("click", () => {
    for (const row of expandable) {
      if (allOpen) expanded.delete(row.document_id);
      else expanded.add(row.document_id);
    }
    rerender();
  });
  return chip;
}

// The same three identity columns and the same field values the rendered rows show, without the link,
// the sample count or the empty-cell dash: the paper row, then its samples when it is expanded.
function copiedRows(row, fields, leading) {
  const paper = row.paper_row ?? {};
  const name = row.name ?? row.document_id ?? "";
  const line = (source, first) =>
    tsvRow(
      leading,
      [first, source.sample_id ?? "", `${source.available_fields ?? 0} / ${source.agree_fields ?? 0}`],
      fields,
      (field) => source[field.name],
    );
  const lines = [line(paper, name)];
  if (expanded.has(row.document_id)) lines.push(...(row.sample_rows ?? []).map((sample) => line(sample, name)));
  return lines;
}

// The value carries its own unit, as it does in the per-document table: the header no longer states it.
function valueCell(value, field) {
  if (value == null) return `<td class="cell empty">—</td>`;
  return `<td class="cell">${valueHtml(value, field)}</td>`;
}
