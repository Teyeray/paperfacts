// Home view: the whole library as one table. Each row is a paper's selected sample row (the same
// `paper_row` the Excel export puts on the 论文数据 sheet), so the mined result for the corpus is
// visible without opening a single document.
//
// There are no quality rows here -- those are per-document -- so a cell is just its value, and an
// empty cell is "this paper has no committed value for this field".

import { api } from "./api.js";
import { escapeHtml, toast } from "./html.js";
import { documentHash } from "./router.js";
import { state } from "./state.js";
import { copyButton, copyTable, headRow, shownValue, toggleChip, tsvHeader, visibleFields } from "./table.js";

let showAllFields = false;

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

  const paperRows = rows.map((row) => row.paper_row ?? {});
  const fields = visibleFields(data.fields, paperRows, showAllFields);

  const head = document.createElement("div");
  head.className = "results-head";
  head.innerHTML = `<h2>结果总表（按论文）</h2>`;
  const chips = document.createElement("div");
  chips.className = "chips";
  chips.append(toggleChip(data.fields, showAllFields, () => { showAllFields = !showAllFields; renderCorpus(root); }));
  head.append(chips);
  const leading = ["论文", "样品", "可用/一致"];
  head.append(copyButton(() => copyTable(tsvHeader(leading, fields), rows.map((row) => rowValues(row, fields)))));
  const download = document.createElement("a");
  download.className = "download";
  download.href = "/api/dataset.xlsx";
  download.setAttribute("download", "");
  download.textContent = "下载全部 Excel";
  head.append(download);

  const note = document.createElement("p");
  note.className = "results-note muted";
  note.textContent = `${rows.length} 篇论文各取一个完整样品行；空白单元格是流水线拒绝猜测的取值，不是 0。`;

  const table = document.createElement("table");
  table.className = "facts-table results-table";
  const thead = document.createElement("thead");
  thead.append(headRow(leading, fields));
  const tbody = document.createElement("tbody");
  for (const row of rows) tbody.append(paperRow(row, fields));
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
  const cells = fields.map((field) => valueCell(paper[field.name]));
  tr.innerHTML =
    `<td class="label" title="${name}"><a href="${escapeHtml(documentHash(row.document_id))}">${name}</a></td>` +
    `<td class="mono">${sampleId}<small>${escapeHtml(row.sample_count ?? 0)} 个样品</small></td>` +
    `<td class="mono">${escapeHtml(paper.available_fields ?? 0)} / ${escapeHtml(paper.agree_fields ?? 0)}</td>` +
    cells.join("");
  return tr;
}

// The same three identity columns and the same field values the rendered row shows, without the link,
// the sample count or the empty-cell dash.
function rowValues(row, fields) {
  const paper = row.paper_row ?? {};
  return [
    row.name ?? row.document_id ?? "",
    paper.sample_id ?? "",
    `${paper.available_fields ?? 0} / ${paper.agree_fields ?? 0}`,
    ...fields.map((field) => paper[field.name]),
  ];
}

function valueCell(value) {
  if (value == null) return `<td class="cell empty">—</td>`;
  return `<td class="cell">${escapeHtml(shownValue(value))}</td>`;
}
