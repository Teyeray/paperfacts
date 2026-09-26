// Home view: the whole library as one table. Each row is a paper's selected sample row (the same
// `paper_row` the Excel export puts on the 论文数据 sheet), so the mined result for the corpus is
// visible without opening a single document. A paper with several samples can be expanded in place to
// show every sample row beneath it; the one the paper row was chosen from is marked.
//
// There are no quality rows here -- those are per-document -- so a cell is just its value, and an
// empty cell is "this paper has no committed value for this field".

import { api } from "./api.js";
import { escapeHtml, keepFocus, toast } from "./html.js";
import { documentHash } from "./router.js";
import { entityGroups, inEntity, state } from "./state.js";
import { chosenFields, fieldPicker, toggleChip, visibleFields } from "./fieldpicker.js";
import { bodyRow, column, fieldColumn, headRow, plainCell } from "./table.js";
import { copyButton, copyTable } from "./tsv.js";

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

// With several entity types the corpus table shows the primary entity only -- its rows and fields beside the
// paper-level ones -- an explicit limit of this view (the paper row is always one of its rows); every entity is in
// each document's own page and in the workbook.
function primaryOnly(data) {
  const groups = entityGroups();
  if (groups.length < 2) return { data, note: "" };
  const [primary] = groups;
  return {
    data: {
      ...data,
      fields: (data?.fields ?? []).filter((field) => field.scope !== "sample" || inEntity(primary, field)),
      rows: (data?.rows ?? []).map((row) => {
        const samples = (row.sample_rows ?? []).filter((sample) => inEntity(primary, sample));
        return { ...row, sample_rows: samples, sample_count: samples.length };
      }),
    },
    note: `这里只列出${primary.label}；${groups.slice(1).map((group) => group.label).join("、")}见各论文页面或 Excel。`,
  };
}

// What this table calls a row: the profile's entity, or with several entity types the primary one.
const primaryLabel = () => entityGroups()[0].label;

export function renderCorpus(root) {
  root.innerHTML = "";
  const { data, note: entityNote } = primaryOnly(state.corpus);
  const rows = data?.rows ?? [];
  if (!rows.length) return;

  // Columns are decided over every sample, so a field that only an expanded sample has still gets one.
  const allRows = rows.flatMap((row) => [row.paper_row ?? {}, ...(row.sample_rows ?? [])]);
  const chosen = chosenFields(data.fields);
  const fields = visibleFields(chosen, allRows, showAllFields);
  const expandable = rows.filter((row) => (row.sample_rows ?? []).length > 1);
  const rerender = () => keepFocus(root, () => renderCorpus(root));
  const columns = corpusColumns(fields);
  // What the table shows, in order: every paper row, plus the sample rows of the papers expanded. The
  // rendered rows and the clipboard copy are both built from this one list.
  const items = rows.flatMap((row) => [
    { kind: "paper", row, source: row.paper_row ?? {} },
    ...(expanded.has(row.document_id) ? (row.sample_rows ?? []).map((sample) => ({ kind: "sample", row, source: sample })) : []),
  ]);

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
  head.append(copyButton(() => copyTable(columns, items)));
  const download = document.createElement("a");
  download.className = "download";
  download.href = "/api/dataset.xlsx";
  // Empty: the server's Content-Disposition names the file after the profile.
  download.setAttribute("download", "");
  download.textContent = "下载全部 Excel";
  head.append(download);

  const note = document.createElement("p");
  note.className = "results-note muted";
  const entity = primaryLabel();
  note.textContent =
    `${rows.length} 篇论文各取一个完整${entity}行，点${entity}数可展开该论文的全部${entity}；` +
    "空白单元格是流水线拒绝猜测的取值，不是 0。" + entityNote;

  const table = document.createElement("table");
  table.className = "facts-table results-table";
  const thead = document.createElement("thead");
  thead.append(headRow(columns));
  const tbody = document.createElement("tbody");
  for (const item of items) {
    const open = item.kind === "paper" && expanded.has(item.row.document_id);
    tbody.append(bodyRow(columns, item, item.kind === "sample" ? "sample-row" : open ? "expanded" : ""));
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

// A paper row links to its document and carries the sample count (a button when there is more than one
// sample to expand into); a sample row is indented under it and marks the sample the paper row came from.
// The clipboard gets the paper's name on every line, so a pasted sample row still says whose it is.
function corpusColumns(fields) {
  const isPaper = (item) => item.kind === "paper";
  const name = (item) => item.row.name ?? item.row.document_id ?? "";
  return [
    column(
      "论文",
      (item) => {
        if (!isPaper(item)) {
          const label = escapeHtml(item.source.sample_label ?? "");
          return `<td class="label indent" title="${label}">${label || "&nbsp;"}</td>`;
        }
        const text = escapeHtml(name(item));
        return `<td class="label" title="${text}"><a href="${escapeHtml(documentHash(item.row.document_id))}">${text}</a></td>`;
      },
      name,
    ),
    column(
      primaryLabel(),
      (item) => `<td class="mono">${escapeHtml(item.source.sample_id ?? "")}${isPaper(item) ? sampleCount(item.row) : chosenMark(item)}</td>`,
      (item) => item.source.sample_id ?? "",
    ),
    column(
      "可用/一致",
      (item) => `<td class="mono">${escapeHtml(`${item.source.available_fields ?? 0} / ${item.source.agree_fields ?? 0}`)}</td>`,
      (item) => `${item.source.available_fields ?? 0} / ${item.source.agree_fields ?? 0}`,
    ),
    ...fields.map((field) => fieldColumn(field, (item) => item.source[field.name] ?? null, (_, value) => plainCell(value, field))),
  ];
}

// Only a paper with more than one sample has anything to expand into.
function sampleCount(row) {
  const count = (row.sample_rows ?? []).length || row.sample_count || 0;
  const entity = escapeHtml(primaryLabel());
  if (count <= 1) return `<small>${escapeHtml(count)} 个${entity}</small>`;
  const open = expanded.has(row.document_id);
  const id = escapeHtml(row.document_id);
  return (
    `<button type="button" class="expand" data-doc="${id}" data-focus="expand:${id}" aria-expanded="${open}"` +
    ` title="${open ? "收起" : "展开"}全部${entity}">${open ? "▾" : "▸"} ${escapeHtml(count)} 个${entity}</button>`
  );
}

const chosenMark = (item) =>
  item.source.sample_id === item.row.paper_row?.sample_id ? `<small class="chosen">论文行取自此${escapeHtml(primaryLabel())}</small>` : "";

// One chip that expands every expandable paper, or collapses them all once they are all open.
function expandAllChip(expandable, rerender) {
  const allOpen = expandable.every((row) => expanded.has(row.document_id));
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = `chip${allOpen ? " on" : ""}`;
  chip.dataset.focus = "expand-all";
  chip.textContent = `${allOpen ? "收起" : "展开"}全部${primaryLabel()}`;
  chip.addEventListener("click", () => {
    for (const row of expandable) {
      if (allOpen) expanded.delete(row.document_id);
      else expanded.add(row.document_id);
    }
    rerender();
  });
  return chip;
}
