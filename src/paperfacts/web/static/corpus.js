// Home view: the whole library as one table. Each row is a paper's selected sample row (the same
// `paper_row` the Excel export puts on the 论文数据 sheet), so the mined result for the corpus is
// visible without opening a single document. A paper with several samples can be expanded in place to
// show every sample row beneath it; the one the paper row was chosen from is marked.
//
// There are no quality rows here -- those are per-document -- so a cell is just its value, and an
// empty cell is "this paper has no committed value for this field".
//
// With several entity types a chip per entity picks what a row is: the primary entity gives the table above (the
// paper row is always one of its rows); any other gives one row per sample of that entity across the papers, beside
// the paper it came from, with that entity's fields. The choice is kept per profile for as long as the page lives.

import { profileApi, profileHref } from "./api.js";
import { escapeHtml, keepFocus } from "./html.js";
import { documentHash } from "./router.js";
import { entityGroups, inEntity, state } from "./state.js";
import { chosenFields, fieldPicker, toggleChip, visibleFields } from "./fieldpicker.js";
import { column, fieldColumn, plainCell, resultsTable } from "./table.js";
import { copyButton, copyTable } from "./tsv.js";

let showAllFields = false;
// Which papers are expanded. Kept across re-renders (a field toggle, a refresh) for as long as the page
// lives; a paper that left the library simply stops matching.
const expanded = new Set();
// profile key -> the entity name the home table shows under it; absent: the primary.
const shownEntity = new Map();

// The table under `profile`; the home view stores it (state.corpus) only once it knows the view is still current.
export const loadCorpus = (profile) => profileApi(profile, "/api/dataset");

// The entity the table shows under the profile on screen, among its groups (the primary when none was picked).
function currentEntity(groups) {
  const name = shownEntity.get(state.corpusProfile ?? "");
  return groups.find((group) => group.name === name) ?? groups[0];
}

// The table's data for one entity group. The primary keeps the paper rows (paper-level fields beside its own); any
// other entity is its sample rows alone, each carrying its paper, with its own fields.
function entityView(data, group, primary) {
  const fields = data?.fields ?? [];
  const papers = data?.rows ?? [];
  if (group === primary) {
    return {
      kind: "paper",
      fields: fields.filter((field) => field.scope !== "sample" || inEntity(group, field)),
      rows: papers.map((row) => {
        const samples = (row.sample_rows ?? []).filter((sample) => inEntity(group, sample));
        return { ...row, sample_rows: samples, sample_count: samples.length };
      }),
    };
  }
  return {
    kind: "sample",
    fields: fields.filter((field) => field.scope === "sample" && inEntity(group, field)),
    rows: papers.flatMap((row) =>
      (row.sample_rows ?? []).filter((sample) => inEntity(group, sample)).map((sample) => ({ row, sample })),
    ),
  };
}

// What the paper table calls a row: the profile's entity, or with several entity types the primary one.
const primaryLabel = () => entityGroups()[0].label;

export function renderCorpus(root) {
  root.innerHTML = "";
  if (!(state.corpus?.rows ?? []).length) return;
  const groups = entityGroups();
  const group = currentEntity(groups);
  const rerender = () => keepFocus(root, () => renderCorpus(root));
  const data = entityView(state.corpus, group, groups[0]);
  const entityChips = groups.length < 2 ? null : entitySwitch(groups, group, rerender);
  if (data.kind === "sample") {
    renderEntityRows(root, data, group, entityChips, rerender);
    return;
  }
  const rows = data.rows;

  // Columns are decided over every sample, so a field that only an expanded sample has still gets one.
  const allRows = rows.flatMap((row) => [row.paper_row ?? {}, ...(row.sample_rows ?? [])]);
  const chosen = chosenFields(data.fields);
  const fields = visibleFields(chosen, allRows, showAllFields);
  const expandable = rows.filter((row) => (row.sample_rows ?? []).length > 1);
  const columns = corpusColumns(fields);
  // What the table shows, in order: every paper row, plus the sample rows of the papers expanded. The
  // rendered rows and the clipboard copy are both built from this one list.
  const items = rows.flatMap((row) => [
    { kind: "paper", row, source: row.paper_row ?? {} },
    ...(expanded.has(row.document_id) ? (row.sample_rows ?? []).map((sample) => ({ kind: "sample", row, source: sample })) : []),
  ]);

  const chips = [
    toggleChip(chosen, showAllFields, () => { showAllFields = !showAllFields; rerender(); }),
    fieldPicker(data.fields, rerender),
  ];
  if (expandable.length) chips.push(expandAllChip(expandable, rerender));
  const entity = primaryLabel();
  const note =
    `${rows.length} 篇论文各取一个完整${entity}行，点${entity}数可展开该论文的全部${entity}；` +
    "空白单元格是流水线拒绝猜测的取值，不是 0。";

  const table = resultsTable(columns, items, {
    rowClass: (item) => (item.kind === "sample" ? "sample-row" : expanded.has(item.row.document_id) ? "expanded" : ""),
  });
  table.querySelector("tbody").addEventListener("click", (event) => {
    const button = event.target.closest("button.expand");
    if (!button) return;
    const id = button.dataset.doc;
    if (expanded.has(id)) expanded.delete(id);
    else expanded.add(id);
    rerender();
  });
  root.append(resultsSection(entityChips, chips, columns, items, note, table));
}

// A secondary entity: one row per sample of it, beside the paper it came from; no paper row, nothing to expand.
function renderEntityRows(root, view, group, entityChips, rerender) {
  const items = view.rows;
  const chosen = chosenFields(view.fields);
  const fields = visibleFields(chosen, items.map((item) => item.sample), showAllFields);
  const columns = entityColumns(fields, group);
  const chips = [
    toggleChip(chosen, showAllFields, () => { showAllFields = !showAllFields; rerender(); }),
    fieldPicker(view.fields, rerender),
  ];
  const papers = new Set(items.map((item) => item.row.document_id)).size;
  const note = `${papers} 篇论文共 ${items.length} 个${group.label}，每行一个；空白单元格是流水线拒绝猜测的取值，不是 0。`;
  root.append(resultsSection(entityChips, chips, columns, items, note, resultsTable(columns, items)));
}

// What every home table has: the entity chips (with several entity types), its own chips, the copy button and the
// workbook link over a note and the table. The clipboard copy is built from the same columns and items as the rows.
function resultsSection(entityChips, chips, columns, items, noteText, table) {
  // No heading of its own: on the home view the page title above the table already names it.
  const head = document.createElement("div");
  head.className = "results-head";
  if (entityChips) head.append(entityChips);
  const own = document.createElement("div");
  own.className = "chips";
  own.append(...chips);
  head.append(own);
  head.append(copyButton(() => copyTable(columns, items)));
  const download = document.createElement("a");
  download.className = "download";
  download.href = profileHref(state.corpusProfile, "/api/dataset.xlsx");
  // Empty: the server's Content-Disposition names the file after the profile.
  download.setAttribute("download", "");
  download.textContent = "下载全部 Excel";
  head.append(download);

  const note = document.createElement("p");
  note.className = "results-note muted";
  note.textContent = noteText;

  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.append(table);

  const section = document.createElement("section");
  section.className = "results corpus";
  section.setAttribute("aria-label", "结果总表");
  section.append(head, note, wrap);
  return section;
}

// One chip per entity type; the one shown is pressed. Each re-renders the table and keeps the keyboard on itself.
function entitySwitch(groups, current, rerender) {
  const chips = document.createElement("div");
  chips.className = "chips entity-switch";
  chips.setAttribute("role", "group");
  chips.setAttribute("aria-label", "按实体类型查看");
  for (const group of groups) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `chip${group === current ? " on" : ""}`;
    chip.dataset.focus = `entity:${group.name}`;
    chip.setAttribute("aria-pressed", String(group === current));
    chip.textContent = group.label;
    chip.addEventListener("click", () => {
      shownEntity.set(state.corpusProfile ?? "", group.name);
      rerender();
    });
    chips.append(chip);
  }
  return chips;
}

// A secondary entity's rows: the paper (a link), the sample's id, its counts, and its fields; a reference column
// shows the id of the row it names, labelled with that row's entity, as it does everywhere.
function entityColumns(fields, group) {
  const name = (item) => item.row.name ?? item.row.document_id ?? "";
  const counts = (item) => `${item.sample.available_fields ?? 0} / ${item.sample.agree_fields ?? 0}`;
  return [
    column(
      "论文",
      (item) => {
        const text = escapeHtml(name(item));
        return `<td class="label" title="${text}"><a href="${escapeHtml(documentHash(item.row.document_id))}">${text}</a></td>`;
      },
      name,
    ),
    column(group.label, (item) => `<td class="mono">${escapeHtml(item.sample.sample_id ?? "")}</td>`, (item) => item.sample.sample_id ?? ""),
    column("可用/一致", (item) => `<td class="mono">${escapeHtml(counts(item))}</td>`, counts),
    ...fields.map((field) => fieldColumn(field, (item) => item.sample[field.name] ?? null, (_, value) => plainCell(value, field))),
  ];
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
