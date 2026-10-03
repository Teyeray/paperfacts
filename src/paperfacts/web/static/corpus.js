// Home view: the whole library as one table. Each row is a paper's selected sample row (the same
// `paper_row` the Excel export puts on the 论文数据 sheet), so the mined result for the corpus is
// visible without opening a single document. A paper with several samples can be expanded in place to
// show every sample row beneath it; the one the paper row was chosen from is marked.
//
// There are no quality rows here -- those are per-document -- so a cell is just its value, and an
// empty cell is "this paper has no committed value for this field".
//
// With several entity types a chip per entity picks what a row is: the primary entity gives the table above (the
// paper row is always one of its rows); any other gives one row per sample of that entity across the papers, with that
// entity's fields. A search or an active filter (explorer.js, filters.js) turns the primary's table into the same
// flattened rows: one per sample, its id first and its paper second, so every row is a sample that matched. A paper
// with no sample of the entity is one row of its own, so a paper found by its name is still listed.
//
// A click on a column's header sorts the table by it (ascending, descending, off). A paper row is ordered by the value
// it shows (its chosen sample's), and an expanded paper's sample rows stay under it. The sort, the entity shown, the
// search and the filters are the home query's (explorer.js): every draw reads them from it, and getSort/setSort/
// onSortChange are the hooks app.js binds the sort to.

import { profileApi, profileHref } from "./api.js";
import { escapeHtml, keepFocus } from "./html.js";
import { documentHash } from "./router.js";
import { entityGroups, inEntity, state } from "./state.js";
import { chosenFields, fieldPicker, toggleChip, visibleFields } from "./fieldpicker.js";
import {
  activeChips,
  clearAllButton,
  explored,
  filterToggle,
  flatItems,
  highlight,
  isExploring,
  panelOpen,
  readExplore,
  searchBar,
  updateQuery,
  writeFilters,
} from "./explorer.js";
import { filterChips, filterPanel, itemValue, usesDocs } from "./filters.js";
import {
  column,
  densitySwitch,
  fieldColumn,
  nextSort,
  plainCell,
  resultsTable,
  sortItems,
  sortValue,
  sortableHeads,
} from "./table.js";
import { copyButton, copyTable } from "./tsv.js";

let showAllFields = false;
// Which papers are expanded. Kept across re-renders (a field toggle, a refresh) for as long as the page
// lives; a paper that left the library simply stops matching.
const expanded = new Set();

// { key, dir: "asc" | "desc" } or null (the server's order), set from the home query on every draw. A key no column of
// the table on screen has is no sort.
let sort = null;
let sortListener = null;
export const getSort = () => sort;
export function setSort(next) {
  const valid = next && typeof next.key === "string" && (next.dir === "asc" || next.dir === "desc");
  sort = valid ? { key: next.key, dir: next.dir } : null;
}
// Told of every sort the reader makes by clicking a header (not of setSort's own writes).
export const onSortChange = (listener) => {
  sortListener = listener;
};

// The table's items in the current sort (the server's order when there is none).
function sorted(columns, items) {
  const current = getSort();
  const by = current && columns.find((c) => c.key === current.key);
  return by ? sortItems(items, by, current.dir) : items;
}

function sortHandler(rerender) {
  return (key) => {
    setSort(nextSort(getSort(), key));
    sortListener?.(getSort());
    rerender();
  };
}

// The table under `profile`; the home view stores it (state.corpus) only once it knows the view is still current.
export const loadCorpus = (profile) => profileApi(profile, "/api/dataset");

// The entity the home query asks for, among the groups (the primary when it names none, or one the profile lacks).
const currentEntity = (groups, name) => groups.find((group) => group.name === name) ?? groups[0];

// The fields of one entity group's table: the primary's sample fields beside the paper-level ones; any other entity's
// own sample fields alone.
const entityFields = (fields, group, primary) =>
  (fields ?? []).filter((field) =>
    group === primary ? field.scope !== "sample" || inEntity(group, field) : field.scope === "sample" && inEntity(group, field),
  );

// What the paper table calls a row: the profile's entity, or with several entity types the primary one.
const primaryLabel = () => entityGroups()[0].label;

// A redraw of the table on screen that keeps the keyboard where it was; a range box that had it gets its caret back
// at the end of what was typed.
function redraw(root) {
  keepFocus(root, () => renderCorpus(root));
  const active = document.activeElement;
  if (active?.matches?.("input.range-num")) active.setSelectionRange(active.value.length, active.value.length);
}

// The home query changed (typed into the search box, a filter, a sort, the address) or, with a status filter on, the
// rail's list did: the table on screen is redrawn from what is already loaded, never re-fetched. Only a table of the
// profile on screen, on the home view, is redrawn.
export function refreshCorpus({ docsChanged = false } = {}) {
  const root = document.getElementById("corpus-view");
  if (!root || state.homeQuery == null || !state.corpus || state.corpusProfile !== state.profileName) return;
  if (docsChanged && !usesDocs(readExplore().filters)) return;
  redraw(root);
}

export function renderCorpus(root) {
  searchBar(root);
  for (const child of [...root.children]) if (!child.classList.contains("explorer-search")) child.remove();
  if (!(state.corpus?.rows ?? []).length) return;
  const explore = readExplore();
  setSort(explore.sort);
  const groups = entityGroups();
  const group = currentEntity(groups, explore.entity);
  const rerender = () => redraw(root);
  const fields = entityFields(state.corpus.fields, group, groups[0]);
  const entityChips = groups.length < 2 ? null : entitySwitch(groups, group);
  // The panel counts and scales over every row of the entity, whatever the search and filters leave.
  const all = flatItems(state.corpus.rows, group);
  const samples = all.filter((item) => item.kind === "sample");
  const papers = state.corpus.rows.map((row) => ({ kind: "paper", row, source: row.paper_row ?? {}, categories: row.paper_categories ?? {} }));
  const open = panelOpen();
  const explorer = {
    toggle: filterToggle(filterChips(explore.filters, fields).length, rerender),
    active: activeChips(explore, fields),
  };
  const body = document.createElement("div");
  body.className = `explorer${open ? " panel-open" : ""}`;
  if (open) body.append(filterPanel(fields, { samples, papers }, explore.filters, writeFilters));
  const exploring = isExploring(explore, fields);
  if (group !== groups[0] || exploring) {
    // A secondary entity is always one row per sample; a paper with none of them is listed only when it was searched for.
    const items = exploring ? explored(all, explore, fields) : samples;
    body.append(flatSection(items, fields, group, entityChips, explore, explorer, { samples, exploring }));
  } else {
    body.append(paperSection(fields, group, entityChips, explorer));
  }
  root.append(body);
}

// The primary entity unsearched and unfiltered: one row per paper, each expandable into its samples.
function paperSection(allFields, group, entityChips, explorer) {
  const rerender = () => redraw(document.getElementById("corpus-view"));
  const rows = state.corpus.rows.map((row) => {
    const samples = (row.sample_rows ?? []).filter((sample) => inEntity(group, sample));
    return { ...row, sample_rows: samples, sample_count: samples.length };
  });

  // Columns are decided over every sample, so a field that only an expanded sample has still gets one.
  const allRows = rows.flatMap((row) => [row.paper_row ?? {}, ...(row.sample_rows ?? [])]);
  const chosen = chosenFields(allFields);
  const fields = visibleFields(chosen, allRows, showAllFields);
  const expandable = rows.filter((row) => (row.sample_rows ?? []).length > 1);
  const columns = corpusColumns(fields);
  // What the table shows, in order: every paper row (sorted by the value it shows), each followed by the sample rows
  // of the papers expanded. The rendered rows and the clipboard copy are both built from this one list.
  const papers = sorted(columns, rows.map((row) => ({ kind: "paper", row, source: row.paper_row ?? {} })));
  const items = papers.flatMap((paper) => [
    paper,
    ...(expanded.has(paper.row.document_id)
      ? (paper.row.sample_rows ?? []).map((sample) => ({ kind: "sample", row: paper.row, source: sample }))
      : []),
  ]);

  const chips = [
    fieldPicker(allFields, rerender),
    toggleChip(chosen, showAllFields, () => { showAllFields = !showAllFields; rerender(); }),
    densitySwitch(),
  ];
  if (expandable.length) chips.push(expandAllChip(expandable, rerender));
  const entity = primaryLabel();
  const samples = rows.reduce((sum, row) => sum + (row.sample_count ?? 0), 0);
  const count = `${rows.length} 篇论文 · ${samples} 个${entity}`;
  const note =
    `每篇论文取一个完整${entity}行，点${entity}数可展开该论文的全部${entity}；点列名排序；` +
    "空白单元格是流水线拒绝猜测的取值，不是 0。";

  const table = resultsTable(columns, items, {
    rowClass: (item) => (item.kind === "sample" ? "sample-row" : expanded.has(item.row.document_id) ? "expanded" : ""),
  });
  sortableHeads(table.tHead.rows[0], columns, getSort(), sortHandler(rerender));
  table.querySelector("tbody").addEventListener("click", (event) => {
    const button = event.target.closest("button.expand");
    if (!button) return;
    const id = button.dataset.doc;
    if (expanded.has(id)) expanded.delete(id);
    else expanded.add(id);
    rerender();
  });
  return resultsSection(entityChips, chips, columns, items, { count, note }, table, explorer);
}

// One row per sample of `group` (and per paper without one, when searched for): the sample's id, its paper, its counts
// and the fields; with 含条件描述 its conditions too. What matched the search is marked.
function flatSection(found, allFields, group, entityChips, explore, explorer, { samples, exploring }) {
  const rerender = () => redraw(document.getElementById("corpus-view"));
  const chosen = chosenFields(allFields);
  // The columns are decided over every row, not the ones found, so they do not come and go while the reader types.
  const rows = [...state.corpus.rows.map((row) => row.paper_row ?? {}), ...samples.map((item) => item.source)];
  const fields = visibleFields(chosen, rows, showAllFields);
  const columns = flatColumns(fields, group, explore);
  const items = sorted(columns, found);
  const chips = [
    fieldPicker(allFields, rerender),
    toggleChip(chosen, showAllFields, () => { showAllFields = !showAllFields; rerender(); }),
    densitySwitch(),
  ];
  const papers = new Set(items.map((item) => item.row.document_id)).size;
  const shown = items.filter((item) => item.kind === "sample").length;
  const total = state.corpus.rows.length;
  const count = `${papers} 篇论文 · ${shown} 个${group.label}${exploring ? `（共 ${total} 篇）` : ""}`;
  const note =
    `每行一个${group.label}，后面是它所在的论文${exploring ? "，只列出符合搜索和筛选的" : ""}；点列名排序；` +
    "空白单元格是流水线拒绝猜测的取值，不是 0。";
  const table = resultsTable(columns, items, { className: "results-table flat" });
  sortableHeads(table.tHead.rows[0], columns, getSort(), sortHandler(rerender));
  const empty = items.length ? null : emptyResult(group);
  return resultsSection(entityChips, chips, columns, items, { count, note }, table, { ...explorer, empty });
}

// Nothing passed: say so where the table would be, with the way out.
function emptyResult(group) {
  const box = document.createElement("div");
  box.className = "table-empty explore-empty";
  box.append(`没有符合搜索和筛选条件的${group.label}或论文。`, clearAllButton());
  return box;
}

// What every home table has: one toolbar row -- the explorer's slot (筛选), the entity chips (with several entity
// types), its own controls, then the row count, the copy button and the workbook link -- over the active filters, a
// note and the table. The clipboard copy is built from the same columns and items as the rows: what is on screen.
function resultsSection(entityChips, chips, columns, items, { count, note: noteText }, table, { toggle, active, empty = null }) {
  // No heading of its own: on the home view the page title above the table already names it.
  const head = document.createElement("div");
  head.className = "results-head table-toolbar";
  const explore = document.createElement("div");
  explore.className = "toolbar-slot";
  explore.dataset.slot = "explore";
  explore.append(toggle);
  head.append(explore);
  if (entityChips) head.append(entityChips);
  const own = document.createElement("div");
  own.className = "chips";
  own.append(...chips);
  head.append(own);
  const end = document.createElement("div");
  end.className = "toolbar-end";
  const counter = document.createElement("span");
  counter.className = "row-count";
  counter.textContent = count;
  const copy = copyButton(() => copyTable(columns, items));
  copy.classList.add("secondary");
  end.append(counter, copy);
  const download = document.createElement("a");
  download.className = "download secondary";
  download.href = profileHref(state.corpusProfile, "/api/dataset.xlsx");
  // Empty: the server's Content-Disposition names the file after the profile.
  download.setAttribute("download", "");
  download.title = "全部论文的结果，不受搜索和筛选影响";
  download.textContent = "下载全部 Excel";
  end.append(download);
  head.append(end);

  const note = document.createElement("p");
  note.className = "results-note muted";
  note.textContent = noteText;

  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  // Nothing found still shows the header row, with the way out under it.
  wrap.append(table, ...(empty ? [empty] : []));

  const section = document.createElement("section");
  section.className = "results corpus";
  section.setAttribute("aria-label", "结果总表");
  section.append(head, ...(active ? [active] : []), note, wrap);
  return section;
}

// One chip per entity type; the one shown is pressed. Each writes the home query, whose redraw keeps the keyboard on it.
function entitySwitch(groups, current) {
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
    chip.addEventListener("click", () => updateQuery({ e: group === groups[0] ? "" : group.name }));
    chips.append(chip);
  }
  return chips;
}

// A flattened row: the sample's id (its label under it), the paper (a link), the conditions when the search reads them,
// its counts, and the fields, a paper-level one read from the paper; what matched the search is marked. A reference
// column shows the id of the row it names, labelled with that row's entity, as it does everywhere. A paper listed
// without a sample has no id.
function flatColumns(fields, group, explore) {
  const q = explore.q;
  const name = (item) => item.row.name ?? item.row.document_id ?? "";
  const id = (item) => (item.kind === "sample" ? String(item.source.sample_id ?? "") : "");
  const label = (item) => (item.kind === "sample" ? String(item.source.sample_label ?? "") : "");
  const conditions = (item) => (item.kind === "sample" ? String(item.source.conditions ?? "") : "");
  const counts = (item) => `${item.source.available_fields ?? 0} / ${item.source.agree_fields ?? 0}`;
  const review = (item) => (item.row.article_type === "review" ? `<span class="article-tag">综述</span>` : "");
  const columns = [
    sortBy(
      column(
        group.label,
        (item) => {
          if (item.kind !== "sample") return `<td class="mono"><small>无${escapeHtml(group.label)}</small></td>`;
          const sub = label(item) && label(item) !== id(item) ? `<small>${highlight(label(item), q)}</small>` : "";
          return `<td class="mono" title="${escapeHtml(id(item))}">${highlight(id(item), q)}${sub}</td>`;
        },
        id,
      ),
      "entity",
      id,
    ),
    sortBy(
      column(
        "论文",
        (item) =>
          `<td class="label" title="${escapeHtml(name(item))}"><a href="${escapeHtml(documentHash(item.row.document_id))}">` +
          `${highlight(name(item), q)}</a>${review(item)}</td>`,
        name,
      ),
      "paper",
      name,
    ),
  ];
  if (explore.cond) {
    columns.push(
      column("条件", (item) => `<td class="muted cond" title="${escapeHtml(conditions(item))}"><div class="clamp">${highlight(conditions(item), q)}</div></td>`, conditions),
    );
  }
  return [
    ...columns,
    sortBy(column("可用/一致", (item) => `<td class="mono">${escapeHtml(counts(item))}</td>`, counts), "counts", (item) => item.source.available_fields),
    ...fields.map((field) => fieldColumn(field, (item) => itemValue(field, item), (_, value) => plainCell(value, field))),
  ];
}

// A paper row links to its document and carries the sample count (a button when there is more than one
// sample to expand into); a sample row is indented under it and marks the sample the paper row came from.
// The clipboard gets the paper's name on every line, so a pasted sample row still says whose it is.
function corpusColumns(fields) {
  const isPaper = (item) => item.kind === "paper";
  const name = (item) => item.row.name ?? item.row.document_id ?? "";
  const id = (item) => item.source.sample_id ?? "";
  return [
    sortBy(
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
      "paper",
      name,
    ),
    sortBy(
      column(
        primaryLabel(),
        (item) =>
          `<td class="mono" title="${escapeHtml(id(item))}">${escapeHtml(id(item))}${isPaper(item) ? sampleCount(item.row) : chosenMark(item)}</td>`,
        id,
      ),
      "entity",
      id,
    ),
    sortBy(
      column(
        "可用/一致",
        (item) => `<td class="mono">${escapeHtml(`${item.source.available_fields ?? 0} / ${item.source.agree_fields ?? 0}`)}</td>`,
        (item) => `${item.source.available_fields ?? 0} / ${item.source.agree_fields ?? 0}`,
      ),
      "counts",
      (item) => item.source.available_fields,
    ),
    ...fields.map((field) => fieldColumn(field, (item) => item.source[field.name] ?? null, (_, value) => plainCell(value, field))),
  ];
}

// An identity column the table can be sorted by, under a stable key.
const sortBy = (col, key, value) => ({ ...col, key, sort: (item) => sortValue(value(item)) });

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
