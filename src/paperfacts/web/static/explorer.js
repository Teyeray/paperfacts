// The home view's explorer: the search box over the table, the 筛选 button that shows the filter panel (filters.js),
// the active filters as chips, and the home query (`#/?q=…`) that holds all of it. What the table shows is read from
// state.homeQuery whenever it is drawn, so a reload, a shared link or the back button gives the same table; a control
// writes the query with the router's setHomeQuery (no history entry, no table load) and the router redraws through
// onHomeQuery. A profile switch is a new view, which keeps no query.
//
// In the query: `q` (the search), `cond=1` (search the conditions text too), `e` (the entity shown, absent: the
// primary), `sort` (a column key, `-key` descending) and the filters (filters.js).
//
// The search matches a sample's id, its label and its paper's name, case-insensitively after NFKC folding (so a
// full-width "ＰＥＴ" finds "PET"); with 含条件描述 its conditions text too. It matches the text as written: there is
// no translation, so a Chinese word does not find a value the paper wrote in English.

import { escapeHtml } from "./html.js";
import { homeQueryFromHash, setHomeQuery } from "./router.js";
import { inEntity, state, uiCopy } from "./state.js";
import { filterChips, filterParams, isFiltering, noFilters, passes, readFilters } from "./filters.js";

const SEARCH_DEBOUNCE_MS = 120;
const PANEL_KEY = "paperfacts.filters-open";

// NFKC, then lower case: the one fold both the query and the text go through.
export const fold = (text) => String(text ?? "").normalize("NFKC").toLowerCase();

// What the home query asks the table for.
export function readExplore(query = state.homeQuery) {
  const sort = query?.sort ?? "";
  const desc = sort.startsWith("-");
  const key = desc ? sort.slice(1) : sort;
  return {
    q: (query?.q ?? "").trim(),
    cond: query?.cond === "1",
    entity: query?.e ?? null,
    sort: key ? { key, dir: desc ? "desc" : "asc" } : null,
    filters: readFilters(query),
  };
}

// Writes `params` over the current home query (a value "" drops its key). Only the home view has a query; a control
// of a table that is no longer on screen writes nothing.
export function updateQuery(params) {
  if (homeQueryFromHash() === null) return;
  setHomeQuery({ ...(state.homeQuery ?? {}), ...params });
}

export const sortParam = (sort) => ({ sort: sort ? `${sort.dir === "desc" ? "-" : ""}${sort.key}` : "" });
export const writeFilters = (filters) => updateQuery(filterParams(filters, state.homeQuery ?? {}));

// Whether the table is narrowed at all: a search or an active filter on one of `fields`.
export const isExploring = (explore, fields) => explore.q !== "" || isFiltering(explore.filters, fields);

// ---------- rows ----------

// The table's rows for one entity group, one per sample of it, each with the categories the server computed for that
// sample; a paper with no sample of it is one paper item, so a paper found by name still has a row.
export function flatItems(rows, group) {
  return rows.flatMap((row) => {
    const samples = (row.sample_rows ?? [])
      .map((source, index) => ({ kind: "sample", row, source, categories: row.sample_categories?.[index] ?? {} }))
      .filter((item) => inEntity(group, item.source));
    return samples.length ? samples : [{ kind: "paper", row, source: row.paper_row ?? {}, categories: row.paper_categories ?? {} }];
  });
}

// Whether `item` is found by the search: its paper's name, and for a sample its id, label and (with `cond`) conditions.
function found(item, needle, cond) {
  if (!needle) return true;
  const texts = [item.row.name];
  if (item.kind === "sample") texts.push(item.source.sample_id, item.source.sample_label, ...(cond ? [item.source.conditions] : []));
  return texts.some((text) => fold(text).includes(needle));
}

// The items that pass the search and every filter among `fields`.
export function explored(items, explore, fields) {
  const needle = fold(explore.q);
  return items.filter((item) => found(item, needle, explore.cond) && passes(explore.filters, fields, item));
}

// `text` escaped, with each match of the search marked. The marks are placed on the text as written, so they are
// drawn only where folding kept every character's place (it nearly always does); the match itself never depends on it.
export function highlight(text, q) {
  const raw = String(text ?? "");
  const needle = fold(q);
  const folded = fold(raw);
  if (!needle || folded.length !== raw.length) return escapeHtml(raw);
  let html = "";
  let at = 0;
  for (let hit = folded.indexOf(needle); hit >= 0; hit = folded.indexOf(needle, at)) {
    html += `${escapeHtml(raw.slice(at, hit))}<mark>${escapeHtml(raw.slice(hit, hit + needle.length))}</mark>`;
    at = hit + needle.length;
  }
  return html + escapeHtml(raw.slice(at));
}

// ---------- the search box ----------

let searchTimer = null;

// The search box over the table, built once and kept across every redraw of the table beneath it, so typing never
// loses the focus or the caret. Its value follows the query whenever the reader is not typing in it.
export function searchBar(root) {
  let bar = root.querySelector(":scope > .explorer-search");
  if (!bar) {
    bar = buildSearchBar();
    root.prepend(bar);
  }
  const explore = readExplore();
  const input = bar.querySelector("input[type=search]");
  if (document.activeElement !== input && fold(input.value).trim() !== fold(explore.q)) {
    clearTimeout(searchTimer);
    input.value = explore.q;
  }
  input.placeholder = `搜索${uiCopy("entity_label_zh")}编号、名称或论文名`;
  bar.querySelector("input[type=checkbox]").checked = explore.cond;
  bar.querySelector(".search-clear").hidden = !input.value;
  return bar;
}

function buildSearchBar() {
  const bar = document.createElement("div");
  bar.className = "explorer-search";
  bar.setAttribute("role", "search");
  bar.innerHTML = `
    <div class="search-field">
      <input type="search" id="corpus-search" data-focus="corpus-search" autocomplete="off" spellcheck="false" aria-describedby="corpus-search-help">
      <button type="button" class="search-clear" data-focus="search-clear" hidden>清除</button>
    </div>
    <label class="search-cond"><input type="checkbox" data-focus="search-cond"> 含条件描述</label>
    <p class="search-help muted" id="corpus-search-help">不区分大小写和全角半角；按原文匹配，不做翻译（中文查不到用英文写的内容）。</p>`;
  const input = bar.querySelector("input[type=search]");
  input.setAttribute("aria-label", "搜索");
  const clear = bar.querySelector(".search-clear");
  input.addEventListener("input", () => {
    clear.hidden = !input.value;
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => updateQuery({ q: input.value.trim() }), SEARCH_DEBOUNCE_MS);
  });
  clear.addEventListener("click", () => {
    clearTimeout(searchTimer);
    input.value = "";
    clear.hidden = true;
    updateQuery({ q: "" });
    input.focus();
  });
  bar.querySelector("input[type=checkbox]").addEventListener("change", (event) => {
    updateQuery({ cond: event.target.checked ? "1" : "" });
  });
  return bar;
}

// ---------- the panel's button and the chips ----------

// Whether the filter panel is open: one choice per browser, open by default on a wide screen.
export function panelOpen() {
  try {
    const stored = localStorage.getItem(PANEL_KEY);
    if (stored !== null) return stored === "1";
  } catch {
    // no storage: the default below
  }
  return window.matchMedia("(min-width: 961px)").matches;
}

function setPanelOpen(open) {
  try {
    localStorage.setItem(PANEL_KEY, open ? "1" : "0");
  } catch {
    // not remembered past this page
  }
}

// 筛选, with the number of active filters; it opens and closes the panel and redraws.
export function filterToggle(count, rerender) {
  const open = panelOpen();
  const button = document.createElement("button");
  button.type = "button";
  button.className = `chip filter-toggle${count ? " on" : ""}`;
  button.dataset.focus = "filter-toggle";
  button.setAttribute("aria-expanded", String(open));
  button.setAttribute("aria-controls", "filter-panel");
  button.innerHTML = `筛选${count ? `<span class="n">${count}</span>` : ""}`;
  button.addEventListener("click", () => {
    setPanelOpen(!panelOpen());
    rerender();
  });
  return button;
}

// The active filters over the table, each removable, and 清除全部 (search included); null when nothing narrows it.
export function activeChips(explore, fields) {
  const chips = filterChips(explore.filters, fields);
  if (!chips.length && !explore.q) return null;
  const row = document.createElement("div");
  row.className = "active-filters chips";
  row.setAttribute("aria-label", "生效的筛选");
  if (explore.q) {
    row.append(removable("search", `搜索：${explore.q}`, () => updateQuery({ q: "" })));
  }
  for (const chip of chips) row.append(removable(chip.key, chip.text, () => writeFilters(chip.without)));
  row.append(clearAllButton());
  return row;
}

function removable(key, text, remove) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "chip removable";
  button.dataset.focus = `remove:${key}`;
  button.title = "去掉这个条件";
  button.textContent = text;
  const mark = document.createElement("span");
  mark.className = "x";
  mark.setAttribute("aria-hidden", "true");
  mark.textContent = "×";
  button.append(mark);
  button.addEventListener("click", remove);
  return button;
}

// Drops the search and every filter; the sort, the entity shown and 含条件描述 stay.
export function clearAllButton() {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "picker-link clear-all";
  button.dataset.focus = "clear-all";
  button.textContent = "清除全部";
  button.addEventListener("click", () => {
    clearTimeout(searchTimer);
    updateQuery({ q: "", ...filterParams(noFilters(), state.homeQuery ?? {}) });
    document.getElementById("corpus-search")?.focus();
  });
  return button;
}
