// The home view's filters: what each kind of field can be filtered by, the test a row must pass, and the panel left
// of the table that sets them. A filter lives in the home query (explorer.js writes it), so this module only reads a
// query and hands back the next one; it never touches the URL itself.
//
// One rule per kind, decided by the column and never by a value's shape:
//  - numeric and interval: a range in the column's canonical unit. A blank cell never passes a range; an interval
//    passes when its two ends overlap the range; a list (`many`) passes when any of its elements does.
//  - a field with categories: checkboxes on the canonical categories the server computed for every value
//    (CorpusRow.paper_categories / sample_categories). The browser never canonicalises a spelling itself.
//  - boolean: 是 / 否.
//  - date and reference: no filter in this iteration (the panel says so).
// Paper level: the article type (综述 / 研究) and the rail's status (有冲突 / 未完成), joined with state.docs.
// Every active filter must pass (AND); within one filter any ticked choice may (OR).
//
// In the query: `f.<field>=<min>~<max>` (either end may be empty) for a range, `f.<field>=<a>|<b>` for ticked
// categories or 是/否 ("1" / "0"), `type=review|research`, `conflict=1`, `unfinished=1`.

import { escapeHtml, fmt } from "./html.js";
import { finishedHere } from "./library.js";
import { state } from "./state.js";

const FIELD_PREFIX = "f.";
const PICK_SEPARATOR = "|";
// An article type the server leaves null is an ordinary research paper.
const ARTICLE_TYPES = [
  { value: "review", label: "综述" },
  { value: "research", label: "研究" },
];
const STATUSES = [
  { key: "conflict", label: "有冲突", test: (doc) => (doc?.counts?.conflict ?? 0) > 0 },
  { key: "unfinished", label: "未完成", test: (doc) => Boolean(doc) && !finishedHere(doc) },
];
const BOOLEAN_CHOICES = [
  { value: "1", label: "是" },
  { value: "0", label: "否" },
];
const RANGE_KINDS = new Set(["numeric", "interval"]);
const UNFILTERED_KINDS = { date: "日期字段暂不支持筛选", reference: "引用字段暂不支持筛选" };
// A slider runs over this many steps between the corpus's smallest and largest value.
const STEPS = 1000;
const BINS = 24;

// What a column can be filtered by: "range", "pick", "none" (said in the panel) or null (left out of the panel).
export function filterKind(field) {
  if (RANGE_KINDS.has(field.kind)) return "range";
  if (field.kind === "boolean") return "pick";
  if ((field.categories ?? []).length) return "pick";
  if (UNFILTERED_KINDS[field.kind]) return "none";
  return null;
}

// ---------- the filters in the query ----------

// The filters a home query holds: { fields: Map(name -> its raw spec), types: Set, conflict, unfinished }.
export function readFilters(query) {
  const fields = new Map();
  for (const [key, value] of Object.entries(query ?? {})) {
    if (key.startsWith(FIELD_PREFIX) && value !== "") fields.set(key.slice(FIELD_PREFIX.length), value);
  }
  return {
    fields,
    types: new Set(splitPicks(query?.type).filter((type) => ARTICLE_TYPES.some((known) => known.value === type))),
    conflict: query?.conflict === "1",
    unfinished: query?.unfinished === "1",
  };
}

// The query keys the filters own, each with its value ("" drops the key).
export function filterParams(filters, previous = {}) {
  const params = {};
  for (const key of Object.keys(previous)) if (key.startsWith(FIELD_PREFIX)) params[key] = "";
  for (const [name, spec] of filters.fields) params[FIELD_PREFIX + name] = spec;
  params.type = [...filters.types].join(PICK_SEPARATOR);
  params.conflict = filters.conflict ? "1" : "";
  params.unfinished = filters.unfinished ? "1" : "";
  return params;
}

export const noFilters = () => ({ fields: new Map(), types: new Set(), conflict: false, unfinished: false });

const withField = (filters, name, spec) => {
  const fields = new Map(filters.fields);
  if (spec) fields.set(name, spec);
  else fields.delete(name);
  return { ...filters, fields };
};

function splitPicks(text) {
  return String(text ?? "")
    .split(PICK_SEPARATOR)
    .filter(Boolean);
}

// A range spec as { min, max } (null: that end is open), or null when it constrains nothing or does not parse.
export function parseRange(spec) {
  const match = /^([^~]*)~([^~]*)$/.exec(spec ?? "");
  if (!match) return null;
  const end = (text) => (text.trim() === "" ? null : Number(text));
  let [min, max] = [end(match[1]), end(match[2])];
  if ((min !== null && !Number.isFinite(min)) || (max !== null && !Number.isFinite(max))) return null;
  if (min === null && max === null) return null;
  if (min !== null && max !== null && min > max) [min, max] = [max, min];
  return { min, max };
}

const rangeSpec = (min, max) => (min === null && max === null ? "" : `${min ?? ""}~${max ?? ""}`);

// The filters that actually constrain the table on screen, among `fields` (the shown entity's columns): a filter on
// a field the view does not have, or one that does not parse, is not active.
function activeFieldFilters(filters, fields) {
  const active = [];
  for (const field of fields) {
    const spec = filters.fields.get(field.name);
    if (spec == null) continue;
    const kind = filterKind(field);
    if (kind === "range") {
      const range = parseRange(spec);
      if (range) active.push({ field, kind, range });
    } else if (kind === "pick") {
      const picks = new Set(splitPicks(spec).filter((value) => choicesOf(field).some((choice) => choice.value === value)));
      if (picks.size) active.push({ field, kind, picks });
    }
  }
  return active;
}

export const isFiltering = (filters, fields) =>
  activeFieldFilters(filters, fields).length > 0 || filters.types.size > 0 || filters.conflict || filters.unfinished;

// Which papers each status holds in the rail's list, as one string: the status filters and the panel's status counts
// read nothing else of it, so a refresh of the list that leaves this unchanged leaves the home table as it is.
export const statusSignature = () =>
  STATUSES.map((status) =>
    state.docs
      .filter((doc) => status.test(doc))
      .map((doc) => doc.document_id)
      .sort()
      .join(","),
  ).join("|");

// The choices of a pick filter: the column's categories, or 是/否.
function choicesOf(field) {
  if (field.kind === "boolean") return BOOLEAN_CHOICES;
  return (field.categories ?? []).map((category) => ({ value: category, label: category }));
}

// ---------- what a row holds ----------
//
// An item is a row of the table: `{ kind: "sample", row, source, categories }` (one sample of a paper, with the
// categories the server computed for it) or `{ kind: "paper", row, source, categories }` (a paper with no sample of
// the shown entity). A paper-level field is read from the paper's row, a sample-level one from the sample.

export function itemValue(field, item) {
  if (field.scope === "paper") return item.row.paper_row?.[field.name] ?? null;
  return item.kind === "sample" ? item.source?.[field.name] ?? null : null;
}

function itemCategories(field, item) {
  if (field.scope === "paper") return item.row.paper_categories?.[field.name] ?? [];
  return item.kind === "sample" ? item.categories?.[field.name] ?? [] : [];
}

// A value's elements: each of a list, else the value alone.
const elements = (value, field) => (field.cardinality === "many" && Array.isArray(value) ? value : [value]);

// One element as the range it covers ({ low, high }, ends possibly infinite), or null when it holds no number.
function span(element, field) {
  if (field.kind === "interval" && Array.isArray(element)) {
    const [low, high] = element.map((end) => (typeof end === "number" && Number.isFinite(end) ? end : null));
    if (low === null && high === null) return null;
    return { low: low ?? -Infinity, high: high ?? Infinity };
  }
  return typeof element === "number" && Number.isFinite(element) ? { low: element, high: element } : null;
}

const spans = (field, item) => elements(itemValue(field, item), field).map((element) => span(element, field)).filter(Boolean);

function passesRange(field, item, range) {
  const min = range.min ?? -Infinity;
  const max = range.max ?? Infinity;
  return spans(field, item).some(({ low, high }) => low <= max && high >= min);
}

function passesPick(field, item, picks) {
  if (field.kind === "boolean") {
    return elements(itemValue(field, item), field).some(
      (value) => typeof value === "boolean" && picks.has(value ? "1" : "0"),
    );
  }
  return itemCategories(field, item).some((category) => picks.has(category));
}

const docOf = (row) => state.docs.find((doc) => doc.document_id === row.document_id);
const articleType = (row) => (row.article_type === "review" ? "review" : "research");

// Whether `item` passes every active filter among `fields`.
export function passes(filters, fields, item) {
  if (filters.types.size && !filters.types.has(articleType(item.row))) return false;
  for (const status of STATUSES) if (filters[status.key] && !status.test(docOf(item.row))) return false;
  return activeFieldFilters(filters, fields).every((active) =>
    active.kind === "range" ? passesRange(active.field, item, active.range) : passesPick(active.field, item, active.picks),
  );
}

// ---------- the chips over the table ----------

const fieldTitle = (field) => field.label || field.name;
const withUnit = (value, field) => (field.unit ? `${fmt(value)} ${field.unit}` : fmt(value));

function rangeText(field, { min, max }) {
  if (min === null) return `${fieldTitle(field)} ≤ ${withUnit(max, field)}`;
  if (max === null) return `${fieldTitle(field)} ≥ ${withUnit(min, field)}`;
  return `${fieldTitle(field)} ${fmt(min)}–${withUnit(max, field)}`;
}

// Every active filter as { key, text, without }: its chip's words and the filters once it is removed.
export function filterChips(filters, fields) {
  const chips = activeFieldFilters(filters, fields).map((active) => ({
    key: `f:${active.field.name}`,
    text:
      active.kind === "range"
        ? rangeText(active.field, active.range)
        : `${fieldTitle(active.field)}：${choicesOf(active.field)
            .filter((choice) => active.picks.has(choice.value))
            .map((choice) => choice.label)
            .join("、")}`,
    without: withField(filters, active.field.name, ""),
  }));
  if (filters.types.size) {
    const names = ARTICLE_TYPES.filter((type) => filters.types.has(type.value)).map((type) => type.label);
    chips.push({ key: "type", text: `文献类型：${names.join("、")}`, without: { ...filters, types: new Set() } });
  }
  for (const status of STATUSES) {
    if (filters[status.key]) chips.push({ key: status.key, text: status.label, without: { ...filters, [status.key]: false } });
  }
  return chips;
}

// ---------- the panel ----------

// The panel for the columns `fields` of the view on screen (grouped by the profile's groups), over its unfiltered rows: `samples` (the shown entity's
// sample items) and `papers` (one item per paper, for the paper-level fields and filters). `commit(filters)` writes the
// next filters.
export function filterPanel(fields, { samples, papers }, filters, commit) {
  const panel = document.createElement("aside");
  panel.className = "filter-panel";
  panel.id = "filter-panel";
  panel.setAttribute("aria-label", "筛选");
  panel.append(paperSection(papers, filters, commit));
  for (const { title, fields: members } of fieldSections(fields)) {
    // A field no row holds a value of offers nothing to filter by and is left out, unless a filter on it is set.
    const itemsOf = (field) => (field.scope === "paper" ? papers : samples);
    const shown = members.filter(
      (field) =>
        filterKind(field) !== null &&
        (filters.fields.has(field.name) || itemsOf(field).some((item) => itemValue(field, item) != null)),
    );
    if (!shown.length) continue;
    const section = groupSection(title);
    for (const field of shown) {
      const items = itemsOf(field);
      const kind = filterKind(field);
      if (kind === "range") section.append(rangeFilter(field, items, filters, commit));
      else if (kind === "pick") section.append(pickFilter(field, items, filters, commit));
      else section.append(unfilteredLine(field));
    }
    panel.append(section);
  }
  return panel;
}

// The view's fields grouped by the profile's groups, in the profile's order; a field of a group the page does not
// know (the profile's view has not loaded) goes under its group's own name.
function fieldSections(fields) {
  const groups = state.profile?.groups ?? [];
  const sections = groups.map((group) => ({ name: group.name, title: group.label_zh || group.name, fields: [] }));
  for (const field of fields) {
    let section = sections.find((candidate) => candidate.name === field.group);
    if (!section) {
      section = { name: field.group, title: field.group ?? "其他", fields: [] };
      sections.push(section);
    }
    section.fields.push(field);
  }
  return sections;
}

function groupSection(title) {
  const section = document.createElement("section");
  section.className = "filter-group";
  const heading = document.createElement("h4");
  heading.textContent = title;
  section.append(heading);
  return section;
}

// One checkbox: its label, how many rows hold it, and what ticking or unticking it commits.
function checkbox(focus, label, count, checked, onChange) {
  const wrap = document.createElement("label");
  wrap.className = "filter-check";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = checked;
  input.dataset.focus = focus;
  input.addEventListener("change", () => onChange(input.checked));
  const text = document.createElement("span");
  text.textContent = label;
  const n = document.createElement("span");
  n.className = "n";
  n.textContent = String(count);
  wrap.append(input, text, n);
  return wrap;
}

function fieldset(legend, title = "") {
  const box = document.createElement("fieldset");
  box.className = "filter";
  const head = document.createElement("legend");
  head.innerHTML = legend;
  if (title) box.title = title;
  box.append(head);
  return box;
}

// Paper level: the article type and the rail's status, counted over papers.
function paperSection(papers, filters, commit) {
  const section = groupSection("论文");
  const types = fieldset("文献类型");
  for (const type of ARTICLE_TYPES) {
    const count = papers.filter((item) => articleType(item.row) === type.value).length;
    types.append(
      checkbox(`type:${type.value}`, type.label, count, filters.types.has(type.value), (on) => {
        const next = new Set(filters.types);
        if (on) next.add(type.value);
        else next.delete(type.value);
        commit({ ...filters, types: next });
      }),
    );
  }
  const status = fieldset("状态", "按左侧文档库的处理状态");
  for (const entry of STATUSES) {
    const count = papers.filter((item) => entry.test(docOf(item.row))).length;
    status.append(checkbox(`status:${entry.key}`, entry.label, count, filters[entry.key], (on) => commit({ ...filters, [entry.key]: on })));
  }
  section.append(types, status);
  return section;
}

function legendFor(field) {
  const unit = field.unit ? ` <small>${escapeHtml(field.unit)}</small>` : "";
  return `${escapeHtml(fieldTitle(field))}${unit}`;
}

function pickFilter(field, items, filters, commit) {
  const box = fieldset(legendFor(field), field.description ?? "");
  const picks = new Set(splitPicks(filters.fields.get(field.name)));
  for (const choice of choicesOf(field)) {
    const count = items.filter((item) => passesPick(field, item, new Set([choice.value]))).length;
    box.append(
      checkbox(`pick:${field.name}:${choice.value}`, choice.label, count, picks.has(choice.value), (on) => {
        const next = new Set(picks);
        if (on) next.add(choice.value);
        else next.delete(choice.value);
        commit(withField(filters, field.name, [...next].join(PICK_SEPARATOR)));
      }),
    );
  }
  return box;
}

function unfilteredLine(field) {
  const line = document.createElement("p");
  line.className = "filter-none muted";
  line.title = UNFILTERED_KINDS[field.kind];
  line.textContent = `${fieldTitle(field)} · 不可筛选`;
  return line;
}

// The scale a range filter's slider and histogram run on: linear, or logarithmic when the values span more than three
// decades (and are all positive, which a logarithm needs).
function scaleOf(min, max) {
  const log = min > 0 && max / min > 1e3;
  const [low, high] = log ? [Math.log10(min), Math.log10(max)] : [min, max];
  const width = high - low || 1;
  return {
    log,
    toStep: (value) => Math.round((((log ? Math.log10(value) : value) - low) / width) * STEPS),
    fromStep: (step) => {
      const position = low + (width * step) / STEPS;
      return log ? 10 ** position : position;
    },
  };
}

// A slider's value, rounded to what a reader would type.
const rounded = (value) => Number(value.toPrecision(3));

function rangeFilter(field, items, filters, commit) {
  const box = fieldset(legendFor(field), field.description ?? "");
  box.classList.add("range-filter");
  box.dataset.field = field.name;
  const points = [];
  let blank = 0;
  for (const item of items) {
    const found = spans(field, item);
    if (!found.length) blank += 1;
    for (const { low, high } of found) {
      // A histogram point per element: an interval at its middle, or at its one closed end.
      if (Number.isFinite(low) && Number.isFinite(high)) points.push((low + high) / 2);
      else points.push(Number.isFinite(low) ? low : high);
    }
  }
  const range = parseRange(filters.fields.get(field.name));
  const note = document.createElement("p");
  note.className = "filter-note muted";
  note.textContent = blank ? `${blank} 个无确定值，不参与筛选` : "";
  if (!points.length) {
    const none = document.createElement("p");
    none.className = "filter-note muted";
    none.textContent = "没有可筛选的取值";
    box.append(none, note);
    return box;
  }
  const min = Math.min(...points);
  const max = Math.max(...points);
  if (min === max) {
    // One value across the corpus: nothing to slide over, but the boxes still take a range.
    const one = document.createElement("p");
    one.className = "filter-note muted";
    one.textContent = `取值都是 ${withUnit(min, field)}`;
    box.append(one);
  }
  const scale = scaleOf(min, max);
  if (scale.log) {
    const tag = document.createElement("small");
    tag.className = "scale-tag";
    tag.textContent = "对数刻度";
    box.querySelector("legend").append(" ", tag);
  }

  const histogram = document.createElement("div");
  histogram.className = "histogram";
  histogram.setAttribute("aria-hidden", "true");
  const counts = new Array(BINS).fill(0);
  for (const point of points) counts[binOf(scale.toStep(point))] += 1;
  const tallest = Math.max(...counts);
  const bars = counts.map((count) => {
    const bar = document.createElement("span");
    bar.className = "bar";
    bar.style.height = count ? `${Math.max(8, (count / tallest) * 100)}%` : "0";
    histogram.append(bar);
    return bar;
  });

  const [lowInput, highInput] = ["lo", "hi"].map((end) => {
    const input = document.createElement("input");
    input.type = "range";
    input.min = "0";
    input.max = String(STEPS);
    input.step = "1";
    input.dataset.focus = `range-${end}:${field.name}`;
    input.setAttribute("aria-label", `${fieldTitle(field)} ${end === "lo" ? "下限" : "上限"}`);
    return input;
  });
  lowInput.value = String(range?.min == null ? 0 : clampStep(scale.toStep(Math.max(range.min, min))));
  highInput.value = String(range?.max == null ? STEPS : clampStep(scale.toStep(Math.min(range.max, max))));
  const slider = document.createElement("div");
  slider.className = "dual-range";
  slider.append(lowInput, highInput);

  const [minText, maxText] = ["min", "max"].map((end) => {
    const input = document.createElement("input");
    input.type = "text";
    input.inputMode = "decimal";
    input.className = "range-num";
    input.dataset.focus = `${end}:${field.name}`;
    input.placeholder = fmt(end === "min" ? min : max);
    input.value = range?.[end] == null ? "" : String(range[end]);
    input.setAttribute("aria-label", `${fieldTitle(field)} ${end === "min" ? "最小值" : "最大值"}`);
    return input;
  });
  const inputs = document.createElement("div");
  inputs.className = "range-inputs";
  const dash = document.createElement("span");
  dash.textContent = "–";
  inputs.append(minText, dash, maxText);

  // What the slider shows while it is dragged: its ends in the text boxes and the bars inside them. Written to the
  // query only once released, so a drag is one query write and the table is not redrawn under the pointer.
  const paint = () => {
    const [low, high] = orderedSteps(lowInput, highInput).map(binOf);
    bars.forEach((bar, bin) => bar.classList.toggle("in", bin >= low && bin <= high));
  };
  const slide = () => {
    const [low, high] = orderedSteps(lowInput, highInput);
    minText.value = low <= 0 ? "" : String(rounded(scale.fromStep(low)));
    maxText.value = high >= STEPS ? "" : String(rounded(scale.fromStep(high)));
    paint();
  };
  const commitText = () => {
    const typed = parseRange(`${minText.value.trim()}~${maxText.value.trim()}`);
    const blankBoth = !minText.value.trim() && !maxText.value.trim();
    if (!typed && !blankBoth) return; // not a number yet: left as typed, nothing committed
    commit(withField(filters, field.name, typed ? rangeSpec(typed.min, typed.max) : ""));
  };
  for (const input of [lowInput, highInput]) {
    input.addEventListener("input", slide);
    input.addEventListener("change", () => {
      slide();
      commitText();
    });
  }
  for (const input of [minText, maxText]) {
    input.addEventListener("change", commitText);
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") commitText();
    });
  }
  paint();
  box.append(...(min === max ? [] : [histogram, slider]), inputs, note);
  return box;
}

const clampStep = (step) => Math.min(STEPS, Math.max(0, step));
// The histogram bar a slider step falls in (the last bar holds the largest value).
const binOf = (step) => Math.min(BINS - 1, Math.floor((clampStep(step) / STEPS) * BINS));
const orderedSteps = (a, b) => [Number(a.value), Number(b.value)].sort((x, y) => x - y);
