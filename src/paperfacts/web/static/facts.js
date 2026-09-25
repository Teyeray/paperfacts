// Fact comparison: KPI summary tiles, status filters, the row-by-row comparison table, and
// selecting a row to highlight both lanes' source blocks in the viewer.
//
// The selected fact is one value, `state.selectedFact`, mirrored in the URL: a filter that hides its row does
// not unselect it, and anything else that takes over the viewer releases it (and the URL) explicitly.

import { caveats, escapeHtml, fmt, keepFocus, onActivate, toast } from "./html.js";
import { documentHash } from "./router.js";
import { LANES, LANE_LABEL, STATUS, STATUS_ORDER, noSamplesReason, slot, state } from "./state.js";
import { revealViewer } from "./viewer.js";

export function renderKpis(root) {
  root.innerHTML = "";
  const counts = state.report?.counts;
  if (!counts) { root.innerHTML = `<div class="muted">尚无比较结果。</div>`; return; }
  const missingNote = Object.entries(counts.missing_by_backend ?? {}).map(([b, n]) => `${LANE_LABEL[b] ?? b} 缺 ${n}`).join(" · ") || STATUS.missing.note;
  const matchNote = `${counts.samples_unmatched} 未配对${counts.low_confidence_matches ? ` · ${counts.low_confidence_matches} 低置信度` : ""}${counts.matching_failed ? " · 匹配失败" : ""}`;
  const tiles = [
    ...STATUS_ORDER.map((status) => [status, STATUS[status].label, counts[status], status === "missing" ? missingNote : STATUS[status].note]),
    ["samples", "样品配对", counts.samples_matched, matchNote],
  ];
  // Only worth a tile when it happened: a lane that placed every value has nothing to report here.
  const unplaced = Object.entries(counts.unattributed_by_backend ?? {});
  if (unplaced.length) {
    const total = unplaced.reduce((sum, [, n]) => sum + n, 0);
    tiles.push(["unattributed", "未归属", total, unplaced.map(([b, n]) => `${LANE_LABEL[b] ?? b} ${n}`).join(" · ")]);
  }
  for (const [cls, label, value, note] of tiles) {
    const div = document.createElement("div");
    div.className = `kpi ${cls}`;
    div.innerHTML = `<div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(value)}</div><div class="note" title="${escapeHtml(note)}">${escapeHtml(note)}</div>`;
    root.append(div);
  }
}

export function renderFilters(root) {
  root.innerHTML = "";
  const rows = state.report?.comparisons ?? [];
  if (!rows.length) return;
  const counts = {};
  for (const c of rows) counts[c.status] = (counts[c.status] ?? 0) + 1;
  const chip = (key, label, n) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip" + (state.filter === key ? " on" : "");
    b.dataset.focus = `filter:${key ?? "all"}`;
    b.setAttribute("aria-pressed", String(state.filter === key));
    b.innerHTML = `${escapeHtml(label)}<span class="n">${n}</span>`;
    b.addEventListener("click", () => applyFilter(key));
    return b;
  };
  root.append(chip(null, "全部", rows.length));
  for (const status of STATUS_ORDER) if (counts[status]) root.append(chip(status, STATUS[status].label, counts[status]));
}

// Filtering only redraws the filter chips and the table rows; the viewer, sample records, and progress bar are untouched
export function applyFilter(key) {
  state.filter = state.filter === key ? null : key;
  const filters = slot("filters");
  keepFocus(filters, () => renderFilters(filters));
  renderRows(slot("rows"), slot("rows-empty"));
}

export function renderRows(tbody, emptyNode) {
  tbody.innerHTML = "";
  const all = state.report?.comparisons ?? [];
  const hasSamples = LANES.some((lane) => state.lanes[lane]?.samples?.length);
  emptyNode.textContent = !state.report
    ? "还没有比较报告。"
    : hasSamples ? "两路都没有抽到可以比较的取值。" : noSamplesReason();
  emptyNode.classList.toggle("hidden", all.length > 0);
  for (const [index, c] of all.entries()) {
    if (state.filter && c.status !== state.filter) continue;
    const tr = document.createElement("tr");
    tr.dataset.index = String(index);
    tr.tabIndex = 0;
    tr.setAttribute("aria-selected", String(index === state.selectedFact));
    if (index === state.selectedFact) tr.classList.add("selected");
    const label = STATUS[c.status]?.label ?? c.status;
    const missing = c.status === "missing" && c.missing_in ? `<small>${escapeHtml(LANE_LABEL[c.missing_in] ?? c.missing_in)} 缺</small>` : "";
    const conf = c.match_confidence != null ? `<div class="conf">配对置信度 ${c.match_confidence.toFixed(2)}</div>` : "";
    const condition = String(c.condition ?? "");
    const detail = String(c.detail ?? "");
    tr.innerHTML = `
      <td><span class="status ${escapeHtml(c.status)}">${escapeHtml(label)}</span>${missing}${conf}</td>
      <td class="mono">${escapeHtml(scopeLabel(c.scope))}</td>
      <td class="field"><b>${escapeHtml(c.field)}</b></td>
      <td class="cond muted"><div class="clamp" title="${escapeHtml(condition)}">${escapeHtml(condition)}</div></td>
      <td class="val">${valueCell(c.a)}</td>
      <td class="val">${valueCell(c.b)}</td>
      <td class="detail"><div class="clamp" title="${escapeHtml(detail)}">${escapeHtml(detail)}</div></td>`;
    onActivate(tr, () => selectFact(index, { reveal: true }));
    tbody.append(tr);
  }
}

// scope is "target", "sample:<a>|<b>" (each lane's own sample_id) or "unattributed" (both lanes
// extracted the value but neither could place it on a sample)
function scopeLabel(scope) {
  if (scope === "target") return "靶材（论文级）";
  if (scope === "unattributed") return "未归属（两路均未对应到样品）";
  const m = scope.match(/^sample:(.*)$/);
  if (!m) return scope;
  const [a, b] = m[1].split("|");
  return b && b !== a ? `${a} ↔ ${b}` : a;
}

function valueCell(field) {
  if (!field) return `<span class="muted">—</span>`;
  const raw = `${field.value_raw} ${field.unit_raw ?? ""}`.trim();
  const norm = field.value != null ? `= ${fmt(field.value)} ${field.unit ?? ""}` : (field.normalization_note ? `(${field.normalization_note})` : "");
  return `${escapeHtml(raw)}<small>${escapeHtml(norm)}</small>${caveats(field)}`;
}

function markRows() {
  for (const row of slot("rows")?.querySelectorAll("tr[data-index]") ?? []) {
    const selected = Number(row.dataset.index) === state.selectedFact;
    row.classList.toggle("selected", selected);
    row.setAttribute("aria-selected", String(selected));
  }
}

// A reader's click (or Enter): highlight both lanes' blocks, put the fact in the URL (without a history
// entry), and bring the viewer on screen when it sits below the fold.
function selectFact(index, { reveal = false } = {}) {
  const comparison = state.report?.comparisons?.[index];
  if (!comparison) return;
  state.selectedFact = index;
  markRows();
  const ids = [...(comparison.a?.source_ids ?? []), ...(comparison.b?.source_ids ?? [])];
  state.viewer?.highlight(ids);
  history.replaceState(null, "", documentHash(state.current, index));
  if (!reveal) return;
  if (!ids.length) toast("这条事实没有 source_id（模型没有引用来源块）");
  revealViewer();
}

// Something else took over the viewer (a result cell, a chart reading, a record's source): the fact is no
// longer what it shows, so it leaves the selection and the URL.
export function releaseFact() {
  if (state.selectedFact == null) return;
  state.selectedFact = null;
  markRows();
  if (state.current) history.replaceState(null, "", documentHash(state.current));
}

// The router's entry point, after a load or on Back/Forward: the URL is the truth. No fact, or one that does
// not exist, clears the selection and its highlight; an unknown index is also dropped from the URL.
export function selectRowByIndex(index) {
  const comparisons = state.report?.comparisons ?? [];
  if (index == null || !comparisons[index]) {
    const hadFact = state.selectedFact != null;
    state.selectedFact = null;
    markRows();
    if (hadFact) state.viewer?.highlight([], { jump: false });
    if (index != null && state.current) history.replaceState(null, "", documentHash(state.current));
    return;
  }
  // A shared link must show its row, whatever filter this page happened to have on.
  if (state.filter && comparisons[index].status !== state.filter) {
    state.filter = null;
    renderFilters(slot("filters"));
    renderRows(slot("rows"), slot("rows-empty"));
  }
  selectFact(index);
  slot("rows")?.querySelector(`tr[data-index="${index}"]`)?.scrollIntoView({ block: "nearest" });
}
