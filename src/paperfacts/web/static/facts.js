// Fact comparison: KPI summary tiles, status filters, the row-by-row comparison table, and
// selecting a row to highlight both lanes' source blocks in the viewer.

import { caveats, escapeHtml, fmt, toast } from "./html.js";
import { documentHash } from "./router.js";
import { LANE_LABEL, STATUS_LABEL, STATUS_NOTE, VERDICT_CLASS, VERDICT_LABEL, slot, state } from "./state.js";

export function renderKpis(root) {
  root.innerHTML = "";
  const counts = state.report?.counts;
  if (!counts) { root.innerHTML = `<div class="muted">尚无比较结果。</div>`; return; }
  const missingNote = Object.entries(counts.missing_by_backend ?? {}).map(([b, n]) => `${LANE_LABEL[b] ?? b} 缺 ${n}`).join(" · ") || STATUS_NOTE.missing;
  const matchNote = `${counts.samples_unmatched} 未配对${counts.low_confidence_matches ? ` · ${counts.low_confidence_matches} 低置信度` : ""}${counts.matching_failed ? " · 匹配失败" : ""}`;
  const tiles = [
    ["agree", "一致", counts.agree, STATUS_NOTE.agree],
    ["conflict", "冲突", counts.conflict, STATUS_NOTE.conflict],
    ["ambiguous", "不确定", counts.ambiguous, STATUS_NOTE.ambiguous],
    ["missing", "单路缺失", counts.missing, missingNote],
    ["samples", "样品配对", counts.samples_matched, matchNote],
  ];
  // Only worth a tile when it happened: a lane that placed every value has nothing to report here.
  const unplaced = Object.entries(counts.unattributed_by_backend ?? {});
  if (unplaced.length) {
    const total = unplaced.reduce((sum, [, n]) => sum + n, 0);
    tiles.push(["unattributed", "未归属", total, unplaced.map(([b, n]) => `${LANE_LABEL[b] ?? b} ${n}`).join(" · ")]);
  }
  // The VLM's tally, when the stage ran: how many disputed readings the page confirmed or denied.
  const v = state.validation?.counts;
  if (v) {
    const note = `否定 ${v.contradicted} · 不可读 ${v.illegible} · 未核验 ${v.not_checked}${v.error ? ` · 失败 ${v.error}` : ""}`;
    tiles.push(["vlm", `视觉确认 / ${v.total}`, v.confirmed, note]);
  }
  for (const [cls, label, value, note] of tiles) {
    const div = document.createElement("div");
    div.className = `kpi ${cls}`;
    div.innerHTML = `<div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(value)}</div><div class="note">${escapeHtml(note)}</div>`;
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
    b.setAttribute("aria-pressed", String(state.filter === key));
    b.innerHTML = `${escapeHtml(label)}<span class="n">${n}</span>`;
    b.addEventListener("click", () => applyFilter(key));
    return b;
  };
  root.append(chip(null, "全部", rows.length));
  for (const status of ["agree", "conflict", "ambiguous", "missing"]) if (counts[status]) root.append(chip(status, STATUS_LABEL[status], counts[status]));
}

// Filtering only redraws the filter chips and the table rows; the viewer, sample records, and progress bar are untouched
export function applyFilter(key) {
  state.filter = state.filter === key ? null : key;
  renderFilters(slot("filters"));
  renderRows(slot("rows"), slot("rows-empty"));
}

export function renderRows(tbody, emptyNode) {
  tbody.innerHTML = "";
  const all = state.report?.comparisons ?? [];
  emptyNode.classList.toggle("hidden", Boolean(state.report));
  for (const [index, c] of all.entries()) {
    if (state.filter && c.status !== state.filter) continue;
    const tr = document.createElement("tr");
    tr.dataset.index = String(index);
    const statusText = c.status === "missing" && c.missing_in ? `MISSING · ${LANE_LABEL[c.missing_in]} 缺` : STATUS_LABEL[c.status];
    const conf = c.match_confidence != null ? `<div class="conf">配对置信度 ${c.match_confidence.toFixed(2)}</div>` : "";
    tr.innerHTML = `
      <td><span class="status ${c.status}">${escapeHtml(statusText)}</span>${conf}</td>
      <td class="mono">${escapeHtml(scopeLabel(c.scope))}</td>
      <td><b>${escapeHtml(c.field)}</b></td>
      <td class="muted">${escapeHtml(c.condition ?? "")}</td>
      <td class="val">${valueCell(c.a, verdictFor(state.report.backend_a, ownerOf(c.scope, "a"), c.a))}</td>
      <td class="val">${valueCell(c.b, verdictFor(state.report.backend_b, ownerOf(c.scope, "b"), c.b))}</td>
      <td class="detail">${escapeHtml(c.detail)}</td>`;
    tr.addEventListener("click", () => selectRow(tr, index));
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

// The owner a value has on one side of a comparison row, spelled as validate.value_key spells it:
// "target" and "unattributed" as they are, "sample:<a>|<b>" split into that side's own sample id.
function ownerOf(scope, side) {
  if (scope === "target" || scope === "unattributed") return scope;
  const m = scope.match(/^sample:(.*)$/);
  if (!m) return scope;
  const [a, b] = m[1].split("|");
  return `sample:${side === "a" ? a : (b ?? a)}`;
}

// Rebuilds validate.value_key for one side of a row: backend|owner|field|value_raw|condition|source_ids.
// The field order is the contract with the backend; keep the two in step.
function verdictFor(backend, owner, field) {
  const values = state.validation?.values;
  if (!values || !field) return null;
  const key = [backend, owner, field.field, field.value_raw, field.condition ?? "", (field.source_ids ?? []).join(",")].join("|");
  return values.find((value) => value.key === key) ?? null;
}

function verdictBadge(verdict) {
  if (!verdict) return "";
  const read = verdict.transcription ? `VLM 读到：${verdict.transcription.slice(0, 300)}` : verdict.detail;
  return `<span class="flag ${VERDICT_CLASS[verdict.verdict] ?? ""}" title="${escapeHtml(read)}">${escapeHtml(VERDICT_LABEL[verdict.verdict] ?? verdict.verdict)}</span>`;
}

function valueCell(field, verdict = null) {
  if (!field) return `<span class="muted">—</span>`;
  const raw = `${field.value_raw} ${field.unit_raw ?? ""}`.trim();
  const norm = field.value != null ? `= ${fmt(field.value)} ${field.unit ?? ""}` : (field.normalization_note ? `(${field.normalization_note})` : "");
  return `${escapeHtml(raw)}<small>${escapeHtml(norm)}</small>${caveats(field)}${verdictBadge(verdict)}`;
}

function selectRow(tr, index) {
  const comparison = state.report.comparisons[index];
  for (const row of tr.parentElement.querySelectorAll("tr.selected")) row.classList.remove("selected");
  tr.classList.add("selected");
  const ids = [...(comparison.a?.source_ids ?? []), ...(comparison.b?.source_ids ?? [])];
  state.viewer?.highlight(ids);
  if (!ids.length) toast("这条事实没有 source_id（模型没有引用来源块）");
  // the selected fact goes into the URL (without adding a history entry); refreshing or sharing the link returns to this same row
  history.replaceState(null, "", documentHash(state.current, index));
}

export function selectRowByIndex(index) {
  if (index == null) return;
  const tr = slot("rows")?.querySelector(`tr[data-index="${index}"]`);
  if (!tr) return;
  selectRow(tr, index);
  tr.scrollIntoView({ block: "nearest" });
}
