// Chart readings: values the vision model read off property-vs-condition charts (the opt-in figures stage).
// They are approximate, never compared across lanes and never put in the results table, so they get a
// section of their own in a neutral colour; blue and orange belong to the two lanes, and a chart belongs to
// neither.

import { escapeHtml, fmt, toast } from "./html.js";
import { LANES, slot, state } from "./state.js";

export function renderFigures(root) {
  const rows = state.figures?.rows ?? [];
  root.classList.toggle("hidden", !rows.length);
  const warn = slot("figures-warning", root);
  const notes = [];
  if (state.figures?.stale) notes.push("这些读数是在旧设置下读出的（模型、提示或字段表已变），重新读图前仅供参考。");
  if (state.figures?.orphaned?.length) notes.push(`${state.figures.orphaned.length} 个图块在当前解析里已不存在，无法在页面上定位。`);
  warn.textContent = notes.join(" ");
  warn.classList.toggle("hidden", !notes.length);
  const body = slot("figure-rows", root);
  body.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    const value = row.value != null
      ? `${fmt(row.value)} <span class="unit">${escapeHtml(row.unit ?? "")}</span>`
      : `<span class="muted">无法换算</span>`;
    tr.innerHTML = `<td>${escapeHtml(row.figure ?? "图")}<small>第 ${escapeHtml(row.page)} 页 · 子图 ${escapeHtml(row.panel)}</small></td>`
      + `<td>${escapeHtml(row.field)}</td><td>${escapeHtml(row.series ?? "")}</td><td class="x">${escapeHtml(row.x ?? "")}</td>`
      + `<td class="val">${value}<span class="flag approx" title="从图上读出，不是论文写出的数">近似值 ${escapeHtml(row.precision)}</span></td>`
      + `<td class="mono">${escapeHtml(row.value_raw ?? "")}</td><td class="note">${escapeHtml(row.detail ?? "")}</td>`;
    tr.title = row.caption ?? "";
    tr.addEventListener("click", () => locate(row));
    body.append(tr);
  }
}

// The row cites the figure block; its page and box come from the artifact the page already loaded.
function locate(row) {
  const lane = LANES.find((l) => row.source_id.startsWith(`${l}_`));
  const block = state.artifacts[lane]?.blocks?.find((b) => b.source_id === row.source_id);
  if (!block || !state.viewer) {
    toast("找不到这张图在页面上的位置", true);
    return;
  }
  state.viewer.showRegion({ page: block.page, bbox: block.bbox, label: row.figure ?? row.source_id });
  slot("viewer")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
