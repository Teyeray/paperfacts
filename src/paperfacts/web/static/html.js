// Small utilities: HTML escaping, number formatting for normalized values, and the bottom-right toast.
// Any dynamic text spliced into innerHTML (filenames, raw extracted values, model-provided notes) must go through escapeHtml first.

export function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// Very large / very small numbers use scientific notation; everything else keeps 6 significant figures with trailing zeros stripped
export const fmt = (n) => (Math.abs(n) >= 1e5 || (Math.abs(n) < 1e-3 && n !== 0) ? n.toExponential(3) : Number(n.toPrecision(6)).toString());

// Two caveats worth showing next to a value: it could not be located in the block it cites, and repeated
// extraction passes disagreed about it. Both are cheap to compute and useless if nobody sees them.
export function caveats(field) {
  const marks = [];
  if (field.grounded === false) marks.push(`<span class="flag bad" title="这个取值没能在它引用的来源块里找到">引用未核实</span>`);
  if (field.agreement != null && field.agreement < 1) {
    marks.push(`<span class="flag warn" title="重复抽取时只有部分轮次给出了这个取值">一致度 ${Math.round(field.agreement * 100)}%</span>`);
  }
  return marks.join("");
}

export function toast(message, isError = false) {
  const div = document.createElement("div");
  div.className = "toast" + (isError ? " error" : "");
  div.textContent = message;
  document.body.append(div);
  setTimeout(() => div.remove(), isError ? 6000 : 3000);
}
