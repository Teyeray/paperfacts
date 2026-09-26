// Small utilities: HTML escaping, number formatting for normalized values, focus keeping, and the toast.
// Any dynamic text spliced into innerHTML (filenames, raw extracted values, model-provided notes) must go through escapeHtml first.

export function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// An element built from text only: `text` and `title` land through textContent and the title property, children are
// nodes or strings (appended as text), so untrusted text never reaches the parser.
export function el(tag, { className = "", text = null, title = null } = {}, ...children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = String(text);
  if (title) node.title = String(title);
  node.append(...children);
  return node;
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

// Re-render `root` without dropping the keyboard focus. A control that re-renders its own table replaces
// itself, and the browser then puts focus on <body>: a keyboard user would have to tab back from the top of
// the page after every toggle. Controls that should survive carry a stable `data-focus` key.
export function keepFocus(root, render) {
  const active = document.activeElement;
  const key = active && root.contains(active) ? active.dataset?.focus : null;
  render();
  if (key) root.querySelector(`[data-focus="${CSS.escape(key)}"]`)?.focus();
}

// Enter and Space act on a focusable row or cell the way a click does.
export function onActivate(element, action) {
  element.addEventListener("click", action);
  element.addEventListener("keydown", (event) => {
    if (event.target !== element || (event.key !== "Enter" && event.key !== " ")) return;
    event.preventDefault();
    action();
  });
}

// Arrowing through a closed <select> fires a change per key. `action` runs at once for a pick with the mouse; a change
// made from the keyboard waits until the reader settles on an option, presses Enter or leaves the select.
const SETTLE_MS = 400;

export function onSettledChange(select, action) {
  let keyed = false;
  let timer = null;
  const run = () => {
    clearTimeout(timer);
    timer = null;
    action();
  };
  select.addEventListener("pointerdown", () => {
    keyed = false;
  });
  select.addEventListener("keydown", (event) => {
    keyed = true;
    if (event.key === "Enter" && timer) run();
  });
  select.addEventListener("blur", () => {
    if (timer) run();
  });
  select.addEventListener("change", () => {
    const fromKeyboard = keyed;
    keyed = false;
    if (!fromKeyboard) {
      run();
      return;
    }
    clearTimeout(timer);
    timer = setTimeout(run, SETTLE_MS);
  });
}

// Toasts live in one polite live region, so a screen reader hears the errors a sighted reader sees.
function toastRegion() {
  let region = document.getElementById("toasts");
  if (!region) {
    region = document.createElement("div");
    region.id = "toasts";
    region.className = "toasts";
    region.setAttribute("role", "status");
    region.setAttribute("aria-live", "polite");
    document.body.append(region);
  }
  return region;
}

export function toast(message, isError = false) {
  const div = document.createElement("div");
  div.className = "toast" + (isError ? " error" : "");
  div.textContent = message;
  toastRegion().append(div);
  setTimeout(() => div.remove(), isError ? 6000 : 3000);
}
