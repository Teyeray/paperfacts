// Which field columns the reader wants, and which of those are worth showing. Both results tables --
// the per-sample table inside a document and the corpus table on the home view -- share this module,
// so a column hidden on one stays hidden on the other.

import { uiCopy } from "./state.js";

// A field earns a column when at least one row put a value in it; the toggle brings the rest back so the
// full table stays inspectable without making the default view mostly blank. Shared with the corpus table,
// which applies the same rule across papers instead of across samples.
export function visibleFields(fields, rows, showAll) {
  const all = fields ?? [];
  if (showAll) return all;
  const present = (rows ?? []).filter(Boolean);
  return all.filter((field) => present.some((row) => row[field.name] != null));
}

// ---------- field column picker ----------
//
// Which field columns the reader wants at all, independent of whether they happen to be empty. The
// choice is one per browser and shared by both results tables, so a column hidden on the home table
// stays hidden inside a document. Storage may be unavailable (private mode, blocked site data); every
// failure degrades to "show every field", never to an empty table.

const PICKER_KEY = "paperfacts.chosen-fields";
// Fields carry `scope`, not the config's richer `group`, so the picker groups by the distinction the
// dataset actually exposes: what belongs to the paper and what belongs to a sample.
const SCOPE_LABEL = { paper: () => uiCopy("paper_level_label_zh"), sample: () => `${uiCopy("entity_label_zh")}级` };
const SCOPE_ORDER = ["paper", "sample"];

// null means "no choice stored" -- every field is shown, including ones added after the last choice.
function readChosen() {
  try {
    const raw = window.localStorage.getItem(PICKER_KEY);
    if (!raw) return null;
    const names = JSON.parse(raw);
    return Array.isArray(names) ? new Set(names.map(String)) : null;
  } catch {
    return null;
  }
}

function writeChosen(names) {
  try {
    if (names === null) window.localStorage.removeItem(PICKER_KEY);
    else window.localStorage.setItem(PICKER_KEY, JSON.stringify([...names]));
  } catch {
    // A browser that refuses storage still gets a working picker for this page view.
  }
}

// The chosen subset of the field list, in the dataset's own column order.
export function chosenFields(fields) {
  const all = fields ?? [];
  const chosen = readChosen();
  if (chosen === null) return all;
  const kept = all.filter((field) => chosen.has(field.name));
  // A stored choice that matches nothing is a choice made before the fields were renamed, not a request
  // for an empty table; that request is an empty set, which is honoured. Fall back to showing everything.
  if (!kept.length && chosen.size) {
    writeChosen(null);
    return all;
  }
  return kept;
}

// Re-rendering the table rebuilds the chip, so the open/closed state lives outside the element.
let pickerOpen = false;

export function fieldPicker(fields, onChange) {
  const all = fields ?? [];
  const chosen = readChosen();
  const isOn = (field) => chosen === null || chosen.has(field.name);

  const wrap = document.createElement("div");
  wrap.className = "field-picker";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "chip" + (chosen === null ? "" : " on");
  // `data-focus` keys: every control here rebuilds the table it sits in, and the caller's keepFocus puts
  // the keyboard focus back on the control with the same key.
  button.dataset.focus = "picker";
  button.setAttribute("aria-expanded", "false");
  button.setAttribute("aria-haspopup", "dialog");
  button.innerHTML = `选择字段<span class="n">${all.filter(isOn).length}/${all.length}</span>`;

  // Hiding every column is a legitimate request, but a table with only its identity columns looks broken.
  // Say why it is empty next to the control that caused it, so the way back is one click away.
  const note = document.createElement("span");
  note.className = "picker-note";
  note.hidden = all.length === 0 || all.some(isOn);
  note.textContent = "已隐藏全部字段列";

  const pop = document.createElement("div");
  pop.className = "picker-pop";
  pop.setAttribute("role", "dialog");
  pop.setAttribute("aria-label", "选择要显示的字段");
  pop.hidden = true;

  // Both links write an explicit set: "全选" stores every current name rather than clearing the key, so
  // the choice stays what the reader saw. Only unchecking everything leaves an empty table by request.
  const actions = document.createElement("div");
  actions.className = "picker-actions";
  actions.append(
    linkButton("全选", "pick-all", () => commit(new Set(all.map((field) => field.name)))),
    linkButton("清空", "pick-none", () => commit(new Set())),
  );
  pop.append(actions);

  for (const scope of SCOPE_ORDER) {
    const group = all.filter((field) => (field.scope ?? "sample") === scope);
    if (!group.length) continue;
    const box = document.createElement("div");
    box.className = "picker-group";
    const title = document.createElement("h4");
    title.textContent = SCOPE_LABEL[scope]();
    box.append(title);
    for (const field of group) box.append(checkbox(field, isOn(field)));
    pop.append(box);
  }

  function checkbox(field, on) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = on;
    input.value = field.name;
    input.dataset.focus = `field:${field.name}`;
    input.addEventListener("change", () => {
      const boxes = [...pop.querySelectorAll("input[type=checkbox]")];
      commit(new Set(boxes.filter((box) => box.checked).map((box) => box.value)));
    });
    const text = document.createElement("span");
    const name = field.label ? `${field.label} ${field.name}` : field.name;
    text.textContent = field.unit ? `${name}（${field.unit}）` : name;
    label.append(input, text);
    return label;
  }

  function commit(names) {
    writeChosen(names);
    // onChange discards this element; its document-level listeners would otherwise outlive it.
    detach();
    onChange();
  }

  const onKey = (event) => {
    if (event.key === "Escape") {
      close();
      button.focus();
    }
  };
  const onOutside = (event) => {
    if (!wrap.contains(event.target)) close();
  };

  function open(focusFirst) {
    pop.hidden = false;
    pickerOpen = true;
    button.setAttribute("aria-expanded", "true");
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onOutside);
    if (focusFirst) pop.querySelector("input[type=checkbox]")?.focus();
  }

  function detach() {
    document.removeEventListener("keydown", onKey);
    document.removeEventListener("pointerdown", onOutside);
  }

  function close() {
    pop.hidden = true;
    pickerOpen = false;
    button.setAttribute("aria-expanded", "false");
    detach();
  }

  button.addEventListener("click", () => (pop.hidden ? open(true) : close()));
  wrap.append(button, note, pop);
  // A checkbox re-renders the table, which replaces this chip; the popover stays where the reader left it.
  if (pickerOpen) open(false);
  return wrap;
}

function linkButton(text, focusKey, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "picker-link";
  button.dataset.focus = focusKey;
  button.textContent = text;
  button.addEventListener("click", onClick);
  return button;
}

export function toggleChip(fields, showAll, onToggle) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "chip" + (showAll ? " on" : "");
  button.dataset.focus = "show-empty";
  button.setAttribute("aria-pressed", String(showAll));
  button.innerHTML = `显示空字段<span class="n">${(fields ?? []).length}</span>`;
  button.addEventListener("click", onToggle);
  return button;
}
