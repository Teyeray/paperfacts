// The document page's middle column is four tabs: the results table first, the rest on demand. A tab is pure
// UI state (state.tab): it never enters the URL, whose hash holds the selected fact (#/doc/<id>/fact/<n>) --
// a section in the hash would need its own router grammar and could clear that fact. Switching documents or
// profiles resets it (document.js's switching block); a job-finish reload of the same document keeps it.

import { state, uiCopy } from "./state.js";

export const TABS = [
  { key: "results", label: () => "结果表" },
  { key: "figures", label: () => "图中读数" },
  { key: "facts", label: () => "事实对照" },
  { key: "samples", label: () => `${uiCopy("entity_label_zh")}记录` },
];

// Shows the tab's panel and marks its button; an unknown key (or none) falls back to the results table.
// Sticky-preservation rule: at >=961px the tab bar is position:sticky under the topbar. Switching to a much
// shorter panel can shrink the document so far that the browser clamps the scroll position below the bar's
// natural offset, un-pinning it -- the bar visibly jumps down. When the bar was pinned before the swap, pin
// it again after (scrolling so its top sits at the topbar edge); if the document is now too short to scroll
// that far, the browser clamps and the bar rests naturally, which is correct.
export function activateTab(key) {
  if (!TABS.some((tab) => tab.key === key)) key = "results";
  state.tab = key;
  const view = document.getElementById("document-view");
  const bar = view.querySelector('[data-slot="tabs"]');
  const wide = matchMedia("(min-width: 961px)").matches;
  const topbar = 56;
  const wasPinned = wide && bar != null && bar.getBoundingClientRect().top <= topbar + 1;
  for (const tab of TABS) {
    const selected = tab.key === key;
    const button = view.querySelector(`[data-slot="tabs"] [data-tab="${tab.key}"]`);
    button?.setAttribute("aria-selected", String(selected));
    button?.setAttribute("tabindex", selected ? "0" : "-1"); // roving tabindex: only the selected tab is in the tab order
    view.querySelector(`[data-panel="${tab.key}"]`)?.toggleAttribute("hidden", !selected);
  }
  if (wasPinned && bar && bar.getBoundingClientRect().top > topbar) {
    const barDocTop = bar.getBoundingClientRect().top + window.scrollY;
    window.scrollTo(0, Math.max(0, barDocTop - topbar));
  }
}

// One role=tab button per panel, with aria-controls, a roving tabindex and Arrow/Home/End to move both
// focus and selection (the tab panels themselves are eager, static template nodes: switching only hides).
export function renderTabBar(slotNode) {
  slotNode.replaceChildren();
  for (const tab of TABS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tab";
    button.dataset.tab = tab.key;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-controls", `panel-${tab.key}`);
    button.textContent = tab.label();
    button.addEventListener("click", () => activateTab(tab.key));
    slotNode.append(button);
  }
  slotNode.addEventListener("keydown", (event) => {
    const index = TABS.findIndex((tab) => tab.key === state.tab);
    const step = { ArrowRight: 1, ArrowLeft: -1, Home: -Infinity, End: Infinity }[event.key];
    if (step === undefined) return;
    event.preventDefault();
    const next = step === Infinity ? TABS.length - 1 : step === -Infinity ? 0 : (index + step + TABS.length) % TABS.length;
    activateTab(TABS[next].key);
    slotNode.children[next]?.focus();
  });
  activateTab(state.tab ?? "results");
}
