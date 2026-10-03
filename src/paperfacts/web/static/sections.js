// The document page's section bar: one button per section, the section being read marked. A click scrolls; nothing
// here touches the URL, whose hash holds the selected fact (#/doc/<id>/fact/<n>) -- a section in the hash would need
// its own router grammar and could clear that fact.

import { uiCopy } from "./state.js";

const SECTIONS = [
  ["results", () => "结果"],
  ["figures", () => "图中读数"],
  ["evidence", () => "证据"],
  ["samples", () => `${uiCopy("entity_label_zh")}与通道`],
  ["log", () => "日志"],
];

// The bar on screen: its buttons by section, the observers that follow the scroll, and the section last clicked (see
// pick). Replaced whole by every render, so a redrawn page never keeps observing the nodes it threw away.
let bar = null;

export function renderSectionBar(root, page) {
  stopSectionBar();
  root.replaceChildren();
  const entries = [];
  for (const [key, label] of SECTIONS) {
    const section = page.querySelector(`[data-section="${key}"]`);
    if (!section) continue;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "section-link";
    button.dataset.goto = key;
    button.textContent = label();
    button.addEventListener("click", () => goTo(key));
    root.append(button);
    entries.push({ key, section, button });
  }
  bar = { root, entries, clicked: null, observers: [], onLine: new Set(), atEnd: false };
  syncSectionBar();
  // What the observers last reported is kept and read, never the geometry at callback time: a smooth scroll goes on
  // moving after the last crossing it causes, and a reading taken then would be stale once it stops.
  const line = readingLine();
  const crossing = new IntersectionObserver(
    (changes) => {
      for (const { isIntersecting, target: section } of changes) bar?.onLine[isIntersecting ? "add" : "delete"](section);
      mark();
    },
    { rootMargin: `-${line}px 0px -${Math.max(window.innerHeight - line - 1, 0)}px 0px` },
  );
  const end = new IntersectionObserver((changes) => {
    if (bar) bar.atEnd = changes.at(-1).isIntersecting;
    mark();
  });
  for (const { section } of entries) crossing.observe(section);
  end.observe(page.querySelector("[data-section-end]") ?? page);
  bar.observers.push(crossing, end);
}

export function stopSectionBar() {
  for (const observer of bar?.observers ?? []) observer.disconnect();
  bar = null;
}

// A section that is not drawn (the log before any job) has no button either.
export function syncSectionBar() {
  for (const { section, button } of bar?.entries ?? []) button.hidden = section.classList.contains("hidden");
  mark();
}

function goTo(key) {
  const entry = bar?.entries.find((item) => item.key === key);
  if (!entry) return;
  bar.clicked = key;
  if (entry.section.tagName === "DETAILS") entry.section.open = true; // a folded section is opened to be read
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  entry.section.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  mark();
}

// Where a section counts as being read: just under the sticky topbar and, where it sticks (wide screens), the bar
// itself -- measured by its height, since before the page is scrolled it has not reached its stuck place yet. A
// click scrolls a section to 8px under them (its scroll-margin-top in app.css); the line is lower than that, so a
// section scrolled to is past it whatever the rounding.
function readingLine() {
  const topbar = document.querySelector(".topbar")?.getBoundingClientRect().bottom ?? 0;
  const sticks = bar && getComputedStyle(bar.root).position === "sticky";
  return Math.ceil(topbar + (sticks ? bar.root.getBoundingClientRect().height : 0) + 24);
}

// The section on the reading line. On none (the line between two sections, or above the first), the last whose top
// has passed it, else the first. Once the page's end is on screen the short sections at the bottom can never reach the
// line, so there the one last clicked wins while it is on screen, else the last one shown. A page too short to scroll
// has its end in view from the start and is read from the line as usual.
function pick() {
  const shown = bar.entries.filter(({ button }) => !button.hidden);
  if (!shown.length) return null;
  const line = readingLine();
  if (bar.atEnd && window.scrollY > 0) {
    const clicked = shown.find(({ key, section }) => key === bar.clicked && section.getBoundingClientRect().bottom > line);
    return clicked ?? shown.at(-1);
  }
  const onLine = shown.find(({ section }) => bar.onLine.has(section));
  if (onLine) return onLine;
  let current = shown[0];
  for (const entry of shown) if (entry.section.getBoundingClientRect().top <= line) current = entry;
  return current;
}

function mark() {
  if (!bar) return;
  const current = pick();
  for (const entry of bar.entries) {
    const on = entry === current;
    entry.button.classList.toggle("on", on);
    if (on) entry.button.setAttribute("aria-current", "location");
    else entry.button.removeAttribute("aria-current");
  }
}
