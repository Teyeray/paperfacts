// theme.js — manual light/dark override on top of the OS preference.
import { readStored, removeStored, writeStored } from "./html.js";

const KEY = "pf-theme";
const ORDER = ["auto", "light", "dark"];

export function initTheme() {
  const saved = readStored(KEY);
  if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
}

export function cycleTheme() {
  const cur = document.documentElement.dataset.theme ?? "auto";
  const next = ORDER[(ORDER.indexOf(cur) + 1) % ORDER.length];
  if (next === "auto") {
    delete document.documentElement.dataset.theme;
    removeStored(KEY);
  } else {
    document.documentElement.dataset.theme = next;
    writeStored(KEY, next);
  }
  return next;
}
