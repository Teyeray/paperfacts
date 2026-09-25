// Router: #/doc/<id16>(/fact/<n>)?. The selected fact goes into the URL; refreshing or sharing the link returns to the same one.
//
// It also owns the view generation (state.generation): every navigation to another document or to home
// bumps it, which retires every load, poll and finish handler still running for the view being left. Moving
// between facts of the open document is not a new view and bumps nothing, so its job keeps being polled.

import { state } from "./state.js";

// Anything shaped like a document link is routed as one, so a malformed id gets a clear "no such document"
// rather than silently falling back to home.
const ROUTE = /^#\/doc\/([^/]*)(?:\/fact\/(\d+))?\/?$/;
const DOCUMENT_ID = /^[0-9a-f]{16}$/;
let handlers = { onDocument: () => {}, onEmpty: () => {}, onMissing: () => {} };
let routed; // the document id the view on screen was routed to (null: home; undefined: nothing yet)
let reloadNext = false;

export function installRouter(next) {
  handlers = next;
  window.addEventListener("hashchange", () => route());
}

export function route({ reload = false } = {}) {
  const match = location.hash.match(ROUTE);
  const id = match ? decoded(match[1]) : null;
  const fresh = reload || reloadNext || id !== routed;
  reloadNext = false;
  routed = id;
  if (fresh) state.generation += 1;
  if (id === null) handlers.onEmpty();
  else if (!DOCUMENT_ID.test(id)) handlers.onMissing(id);
  else handlers.onDocument(id, factFromHash(), { reload: fresh });
}

// `reload` re-reads the view even when the hash says it is already on screen (a re-upload of the open
// document, a bulk run that queued it): the browser fires no hashchange for an unchanged hash.
export function navigate(hash, { reload = false } = {}) {
  if (location.hash === hash) route({ reload });
  else {
    reloadNext = reload;
    location.hash = hash;
  }
}

export const reloadView = () => route({ reload: true });

// The fact index the URL asks for right now, read at the moment it is needed: a load that started earlier
// must select what the address bar says when it lands, not what it said when it began.
export function factFromHash() {
  const match = location.hash.match(ROUTE);
  return match?.[2] == null ? null : Number(match[2]);
}

// A hand-typed link may hold a broken %-escape; it is still just an id that names no document.
function decoded(text) {
  try {
    return decodeURIComponent(text);
  } catch {
    return text;
  }
}

export const documentHash = (id, factIndex = null) => (factIndex == null ? `#/doc/${id}` : `#/doc/${id}/fact/${factIndex}`);
