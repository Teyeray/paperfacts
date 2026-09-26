// Router: (#/p/<profile>)?/doc/<id16>(/fact/<n>)?, and #/check (the profile check page). The selected fact goes into the URL; refreshing or sharing the link
// returns to the same one. The profile prefix names the domain profile the view is shown under; it is left out for
// the server's default, so every link written before profiles existed still opens as it did.
//
// It also owns the view generation (state.generation): every navigation to another document, to home or to another
// profile bumps it, which retires every load, poll and finish handler still running for the view being left. Moving
// between facts of the open document is not a new view and bumps nothing, so its job keeps being polled.

import { state } from "./state.js";

// Anything shaped like a profile prefix or a document link is routed as one, so a malformed name or id gets a clear
// "no such profile / document" rather than silently falling back to the default's home.
const PREFIX = /^#\/p\/([^/]*)(\/.*)?$/;
const DOCUMENT = /^\/doc\/([^/]*)(?:\/fact\/(\d+))?\/?$/;
const DOCUMENT_ID = /^[0-9a-f]{16}$/;
// The check page is profile-free; under a prefix it is the same page.
const CHECK = /^\/check\/?$/;
const PROFILE_NAME = /^[a-z][a-z0-9_]{0,39}$/; // profile_loader.IDENTIFIER
let handlers = {
  onDocument: () => {},
  onEmpty: () => {},
  onMissing: () => {},
  onMissingProfile: () => {},
  onProfile: () => {},
  onCheck: () => {},
};
let routed; // the view on screen: "<profile>\n<document id>" (undefined: nothing yet)
let reloadNext = false;

export function installRouter(next) {
  handlers = { ...handlers, ...next };
  window.addEventListener("hashchange", () => route());
}

// The hash split into the profile it names (null: none, the default) and the rest. A prefix naming the default is
// accepted as the default and left as typed.
function parse(hash) {
  const prefix = hash.match(PREFIX);
  const raw = prefix ? decoded(prefix[1]) : null;
  const rest = prefix ? prefix[2] ?? "/" : hash.slice(1);
  const document = rest.match(DOCUMENT);
  return {
    raw,
    profile: raw === state.defaultProfile ? null : raw,
    id: document ? decoded(document[1]) : null,
    fact: document?.[2] == null ? null : Number(document[2]),
    check: CHECK.test(rest),
  };
}

export function route({ reload = false } = {}) {
  const view = parse(location.hash);
  const key = `${view.profile ?? ""}\n${view.check ? "check" : view.id ?? ""}`;
  const fresh = reload || reloadNext || key !== routed;
  reloadNext = false;
  routed = key;
  if (fresh) state.generation += 1;
  const refusal = view.raw === null ? null : profileRefusal(view.raw);
  if (refusal) {
    handlers.onMissingProfile(view.raw, refusal);
    return;
  }
  if (view.profile !== state.profileName) {
    state.profileName = view.profile;
    handlers.onProfile(view.profile);
  }
  if (view.check) handlers.onCheck();
  else if (view.id === null) handlers.onEmpty();
  else if (!DOCUMENT_ID.test(view.id)) handlers.onMissing(view.id);
  else handlers.onDocument(view.id, view.fact, { reload: fresh });
}

// Why a named profile cannot be shown, or null when it can. Without the server's list (it did not load) a well-formed
// name is tried, and the server says whether it exists.
function profileRefusal(name) {
  if (!PROFILE_NAME.test(name)) return { reason: "missing" };
  const list = state.profiles;
  if (!list) return null;
  const invalid = list.invalid?.find((entry) => entry.name === name);
  if (invalid) return { reason: "invalid", errors: invalid.errors ?? [] };
  const served = list.profiles?.find((entry) => entry.name === name);
  if (!served) return { reason: "missing" };
  if (!served.runnable) return { reason: "not_runnable", errors: [served.not_runnable ?? ""] };
  return null;
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
export const factFromHash = () => parse(location.hash).fact;

// The document the URL names right now (null: none), whether or not its view has landed yet.
export const documentFromHash = () => parse(location.hash).id;

// A hand-typed link may hold a broken %-escape; it is still just a name that names nothing.
function decoded(text) {
  try {
    return decodeURIComponent(text);
  } catch {
    return text;
  }
}

// The address of a view: home (no `id`) or a document, under `profile` (null or the default's name: no prefix).
export function hashFor({ profile = null, id = null, fact = null } = {}) {
  const prefix = profile == null || profile === state.defaultProfile ? "" : `/p/${encodeURIComponent(profile)}`;
  if (id == null) return prefix ? `#${prefix}` : "#/";
  return `#${prefix}/doc/${id}${fact == null ? "" : `/fact/${fact}`}`;
}

// A document of the profile on screen.
export const documentHash = (id, factIndex = null) => hashFor({ profile: state.profileName, id, fact: factIndex });
