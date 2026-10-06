// Router: (#/p/<profile>)?(/?<query> | /doc/<id16>(/fact/<n>)? | /profile | /check)?. The selected fact goes into the
// URL; refreshing or sharing the link returns to the same one. `/profile` is the read-only page of the profile itself;
// `/check` is the profile check page, profile-free (under a prefix it is the same page, so leaving it returns to that
// profile). The profile prefix names the domain profile the view is shown under; it is left out for
// the server's default, so every link written before profiles existed still opens as it did.
//
// The home view carries a query (`#/?q=…`, `#/p/<name>/?q=…`): what its table is asked to show. The query is cut off
// before the path is matched, so `#/p/<name>?q=…` is that profile's home too, never a profile named "<name>?q=…"; a
// query on any other view is ignored. A change of the query alone is not a new view: the home handler is not re-run
// (no fetch), `onHomeQuery` gets the new query. A control bound to the query writes it with `setHomeQuery`, through
// history.replaceState: no hashchange, no history entry per keystroke. The router's memory of the query moves with
// that write, so the next hashchange is judged against what the router last saw, never against the address bar.
//
// It also owns the view generation (state.generation): every navigation to another document, to home or to another
// profile bumps it, which retires every load, poll and finish handler still running for the view being left. Moving
// between facts of the open document is not a new view and bumps nothing, so its job keeps being polled.

import { state } from "./state.js";

// Anything shaped like a profile prefix or a document link is routed as one, so a malformed name or id gets a clear
// "no such profile / document" rather than silently falling back to the default's home.
const PREFIX = /^#\/p\/([^/]*)(\/.*)?$/;
const DOCUMENT = /^\/doc\/([^/]*)(?:\/fact\/(\d+))?\/?$/;
const PAGE = /^\/(profile|check)\/?$/; // the pages beside the documents
const DOCUMENT_ID = /^[0-9a-f]{16}$/;
const PROFILE_NAME = /^[a-z][a-z0-9_]{0,39}$/; // profile_loader.IDENTIFIER
let handlers = {
  onDocument: () => {},
  onEmpty: () => {},
  onHomeQuery: () => {},
  onMissing: () => {},
  onMissingProfile: () => {},
  onProfile: () => {},
  onProfilePage: () => {},
  onCheck: () => {},
};
let routed; // the view on screen: "<profile>\n<document id | page>" (undefined: nothing yet)
let routedQuery; // the home query on screen, canonical (undefined: the view is not home, or nothing yet)
let reloadNext = false;
// profile key -> the last home query seen under it, canonical: what the link home carries while a document is open.
const homeQueries = new Map();

export function installRouter(next) {
  handlers = { ...handlers, ...next };
  window.addEventListener("hashchange", () => route());
}

// A query as the router compares and writes it: URLSearchParams' own spelling (so `a%20b` and `a+b` are one query),
// without the keys whose value is empty (an emptied search box leaves a bare home address). `params` is a query
// string, a URLSearchParams or a plain object.
function canonical(params) {
  const search = new URLSearchParams();
  for (const [key, value] of new URLSearchParams(params)) if (value !== "") search.append(key, value);
  return search;
}

// The hash split into the profile it names (null: none, the default), the rest, and the query cut off before either
// is matched. A prefix naming the default is accepted as the default and left as typed.
function parse(hash) {
  const mark = hash.indexOf("?");
  const path = mark < 0 ? hash : hash.slice(0, mark);
  const search = canonical(mark < 0 ? "" : hash.slice(mark + 1));
  const prefix = path.match(PREFIX);
  const raw = prefix ? decoded(prefix[1]) : null;
  const rest = prefix ? prefix[2] ?? "/" : path.slice(1);
  const document = rest.match(DOCUMENT);
  return {
    raw,
    profile: raw === state.defaultProfile ? null : raw,
    page: rest.match(PAGE)?.[1] ?? null,
    id: document ? decoded(document[1]) : null,
    fact: document?.[2] == null ? null : Number(document[2]),
    query: Object.fromEntries(search),
    queryString: search.toString(),
  };
}

const isHome = (view) => view.page === null && view.id === null;

// The topbar's 总表 link marks the home view, the way the section bar marks its current section (a route the
// router can name without asking any view's DOM). It is set before the handlers run, so even a refused profile
// or a missing document still marks the home link truthfully.
function setHomeCurrent(home) {
  const link = document.querySelector('.topnav [data-nav="home"]');
  if (home) link?.setAttribute("aria-current", "page");
  else link?.removeAttribute("aria-current");
}

export function route({ reload = false } = {}) {
  const view = parse(location.hash);
  const home = isHome(view);
  setHomeCurrent(home);
  const key = `${view.profile ?? ""}\n${view.page ?? view.id ?? ""}`;
  const fresh = reload || reloadNext || key !== routed;
  const queryChanged = home && view.queryString !== routedQuery;
  reloadNext = false;
  routed = key;
  routedQuery = home ? view.queryString : undefined;
  if (fresh) state.generation += 1;
  const refusal = view.raw === null ? null : profileRefusal(view.raw);
  if (refusal) {
    state.homeQuery = null;
    handlers.onMissingProfile(view.raw, refusal);
    return;
  }
  if (view.profile !== state.profileName) {
    state.profileName = view.profile;
    handlers.onProfile(view.profile);
  }
  state.homeQuery = home ? view.query : null;
  if (view.page === "profile") handlers.onProfilePage();
  else if (view.page === "check") handlers.onCheck();
  else if (home) {
    homeQueries.set(view.profile ?? "", view.queryString);
    if (!fresh && queryChanged) handlers.onHomeQuery(view.query);
    else handlers.onEmpty();
  } else if (!DOCUMENT_ID.test(view.id)) handlers.onMissing(view.id);
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

// The home query written by a control bound to it (a search box), as it changes. Written in place with
// history.replaceState, which fires no hashchange, so the router takes note itself and tells `onHomeQuery` as it
// would for a hashchange (the link home follows; the home table is not re-fetched). A write that changes nothing is
// dropped. Only the home view has a query: writing one anywhere else is a programming error.
export function setHomeQuery(params) {
  const view = parse(location.hash);
  if (!isHome(view)) throw new Error("the home query belongs to the home view only");
  const search = canonical(params);
  const queryString = search.toString();
  if (queryString === routedQuery) return;
  history.replaceState(history.state, "", hashFor({ profile: view.profile, query: queryString }));
  routedQuery = queryString;
  homeQueries.set(view.profile ?? "", queryString);
  state.homeQuery = Object.fromEntries(search);
  handlers.onHomeQuery(state.homeQuery);
}

// The home query last seen under `profile` (null: the default), canonical; "" when none was.
export const lastHomeQuery = (profile) => homeQueries.get(profile ?? "") ?? "";

// The fact index the URL asks for right now, read at the moment it is needed: a load that started earlier
// must select what the address bar says when it lands, not what it said when it began.
export const factFromHash = () => parse(location.hash).fact;

// The home query the URL holds right now ({} when it has none), or null when the URL is not the home view's.
export function homeQueryFromHash() {
  const view = parse(location.hash);
  return isHome(view) ? view.query : null;
}

// The document the URL names right now (null: none), whether or not its view has landed yet.
export const documentFromHash = () => parse(location.hash).id;

// The page the URL names beside the documents ("profile": the profile page, "check": the check page), or null.
export const pageFromHash = () => parse(location.hash).page;

// A hand-typed link may hold a broken %-escape; it is still just a name that names nothing.
function decoded(text) {
  try {
    return decodeURIComponent(text);
  } catch {
    return text;
  }
}

// The address of a view: home (no `id`; with `query`, the home query: `#/?…` or `#/p/<name>/?…`), a document, or a
// page (`page: "profile" | "check"`), under `profile` (null or the default's name: no prefix).
export function hashFor({ profile = null, id = null, fact = null, page = null, query = null } = {}) {
  const prefix = profile == null || profile === state.defaultProfile ? "" : `/p/${encodeURIComponent(profile)}`;
  if (page != null) return `#${prefix}/${page}`;
  if (id == null) {
    const search = query == null ? "" : canonical(query).toString();
    if (search) return `#${prefix}/?${search}`;
    return prefix ? `#${prefix}` : "#/";
  }
  return `#${prefix}/doc/${id}${fact == null ? "" : `/fact/${fact}`}`;
}

// A document of the profile on screen.
export const documentHash = (id, factIndex = null) => hashFor({ profile: state.profileName, id, fact: factIndex });
