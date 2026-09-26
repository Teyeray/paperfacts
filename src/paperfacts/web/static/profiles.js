// Domain profiles: the header switcher, the served list, and each profile's view (its title, UI copy, groups and
// fields: GET /api/profile). The server serves every profile over one library; the page shows one at a time, the
// one the URL is routed to (router.js).
//
// A view's labels and its data must come from the same profile: rows of an entity the previous profile does not
// declare would otherwise vanish from a table drawn in its groups. So `state.profile` is never set on its own: every
// generation-guarded load awaits `profileView(profile)` beside its data and adopts both only while it is current.

import { api, profileApi } from "./api.js";
import { toast } from "./html.js";
import { documentFromHash, hashFor, navigate, pageFromHash, reloadView } from "./router.js";
import { applyUiCopy, state, viewShows } from "./state.js";

// The same pause as the job poll's (job.js), without importing it: job.js draws the document view.
const RETRY_MS = 1500;
const MAX_RETRY_MS = 15000;
const MAX_RETRIES = 5;

// profile key -> { hash, view }: the Promise of its view and the content hash /api/profiles gave that profile when it
// was asked (null: unknown). A failed one is dropped, so the next load asks again.
const views = new Map();
const retries = new Map(); // profile key -> { failures, timer }
const keyOf = (profile) => profile ?? "";
const backoff = (failures) => Math.min(RETRY_MS * 2 ** failures, MAX_RETRY_MS);
// Arrowing through the closed select fires a change per key; the switch waits for the reader to settle (or Enter).
const SWITCH_SETTLE_MS = 400;

// A list that fails at start is asked again with the same growing pause as a profile's view; once one lands, the
// switcher appears and the view on screen is re-read (the router can now tell a served profile from a missing one).
export async function loadProfiles({ failures = 0 } = {}) {
  let list;
  try {
    list = await api("/api/profiles");
  } catch (error) {
    if (failures < MAX_RETRIES) {
      setTimeout(() => loadProfiles({ failures: failures + 1 }).then(reloadView, () => {}), backoff(failures + 1));
    } else {
      toast(`读取领域配置列表失败：${error.message}`, true);
    }
    throw error;
  }
  state.profiles = list;
  state.defaultProfile = list.default ?? null;
  // A view read under other content (or before the list said which) is read again when next asked.
  for (const [key, entry] of views) {
    const hash = servedProfile(key || null)?.content_hash ?? null;
    if (hash !== null && hash !== entry.hash) views.delete(key);
  }
  renderSwitcher();
  return list;
}

// The view of `profile` (null: the default), or null when it could not be read: the page then keeps the generic copy
// (state.uiCopy's defaults) and asks again with a growing pause, redrawing once it lands.
export function profileView(profile) {
  const key = keyOf(profile);
  if (!views.has(key)) {
    const entry = { hash: servedProfile(profile)?.content_hash ?? null, view: null };
    entry.view = profileApi(profile, "/api/profile").catch((error) => {
      if (views.get(key) === entry) views.delete(key);
      scheduleRetry(profile, error);
      return null;
    });
    views.set(key, entry);
  }
  return views.get(key).view;
}

function scheduleRetry(profile, error) {
  const key = keyOf(profile);
  const entry = retries.get(key) ?? { failures: 0, timer: null };
  if (entry.timer) return; // a retry is already on its way
  entry.failures += 1;
  if (entry.failures > MAX_RETRIES) {
    retries.delete(key);
    toast(`读取领域配置失败，页面使用通用名称：${error.message}`, true);
    return;
  }
  entry.timer = setTimeout(async () => {
    entry.timer = null;
    const view = await profileView(profile);
    if (!view) return; // profileView scheduled the next attempt
    retries.delete(key);
    if (viewShows(undefined, profile)) reloadView();
  }, backoff(entry.failures));
  retries.set(key, entry);
}

// Adopted by a load that is still current, with the data it labels: `profile` is the routed name (null: the default).
export function adoptProfile(profile, view) {
  state.shownProfile = profile;
  state.profile = view;
  showProfile(view);
  applyUiCopy(document);
}

// The header names the domain on screen, and says so when its profile is only an example.
function showProfile(view) {
  const title = document.getElementById("profile-title");
  title.textContent = view?.title_zh ?? "";
  title.title = view?.description_zh ?? "";
  if (view?.maturity === "example") {
    const badge = document.createElement("span");
    badge.className = "profile-badge";
    badge.textContent = "示例配置";
    title.append(badge);
  }
  const upload = document.getElementById("upload-profile");
  upload.textContent = view?.title_zh ? `按「${view.title_zh}」处理` : "";
  upload.classList.toggle("hidden", !switcherShown() || !view?.title_zh);
}

// The served profile named `name` (null: the default) as /api/profiles lists it, or null.
export function servedProfile(name) {
  const wanted = name ?? state.defaultProfile;
  return state.profiles?.profiles?.find((entry) => entry.name === wanted) ?? null;
}

export const profileTitle = (name) => servedProfile(name)?.title_zh || name || "";

// A one-profile server looks as it did before profiles: no switcher.
const switcherShown = () => (state.profiles?.profiles?.length ?? 0) + (state.profiles?.invalid?.length ?? 0) > 1;

// Each option says what the reader would get: an example profile, one this server cannot run, one whose file was
// edited since the server started. Words, not colour.
function optionLabel(entry) {
  const notes = [
    entry.maturity === "example" ? "（示例）" : "",
    entry.runnable ? "" : "（不可运行）",
    entry.on_disk_changed ? "（文件已改动，需重启）" : "",
  ];
  return `${entry.title_zh || entry.name}${notes.join("")}`;
}

function renderSwitcher() {
  const wrap = document.getElementById("profile-switch");
  const select = document.getElementById("profile-select");
  select.innerHTML = "";
  for (const entry of state.profiles?.profiles ?? []) {
    const option = document.createElement("option");
    option.value = entry.name;
    option.textContent = optionLabel(entry);
    option.title = entry.description_zh ?? "";
    select.append(option);
  }
  for (const entry of state.profiles?.invalid ?? []) {
    const option = document.createElement("option");
    option.value = entry.name;
    option.disabled = true;
    option.textContent = `${entry.name}（配置有误）`;
    option.title = (entry.errors ?? []).join("\n");
    select.append(option);
  }
  wrap.classList.toggle("hidden", !switcherShown());
  syncSwitcher();
}

// The select shows the routed profile, and the header links keep it.
export function syncSwitcher() {
  const select = document.getElementById("profile-select");
  const name = state.profileName ?? state.defaultProfile;
  if (name != null && [...select.options].some((option) => option.value === name)) select.value = name;
  document.querySelector(".brand").setAttribute("href", hashFor({ profile: state.profileName }));
  document.getElementById("profile-link").setAttribute("href", hashFor({ profile: state.profileName, page: "profile" }));
  // The check page is profile-free; kept under the prefix, leaving it returns to the same profile.
  document.getElementById("check-link").setAttribute("href", hashFor({ profile: state.profileName, page: "check" }));
}

// Switching keeps the reader on the same paper: "show me this one under the other domain" is the point. The fact
// index is dropped, since it numbers another profile's comparisons. On the profile page it shows the other profile's.
//
// A pick with the mouse switches at once. A change made from the keyboard waits until the reader settles on an option,
// presses Enter or leaves the select, so arrowing past a profile does not open it.
export function setupSwitcher() {
  const select = document.getElementById("profile-select");
  let keyed = false;
  let timer = null;
  const go = () => {
    clearTimeout(timer);
    timer = null;
    const name = select.value;
    if (name === (state.profileName ?? state.defaultProfile)) return;
    const id = documentFromHash();
    const profile = name === state.defaultProfile ? null : name;
    navigate(hashFor({ profile, id: /^[0-9a-f]{16}$/.test(id ?? "") ? id : null, page: pageFromHash() }));
  };
  select.addEventListener("pointerdown", () => {
    keyed = false;
  });
  select.addEventListener("keydown", (event) => {
    keyed = true;
    if (event.key === "Enter" && timer) go();
  });
  select.addEventListener("blur", () => {
    if (timer) go();
  });
  select.addEventListener("change", () => {
    const fromKeyboard = keyed;
    keyed = false;
    if (!fromKeyboard) {
      go();
      return;
    }
    clearTimeout(timer);
    timer = setTimeout(go, SWITCH_SETTLE_MS);
  });
}
