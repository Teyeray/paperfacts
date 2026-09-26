// Round trip to /api. A non-2xx response throws an Error carrying `status`; the message comes
// from the backend's `detail` (the status code is the contract: 404 / 409 / 413 / 422 each mean something specific).
//
// Every answer that depends on a domain profile is asked through `profileApi`, which names the profile the view was
// routed to. `null` is the server's default profile and sends no parameter on a GET, so a default page asks exactly
// what it asked before profiles existed; a POST names the profile whenever the page knows its name, so a server
// restarted with another default never runs a queued paper under a profile the reader did not see. A caller that
// forgot to pass one (`undefined`) throws instead of silently asking the default. Plain `api` is only for the
// routes every profile shares, and refuses any other path.

import { state } from "./state.js";

// The profile the server answered under, echoed on every per-profile route.
const PROFILE_HEADER = "X-PaperFacts-Profile";
// health, the profile list, the check of a pasted profile, jobs, and a document's parse output and pages: the same
// under every profile.
const PROFILE_FREE =
  /^\/api\/(?:health|profiles(?:\/[^?]*)?|profile-check|jobs(?:\/[^/?]+)?|documents\/[^/?]+\/(?:jobs|artifact\/[^/?]+|pages\/[^/?]+))(?:\?.*)?$/;

export async function api(path, options = {}) {
  if (!PROFILE_FREE.test(path)) throw new Error(`${path} depends on the profile: ask it through profileApi`);
  return request(path, options, undefined);
}

// `path` asked under `profile` (null: the default).
export async function profileApi(profile, path, options = {}) {
  return request(profileUrl(profile, path, options.method), options, profile);
}

// The same address as a link (a download), for which no request is made here.
export const profileHref = (profile, path) => profileUrl(profile, path, "GET");

function profileUrl(profile, path, method = "GET") {
  if (profile === undefined) throw new Error(`no profile given for ${path}`);
  const name = profile ?? (method === "GET" ? null : state.defaultProfile);
  return name == null ? path : `${path}${path.includes("?") ? "&" : "?"}profile=${encodeURIComponent(name)}`;
}

async function request(path, options, profile) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail ?? detail; } catch { /* non-JSON error body */ }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  // The answer is drawn as the profile's: one given under another (a server whose default changed under an open
  // page) is refused rather than shown under the wrong labels. A response without the header says nothing.
  if (profile !== undefined) {
    const expected = profile ?? state.defaultProfile;
    const echoed = response.headers.get(PROFILE_HEADER);
    if (echoed && expected && echoed !== expected) {
      const error = new Error(`服务器按领域配置「${echoed}」回答了「${expected}」的请求，请刷新页面`);
      error.status = 0;
      throw error;
    }
  }
  return response.json();
}

// "this artifact doesn't exist yet" is a normal state, not an error: 404 -> null, everything else still throws
export const optional = (promise) => promise.catch((error) => (error.status === 404 ? null : Promise.reject(error)));
