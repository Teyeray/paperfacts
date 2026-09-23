// Round trip to /api. A non-2xx response throws an Error carrying `status`; the message comes
// from the backend's `detail` (the status code is the contract: 404 / 409 / 413 / 422 each mean something specific).

export async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail ?? detail; } catch { /* non-JSON error body */ }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

// "this artifact doesn't exist yet" is a normal state, not an error: 404 -> null, everything else still throws
export const optional = (promise) => promise.catch((error) => (error.status === 404 ? null : Promise.reject(error)));
