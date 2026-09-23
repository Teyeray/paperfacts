// Router: #/doc/<id16>(/fact/<n>)?. The selected fact goes into the URL; refreshing or sharing the link returns to the same one.

const ROUTE = /^#\/doc\/([0-9a-f]{16})(?:\/fact\/(\d+))?$/;
let handlers = { onDocument: () => {}, onEmpty: () => {} };

export function installRouter(next) {
  handlers = next;
  window.addEventListener("hashchange", route);
}

export function route() {
  const match = location.hash.match(ROUTE);
  if (!match) { handlers.onEmpty(); return; }
  handlers.onDocument(match[1], match[2] == null ? null : Number(match[2]));
}

// the browser doesn't fire hashchange when the hash is unchanged (e.g. re-uploading the document that's already open): route manually in that case
export function navigate(hash) {
  if (location.hash === hash) route();
  else location.hash = hash;
}

export const documentHash = (id, factIndex = null) => (factIndex == null ? `#/doc/${id}` : `#/doc/${id}/fact/${factIndex}`);
