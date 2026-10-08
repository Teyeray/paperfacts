# Delete-document: a per-document delete control in the left rail (文献栏)

**Date:** 2026-10-07 · **Status:** Shipped (this document mirrors the approved design; working copy
in `.pi/plans/2026-10-07-delete-doc/plan.md`)

## Intent

Each document row in the left rail gains a delete button. Deleting a document removes its entire
`data/docs/<16-hex>/` directory — the uploaded PDF, parses, extractions, comparisons, datasets,
figures, crops, overlays and page renders — after one confirmation dialog. Deletion is **hard**
(irreversible), **cross-profile** (the directory is shared by every profile), and **refused while a
job is queued or running** for that document.

User request: 「左侧文献栏是否增加一个删除文献的按钮」.

## Decisions

- **Hard delete, no trash** — the listing is a validated directory scan, so it self-heals with no
  index rewrite; a trash dir would be a new on-disk concept needing its own lifecycle rule.
- **409 while active, not delete-anyway** — no cancel exists, and a vanishing directory mid-job
  would end the job `failed` with `FileNotFoundError`. `JobManager.is_active(document_id)` answers
  it.
- **`DELETE /api/documents/{id}`**, profile-free via `default_library` (same as the artifact/page
  routes): existence and the directory are the same under every profile. The method-based
  same-origin/auth middleware covers it for free; no route-order collision with the literal
  `run-all`.
- **Rail row = two sibling buttons in `li.doc-row`** — a `<button>` may never nest in
  `<button class="doc-item">`, whose markup stays byte-identical (five e2e selectors pin it). The
  delete control is hover/focus-revealed, always visible on touch, `aria-label`/`title`
  `删除：<name>`.
- **Native `<dialog id="delete-dialog">`** mirrors `#run-all-dialog`; 取消 sends no request. UI
  copy is chrome (hardcoded Chinese): 删除文档 / 确定删除「…」…不可恢复。/ 删除 / 取消 /
  已删除「…」/ 删除失败：….
- **Deleting the open paper navigates home**, then `loadLibrary()` refreshes the rail and home
  counts; focus lands on `#rail-search` (the deleted row is gone).
- **`llm_cache` and `data/exports` untouched** — content-addressed / rebuilt on demand.

## Testing

pytest: `JobManager.is_active` gated-job unit; `Library.delete` storage tests (directory removal,
malformed-id `KeyError`, counts-cache eviction, llm_cache survival, display name); route tests
(success + 404 unknown/malformed/repeat + gated-job 409 + cross-origin 403). e2e: one `web_races.py`
check that uploads and deletes **its own** PDF only (never a seeded doc), asserting cancel sends no
request and confirm removes the row, toasts, and shows the missing view on a stale link.
