# Plan — delete-document (per-document delete in the left rail)

**Date:** 2026-10-07 · **Design:** `docs/superpowers/specs/2026-10-07-delete-document-design.md` ·
**Base:** `7890959` (working copy: `.pi/plans/2026-10-07-delete-doc/plan.md`)

One PR, backend first.

## Backend

- [x] `JobManager.is_active(document_id)` (`web/jobs.py`), any profile, under the lock; gated-job
      unit test in `tests/test_web_jobs.py`.
- [x] `Library.delete(document_id)` (`web/documents.py`): `_require_key` guard, `shutil.rmtree` of
      `layout.doc_dir`, own `_counts_cache` eviction under `_counts_lock`, returns the display
      name; `DuplicateDocument` docstring loses the false "there is no delete". Tests in
      `tests/test_web_documents.py` (removal + listing, malformed ids, cache eviction, llm_cache
      survival, upload display name).
- [x] `DELETE /api/documents/{document_id}` (`web/app.py`): `DeletedDocument(document_id, name)`
      model, `require_document` → 404, `manager.is_active` → 409, rmtree in the threadpool;
      route-table line in the module docstring. Tests in `tests/test_web_app.py` (success, 404
      unknown/malformed/repeat, gated-job 409 with the directory untouched, llm_cache survival) and
      a cross-origin 403 in `tests/test_web_edge.py`.

## Frontend

- [x] `api.js`: `PROFILE_FREE` gains an optional per-document suffix (`documents/<id>` bare, not
      the list route).
- [x] `index.html`: `#delete-dialog` cloned from `#run-all-dialog`; `app.css`: `.doc-row` flex row,
      hover/focus-revealed `.doc-delete`, touch always-visible, `.primary.danger`; `.doc-item`
      drops `width: 100%` for `flex: 1; min-width: 0`.
- [x] `library.js`: `libraryItem` sets `li.className = "doc-row"`, appends the sibling delete
      button (byte-identical `.doc-item` markup); `setupDelete()` mirrors `setupRunAll`'s dialog
      lifecycle, toasts, navigates home when the open paper is deleted, refreshes via
      `loadLibrary()`, refocuses `#rail-search`; wired in `app.js`.

## Verification

- [x] e2e check in `tests/e2e/web_races.py`: uploads its own `delete-me.pdf`, cancel posts no
      request, confirm removes the row + toasts 已删除「…」, `#doc-count` still reads, stale link
      shows the missing view, server no longer lists the id. Never touches a seeded doc.
- [x] Screenshots: `rail-delete-hover` and `delete-dialog` variants added to
      `scripts/ui_screenshots.py` (desktop-only; the dialog shot only opens, never confirms).
- [x] Gates: `uv run pytest`, `uv run ruff check src tests runners eval`,
      `uv run ruff format --check src tests runners eval`, `make e2e` (twice).
