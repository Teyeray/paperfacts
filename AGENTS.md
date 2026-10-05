# PaperFacts agent instructions

## Scope and source of truth

- Read `CLAUDE.md` before editing code. It is the existing source of truth for architecture,
  dependency isolation, cache fingerprints, storage, and verification commands.
- Read `README.md` for the public interface; verify capability claims against the actual call graph.
- User instructions take precedence. Planning and implementation are separate tasks: a request for
  an action plan permits documentation work, not product changes or model calls.
- Discuss substantial changes and obtain approval of the concrete plan before implementation.

## Directory conventions

- Keep production modules flat under `src/paperfacts/`; `web/` remains the only subpackage.
- Store reviewable designs in `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md` and actionable
  task lists in `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`. Link each plan to its design.
- When the user requests HTML, the plan may instead use the same dated name with `.html` in
  `docs/superpowers/plans/`. Keep it self-contained, offline-readable and printable; link its design.
  Mark superseded designs/plans with a link to their replacement and retain them for history.
- Keep deterministic tests under `tests/`; use synthetic or distributable small fixtures there.
- Keep evaluation definitions and scoring code under `eval/`. Record fixture provenance and distinguish
  synthetic scorer tests from manually checked real-paper evidence.
- Keep source PDFs, crops, private model responses, caches, and evaluation run outputs under ignored
  `data/` or `output/`, following `storage.py`. Do not put them in documentation or tracked fixtures.
- Name evaluation outputs by date and run identifier; preserve the source SHA, model configuration,
  code revision, and reviewed gold revision. Do not automatically clean earlier results.
- New directories must have a purpose and naming/lifecycle rule documented here or in their parent
  documentation before they are populated. Removal always requires user permission.

## Boundaries

- Ask before deleting files/directories/history; editing `.env`, secrets, tokens, or CI/CD;
  database schema changes or migrations; `git push`, rebase, hard reset, or force push;
  global dependency installation or system changes; and public publication or production deployment.
- Never log or commit credentials. Do not copy the Reader repository's runtime data into this project.
- Preserve the two parser lanes' evidence and comparison semantics unless an approved plan changes them.
- A visual transcription match is not proof of sample, field, unit, or condition attribution.

## Verification

- For code changes, run the focused tests first and the required project checks before completion:
  `uv run pytest`, `uv run ruff check src tests runners`, and
  `uv run ruff format --check src tests runners`.
- Relevant frontend changes also require the existing browser regression suite described in `CLAUDE.md`.
- Real parser/model evaluation is a separate, explicitly identified activity; a fake-client test does
  not establish OCR accuracy or scientific validity.
- For documentation-only work, check links, referenced paths, internal consistency, and
  `git diff --check`; do not invoke paid APIs or rebuild runtime environments just to validate prose.
