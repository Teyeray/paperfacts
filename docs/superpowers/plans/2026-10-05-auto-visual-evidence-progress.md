# Implementation record — automatic visual evidence

Plan: [2026-10-05-auto-visual-evidence.html](2026-10-05-auto-visual-evidence.html)
Design: [2026-10-05-auto-visual-evidence-design.md](../specs/2026-10-05-auto-visual-evidence-design.md)
Code baseline: `ef0ebb6f2c62d338f5fc9eb9e25a75ef6c322a60`, existing `dev_JW` checkout.

## Authorized scope

The user requested tool development before concrete paper analysis. Implement T1/T2 and the deterministic
experiment/replay tools from T3 using synthetic PDFs and fake vision clients. T0's real inputs and reviewed
labels remain pending. No live model requests, parser runs, scientific acceptance claims, T4 integration,
or T5 production adoption in this execution.

## Execution decisions

- Preserve the existing uncommitted plan/design revision. Work in this suitable checkout; do not reset,
  clean, delete, push, or change credentials.
- Record progress here in the existing plans directory. Keep generated test logs in temporary storage;
  do not create a second project scratch hierarchy or delete prior artifacts.
- Parallelize independent PDF/candidate, crop/experiment, and strict-reading work; one owner per file.
- Keep candidate selection and image reading in two flat modules, each with its own focused tests.
  They introduce no framework, subpackage, workflow stage, or new dependency.
- Freeze interfaces before dependent work: `read_native_pages` returns page text and normalized text
  regions; `CropStore` preserves its public methods while including `max_pixels` in disk identity.
- Genuine PDF/vision correctness still requires the separately authorized and reviewed T3 experiment.

## Progress

- Baseline: sandboxed tests stalled in an existing TestClient call because Unix socket sends were denied.
  A standard-library reproduction and the same test outside the sandbox confirmed the cause. Unrestricted
  offline baseline: 4770 passed, 11 skipped, 1 deselected, 1 failed. The failure was an existing CLI test
  patching the workflow's parser binding instead of the CLI's imported binding. The test now patches its
  actual call site and asserts both fake backend calls; its focused rerun passed. No parser installation
  or dependency/configuration change was needed.
- T0: output directory rules documented in `eval/README.md`; actual corpus and gold review pending.
- T1: implemented native text/coordinate extraction and deterministic risk/discovery candidates. Separate
  quotas, deduplication, caption/footnote context and explicit unresolved coverage are tested synthetically.
- T2: implemented profile-scoped independent reading, strict JSON, original-value preservation, exact
  sample mapping, at most one known-region zoom, and per-candidate errors. Reports bind actual PDF/image
  digests and preserve the raw answer; no field is adopted into a dataset.
- T3 tooling: freeze/replay entry and reviewed-scope scoring passed synthetic engineering verification. The frozen
  manifest records explicit files and source/model settings; replay defaults offline and never reruns A/B.
  Real experiments, reviewed complete-fact labels and scientific acceptance remain pending.
- T4/T5: not started; original gates retained.

## Verification

New behavior was tested RED then GREEN using synthetic PDFs, frozen synthetic lanes and fake clients.
Boundary review covered rotated/CropBox coordinates, region context, multi-entity references, strict cached
responses, qualifier preservation, partial sample-matching failure, and merged-SI crop identity.
Final verification on 2026-10-05:

- `UV_CACHE_DIR=/tmp/paperfacts-uv-cache UV_OFFLINE=1 uv run pytest`: **4904 passed, 11 skipped,
  1 deselected**, 64.01 seconds. It ran outside the restrictive socket sandbox. The default skips/deselection
  are parser/browser integration tests; no real-paper scientific evaluation is claimed. Two existing
  Starlette/AnyIO deprecation warnings remain.
- `uv run ruff check src tests runners eval/visual_evidence.py eval/visual_evidence_score.py`: passed.
- `uv run ruff format --check src tests runners eval/visual_evidence.py eval/visual_evidence_score.py`:
  passed, 192 files already formatted.
- `git diff --check`: passed. HTML's 28 references, IDs/ARIA and offline assets checked; Markdown's
  local links resolve. Both experiment CLI help and the scorer's JSON-schema output were checked.

No new dependencies, production settings, database changes, frontend behavior changes, live model calls,
real-paper analysis, or Git publication were part of this execution. Existing crops/history were retained.
Code changes remain in the current checkout for review.

## Next increment — offline adoption and G2 replay

User approval: implement the proposed pure adoption decisions and G2 offline replay without real data/API,
keeping complexity small. Existing workspace changes are retained.

Ruling: add one flat `visual_adoption.py` module and one `eval/visual_adoption.py` entry, with synthetic tests.
Do not modify the hashed A/B decision modules or wire this into workflow/batch. Reuse dataset scopes,
profile unit conversion and kind rules. Output a cell-level counterfactual audit (before/after and reasons),
not a modified production dataset or workbook; export/lifecycle integration remains a later gated step.

Initial rule scope: explicitly listed single numeric sample fields; only missing/conflict reasons; printed
facts with unique existing sample attribution, complete crop context, compatible conditions and an exact,
finite in-range value. Recompute attribution/normalization from raw evidence. Conflicting C observations,
new samples, unknown context and existing populated cells are refused. Policy revision and input hashes
travel with replay output. Gold labels affect scoring only, never adoption decisions.

Private replay input JSON remains in the existing ignored output/visual-evidence run directory. The CLI
reads frozen file references and prints JSON; it does not create a new output hierarchy or overwrite inputs.
Real scientific acceptance is never inferred from a synthetic/reviewed-scope metric pass.

Implementation and independent review:

- Added the pure `paperfacts.visual_adoption.replay_document` function and standalone
  `eval/visual_adoption.py --replay/--schema` entry. No new dependencies, settings, storage hierarchy,
  production decision keys, workflow wiring or dataset mutation.
- Decisions carry original observation indices and before/after values. Frozen input metadata/digests
  are checked, cached C attribution/numeric values are recomputed, and gold is inaccessible to the pure
  decision function. Shared-missing and correction goals have separate counters, with explicit zero
  coverage for every policy field/reason scope.
- The fresh reviewer found two important edge cases. Both were reproduced RED and fixed GREEN:
  `test_every_supported_uncertainty_spelling_is_not_exact` covers ASCII/parenthesized uncertainty;
  `test_after_clause_is_not_silently_removed_before_adoption` prevents embedded state loss. Existing
  numeric reader syntax is reused; the A/B parsing/decision modules remain unchanged.
- Final review ruling: actual pixel readability, unit-multiplier interpretation, scientific attribution,
  holdout eligibility and production integration remain outside this synthetic increment. These require
  reviewed original papers and later integration work; a metric pass never certifies them.
- No minor findings were deferred. The 52 focused adoption/replay tests pass. Final full-suite verification
  is recorded below after completion.

Final increment verification on 2026-10-05:

- Focused adoption/replay tests: **52 passed** using synthetic inputs only.
- Required full suite after review fixes: **4961 passed, 11 skipped, 1 deselected**, 66.47 seconds.
  Command: `UV_CACHE_DIR=/tmp/paperfacts-uv-cache UV_OFFLINE=1 uv run pytest`, outside the restrictive
  Unix-socket sandbox. Two pre-existing Starlette/AnyIO deprecation warnings remain.
- `uv run ruff check src tests runners eval/visual_evidence.py eval/visual_evidence_score.py eval/visual_adoption.py`:
  passed; matching `ruff format --check`: **196 files already formatted**.
- CLI help and schema checked; local documentation links, HTML IDs and offline asset references checked.
  External reference links were not fetched. `git diff --check`: passed.
- No live API calls, real corpus evaluation, global dependency installation, credentials changes,
  production adoption, deletion, commit or push. Earlier workspace changes were preserved.

Status: this offline increment is complete. Real G1/G2 evidence, lifecycle integration, and consistent
runtime/export/reload adoption remain subsequent gated work, not claims made by these synthetic tests.

## Next increment — persisted experiment lifecycle and export

Authorized: persist complete/partial/failed experimental C attempts, reuse matching results without
repeated requests, explicitly retry while preserving history, generate independent experimental dataset
copies with consistent derived rows, and demonstrate everything with synthetic inputs. No production wiring.

Pre-flight: frozen experiment manifests bind PDF/A/B/profile/model/render/code; each strategy's state
will bind the manifest digest and its report digest. Repeated runs read those states before constructing a
client. Changed frozen inputs require a new run (existing freeze contract), never mutation of an old run.
Export consumes only matching saved reports and always reconstructs the baseline from A/B.

Ruling: extend the existing experiment entry and storage layout rather than add a scheduler/cache layer.
Numbered retry reports/states retain earlier results. A persisted interrupted attempt requires explicit
retry. Runtime state lives only in the existing ignored run directory. One process owns a run directory.

Ruling: add one flat `visual_snapshot.py` for independent experimental dataset copies; reuse consolidation,
payload loading and workbook export. Do not change hashed A/B modules just to add unused experimental
hooks. Recompute the small row-count/representative-selection step after adoption; test it against baseline
behavior, including entities and tie-breaking. Cost: this small selection rule must stay aligned with dataset.py.

Directory convention: each export uses a new caller-named directory under its experiment run, containing
`<document-key>.snapshot.json` and `dataset.xlsx`. Never overwrite an earlier export. The synthetic demo
uses `inputs/` beneath its explicit date/run root for generated PDF/profile/A/B fixtures, and named run/export
subdirectories; all are labelled synthetic, ignored, and retained. `eval/visual_demo.py` owns only these
synthetic fixtures, not private-paper data. No new package, dependency, database or UI.

Status: implementation in progress, tests first; final review and verification pending.


Lifecycle/export increment completed on 2026-10-05:

- Extended the existing experiment entry with persisted strategy attempts, status, explicit retry and
  independent export. Failed/partial/interrupted attempts are reused without implicit model calls;
  retries retain the prior numbered reports and states. Offline misses retain other reading outcomes.
- Added `visual_snapshot.py`, rebuilding from A/B, recalculating quality/source/condition rows,
  available/agree counts and the whole representative sample. It saves a labelled experiment wrapper,
  reloads the snapshot, and uses the existing workbook exporter. Experimental outputs never overwrite
  production datasets or earlier exports.
- Added `eval/visual_demo.py`, generated synthetic PDF/profile/artifacts/lanes and fixed local replies.
  Actual retained demo: `output/visual-evidence/2026-10-05-lifecycle-demo/`. It showed failed and partial
  initial states, 0 repeat requests, 2 adoptions, 2 non-adoptions, and 0 stale adoptions. All outputs are
  Git-ignored. They establish engineering behavior only.
- Fresh read-only final review found no Critical/Important findings and no deferred minors; reviewer
  independently ran 42 focused tests. Ruling: recovery is per strategy attempt, not per-region checkpointing.
  Retry refreshes the whole non-complete attempt; the tradeoff is some repeated local/model work on an
  explicit retry. This avoids an additional checkpoint scheduler and is bounded by the existing region cap.
- Ruling: real OCR/scientific attribution, G1/G2 acceptance, production integration and concurrent writers
  remain outside this increment. A run directory has one owner; these files do not promise adversarial
  tamper protection. Existing input/report digest checks target reproducibility and accidental drift.

Verification after the final code changes:

- Visual reading/adoption/scoring/lifecycle/export/demo tests: **158 passed**.
- Full required suite: **4985 passed, 11 skipped, 1 deselected**, 67.43 seconds; run outside the restrictive
  Unix-socket sandbox. Two pre-existing Starlette/AnyIO deprecation warnings remain.
- Ruff check across `src tests runners` and all four visual evaluation entries passed; format check:
  **200 files already formatted**. CLI help, local documentation links, HTML IDs/offline assets and
  `git diff --check` passed.
- No real data/model API use, dependency installation, credentials/configuration changes, production
  wiring, database migration, deletion, commit or push. Earlier workspace changes preserved.

Status: the authorized lifecycle, independent export and synthetic-demo increment is complete.
