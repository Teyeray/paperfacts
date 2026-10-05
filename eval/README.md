# Gold evaluation set

## Visual evidence tooling

The visual-evidence experiment is separate from the two-lane score below. Its tooling can be developed and
tested with synthetic PDFs and fake clients before real-paper inputs are available. Such checks establish
engineering behavior only, never OCR accuracy or the G1/G2 scientific acceptance gates.

Private experiment inputs and outputs belong under the ignored
`output/visual-evidence/YYYY-MM-DD-run-id/` directory: a frozen input manifest, candidate/observation reports,
reviewed fact labels, and private response references. A run id identifies one immutable experiment setup;
use a new run id when inputs, selection policy, prompts, or reviewed labels change. Replays reuse request
caches; they do not overwrite unrelated runs or delete history. Paths and atomic writes are defined in
`storage.py`. Do not put source PDFs, private responses, or generated run outputs in tracked fixtures.

The 2026-10-05 implementation starts with deterministic tools and offline replay only. Real model calls,
paper analysis, G1/G2 acceptance, daily workflow integration (T4), and automatic adoption (T5) are pending.

### Standalone freeze and replay entry

`visual_evidence.py` freezes explicitly named inputs and runs `risk_only` and `balanced` as two separate
side reports. It never calls `workflow.run_document`, reparses a document, re-extracts A/B, or changes the
comparison/dataset. This is the T3 engineering entry only: reviewed complete-fact labels, G1 scoring,
G2 adoption replay, and scientific acceptance are not implemented by this script.

Keep the input JSON in ignored storage. Paths are relative to that JSON file, or absolute. For example,
an `experiment.json` directly under `output/visual-evidence/` can contain:

```json
{
  "profile_path": "../../profiles/tco.json",
  "data_root": "../../data",
  "model": "YOUR_EXISTING_VISION_MODEL",
  "base_url": "https://YOUR_EXISTING_ENDPOINT/v1",
  "render": {"dpi": 200, "max_pixels": 2000000},
  "limit": 4,
  "temperature": 0,
  "max_tokens": 4096,
  "timeout_s": 120,
  "metadata": {
    "split": "development; holdout status not yet established",
    "history": "Record every document's prior extraction, tuning, gold and C-rule use here."
  },
  "reviewed_gold_paths": [],
  "documents": [{
    "id": "FULL_64_HEX_DOCUMENT_ID_OR_EXISTING_16_HEX_KEY",
    "pdf_path": "../../data/docs/DOCUMENT_KEY/source.pdf",
    "identity_path": "../../data/docs/DOCUMENT_KEY/identity.json",
    "artifact_paths": {
      "mineru": "../../data/docs/DOCUMENT_KEY/parsed/mineru.artifact.json",
      "paddleocr_vl": "../../data/docs/DOCUMENT_KEY/parsed/paddleocr_vl.artifact.json"
    },
    "lane_paths": {
      "mineru": "../../data/docs/DOCUMENT_KEY/facts/mineru.EXTRACTOR_KEY.json",
      "paddleocr_vl": "../../data/docs/DOCUMENT_KEY/facts/paddleocr_vl.EXTRACTOR_KEY.json"
    },
    "comparison_path": "../../data/docs/DOCUMENT_KEY/comparisons/EXTRACTOR_KEY.COMPARISON_KEY.json",
    "dataset_path": "../../data/docs/DOCUMENT_KEY/datasets/EXTRACTOR_KEY.COMPARISON_KEY.json"
  }]
}
```

Replace every placeholder with an explicit existing file/model/endpoint. `dataset_path` is optional; when
provided, its exact bytes are frozen for future G2 work but are not scored or modified. `identity_path`
is required for a merged SI document, and optional for an ordinary PDF whose bytes match its document ID.
The manifest keeps SI part hashes/page ranges and document identity separately from the actual merged
PDF's byte SHA. Optional `reviewed_gold_paths` freezes the files that define the reviewed gold revision;
an empty list establishes no real-paper gold evidence. Never put credentials in this JSON or its metadata.

```bash
uv run python eval/visual_evidence.py freeze --config output/visual-evidence/experiment.json \
  --run-dir output/visual-evidence/2026-10-05-example
uv run python eval/visual_evidence.py run --run-dir output/visual-evidence/2026-10-05-example
```

Freeze records file-byte hashes for the config, profile, PDF, A/B artifacts/lanes/comparison, supplied
identity/dataset and reviewed gold. It also records the profile hash, Git revision, dirty source status,
all package Python source hashes, entry/lockfile hashes, a source digest, prompt-module hash and model
parameters. Missing or mismatched lane/parse/profile fingerprints are refused; historical unknown
fingerprints do not silently count as a verified baseline. Lanes with failed extraction questions are
refused as incomplete baselines. Geometry and SI page ranges must match the
actual PDF. The original config and all input files must remain available and unchanged for replay.

`run` verifies **every** frozen document and code snapshot before creating a model client. Default replay
does not load credentials or make HTTP requests; a missing request cache raises `LlmOfflineMiss`, with no
online fallback. Both strategies use the existing `DataLayout(data_root).llm_cache_dir()` and crop paths.
Experiment crops include the frozen PDF byte SHA in their filename, separately from the logical document
ID, so an SI merge with changed bytes cannot silently reuse another merge's pixels.
Each strategy selects at most `limit` (1–4) candidates, with at most two logical reads per candidate and
two client attempts per read. Online execution is available only through an explicit `run ... --online`;
it resolves the existing configured key after preflight. No real online execution is part of the synthetic
tool verification recorded here.

An identical `freeze` is idempotent. A changed setup or nonempty unrelated run directory is refused:
freeze a new dated run ID. Each strategy now saves its report and state immediately. State binds the
manifest digest and exact report bytes, distinguishes complete/partial/failed attempts, and survives a
restart. An interrupted attempt is shown as interrupted and is not implicitly rerun. An offline cache miss
is saved with other candidate results, then raises `LlmOfflineMiss`; it never becomes a complete report.

Repeated `run` reuses matching saved attempts before constructing a client, including partial/failed ones.
`run ... --retry` retries only non-complete attempts, using the existing `refresh` behavior; it does not
rerun completed strategies. Recovery is per strategy attempt: a process interruption does not checkpoint
each finished region, and retry refreshes the whole non-complete attempt. Reports and `.state.json`
receipts from attempt 1 retain their names; retries
add `.retry-0002`, `.retry-0003`, etc. A missing/changed report is stale. Legacy reports without a receipt
are also stale and need explicit retry. Each run directory has one serial owner; this experiment tool is
not a concurrent job scheduler. Report existence or a complete processing state is not G1/G2 acceptance.

```bash
uv run python eval/visual_evidence.py status --run-dir output/visual-evidence/2026-10-05-example
uv run python eval/visual_evidence.py run --run-dir output/visual-evidence/2026-10-05-example --retry
```

Reports also bind canonical A/B lane digests to the exact lane models used for that reading. The scorer
checks these against the reviewed baseline files, in addition to checking those files' byte hashes.
`usage` includes the token counts stored in cached replies; `uncached_usage` and `cached_requests` keep
replay from looking like new token consumption. Neither is a billing estimate.

### Reviewed-scope scoring

`visual_evidence_score.py` scores an explicit human review of complete facts. It does not infer truth from
a matching number, reinterpret the old sparse gold, or determine holdout eligibility. Both original lane
files must be supplied for every document, so the reviewer can inspect the complete frozen A/B baseline.

```bash
uv run python eval/visual_evidence_score.py --schema
uv run python eval/visual_evidence_score.py --review output/visual-evidence/2026-10-05-example/review.json
```

Both commands print JSON to stdout without writing experiment files. The schema gives the complete
input contract. Review paths are relative to the review JSON (or absolute). Required top-level fields are:

| Field | Content |
|---|---|
| `format`, `review_version`, `gold_revision` | Format `1` and explicit annotation/gold revisions |
| `evaluation_set` | `synthetic` (default), `development`, or `c_holdout`; a label, not proof of eligibility |
| `baseline_sources` | Each document's two complete lane files: `document_id`, `backend`, `path`, `sha256` |
| `reports` | The strategy reports being scored, each with `path` and `sha256` |
| `facts` | Reviewed canonical facts, including opportunities that C never found |
| `observations` | Human judgments pointing to actual facts inside the frozen reports |

A `facts` item contains `key`, `document_id`, `scope`, `entity`, `sample`, `field`, `condition`,
`condition_status`, `measurement_state`, `value_raw`, `unit_raw`, `baseline`, and original-image `source`.
The source is `{pdf: {path, sha256}, page, bbox}` with a zero-based page and normalized box. A paper fact
uses null entity/sample. Canonical identity includes the scope, entity, sample, field, condition/status and
measurement state; duplicate identities are rejected.

The reviewer assigns `baseline` after checking the complete A/B files and original image:

- `correction_opportunity`: A/B has a wrong fact or unresolved conflict that can be corrected.
- `shared_missing`: neither lane contains the real fact.
- `already_correct`: an existing correct fact; confirmation adds no gain, a wrong suggestion is penalized.
- `uncertain`: the reviewed truth/baseline cannot yet be settled.
- `excluded`: an explicitly rejected claim (for example a hallucinated or out-of-scope fact), with a
  required `reason` and `penalty_goal` (`correction` or `shared_missing`). It never creates an opportunity.

Each observation judgment contains `report_path`, `reading_index`, `fact_index`, `fact_key`, `verdict`
(`correct`, `wrong`, `uncertain`) and `reason`. These indices refer to `readings[i].facts[j]`; the scorer
includes that complete raw/normalized observation in its output. The human judgment must check the value,
unit, sample/field attribution and measurement conditions together. Unreviewed observations stay
`pending_review`; they are never automatically counted as extra or wrong.

Each strategy counts each canonical fact at most once as correct and once as wrong. Repeated reads add
no gain; a correct and wrong suggestion for the same fact yields zero net gain. Correction and shared
missing have separate opportunity counts and net scores. Both must be positive, both must have real
opportunities, every required document must have a report, and no pending/uncertain review may remain for
`metric_result: pass`. The result covers only the reviewed scope and is never called full-paper recall.

The output preserves input byte hashes, review/gold versions, scorer source hash, per-fact outcomes and
per-observation judgments. It always says `scientific_acceptance: not_established_by_scorer` and
`g2: not_evaluated`: real T0 history/splits, C holdout review, and later dataset-adoption replay remain
separate work. The synthetic tests verify this arithmetic and provenance contract only.

## Existing two-lane evaluation

`gold/<doc_id>.json` holds hand-checked answers for a few papers; `score.py` compares a PaperFacts dataset
(`data/docs/<doc_id>/datasets/*.json`) against them. Values were read off the rendered PDF pages, with the parsed
markdown used only to find where to look.

```bash
uv run python eval/score.py --data-root data --keys <extractor_key>.<comparison_key>   # the run those keys name
uv run python eval/score.py --data-root data --out report.md --json cells.json          # newest dataset of each paper
uv run python eval/score.py --dataset 80c3b69d570c2b6d=/path/to/dataset.json --only 80c3b69d570c2b6d
```

Runs in the package's environment: tolerances, categories and which fields are paper-level come from the profile
(`profiles/tco.json`; `--profile` to point elsewhere), and categories are matched by the package's own
`normalize.canonical_category`. `--keys` scores `datasets/<extractor_key>.<comparison_key>.json`; without it the
newest dataset file of each paper is scored, which is only right while the library holds a single set of keys.
The gold files use `"paper"` as the id of the paper-level record, whatever the profile calls that group. A gold
file that still says `"target"` (the id before round 2) is read the same, with a note on stderr.

## Gold file format

```jsonc
{
  "doc_id": "80c3b69d570c2b6d",          // the 16-hex directory prefix under data/docs/
  "sha256": "...", "title": "...",
  "notes": ["what the paper contains, traps, what was deliberately left out"],
  "paper": { "<paper-level field>": [cell, ...] },   // TCO: component, resistance, density, inch
  "series": { "<field>": [cell, ...] },          // stated once for the whole series; applies to every sample
  "samples": [
    {
      "id": "ICO-30nm-1H2-annealed",
      "entity": "catalyst",                         // required under a profile with entity types (one is enough)
      "description": "30 nm RPD ICO, 1% H2, annealed 180 °C",
      "ambiguous": false, "why": "...",           // optional; see below
      "match": { "fields": {"thickness": 30}, "label": "regex", "label_not": "regex" },
      "exclude_series": ["field"],                 // optional; series fields that do not apply here
      "fields": { "<field>": [cell, ...] }         // overrides the series entry for the same field
    }
  ]
}
```

A **cell** is one value the paper states:

| key | meaning |
|---|---|
| `value` | number in the field's canonical unit (the profile's), or text for `component` / `mode`; `null` = the paper states something that is not a scalar (a range, a lower bound, a power density) |
| `raw` | the words on the page (abridged) |
| `page` | 0-based page index, the same numbering as source ids (`mineru_p3_b1` is page 3) |
| `condition` | measurement condition, e.g. `average 400-800 nm` for transmittance, `O2/Ar` for o2_ratio |
| `ambiguous` + `why` | the paper states it, but it is unclear which sample it belongs to, or it is only approximate |
| `figure_only` | readable only from a figure (including legend or inset text); never required |
| `accept` | text fields only: regexes, any of which accepts a dataset value |

A sample marked `ambiguous` (its existence rests on a figure label, or it is a device layer rather than a
studied film) is never required: all its cells count as ambiguous.

Values are text/table values only. A field the paper does not state has no entry: a dataset value there is an
*extra*. Several cells for one field mean several acceptable answers (e.g. transmittance at two ranges).

## Scoring rules

**Paper-level fields** are compared with the dataset's `paper_row` (every row repeats them).

**Sample alignment.** A dataset row is a candidate for a gold sample only when both name the same `entity` (both
none for a profile without entity types), and a gold sample is scored on its entity's fields only. Under a profile
that declares entity types, a gold sample whose `entity` is missing or names none of them stops the run with the
document and sample id, since it could match no row and would score nothing. It is then a
candidate when every `match.fields` value equals the
row's value within the field tolerance, `match.label` matches `sample_id + " | " + sample_label`
(case-insensitive) and `match.label_not` does not. Candidate pairs are then taken greedily, one-to-one, by the
number of the row's cells that agree with that gold sample (descending), ties by gold order then row order.
A gold sample without a row loses all its required cells as *missing*; a row without a gold sample makes all its
non-empty cells *extra*.

**Cell outcomes** (for each gold sample × sample field, and each paper-level field):

| outcome | when | precision | recall |
|---|---|---|---|
| correct | dataset value matches a required (not ambiguous, not figure-only, not null) gold cell | TP | TP |
| soft | matches only an ambiguous or figure-only gold cell | TP | not counted |
| wrong | has a value, gold has required cells, none match | FP | FN |
| missing | empty, gold has a required cell | – | FN |
| extra | has a value, gold has no cell for this sample/field (or the row is unaligned) | FP | – |
| disputed | has a value, gold has only ambiguous/null cells, none match | not counted | not counted |

Numbers match with `math.isclose(dataset, gold, rel_tol, abs_tol)` using the field's tolerances; a field with `categories` (`mode`)
compares the category each value names, by the pipeline's rule; `component` matches on normalised equality or an `accept` regex.

**List fields** (`cardinality: many`) are scored per element: their gold cells are the elements the list must
hold, not alternatives. Each dataset element matching a required cell no other element matched is *correct* (*soft* if it matches only an
ambiguous or figure-only one), any other element is *extra* (*disputed* when the gold has only ambiguous/null
cells), and each required cell no element matches is *missing*. The gold format is unchanged.

**Reference fields** (`kind: reference`, e.g. the catalyst a reaction test ran on) hold, as gold `value`, the `id`
of the gold sample of the referenced entity. The dataset cell holds a row's `sample_id`, so each gold value is first
replaced by the `sample_id` of the row aligned to the sample it names; a sample no row is aligned to leaves a value
no row matches. The samples holding references are then aligned again with those values, and scored as usual:
correct when the dataset names the row aligned to the gold sample.

Precision = (correct + soft) / (correct + soft + wrong + extra); recall = correct / (correct + wrong + missing).
The report gives micro totals, a macro average over papers (a 36-sample series otherwise dominates), per field
group, per field, per paper, and every non-correct cell with the dataset's `quality_rows` decision, conditions,
detail and source ids so it can be traced.

## Papers

| doc_id | paper | samples | why it is here |
|---|---|---|---|
| e6939c89e983c426 | Zakaria 2022, SnOx two-step (Sci Rep) | 36 | big table, three treatments × two temperatures × six O2/Ar |
| 219df6e1cd7b19fe | An 2020, ICO electrode (Solar Energy) | 10 (5 ambiguous) | two targets, samples partly only in figures, device-layer traps |
| 80c3b69d570c2b6d | Zhao 2026, ultra-thin ICO by RPD | 8 | as-deposited/annealed pairs, per-family gas ratios, several transmittance ranges |
| 534040e6151e0636 | Wang 2023, GZO post-annealing (Materials) | 8 (1 ambiguous) | annealing forming gas vs sputtering gas |
| 5c10f7a0128f15e0 | Bauden 2026, SnO2:Ta (pss b) | 6 | O2-flow series, one sample characterised, literature table |
| e855631c6f46a0ee | Seok 2019, ITO on invar (Metals) | 8 (+1 ambiguous) | two substrates × four thicknesses |
| ffd70c234c43ba93 | semi-transparent perovskite cells | 0 | no TCO deposited: every value is an extra |
| 08562126a9b9aab8 | Guillén 2006, sputtered ITO thickness × vacuum anneal (TSF) | 7 | untuned; `ρ×10^4` table header, two transmittance ranges |
| c3ab31d08acc066b | Damgaci 2024, ITO deposition temperature (Materials) | 6 (+2 ambiguous) | untuned; base pressure called "working pressure", text resistivity contradicts figure |
| 1048c42316a5c9f8 | Seok 2019, GZO-graded ITO (STAM) | 7 (+1 ambiguous) | untuned; one ITO recipe shared by all electrodes but the SI-only GZO film |
| 8977655673fa6d9a | Chen 2025, sputtered ATO electrodes (Solar Energy) | 4 (+7 ambiguous) | untuned; two targets, stability table repeats sheet resistance per state |

## Offline adoption replay (G2 tooling)

`visual_adoption.py` evaluates the experimental pure function in `paperfacts.visual_adoption` against a
frozen dataset. It prints a **counterfactual cell audit**, never rewrites `dataset.json`, the paper row or
workbooks, and never constructs a model client. This can be developed and tested without real papers or
API credentials. Real G2 acceptance and runtime/export integration remain separate work.

```bash
uv run python eval/visual_adoption.py --schema
uv run python eval/visual_adoption.py --replay output/visual-evidence/2026-10-05-example/adoption.json
```

The schema describes the complete input contract. Paths are relative to the manifest, or absolute:

- `format: 1`, `evaluation_set` (`synthetic`, `development`, `c_holdout`) and `gold_revision` identify the review.
- `profile` is `{path, sha256}`; its filename must match its declared profile name.
- `policy` contains `revision`, an explicit `fields` list, and `ambiguous_match_confidence` (default 0.6).
  Every field must be a single numeric sample field. The supported reasons are fixed to `missing` and
  `conflict`; both are separately reported. Changing the manifest changes its recorded SHA.
- Each `cases` item supplies frozen `{path, sha256}` references for `pdf`, `dataset`, `comparison`,
  `lane_a`, `lane_b`, and `evidence` (the original C report). All hashes are checked before decisions.
  Document/profile/parser artifact identities and the C report's canonical A/B digests must agree.
- `gold` contains reviewed cells, identified by `document_id`, `entity`, dataset `sample_id`, and `field`.
  Give the expected numeric `value` in canonical `unit` (null value means no valid scalar), `condition`,
  `condition_status`, `measurement_state`, and human-reviewed `before_correct` (null means unreviewed).
  Its `observations` list uses `reading_index`, `fact_index`, and `verdict` (`correct`, `wrong`, `uncertain`).
  **Correct means the complete fact, including sample, quantity, units and measurement state**, not a
  numeric transcription match. Every adopted pointer needs review; missing labels stay pending.

Adoption recomputes sample attribution and unit conversion from raw C observations. It requires a complete
selected crop and matching strict raw response; narrower zooms and known missing/multipage context stay
unadopted. It refuses populated baseline cells, unknown/ambiguous samples, failed field questions,
non-whitelisted statuses, curves, approximate/range/uncertainty values, conditions embedded in the numeric quote, unclear or conflicting
conditions, invalid
units/ranges, and contradictory C readings. Equivalent repeated readings count as one cell. A receipt
proves traceability only; scientific attribution still needs original-image review.

Output retains before/after values, reason codes and original observation indices, policy revision, the
manifest/file digests and a code-content fingerprint. Gold is used **after** all adoption decisions.
Correction and shared-missing counts include opportunities, triggers, adoptions, improvements, errors,
regressions, pending reviews and distinct document IDs. Field/reason/printed-evidence scopes are reported
separately, including zero coverage; zero adoption is insufficient. Known errors remain errors even when
another observation is pending. A score pass requires positive improvement, actual adoption, no errors or
pending review in both goals and every predeclared rule scope. `scientific_acceptance` is always
`not_established`: this tool cannot certify holdout history, adequate sample size, or human-review quality.

Keep private manifests and any redirected reports in the existing ignored date/run directory above;
use a new filename for each result and retain earlier inputs/reports. Tests use synthetic fixtures only.


## Independent experimental exports and synthetic demo

`visual_snapshot.py` rebuilds the original baseline with the existing `consolidate_document`, recomputes
adoption, and creates an **experimental copy**. It does not accept a previously adopted dataset as input.
It rebuilds quality/source rows, condition summaries, available/agree counts and the whole representative
sample row. The wrapper keeps the policy, baseline/report digests, audit and provenance. Experimental
comparison keys cannot be mistaken for baseline keys. Missing/stale C produces an A/B-only copy.

The standalone export validates the frozen inputs/code and lifecycle receipt, then saves snapshots and
**reloads those JSON files** before invoking the existing Excel exporter. It constructs no model client.
The output directory must be a new directory under the run; no prior export is overwritten. A changed
frozen input/code requires a new freeze; do not edit an old manifest to make it pass.

```bash
uv run python eval/visual_evidence.py export \
  --run-dir output/visual-evidence/2026-10-05-example \
  --policy output/visual-evidence/policy.json \
  --output-dir output/visual-evidence/2026-10-05-example/export-01
```

The policy is the same `{revision, fields, ambiguous_match_confidence}` used by the G2 tool. Default
strategy is `balanced`; `--strategy risk_only` exports the comparator. Output consists of
`<document-key>.snapshot.json` and `dataset.xlsx`. Refusal reasons remain in the audit and quality detail;
accepted sources include image digest and observation indices. Re-export always recomputes from A/B;
it never patches an old adopted copy. Scientific acceptance remains `not_established`.

For a complete demo with **no supplied papers, API credentials or network**:

```bash
uv run python eval/visual_demo.py --output-dir output/visual-evidence/2026-10-05-lifecycle-demo
```

Use a new directory for each demo. It generates a clearly labelled synthetic PDF, frozen synthetic lanes,
a profile and local fixed replies under `inputs/`, saves partial/failed attempts, checks zero repeated
requests, retries while retaining history, exports two adopted cells plus refusals, and writes a stale
snapshot with no adoption. Inspect `summary.json`, `run/` and `run/export-current/dataset.xlsx`. The demo
also uses a local `cache/` for crop files; it does not use or alter private runtime data. These generated
files are retained under the existing ignored output tree. This is engineering verification, not OCR or
scientific evidence. Production CLI/Web/batch behavior is unchanged.
