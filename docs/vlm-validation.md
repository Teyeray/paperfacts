# Visual validation with a vision-language model

*Design note for the `validate` stage added in this fork. What it is for, why it is built the way it is,
what it deliberately does not do, and how to tell whether it helps.*

## 1. The gap it fills

PaperFacts' whole argument is structural independence: two parsers that share nothing read the same PDF,
one extractor asks both texts the same questions, and disagreement between the two answers is the signal.
Where the lanes agree, a number survived two different layout analyses and two different OCR passes.

Every stage of that pipeline, though, works on *text the parsers produced*:

| Stage | What it reads |
|---|---|
| extract | parser text (Markdown with `<!-- source: id -->` markers) |
| grounding | the quoted `value_raw` against the cited block's *parser text* |
| compare | two parser texts, normalised |
| dataset | the comparison verdicts |

Nothing looks at pixels. So the one failure the design cannot see is the correlated one: both parsers
misread the same character. A minus sign lost from `10⁻²` in a table cell; a `±` read as `+`; a row shift
that puts the right number in the wrong column of *both* outputs because both layout models cut the table
the same way. The pipeline reports AGREE, grounding passes (the quote *is* in the parser text — the parser
text is what is wrong), and the number goes into the training set with the highest confidence label the
tool has.

The README's own limitations section names half of this: *"the two lanes share one extractor, so a mistake
made by the language model itself correlates across lanes and AGREE will not catch it."* The other half —
the two lanes can share an OCR mistake too — had no stage that could catch it. This one is that stage.

## 2. What it does, in one paragraph

After `compare` and before `export`, for every value the two lanes could not settle between them — and,
under the default policy, every value that was read out of a table — the stage renders the page region the
value was cited from (the union of its cited blocks, widened by a sliding window of one neighbour on each
side, padded), hands the PNG to a vision-language model with the instruction *transcribe exactly what is
printed here*, and then — in code — checks whether the quoted `value_raw` occurs in that transcription
using the same matcher grounding already uses. The result is one of five verdicts per value, stored with
the transcription and the crop's page, box and digest, and consulted by the dataset in exactly three
places.

Then, still inside the stage, every table a lane's samples were read from is transcribed once as a whole
and the *extraction* model is asked to quote, from that transcription only, the fields those samples still
lack. What it quotes is cleaned and grounded like any other value; what survives is a **fill**, which the
dataset commits in exactly one place: a cell that would otherwise be blank.

```
 comparison report ──→ select_targets (a lane-blind rule) ──→ [value, lane, cited blocks]
                                                                     │
 artifact (blocks + bboxes) ──→ region_for: cited blocks ± context_blocks neighbours on one page, padded
                                                                     │
 source PDF ──→ pdf.render_region (same pixel mapping as the web viewer) ──→ PNG, cached under crops/
                                                                     │
                       VisionClient.complete_vision(system, user = field name only, image) ──→ text
                                                                     │
                       parse_reading (lenient) ──→ {transcription, legible}
                                                                     │
                       adjudicate: grounding.is_grounded(value, {"vlm": transcription})
                                                                     │
        confirmed │ contradicted │ illegible │ not_checked │ error  ──→ validations/<ek>.<ck>.<vk>.json
                                                                     │
                       dataset._decide: three entry points, nothing else

 lane samples ──→ table_regions_of: every cited table block, ± context, once ──→ PNG (same crop store)
                                                                     │
                       VisionClient.complete_vision(system, user = "transcribe the whole table") ──→ text
                                                                     │
                       LlmClient (the extractor): "these samples lack these fields; quote them from this"
                                                                     │
                       response_to_records ──→ adjudicate against the transcription ──→ FilledValue
                                                                     │
                       dataset._decide: one entry point -- a cell that would otherwise be blank
```

Why the fill step is made by the extraction model and not by the VLM: the VLM's job is to read pixels
faithfully, and the extractor's job is to know which printed cell is *this sample's sheet resistance* —
the field table, the quoting rules and the record cleaning all live on that side. Handing the VLM's
transcription to the extractor keeps each model doing the thing it is trusted for, and a filled value
goes through exactly the guardrails a lane value goes through, with the transcription as its one citation.

## 3. The three rules, and why each one

### 3.1 The model transcribes; the code adjudicates

The obvious design is to show the model the crop and ask *"does this region say 12.5 Ω/sq?"* It is also
the wrong one. A vision-language model asked to confirm a specific number tends to confirm it — the number
is in the prompt, the model is helpful, and the region usually does contain *some* number that looks like
it. That is confirmation bias built into the instrument, and it would be invisible: every verdict would
come back `confirmed` and the stage would look like it was working.

So the prompt never carries the value. It asks for a verbatim transcription of the region and names the
field (*"this region was cited as the source of a value of: sheet_resistance"*) so the model knows which
small print deserves care — identically for both lanes, and nothing else. The verdict is then decided by
`grounding.is_grounded`, with the transcription stood in as a block the value cites. That reuse is
deliberate on two counts:

- **Same leniency.** Grounding already knows the two parsers write the same number differently (`$( 4 0
  \times 1 0 \mathrm { c m }$` versus `(40 × 10 cm`) and folds LaTeX, `×`/`x`, superscripts, case and
  decoration before comparing. A model's transcription is a third spelling of the same thing and needs the
  same folds.
- **Same strictness.** Grounding refuses to find a number inside a longer number: `4` does not ground in
  `40 min`, `5 nm` does not ground in `235 nm`. A validator with a looser matcher would confirm values the
  page does not contain.

One fold was added, and it is applied to *both* sides: a multiplication sign between two digits loses the
spaces around it (`1.2 × 10^-4` ≡ `1.2×10⁻⁴`). Parsers copy the PDF's spacing and the extractor quotes it
verbatim; a model writes its own. Grounding refuses to close space gaps for numbers on principle (a space is
a boundary it must not invent), and this fold respects the principle by closing only that one gap, only
between digits, only around `×`. `5 nm` still cannot be found in `235 nm`.

This is the same rule extraction lives by — *the model quotes; the code converts* — applied to the reader
instead of the extractor. `ExtractionResponse` has no `value` field so the model cannot convert; the
validation prompt has no value in it so the model cannot agree.

### 3.2 Both lanes get the same treatment

CLAUDE.md is blunt about extraction: *"any asymmetry there contaminates the disagreement signal, which is
the whole measurement."* A validator that checked lane A's values more often, or more leniently, than lane
B's would do the same damage one stage later.

So which values are checked is a rule, not a model call, and the rule is lane-blind by construction:

| Reason | What is selected |
|---|---|
| `conflict` | **both** sides of every CONFLICT row |
| `ambiguous` | **both** sides of every AMBIGUOUS row |
| `missing` | the one side of every MISSING row (there is no other side) |
| `ungrounded` | every value grounding flagged, in **either** lane |
| `all` (policy) | every value in both lanes |

Each value is listed once, under the first reason that named it, in a fixed order; two runs over the same
inputs produce byte-identical reports. The prompt, model, temperature, crop DPI, padding and pixel cap are
the same for both lanes, and the test suite pins that the two lanes' requests differ in nothing but the
image.

### 3.3 A verdict is a fourth reading, not the truth

`confirmed` means: a model that never saw either parser's text also reads these characters in this
region. `contradicted` means it does not. Both are strong evidence and neither is proof — the model can
misread too. So every verdict carries the transcription and the crop's location, the web page shows the
transcription on hover, and the dataset never deletes anything: a contradicted value is *set aside* in the
decision and named in the 视觉核验 column, and a reviewer can look at the crop and overrule.

The five verdicts are kept apart on purpose:

| Verdict | Meaning |
|---|---|
| `confirmed` | the quoted value occurs in the model's transcription of the cited region |
| `contradicted` | it does not |
| `illegible` | the model said the region was unreadable, or transcribed nothing |
| `not_checked` | the stage could not put the value in front of the model: no valid citation, or no PDF on this machine |
| `error` | the request failed after its retries |

A value nobody could check must never look like a value that was checked and passed — `not_checked` is
not `confirmed` — and a systematic failure must not hide inside a hundred `error`s: a stage in which every
request failed raises, because that is a misconfigured endpoint, not a coincidence.

## 4. Where the verdicts act

In `dataset._decide`, in exactly three places, in this order:

1. **Contradicted values are set aside first**, before the two-lane rules judge anything. If nothing
   survives, the cell is refused as `vlm_contradicted`. This is the "both lanes agree and both are wrong"
   case — the failure the stage exists for — and it takes precedence over AGREE.
2. **A conflict resolves only in one exact shape**: at least one side was contradicted *and every
   surviving side was confirmed*. Then the survivor is committed as `vlm_resolved`. A conflict where both
   sides were confirmed (the page really says both; the lanes each caught one) stays a conflict. A conflict
   where the survivor was never checked stays a conflict — one denial does not promote an unverified value.
3. **A confirmed value counts as trusted even when grounding failed.** Grounding against parser text has
   false negatives — the README's first grounding catch was one, a quote straddling two blocks — and a
   reading of the pixels is the natural appeal. A confirmed-but-ungrounded value is committed with a note.

And nowhere else. A verdict never invents a value, never changes a unit, never overrides
`multiple_conditions`, `multiple_values`, `non_scalar`, a failed sample match or a low match confidence. The
test file `test_dataset_validation.py` pins each entry point and, for each, the neighbouring shape it must
*not* apply to.

Fills have one entry point of their own, deliberately narrower than any of the three above: **a cell for
which no lane holds a value** (or every value it held was contradicted) takes the fill and is committed as
`vlm_filled`, through the same scalar parsing as any other value. A fill never replaces a value a lane
read, never breaks a conflict (two parser readings that disagree are a disagreement to review, not a vote a
third reading can swing), and when both lanes' tables produced one, the first lane in `BACKENDS` order is
taken so the choice is deterministic. The cell cites `vlm:<crop file>`, so a reviewer opens the exact PNG
the transcription came from.

### 4.1 Why tables get more of the model than prose

Running text reaches the extraction LLM whole, in context, and a misread word there tends to surface as the
two lanes disagreeing — the existing machinery already catches it, and the VLM is only needed for the
residue (`disputed`). A table cell is a single token in a grid; the classical parser failures on tables —
a shifted column, a merged header row, a superscript dropped from a unit — produce two lanes that *agree*
on the same wrong number, or two lanes that both leave the cell blank because neither Markdown carried it.
Neither shape is visible to a text-only pipeline. So the default policy checks every table-cited value
whether or not the lanes agreed, and the fill step reads every cited table back as a whole. Text still gets
checked — the disputed rule applies to it as before — just not exhaustively.

### 4.2 The sliding window

A cited block alone is often not enough to read: a table cell without its header row has no unit, a
caption without its table names samples the crop does not show, and grounding itself accepts a quote that
straddles the boundary into the next block. `context_blocks` widens every crop by that many text-bearing
neighbours before and after the cited block, in the page's reading order — the same adjacency grounding
walks — skipping figures and page furniture without counting them and never crossing a page. The default
of 1 brings a table its caption and footnote. The window is part of `validation_key` (a wider crop is a
different question) and the stored verdict names the context blocks alongside the cited ones.

## 5. Why Qwen3-VL, and why not the models that score higher

The obvious pick from the document-parsing leaderboards is not the right pick here, because the property
that matters is **independence from the two lanes**, not benchmark rank.

| Candidate | OmniDocBench | Verdict |
|---|---|---|
| PaddleOCR-VL-1.6 | 96.3 | *Is* lane B. A validator that is lane B confirms lane B. |
| MinerU 2.5 (VLM backend) | 95.8 | Lane A's family. Same problem. |
| GLM-OCR | 94.6 (#1 on v1.5) | Its layout stage is PP-DocLayoutV3 — the same family as lane B's PP-DocLayoutV2. Correlated *layout* errors are exactly the row-shift case the stage must catch. |
| dots.ocr, olmOCR-2, Chandra | 82–84 (olmOCR-bench) | Independent and grounding-capable; tuned for full-page Markdown dumps rather than "read this region". Reasonable alternatives to benchmark against. |
| **Qwen3-VL** (2B–235B, Apache 2.0) | 89.8 (235B) | Independent of both lanes. Native bounding-box grounding, 256K context, served by vLLM and by the same Model Studio endpoint `config.json` already uses. Instruction-follows on a region question. |

Two further points decided it:

- **The task is region transcription, not page parsing.** The OCR specialists' scores measure full-page
  Markdown reconstruction, which is not what the stage asks for. A general VLM that follows an
  instruction about *this crop* is the right instrument, and the transcription is short.
- **Open weights make the hosted pilot and the self-hosted route the same reader.** The default
  `qwen3-vl-32b-instruct` on Model Studio and `Qwen/Qwen3-VL-32B-Instruct` under vLLM on GPU 7 are the same
  weights, so verdicts collected during the API pilot stay comparable when the model moves on-premises.
  The model name is part of `validation_key`, so the two are nevertheless stored as different files — a
  different reader is a different set of verdicts, and the filenames say so.

Honest caveat: PaddleOCR-VL is itself a VLM, so the stage is the *second* vision-language model in the
stack. Different family, different training data, different layout model; independent enough to be a
useful third reader, not a guarantee of anything.

## 6. Cost and what to measure

Under the `disputed` policy the stage sends one vision request per disputed value — on the 14-paper
corpus, the 9 ambiguous rows plus every one-sided and ungrounded value: tens of small requests per paper
against the hundreds extraction makes. The default `tables` policy adds one request per table-cited value,
and the fill step adds one vision request plus one extraction request per cited table per lane — a
handful per paper. Identical crops share one request (the cache stands the image in by digest), and a
re-run replays every answer for free.

Two more numbers the fill step makes measurable: the fraction of blank cells it fills, and — on the
hand-labelled sample — how often a `vlm_filled` value is right. A fill that is confirmed by neither lane
rests on one model's reading of one crop, so its precision must be reported apart from the two-lane cells.

Under `policy = all` it sends one request per value in both lanes. That is the measurement mode, and it
answers the question that justifies the stage: **how often do both lanes agree on a wrong reading?**
Nobody has measured it, because until this stage there was nothing that could. The number to report is the
fraction of AGREE cells whose evidence the VLM contradicted, alongside the validator's own precision and
recall on a hand-labelled sample (a hundred or two values, deliberately including known-bad ones) — a
validator has to be validated too.

## 7. What it is not

- **Not an attribution check.** The stage checks *characters in a region*. It can say `10^2` is not printed
  where a lane says it is; it cannot say a correctly read number belongs to a different sample. That
  failure — the extractor's, correlated across lanes — remains outside the design.
- **Not a third lane.** A third full parse with three-way voting was considered and rejected for now:
  `compare_lanes` is strictly two-lane, the cost is roughly three times a run, and most of the value is in
  the disputed values anyway. The targeted validator can be measured first and the third lane argued from
  the numbers.
- **Not a judge.** It is never asked whether a value is right, for the reason in §3.1.

## 8. File map

| File | What changed |
|---|---|
| `src/paperfacts/validate.py` | new — selection (`select_targets`, `cites_table`), region (`region_for`, `region_of_blocks` with the sliding window), crop store, reading, adjudication, the stage; the fill step (`missing_fields`, `table_regions_of`, `fill_from_tables`, `FilledValue`) |
| `src/paperfacts/llm.py` | `VisionClient` protocol; `complete_vision`, `vision_payload`, `vision_cache_key` |
| `src/paperfacts/pdf.py` | `render_region`, `png_bytes` |
| `src/paperfacts/prompts.py` | `validation_system_prompt`, `validation_user_prompt`, `table_transcription_user_prompt`, `fill_system_prompt`, `fill_user_prompt` |
| `src/paperfacts/config.py` | `vlm.*` constants, `Settings.vlm_*` (incl. `vlm_policy`, `vlm_context_blocks`, `vlm_fill_blanks`), `require_vlm_api_key`, `_parse_bool`, `_parse_policy`, `_non_negative` |
| `src/paperfacts/keys.py` | `validation_key` (both prompts, the window, the policy, the fill switch), `validation_key_for`, `validation_code_fingerprint` |
| `src/paperfacts/storage.py` | `validation_path`, `crops_dir`, `crop_path`; `dataset_json_path` takes the optional third key |
| `src/paperfacts/dataset.py` | verdict lookup; the three entry points in `_decide`; the fill lookup and `_commit_fill` (`vlm_filled`); 视觉核验 column; `validation_key` on the payload |
| `src/paperfacts/workflow.py` | `build_vlm_client`, `read_validation`, `validate_document` (takes the extractor for the fill step), the `validate` stage inside the extractor's client scope, export reads a stored validation |
| `src/paperfacts/cli.py` | `paperfacts validate`; `--policy disputed|tables|all` on `run` and `validate` |
| `src/paperfacts/report.py` | `render_validation` (verdicts, fills, dropped fills) |
| `src/paperfacts/web/documents.py`, `web/app.py` | `Library.validation`, `GET /api/documents/{id}/validation`, three-key dataset lookup, `vlm` in `/api/health` |
| `src/paperfacts/web/static/*` | verdict badges, KPI tile, stage label, skipped state; 视觉裁定 / 视觉补全 cell badges in the results table |
| `config.json` | the `vlm` block |
| `deploy/compose.yaml`, `deploy/host/start_qwen_vlm.sh`, `deploy/.env.example` | `qwen-vlm-server` on GPU 7 |
| `tests/` | `test_validate.py`, `test_llm_vision.py`, `test_pdf_region.py`, `test_dataset_validation.py`, `test_workflow_validate.py`, `test_keys_validation.py`; additions to the config, workflow, CLI and web tests; `FakeVisionClient` in `support/llm.py` |
