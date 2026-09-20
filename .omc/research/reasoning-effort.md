# reasoning_effort: speed against recall (2026-09-20)

Same paper (tsta-20-1599695, 12 pages, both parses cached), same prompts, passage mode, one pass,
`deepseek-v4.1-flash` on the Aliyun MaaS OpenAI-compatible endpoint. Wall-clock is the whole job from
queue to export; the parsers were cache hits.

| reasoning_effort | wall-clock | values (MinerU + Paddle) | samples | agree / missing | table cells committed |
|---|---|---|---|---|---|
| unset (default) | 10 min 30 s | 48 + 64 = 112 | 8 | 35 / 39 | 19 before the conditions fix |
| none | 1 min 39 s | 44 + 33 = 77 | 7 | 26 / 16 | 24 after the conditions fix |
| low | 18 min 50 s | 47 + 58 = 105 | 8 | 33 / 30 (+4 ambiguous) | 34 after the conditions fix |

"none" is fast because the model stops reading closely: the PaddleOCR lane, whose blocks are more
fragmented, lost half its values. "low" was slower than unset on this endpoint and no better. Decision: leave the
parameter unset; get speed from concurrency (lanes and field questions in parallel), which does not change
what the model is asked.

## Concurrency (2026-09-20, after the lanes and field questions were made concurrent)

Different paper (coatings-12-00203, 10 samples, both parses cached, model cache empty for it), reasoning
unset, `llm.concurrency` 4, CLI `paperfacts run`: **3 min 42 s** wall-clock for 42 live model calls. The
inventory question is now the long pole (about 2 minutes per lane, both lanes overlapping); the twenty field
questions of a lane finish within a minute after it. Not the same paper as the 10 min 30 s baseline, so
read it as "a third of the time", not a precise ratio.

## Unattributed values that are correct (2026-09-20)

coatings-12-00203, PaddleOCR lane: `o2_flow_rate` 0.0/0.2/0.4/0.6 sccm went to `unattributed`. Not a
bug. The values come from Table 3, whose rows are devices ("#1 (0.0 sccm)"), and each film exists twice in
the inventory (as-deposited and annealed at 480 °C), so no single sample owns them; the model's condition
text says exactly that. `transmittance 'over 80 %'` is a paper-wide sentence. Both belong where they are.
Do not add a rule that attaches a value to a sample whose name contains the number.

## Where the time goes (2026-09-20, reasoning_tokens now recorded)

coatings-12-00203 again, concurrent pipeline, reasoning unset, 42 live calls, 2 min 39 s wall-clock.
Inventory calls: MinerU 14.4k completion tokens of which 11.5k reasoning; PaddleOCR 20.2k of which 17.0k.
The forty field questions together reason less than the two inventories; most answer with under 300
reasoning tokens. The inventory is therefore the lever, and it is the one question where reasoning may
matter (it decides how many samples exist). Next: a per-stage effort setting for the inventory alone,
measured on sample count and conditions before it ships.

## Inventory-only effort (2026-09-20)

`llm.inventory_reasoning_effort` applied to the coatings paper, everything else unchanged:

| inventory effort | MinerU samples | PaddleOCR samples | matched | agree / missing | wall-clock |
|---|---|---|---|---|---|
| unset | 10 | 10 | 10 | 30 / 26 | 2 min 39 s |
| low | 6 | 8 | 4 (6 unmatched) | 15 / 42 | 1 min 58 s |
| none | 6 | 6 | 6 | 20 / 21 | 1 min 02 s |

Less reasoning makes the inventory merge the as-deposited and 480 °C-annealed films into one sample per
O2 flow, and at "low" the two lanes disagree on which set exists, so matching falls apart. The inventory
is the one question where the hidden reasoning is doing the work. The setting stays unset; the remaining
wall-clock is model generation, not a code path.

## Condition paraphrase in one lane: not fixable lexically (2026-09-20)

A numeric-signature condition key ("at 550 nm" == "550 nm wavelength") was built, reviewed and reverted:
"550 nm" and "550 nm, annealed" share the signature, and optical values before and after annealing at one
wavelength are routine. No word-level rule tells a paraphrase from a qualifier. The within-lane
multiple_conditions refusal stays as it is; the cost is a handful of blank cells per paper.

## Parsers verified live on this Mac (2026-09-20)

Forced re-run of tsta-20-1599695 with the mlx-vlm server on port 8111 (Apple M4, 16 GB) and
`PAPERFACTS_PADDLE_VL_MODEL_NAME=PaddlePaddle/PaddleOCR-VL-1.6`: MinerU 3.4.5 in 24.5 s, PaddleOCR-VL 3.7.0 in
153.6 s, 12 pages, 128 and 191 blocks — the same counts the cached parses from the other machine held.
Without the model name the Paddle pipeline asks the server for "PaddleOCR-VL-1.6-0.9B", which is not a
Hugging Face repository, and the parse fails with a 401; the name must be set wherever the mlx backend is.

## Corpus run (2026-09-20) and the endpoint running dry

All 14 library papers were extracted under the current keys (13 via `paperfacts batch` over the template
zip, the web uploads through run-all); rebuilding comparisons and tables after a key change takes 34 s
for 13 papers because every model answer is cached. The 14th paper (s41598-022-19270-w) stopped mid-way:
from 08:56 every request to the endpoint, even a one-line test, answers HTTP 500 `BalanceError: There are
no suitable services`. 748 answers were bought today. It is not the paper, the payload, or concurrency —
all three were bisected. Nothing runs until the account has balance again; the cache makes the resume free.

## Whole corpus (2026-09-20, 14/14 papers, endpoint back at 09:01)

```
papers 14, samples 169, sample×field cells 2760
  missing               2149  77.9%
  agree                  352  12.8%
  single_source          156   5.7%
  multiple_conditions     48   1.7%
  non_scalar              34   1.2%
  ambiguous               11   0.4%
  multiple_values          6   0.2%
  conflict                 2   0.1%
  ungrounded               2   0.1%
```

The outage was 08:56–09:01; the last paper then took 45 s (36 samples, 108 agreements).
