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
