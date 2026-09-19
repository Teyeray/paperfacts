# reasoning_effort: speed against recall (2026-09-20)

Same paper (tsta-20-1599695, 12 pages, both parses cached), same prompts, passage mode, one pass,
`deepseek-v4.1-flash` on the Aliyun MaaS OpenAI-compatible endpoint. Wall-clock is the whole job from
queue to export; the parsers were cache hits.

| reasoning_effort | wall-clock | values (MinerU + Paddle) | samples | agree / missing | table cells committed |
|---|---|---|---|---|---|
| unset (default) | 10 min 30 s | 48 + 64 = 112 | 8 | 35 / 39 | 19 before the conditions fix |
| none | 1 min 39 s | 44 + 33 = 77 | 7 | 26 / 16 | 24 after the conditions fix |
| low | MinerU lane alone 5 min, more completion tokens than unset | 47 (MinerU) | 8 | not finished when recorded | — |

"none" is fast because the model stops reading closely: the PaddleOCR lane, whose blocks are more
fragmented, lost half its values. "low" was not cheaper than unset on this endpoint. Decision: leave the
parameter unset; get speed from concurrency (lanes and field questions in parallel), which does not change
what the model is asked.
