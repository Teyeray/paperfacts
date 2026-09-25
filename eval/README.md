# Gold evaluation set

`gold/<doc_id>.json` holds hand-checked answers for a few papers; `score.py` compares a PaperFacts dataset
(`data/docs/<doc_id>/datasets/*.json`) against them. Values were read off the rendered PDF pages, with the parsed
markdown used only to find where to look.

```bash
python eval/score.py --data-root data                                # newest dataset of each gold paper
python eval/score.py --data-root data --out report.md --json cells.json
python eval/score.py --dataset 80c3b69d570c2b6d=/path/to/dataset.json --only 80c3b69d570c2b6d
```

Standard library only; tolerances come from `config.json` (`--config` to point elsewhere).

## Gold file format

```jsonc
{
  "doc_id": "80c3b69d570c2b6d",          // the 16-hex directory prefix under data/docs/
  "sha256": "...", "title": "...",
  "notes": ["what the paper contains, traps, what was deliberately left out"],
  "target": { "<target field>": [cell, ...] },   // component, resistance, density, inch
  "series": { "<field>": [cell, ...] },          // stated once for the whole series; applies to every sample
  "samples": [
    {
      "id": "ICO-30nm-1H2-annealed",
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
| `value` | number in the field's canonical unit (`config.json`), or text for `component` / `mode`; `null` = the paper states something that is not a scalar (a range, a lower bound, a power density) |
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

**Target fields** are compared with the dataset's `paper_row` (target values are paper-level; every row repeats
them).

**Sample alignment.** A dataset row is a candidate for a gold sample when every `match.fields` value equals the
row's value within the field tolerance, `match.label` matches `sample_id + " | " + sample_label`
(case-insensitive) and `match.label_not` does not. Candidate pairs are then taken greedily, one-to-one, by the
number of the row's cells that agree with that gold sample (descending), ties by gold order then row order.
A gold sample without a row loses all its required cells as *missing*; a row without a gold sample makes all its
non-empty cells *extra*.

**Cell outcomes** (for each gold sample × sample field, and each target field):

| outcome | when | precision | recall |
|---|---|---|---|
| correct | dataset value matches a required (not ambiguous, not figure-only, not null) gold cell | TP | TP |
| soft | matches only an ambiguous or figure-only gold cell | TP | not counted |
| wrong | has a value, gold has required cells, none match | FP | FN |
| missing | empty, gold has a required cell | – | FN |
| extra | has a value, gold has no cell for this sample/field (or the row is unaligned) | FP | – |
| disputed | has a value, gold has only ambiguous/null cells, none match | not counted | not counted |

Numbers match with `math.isclose(dataset, gold, rel_tol, abs_tol)` using the field's tolerances; `mode` compares
its category set (DC / RF / pulsed DC); `component` matches on normalised equality or an `accept` regex.

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
