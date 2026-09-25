# PaperFacts gold evaluation

Micro: precision 0.50, recall 0.46. Macro over papers: precision 0.30, recall 0.46.

Datasets scored:

- `0000000000000001`: `dataset_1.json`
- `0000000000000002`: `dataset_2.json`

## Overall

| all | correct | soft | wrong | missing | extra | disputed | precision | recall |
|---|---|---|---|---|---|---|---|---|
| all | 6 | 3 | 3 | 4 | 6 | 0 | 0.50 | 0.46 |

## Per field group

| group | correct | soft | wrong | missing | extra | disputed | precision | recall |
|---|---|---|---|---|---|---|---|---|
| film | 3 | 3 | 1 | 1 | 2 | 0 | 0.67 | 0.60 |
| process | 2 | 0 | 1 | 3 | 2 | 0 | 0.40 | 0.33 |
| target | 1 | 0 | 1 | 0 | 2 | 0 | 0.25 | 0.50 |

## Per field

| field | correct | soft | wrong | missing | extra | disputed | precision | recall |
|---|---|---|---|---|---|---|---|---|
| component | 1 | 0 | 0 | 0 | 1 | 0 | 0.50 | 1.00 |
| resistance | 0 | 0 | 0 | 0 | 1 | 0 | 0.00 | – |
| inch | 0 | 0 | 1 | 0 | 0 | 0 | 0.00 | 0.00 |
| mode | 1 | 0 | 1 | 1 | 2 | 0 | 0.25 | 0.33 |
| working_pressure | 1 | 0 | 0 | 2 | 0 | 0 | 1.00 | 0.33 |
| sheet_resistance | 1 | 0 | 1 | 0 | 0 | 0 | 0.50 | 0.50 |
| resistivity | 0 | 1 | 0 | 0 | 0 | 0 | 1.00 | – |
| transmittance | 0 | 1 | 0 | 0 | 0 | 0 | 1.00 | – |
| thickness | 2 | 1 | 0 | 1 | 2 | 0 | 0.60 | 0.67 |

## Per paper

| paper | correct | soft | wrong | missing | extra | disputed | precision | recall |
|---|---|---|---|---|---|---|---|---|
| 0000000000000001 | 6 | 3 | 3 | 4 | 3 | 0 | 0.60 | 0.46 |
| 0000000000000002 | 0 | 0 | 0 | 0 | 3 | 0 | 0.00 | – |

## Wrong, extra, missing and disputed cells

| paper | gold sample | dataset row | field | outcome | dataset value | gold | quality_rows trace |
|---|---|---|---|---|---|---|---|
| 0000000000000001 | (unaligned) stray | stray | thickness | extra | 50 |  | row matches no gold sample; single_source \| mineru_p2_b3 |
| 0000000000000001 | S-annealed | ann | mode | extra | RF |  |  |
| 0000000000000001 | target | target | resistance | extra | 0.001 |  |  |
| 0000000000000001 | S-100 | 100nm-asdep | working_pressure | missing | None | 0.5 (p2) |  |
| 0000000000000001 | S-absent |  | mode | missing | None | RF (p1) | no dataset row aligned to this gold sample |
| 0000000000000001 | S-absent |  | thickness | missing | None | 400 (p3) | no dataset row aligned to this gold sample |
| 0000000000000001 | S-absent |  | working_pressure | missing | None | 0.5 (p2) | no dataset row aligned to this gold sample |
| 0000000000000001 | S-200 | 200nm | mode | wrong | DC | RF (p1) |  |
| 0000000000000001 | S-200 | 200nm | sheet_resistance | wrong | 15 | 10 (p3) | conflict \| 15 vs 16 \| paddleocr_vl_p2_b7 |
| 0000000000000001 | target | target | inch | wrong | 2 | 3 (p2) | agree \| both lanes \| mineru_p1_b2 |
| 0000000000000002 | (unaligned) film | film | mode | extra | DC |  | row matches no gold sample;  |
| 0000000000000002 | (unaligned) film | film | thickness | extra | 10 |  | row matches no gold sample;  |
| 0000000000000002 | target | target | component | extra | ITO |  |  |
