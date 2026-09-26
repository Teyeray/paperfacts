# PaperFacts gold evaluation

Micro: precision 0.99, recall 0.93. Macro over papers: precision 0.99, recall 0.89.

Datasets scored:

- `08562126a9b9aab8`: `/home/yangrm/Projects/paperfacts/data/docs/08562126a9b9aab8/datasets/60039cd934bd.8a8e87040b84.json`
- `1048c42316a5c9f8`: `/home/yangrm/Projects/paperfacts/data/docs/1048c42316a5c9f8/datasets/60039cd934bd.8a8e87040b84.json`
- `219df6e1cd7b19fe`: `/home/yangrm/Projects/paperfacts/data/docs/219df6e1cd7b19fe/datasets/60039cd934bd.8a8e87040b84.json`
- `534040e6151e0636`: `/home/yangrm/Projects/paperfacts/data/docs/534040e6151e0636/datasets/60039cd934bd.8a8e87040b84.json`
- `5c10f7a0128f15e0`: `/home/yangrm/Projects/paperfacts/data/docs/5c10f7a0128f15e0/datasets/60039cd934bd.8a8e87040b84.json`
- `80c3b69d570c2b6d`: `/home/yangrm/Projects/paperfacts/data/docs/80c3b69d570c2b6d/datasets/60039cd934bd.8a8e87040b84.json`
- `8977655673fa6d9a`: `/home/yangrm/Projects/paperfacts/data/docs/8977655673fa6d9a/datasets/60039cd934bd.8a8e87040b84.json`
- `c3ab31d08acc066b`: `/home/yangrm/Projects/paperfacts/data/docs/c3ab31d08acc066b/datasets/60039cd934bd.8a8e87040b84.json`
- `e6939c89e983c426`: `/home/yangrm/Projects/paperfacts/data/docs/e6939c89e983c426/datasets/60039cd934bd.8a8e87040b84.json`
- `e855631c6f46a0ee`: `/home/yangrm/Projects/paperfacts/data/docs/e855631c6f46a0ee/datasets/60039cd934bd.8a8e87040b84.json`
- `ffd70c234c43ba93`: `/home/yangrm/Projects/paperfacts/data/docs/ffd70c234c43ba93/datasets/60039cd934bd.8a8e87040b84.json`

## Overall

| all | correct | soft | wrong | missing | extra | disputed | precision | recall |
|---|---|---|---|---|---|---|---|---|
| all | 654 | 46 | 0 | 48 | 7 | 0 | 0.99 | 0.93 |

## Per field group

| group | correct | soft | wrong | missing | extra | disputed | precision | recall |
|---|---|---|---|---|---|---|---|---|
| film | 155 | 14 | 0 | 10 | 7 | 0 | 0.96 | 0.94 |
| process | 494 | 32 | 0 | 36 | 0 | 0 | 1.00 | 0.93 |
| target | 5 | 0 | 0 | 2 | 0 | 0 | 1.00 | 0.71 |

## Per field

| field | correct | soft | wrong | missing | extra | disputed | precision | recall |
|---|---|---|---|---|---|---|---|---|
| component | 2 | 0 | 0 | 2 | 0 | 0 | 1.00 | 0.50 |
| inch | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 |
| sputtering_time | 47 | 4 | 0 | 0 | 0 | 0 | 1.00 | 1.00 |
| sputtering_power | 67 | 9 | 0 | 0 | 0 | 0 | 1.00 | 1.00 |
| mode | 70 | 15 | 0 | 8 | 0 | 0 | 1.00 | 0.90 |
| ar_flow_rate | 57 | 0 | 0 | 15 | 0 | 0 | 1.00 | 0.79 |
| o2_flow_rate | 12 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 |
| o2_ratio | 43 | 0 | 0 | 2 | 0 | 0 | 1.00 | 0.96 |
| h2_ratio | 8 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 |
| working_pressure | 39 | 0 | 0 | 9 | 0 | 0 | 1.00 | 0.81 |
| target_substrate_distance | 4 | 0 | 0 | 1 | 0 | 0 | 1.00 | 0.80 |
| substrate_temperature | 51 | 2 | 0 | 1 | 0 | 0 | 1.00 | 0.98 |
| annealing_temperature | 45 | 1 | 0 | 0 | 0 | 0 | 1.00 | 1.00 |
| annealing_time | 45 | 1 | 0 | 0 | 0 | 0 | 1.00 | 1.00 |
| rotation_speed | 6 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 |
| sheet_resistance | 20 | 0 | 0 | 3 | 0 | 0 | 1.00 | 0.87 |
| resistivity | 27 | 5 | 0 | 1 | 1 | 0 | 0.97 | 0.96 |
| transmittance | 63 | 2 | 0 | 1 | 6 | 0 | 0.92 | 0.98 |
| thickness | 45 | 7 | 0 | 5 | 0 | 0 | 1.00 | 0.90 |

## Per paper

| paper | correct | soft | wrong | missing | extra | disputed | precision | recall |
|---|---|---|---|---|---|---|---|---|
| 08562126a9b9aab8 | 41 | 0 | 0 | 7 | 0 | 0 | 1.00 | 0.85 |
| 1048c42316a5c9f8 | 32 | 14 | 0 | 18 | 0 | 0 | 1.00 | 0.64 |
| 219df6e1cd7b19fe | 22 | 0 | 0 | 5 | 0 | 0 | 1.00 | 0.81 |
| 534040e6151e0636 | 58 | 0 | 0 | 1 | 6 | 0 | 0.91 | 0.98 |
| 5c10f7a0128f15e0 | 33 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 |
| 80c3b69d570c2b6d | 39 | 2 | 0 | 11 | 1 | 0 | 0.98 | 0.78 |
| 8977655673fa6d9a | 21 | 20 | 0 | 5 | 0 | 0 | 1.00 | 0.81 |
| c3ab31d08acc066b | 45 | 8 | 0 | 1 | 0 | 0 | 1.00 | 0.98 |
| e6939c89e983c426 | 308 | 2 | 0 | 0 | 0 | 0 | 1.00 | 1.00 |
| e855631c6f46a0ee | 55 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 |
| ffd70c234c43ba93 | 0 | 0 | 0 | 0 | 0 | 0 | – | – |

## Wrong, extra, missing and disputed cells

| paper | gold sample | dataset row | field | outcome | dataset value | gold | quality_rows trace |
|---|---|---|---|---|---|---|---|
| 08562126a9b9aab8 | ITO-170nm-ann350 | 170nm-ann350C \| ITO-170nm-ann350 | mode | missing | None | RF (p1) | conflict \| 双路比较存在冲突或歧义，需人工复核; mineru: rfmagnetron sputtering ; paddleocr_vl: rf-magnetron sputtering  \| mineru_p1_b1; paddleocr_vl_p1_b1 |
| 08562126a9b9aab8 | ITO-170nm-asdep | 170nm-as-grown \| ITO-170nm-asdep | mode | missing | None | RF (p1) | conflict \| 双路比较存在冲突或歧义，需人工复核; mineru: rfmagnetron sputtering ; paddleocr_vl: rf-magnetron sputtering  \| mineru_p1_b1; paddleocr_vl_p1_b1 |
| 08562126a9b9aab8 | ITO-350nm-ann250 | 350nm-ann250C \| ITO-350nm-ann250 | mode | missing | None | RF (p1) | conflict \| 双路比较存在冲突或歧义，需人工复核; mineru: rfmagnetron sputtering ; paddleocr_vl: rf-magnetron sputtering  \| mineru_p1_b1; paddleocr_vl_p1_b1 |
| 08562126a9b9aab8 | ITO-350nm-ann350 | 350nm-ann350C \| ITO-350nm-ann350 | mode | missing | None | RF (p1) | conflict \| 双路比较存在冲突或歧义，需人工复核; mineru: rfmagnetron sputtering ; paddleocr_vl: rf-magnetron sputtering  \| mineru_p1_b1; paddleocr_vl_p1_b1 |
| 08562126a9b9aab8 | ITO-350nm-asdep | 350nm-as-grown \| ITO-350nm-asdep | mode | missing | None | RF (p1) | conflict \| 双路比较存在冲突或歧义，需人工复核; mineru: rfmagnetron sputtering ; paddleocr_vl: rf-magnetron sputtering  \| mineru_p1_b1; paddleocr_vl_p1_b1 |
| 08562126a9b9aab8 | ITO-700nm-ann350 | 700nm-ann350C \| ITO-700nm-ann350 | mode | missing | None | RF (p1) | conflict \| 双路比较存在冲突或歧义，需人工复核; mineru: rfmagnetron sputtering ; paddleocr_vl: rf-magnetron sputtering  \| mineru_p1_b1; paddleocr_vl_p1_b1 |
| 08562126a9b9aab8 | ITO-700nm-asdep | 700nm-as-grown \| ITO-700nm-asdep | mode | missing | None | RF (p1) | conflict \| 双路比较存在冲突或歧义，需人工复核; mineru: rfmagnetron sputtering ; paddleocr_vl: rf-magnetron sputtering  \| mineru_p1_b1; paddleocr_vl_p1_b1 |
| 1048c42316a5c9f8 | G10 | GZO-graded-ITO-G10 \| G10 | ar_flow_rate | missing | None | 20 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 1048c42316a5c9f8 | G10 | GZO-graded-ITO-G10 \| G10 | thickness | missing | None | 150 (p1); 10 [amb] (p1) | multiple_conditions \| graded GZO layer thickness; total thickness of GZO-graded ITO film; GZO-graded region thickness \| 同一解析通道记录了多种测量条件，无法唯一确定; mineru: 10 nm; mineru: 150 nm; paddleocr_vl: 10 nm; paddleocr_vl: 150 nm \| mineru_p1_b5; paddleocr_vl_p1_b5 |
| 1048c42316a5c9f8 | G10 | GZO-graded-ITO-G10 \| G10 | working_pressure | missing | None | 0.399966 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 1048c42316a5c9f8 | G15 | GZO-graded-ITO-G15 \| G15 | ar_flow_rate | missing | None | 20 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 1048c42316a5c9f8 | G15 | GZO-graded-ITO-G15 \| G15 | thickness | missing | None | 150 (p1); 15 [amb] (p1) | multiple_conditions \| graded GZO layer thickness; total thickness of GZO-graded ITO film; GZO-graded region thickness \| 同一解析通道记录了多种测量条件，无法唯一确定; mineru: 15 nm; mineru: 150 nm; paddleocr_vl: 15 nm; paddleocr_vl: 150 nm \| mineru_p1_b5; paddleocr_vl_p1_b5 |
| 1048c42316a5c9f8 | G15 | GZO-graded-ITO-G15 \| G15 | working_pressure | missing | None | 0.399966 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 1048c42316a5c9f8 | G20 | GZO-graded-ITO-G20 \| G20 | ar_flow_rate | missing | None | 20 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 1048c42316a5c9f8 | G20 | GZO-graded-ITO-G20 \| G20 | thickness | missing | None | 150 (p1); 20 [amb] (p1) | multiple_conditions \| graded GZO layer thickness; total thickness of GZO-graded ITO film; GZO-graded region thickness \| 同一解析通道记录了多种测量条件，无法唯一确定; mineru: 20 nm; mineru: 150 nm; paddleocr_vl: 20 nm; paddleocr_vl: 150 nm \| mineru_p1_b5; paddleocr_vl_p1_b5 |
| 1048c42316a5c9f8 | G20 | GZO-graded-ITO-G20 \| G20 | working_pressure | missing | None | 0.399966 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 1048c42316a5c9f8 | G5 | GZO-graded-ITO-G5 \| G5 | ar_flow_rate | missing | None | 20 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 1048c42316a5c9f8 | G5 | GZO-graded-ITO-G5 \| G5 | thickness | missing | None | 150 (p1); 5 [amb] (p1) | multiple_conditions \| graded GZO layer thickness; total thickness of GZO-graded ITO film; GZO-graded region thickness \| 同一解析通道记录了多种测量条件，无法唯一确定; mineru: 5 nm; mineru: 150 nm; paddleocr_vl: 5 nm; paddleocr_vl: 150 nm \| mineru_p1_b5; paddleocr_vl_p1_b5 |
| 1048c42316a5c9f8 | G5 | GZO-graded-ITO-G5 \| G5 | working_pressure | missing | None | 0.399966 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 1048c42316a5c9f8 | GZO-ITO-bilayer-15 | GZO-ITO-bilayer-15 \| GZO_ITO_bilayer_15 | ar_flow_rate | missing | None | 20 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 1048c42316a5c9f8 | GZO-ITO-bilayer-15 | GZO-ITO-bilayer-15 \| GZO_ITO_bilayer_15 | working_pressure | missing | None | 0.399966 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 1048c42316a5c9f8 | GZO-ITO-bilayer-20 | GZO-ITO-bilayer-20 \| GZO_ITO_bilayer_20 | ar_flow_rate | missing | None | 20 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 1048c42316a5c9f8 | GZO-ITO-bilayer-20 | GZO-ITO-bilayer-20 \| GZO_ITO_bilayer_20 | working_pressure | missing | None | 0.399966 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 1048c42316a5c9f8 | ITO-monolayer | ITO-monolayer \| ITO_monolayer | ar_flow_rate | missing | None | 20 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 1048c42316a5c9f8 | ITO-monolayer | ITO-monolayer \| ITO_monolayer | working_pressure | missing | None | 0.399966 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 219df6e1cd7b19fe | ICO-150 |  | mode | missing | None | RF (p2) | no dataset row aligned to this gold sample |
| 219df6e1cd7b19fe | ICO-150 |  | substrate_temperature | missing | None | 150 (p3) | no dataset row aligned to this gold sample |
| 219df6e1cd7b19fe | ICO-150 |  | target_substrate_distance | missing | None | 19 (p2) | no dataset row aligned to this gold sample |
| 219df6e1cd7b19fe | ICO-150 |  | thickness | missing | None | 100 (p2) | no dataset row aligned to this gold sample |
| 219df6e1cd7b19fe | ICO-150 |  | working_pressure | missing | None | 0.199983 (p3) | no dataset row aligned to this gold sample |
| 534040e6151e0636 | HN400 | HN400 | transmittance | extra | 90.0 |  | single_source \| 400 nm to 800 nm \| 已排除不是唯一标量的候选（mineru: above 90 %）：不是唯一精确标量（含上下界、区间、尺寸组合或无法解析的文字）; 采用 paddleocr_vl；抽取重复一致率 1；合并重复证据 \| paddleocr_vl_p6_b2 |
| 534040e6151e0636 | HN500 | HN500 | transmittance | extra | 90.0 |  | single_source \| 400 nm to 800 nm \| 已排除不是唯一标量的候选（mineru: above 90 %）：不是唯一精确标量（含上下界、区间、尺寸组合或无法解析的文字）; 采用 paddleocr_vl；抽取重复一致率 1；合并重复证据 \| paddleocr_vl_p6_b2 |
| 534040e6151e0636 | N400 | N400 | transmittance | extra | 90.0 |  | single_source \| 400 nm to 800 nm \| 采用 paddleocr_vl；抽取重复一致率 1；合并重复证据 \| paddleocr_vl_p6_b2 |
| 534040e6151e0636 | N450 | N450 | transmittance | extra | 90.0 |  | single_source \| 400 nm to 800 nm \| 采用 paddleocr_vl；抽取重复一致率 1；合并重复证据 \| paddleocr_vl_p6_b2 |
| 534040e6151e0636 | N500 | N500 | transmittance | extra | 90.0 |  | single_source \| 400 nm to 800 nm \| 采用 paddleocr_vl；抽取重复一致率 1；合并重复证据 \| paddleocr_vl_p6_b2 |
| 534040e6151e0636 | as-deposited | as-deposited \| As-deposited | transmittance | extra | 90.0 |  | single_source \| 400 nm to 800 nm \| 已排除不是唯一标量的候选（mineru: above 90 %）：不是唯一精确标量（含上下界、区间、尺寸组合或无法解析的文字）; 采用 paddleocr_vl；抽取重复一致率 1；合并重复证据 \| paddleocr_vl_p6_b2 |
| 534040e6151e0636 | paper | paper | component | missing | None | 3 wt.% Ga2O3 + 97 wt.% ZnO (p1) | conflict \| 双路比较存在冲突或歧义，需人工复核; mineru: 3 wt.% wt.%; mineru: 97 wt.% wt.%; paddleocr_vl: 3 wt.% gallium oxide (purity 99.95%) and 97 wt.% zinc oxide (purity 99.95%) wt.% \| mineru_p1_b3; paddleocr_vl_p1_b3 |
| 80c3b69d570c2b6d | ITO-100nm-asdep | ITO-100nm-asdeposited \| ITO-100nm-asdep | resistivity | extra | 0.00018600000000000002 |  | agree \| average resistivity of eight 100 nm ITO films, as-deposited (10% O2 / 0.6% H2) \| 采用 mineru；抽取重复一致率 1；合并重复证据 \| mineru_p2_b7; paddleocr_vl_p2_b7 |
| 80c3b69d570c2b6d | ICO-100nm-annealed | ICO-100nm-0.6H2-annealed180C \| ICO-100nm-0.6H2-annealed-180C | ar_flow_rate | missing | None | 40 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 80c3b69d570c2b6d | ICO-100nm-asdep | ICO-100nm-0.6H2-asdeposited \| ICO-100nm-0.6H2-asdep | ar_flow_rate | missing | None | 40 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 80c3b69d570c2b6d | ICO-30nm-0.6H2-annealed | ICO-30nm-0.6H2-annealed180C \| ICO-30nm-0.6H2-annealed-180C | ar_flow_rate | missing | None | 40 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 80c3b69d570c2b6d | ICO-30nm-0.6H2-asdep | ICO-30nm-0.6H2-asdeposited \| ICO-30nm-0.6H2-asdep | ar_flow_rate | missing | None | 40 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 80c3b69d570c2b6d | ICO-30nm-1H2-annealed | ICO-30nm-1H2-annealed180C \| ICO-30nm-1H2-annealed-180C | ar_flow_rate | missing | None | 40 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 80c3b69d570c2b6d | ICO-30nm-1H2-asdep | ICO-30nm-1H2-asdeposited \| ICO-30nm-1H2-asdep | ar_flow_rate | missing | None | 40 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 80c3b69d570c2b6d | ITO-100nm-annealed | ITO-100nm-annealed180C \| ITO-100nm-annealed-180C | ar_flow_rate | missing | None | 40 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 80c3b69d570c2b6d | ITO-100nm-annealed | ITO-100nm-annealed180C \| ITO-100nm-annealed-180C | o2_ratio | missing | None | 10 @share of reactive gas; basis not stated (p2) | missing \| 未提取到该字段；留空，不填 0 |
| 80c3b69d570c2b6d | ITO-100nm-annealed | ITO-100nm-annealed180C \| ITO-100nm-annealed-180C | resistivity | missing | None | 0.000186 (p2) | missing \| 未提取到该字段；留空，不填 0 |
| 80c3b69d570c2b6d | ITO-100nm-asdep | ITO-100nm-asdeposited \| ITO-100nm-asdep | ar_flow_rate | missing | None | 40 (p1) | missing \| 未提取到该字段；留空，不填 0 |
| 80c3b69d570c2b6d | ITO-100nm-asdep | ITO-100nm-asdeposited \| ITO-100nm-asdep | o2_ratio | missing | None | 10 @share of reactive gas; basis not stated (p2) | missing \| 未提取到该字段；留空，不填 0 |
| 8977655673fa6d9a | ATO-140nm | ATO-140nm | working_pressure | missing | None | 1.0 (p2) | multiple_conditions \| optimized sputtering conditions \| 同一解析通道记录了多种测量条件，无法唯一确定; mineru: 1.0 Pa; paddleocr_vl: 0.6–1.2 Pa; paddleocr_vl: 1.0 Pa \| mineru_p1_b12; mineru_p2_b12; paddleocr_vl_p1_b6; paddleocr_vl_p2_b3 |
| 8977655673fa6d9a | ATO-280nm | ATO-280nm | sheet_resistance | missing | None | 583 (p2); 1122.2 [amb] (p3); 1200.1 [amb] (p3) | multiple_conditions \| before heat (85 °C, 170 h); after heat (85 °C, 170 h); before damp (30 ± 10 RH%, 800 h); after damp (30 ± 10 RH%, 800 h); Before Heat; After Heat; Before Damp; After Damp \| 同一解析通道记录了多种测量条件，无法唯一确定; mineru: 583 Ω/sq; mineru: 1122.2 Ω/sq; mineru: 1143.4 Ω/sq; mineru: 1200.1 Ω/sq |
| 8977655673fa6d9a | ATO-ITO-bilayer | ATO/ITO-140/25nm \| ATO/ITO | sheet_resistance | missing | None | 130 (p2); 210.3 [amb] (p3); 223.2 [amb] (p3) | multiple_conditions \| before heat (85 °C, 170 h); after heat (85 °C, 170 h); before damp (30 ± 10 RH%, 800 h); after damp (30 ± 10 RH%, 800 h); Before Heat; After Heat; Before Damp; After Damp \| 同一解析通道记录了多种测量条件，无法唯一确定; mineru: 130 Ω/sq; mineru: 210.3 Ω/sq; mineru: 212.3 Ω/sq; mineru: 223.2 Ω/sq; m |
| 8977655673fa6d9a | ITO-165nm | ITO-165nm | sheet_resistance | missing | None | 21 (p2); 34.1 [amb] (p3); 33.2 [amb] (p3) | multiple_conditions \| before heat (85 °C, 170 h); after heat (85 °C, 170 h); before damp (30 ± 10 RH%, 800 h); after damp (30 ± 10 RH%, 800 h); Before Heat; After Heat; Before Damp; After Damp \| 同一解析通道记录了多种测量条件，无法唯一确定; mineru: 21 Ω/sq; mineru: 34.1 Ω/sq; mineru: 35.5 Ω/sq; mineru: 33.2 Ω/sq; miner |
| 8977655673fa6d9a | ITO-165nm | ITO-165nm | transmittance | missing | None | 81.99 @average 380-780 nm (p4); 82.08 @average 380-780 nm [amb] (p3) | multiple_conditions \| visible spectrum of 380–780 nm; 380–780 nm wavelength range (165 nm ITO front electrode); average over 380–780 nm; before heat (85 °C) test; average over 380–780 nm; after heat (85 °C) test; average over 380–780 nm; before damp (30 ± 10 RH%) test; average over 380–780 nm; afte |
| c3ab31d08acc066b | paper | paper | component | missing | None | ITO, In2O3:SnO2 90:10 (mass) (p0) | multiple_values \| sputtering target \| 多个候选值未经双路一致确认，无法唯一确定; mineru: with a mass ratio of In2O3 to SnO2 of 90% to 10% ; paddleocr_vl: ITO, with a mass ratio of In₂O₃ to SnO₂ of 90% to 10%  \| mineru_p0_b10; paddleocr_vl_p0_b9; paddleocr_vl_p2_b3 |
