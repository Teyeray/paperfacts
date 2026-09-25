A synthetic gold set (`gold/`) and two datasets, scored once by `eval/score.py` as it stood at B0 (8bb2a08,
unchanged up to 74197c9) with the B0 field table from `config.json`:

    cd tests/fixtures/score_b0
    PYTHONPATH=../../../src python ../../../eval/score.py --gold gold --config ../../../config.json \
        --dataset 0000000000000001=dataset_1.json --dataset 0000000000000002=dataset_2.json \
        --out report.md --json cells.json

`cells.json` and `report.md` are that run's output. `tests/test_score.py` scores the same inputs from
`profiles/tco.json` and requires both to be identical. The recording is frozen: re-running it needs the B0
script, since the current one no longer reads `config.json`.
