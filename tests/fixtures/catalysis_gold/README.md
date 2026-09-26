A synthetic gold set for the `catalysis` example profile: two made-up documents, their gold files (`gold/`), a
hand-written dataset for each (`datasets/`), and the cells `eval/score.py` must produce (`expected.json`). It pins
how the scorer treats what the profile uses -- a list scored per element, a date, an interval, a yes/no field and a
reference resolved to the row aligned to the gold sample it names. No real paper is behind it: it measures the
scorer, not extraction quality.
