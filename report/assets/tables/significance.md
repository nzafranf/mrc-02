# Paired bootstrap on seed-averaged per-example F1

| comparison (B - A) | delta F1 | 95% CI | p (two-sided paired bootstrap) |
|---|---|---|---|
| classical_pipeline - no_answer | +7.16 | [3.92, 10.24] | 0.0010 |
| bidaf_noattn - classical_pipeline | +3.11 | [-0.26, 6.14] | 0.0670 |
| bidaf - bidaf_noattn | +0.14 | [-1.56, 1.70] | 0.8790 |
| plm_indobert_full - bidaf | +16.93 | [14.06, 19.59] | 0.0010 |
