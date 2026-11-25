# 08 results on official_val (written by compare.py)

n = 2003 (1282 answerable, 721 unanswerable). Seed 13, full train (8042), epoch and NA threshold chosen on official_val.

| run | EM | F1 | HasAns F1 | NoAns acc | predicted no-answer % | best epoch | NA threshold |
|---|---|---|---|---|---|---|---|
| xlmr_large | 74.4 | 82.4 | 76.5 | 92.9 | 37.5 | 6 | 1.38 |
| mdeberta_v3_base | 72.6 | 80.3 | 73.0 | 93.3 | 38.2 | 6 | -1.12 |
| mbert_base | 69.9 | 77.3 | 71.4 | 87.8 | 36.3 | 8 | 2.95 |
| nusabert_large | 68.4 | 76.4 | 70.0 | 87.8 | 37.1 | 10 | 2.74 |
| indobert_base | 64.3 | 71.7 | 62.5 | 88.1 | 39.0 | 9 | 0.69 |
| albert_indolite_large | 36.0 | 36.1 | 0.2 | 100.0 | 99.8 | 1 | 2.55 |

Paired bootstrap over questions (10000 resamples, F1 points; two-sided p).

| difference | dF1 | 95% CI | p | answerable | unanswerable |
|---|---|---|---|---|---|
| mbert_base - indobert_base | +5.57 | [3.81, 7.30] | 0.0001 | +8.86 | -0.28 |
| mdeberta_v3_base - indobert_base | +8.60 | [6.85, 10.34] | 0.0001 | +10.47 | +5.27 |
| mdeberta_v3_base - mbert_base | +3.03 | [1.50, 4.59] | 0.0001 | +1.62 | +5.55 |
| xlmr_large - indobert_base | +10.68 | [8.98, 12.40] | 0.0001 | +13.95 | +4.85 |
| xlmr_large - mbert_base | +5.11 | [3.52, 6.72] | 0.0001 | +5.09 | +5.13 |
| xlmr_large - mdeberta_v3_base | +2.07 | [0.77, 3.37] | 0.0022 | +3.48 | -0.42 |
| nusabert_large - indobert_base | +4.69 | [3.13, 6.25] | 0.0001 | +7.48 | -0.28 |
| nusabert_large - mbert_base | -0.88 | [-2.53, 0.79] | 0.3044 | -1.38 | +0.00 |
| nusabert_large - mdeberta_v3_base | -3.92 | [-5.49, -2.39] | 0.0001 | -3.00 | -5.55 |
| albert_indolite_large - indobert_base | -35.62 | [-37.96, -33.27] | 0.0001 | -62.36 | +11.93 |
