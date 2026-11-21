# Paired bootstrap on seed-averaged per-question F1

| comparison | n | delta F1 | 95% CI | p (paired bootstrap) |
|---|---|---|---|---|
| qwen0.5b_id1_0shot_n100 - qwen0.5b_en1_0shot_n100 | 100 | +2.53 | [-1.48, 7.08] | 0.2290 |
| qwen0.5b_id1_0shot_n60_first_token - qwen0.5b_en1_0shot_n100 | 60 | +1.92 | [-4.19, 8.19] | 0.5070 |
| qwen0.5b_id1_0shot_n60_seqlogprob - qwen0.5b_en1_0shot_n100 | 60 | -4.06 | [-11.48, 3.53] | 0.2910 |
| qwen0.5b_id1_3shot_fixed_n100 - qwen0.5b_en1_0shot_n100 | 100 | -0.90 | [-8.06, 5.65] | 0.7690 |
