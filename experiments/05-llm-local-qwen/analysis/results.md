# Dev results (NA threshold tuned on dev, dev-optimistic)

| variant | runs | n | EM | F1 | HasAns F1 | NoAns acc | predicted no-answer % | NA threshold |
|---|---|---|---|---|---|---|---|---|
| qwen0.5b_en1_0shot_n100 | 1 | 100 | 43.0 | 49.4 | 17.9 | 92.9 | 76.0 | 0.0515 |
| qwen0.5b_id1_0shot_n100 | 1 | 100 | 46.0 | 51.9 | 17.1 | 100.0 | 79.0 | 0.0583 |
| qwen0.5b_id1_0shot_n60_first_token | 1 | 60 | 45.0 | 52.1 | 20.2 | 100.0 | 76.7 | 0.0583 |
| qwen0.5b_id1_0shot_n60_seqlogprob | 1 | 60 | 45.0 | 46.1 | 10.2 | 100.0 | 93.3 | 0.0321 |
| qwen0.5b_id1_3shot_fixed_n100 | 1 | 100 | 45.0 | 48.5 | 12.9 | 97.6 | 87.0 | 0.000506 |
