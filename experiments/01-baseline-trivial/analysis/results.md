# Dev results (NA threshold tuned on dev, dev-optimistic)

| variant | runs | n | EM | F1 | HasAns F1 | NoAns acc | predicted no-answer % | NA threshold |
|---|---|---|---|---|---|---|---|---|
| no_answer | 1 | 803 | 46.2 | 46.2 | 0.0 | 100.0 | 100.0 | -1e+09 |
| sliding_window | 1 | 803 | 9.2 | 12.5 | 22.9 | 0.3 | 0.7 | untuned |
| sliding_window_thr | 1 | 803 | 46.1 | 46.8 | 4.4 | 96.2 | 88.7 | 0.0939 |
