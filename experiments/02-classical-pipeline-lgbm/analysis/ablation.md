# Feature-group ablation (dev, seed 42)

Each row retrains one stage without one feature group, keeping the other stage at its full-feature model. F1 is overall dev F1 at the dev-tuned no-answer threshold (re-tuned per row). Negative delta means the group helps.

Full model: F1 53.09, EM 50.81, span-only HasAns F1 (ignores stage A, answerable questions) 52.03.

## Stage C (span ranker)

| dropped group | #feat | F1 | dF1 | EM | span F1 (answerable, no thr) | d span F1 |
|---|---|---|---|---|---|---|
| type_match | 13 | 51.08 | -2.01 | 49.81 | 35.49 | -16.54 |
| overlap | 4 | 52.52 | -0.57 | 50.68 | 49.92 | -2.11 |
| context_words | 7 | 52.57 | -0.52 | 50.81 | 49.91 | -2.12 |
| proximity | 9 | 52.64 | -0.45 | 50.93 | 51.51 | -0.53 |
| sentence | 6 | 52.71 | -0.38 | 51.18 | 50.62 | -1.41 |
| shape_boundary | 11 | 53.01 | -0.08 | 51.18 | 49.81 | -2.23 |
| pos_heuristics | 6 | 53.60 | +0.51 | 51.56 | 51.88 | -0.15 |

## Stage A (answerability classifier)

| dropped group | #feat | F1 | dF1 | EM | AUC | NoAns acc |
|---|---|---|---|---|---|---|
| question | 5 | 52.15 | -0.94 | 50.44 | 0.712 | 90.03 |
| entity_type | 4 | 53.01 | -0.08 | 51.31 | 0.723 | 92.45 |
| context | 2 | 53.02 | -0.07 | 50.44 | 0.742 | 78.17 |
| overlap | 7 | 53.02 | -0.07 | 50.68 | 0.744 | 77.09 |
| window | 3 | 53.03 | -0.06 | 50.56 | 0.748 | 77.90 |
| retrieval | 6 | 54.05 | +0.96 | 51.81 | 0.746 | 84.10 |

Note: single seed, 803 dev questions; differences below about 0.5 F1 are within noise.
