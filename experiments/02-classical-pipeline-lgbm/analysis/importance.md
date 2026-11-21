# Feature importance (seed 42, LightGBM gain share, top 15)

## Stage A

| feature | group | gain share |
|---|---|---|
| ctx_coverage | overlap | 0.086 |
| qtype | question | 0.083 |
| q_len | question | 0.081 |
| missing_w | overlap | 0.077 |
| max_overlap_ratio | overlap | 0.068 |
| cap_cover | entity_type | 0.063 |
| win5 | window | 0.055 |
| ctx_tfidf | retrieval | 0.049 |
| max_bm25 | retrieval | 0.045 |
| mean_tfidf | retrieval | 0.044 |
| max_tfidf | retrieval | 0.040 |
| ctx_len | context | 0.035 |
| win10 | window | 0.032 |
| best_overlap_ratio | overlap | 0.032 |
| margin_tfidf | retrieval | 0.030 |

Gain share by group: overlap 0.289, retrieval 0.237, question 0.221, entity_type 0.105, window 0.105, context 0.043

## Stage C

| feature | group | gain share |
|---|---|---|
| type_match | type_match | 0.169 |
| right1_id | context_words | 0.121 |
| left1_id | context_words | 0.088 |
| q_type | type_match | 0.063 |
| c_all_cap | type_match | 0.048 |
| stop_frac | pos_heuristics | 0.043 |
| c_len_chr | shape_boundary | 0.037 |
| verb_like_frac | pos_heuristics | 0.037 |
| c_cap_frac | type_match | 0.033 |
| left2_id | context_words | 0.027 |
| right_is_cap | shape_boundary | 0.021 |
| ov_frac | overlap | 0.020 |
| rel_pos_ctx | shape_boundary | 0.020 |
| d_min | proximity | 0.019 |
| c_len_tok | shape_boundary | 0.019 |

Gain share by group: type_match 0.347, context_words 0.239, shape_boundary 0.145, proximity 0.096, pos_heuristics 0.092, sentence 0.053, overlap 0.028

## Stage B (standardized LR coefficients)

| feature | coef |
|---|---|
| n_overlap | +0.974 |
| bm25_rel | +0.954 |
| n_sent | -0.830 |
| w_overlap | -0.659 |
| tfidf | -0.632 |
| is_first | +0.580 |
| tfidf_rel | -0.548 |
| pos | -0.500 |
| type_hits | +0.406 |
| bm25 | +0.311 |
| overlap_ratio | +0.282 |
| rel_pos | -0.188 |
| bigram_overlap | -0.173 |
| overlap_rel | +0.113 |
| qtype | +0.092 |
| len | +0.066 |
