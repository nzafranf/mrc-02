# F1 by slice (seed-averaged)

| slice | value | n (Qwen2.5-14B-Instruct_fs_id) | F1 Qwen2.5-14B-Instruct_fs_id |
|---|---|---|---|
| q_type | entity_or_def | 350 | 75.1 |
| q_type | location | 101 | 64.1 |
| q_type | number | 153 | 76.9 |
| q_type | other | 31 | 75.7 |
| q_type | person | 168 | 76.4 |
| q_type | time | 194 | 79.7 |
| q_type | yesno_like | 3 | 66.7 |
| ans_type | date_year | 116 | 73.3 |
| ans_type | lowercase_phrase | 196 | 69.0 |
| ans_type | none | 360 | 85.0 |
| ans_type | number | 67 | 76.7 |
| ans_type | other_cap | 165 | 68.0 |
| ans_type | person_like_cap | 96 | 66.7 |
| ans_len_bucket | 1 | 170 | 72.9 |
| ans_len_bucket | 2-3 | 233 | 64.3 |
| ans_len_bucket | 4-6 | 88 | 75.6 |
| ans_len_bucket | >6 | 149 | 72.3 |
| ans_len_bucket | none | 360 | 85.0 |
| lex_overlap_bucket | high | 214 | 78.4 |
| lex_overlap_bucket | low | 348 | 67.9 |
| lex_overlap_bucket | mid | 438 | 79.8 |
| ctx_len_bucket | 120-250 | 156 | 76.4 |
| ctx_len_bucket | 60-120 | 432 | 75.3 |
| ctx_len_bucket | <60 | 401 | 74.8 |
| ctx_len_bucket | >250 | 11 | 87.6 |
| ans_rel_pos_bucket | Q1 | 343 | 74.0 |
| ans_rel_pos_bucket | Q2 | 142 | 66.7 |
| ans_rel_pos_bucket | Q3 | 83 | 59.5 |
| ans_rel_pos_bucket | Q4 | 72 | 69.5 |
| ans_rel_pos_bucket | none | 360 | 85.0 |
| ends_space_qmark | False | 804 | 75.9 |
| ends_space_qmark | True | 196 | 73.1 |
