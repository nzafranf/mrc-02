# F1 by slice (seed-averaged)

| slice | value | n (no_answer) | F1 no_answer | n (sliding_window) | F1 sliding_window | n (sliding_window_thr) | F1 sliding_window_thr |
|---|---|---|---|---|---|---|---|
| q_type | entity_or_def | 281 | 44.5 | 281 | 9.4 | 281 | 45.8 |
| q_type | location | 61 | 42.6 | 61 | 17.2 | 61 | 43.8 |
| q_type | number | 91 | 41.8 | 91 | 9.9 | 91 | 44.3 |
| q_type | other | 24 | 62.5 | 24 | 6.8 | 24 | 45.8 |
| q_type | person | 136 | 46.3 | 136 | 9.5 | 136 | 46.4 |
| q_type | time | 210 | 49.5 | 210 | 18.8 | 210 | 50.5 |
| ans_type | date_year | 88 | 0.0 | 88 | 38.8 | 88 | 4.0 |
| ans_type | lowercase_phrase | 119 | 0.0 | 119 | 6.8 | 119 | 1.2 |
| ans_type | none | 371 | 100.0 | 371 | 0.3 | 371 | 96.2 |
| ans_type | number | 34 | 0.0 | 34 | 26.4 | 34 | 11.3 |
| ans_type | other_cap | 121 | 0.0 | 121 | 24.7 | 121 | 7.1 |
| ans_type | person_like_cap | 70 | 0.0 | 70 | 25.5 | 70 | 2.1 |
| ans_len_bucket | 1 | 133 | 0.0 | 133 | 32.9 | 133 | 7.3 |
| ans_len_bucket | 2-3 | 162 | 0.0 | 162 | 26.3 | 162 | 3.6 |
| ans_len_bucket | 4-6 | 46 | 0.0 | 46 | 16.7 | 46 | 3.3 |
| ans_len_bucket | >6 | 91 | 0.0 | 91 | 5.5 | 91 | 2.1 |
| ans_len_bucket | none | 371 | 100.0 | 371 | 0.3 | 371 | 96.2 |
| lex_overlap_bucket | high | 147 | 35.4 | 147 | 16.7 | 147 | 38.3 |
| lex_overlap_bucket | low | 319 | 42.9 | 319 | 7.9 | 319 | 42.9 |
| lex_overlap_bucket | mid | 337 | 54.0 | 337 | 14.9 | 337 | 54.2 |
| ctx_len_bucket | 120-250 | 103 | 44.7 | 103 | 13.3 | 103 | 43.1 |
| ctx_len_bucket | 60-120 | 369 | 45.5 | 369 | 12.8 | 369 | 45.9 |
| ctx_len_bucket | <60 | 324 | 47.5 | 324 | 12.0 | 324 | 49.1 |
| ctx_len_bucket | >250 | 7 | 42.9 | 7 | 0.0 | 7 | 42.9 |
| ans_rel_pos_bucket | Q1 | 221 | 0.0 | 221 | 26.3 | 221 | 3.4 |
| ans_rel_pos_bucket | Q2 | 112 | 0.0 | 112 | 15.1 | 112 | 4.0 |
| ans_rel_pos_bucket | Q3 | 48 | 0.0 | 48 | 22.1 | 48 | 7.6 |
| ans_rel_pos_bucket | Q4 | 51 | 0.0 | 51 | 26.1 | 51 | 6.2 |
| ans_rel_pos_bucket | none | 371 | 100.0 | 371 | 0.3 | 371 | 96.2 |
| ends_space_qmark | False | 648 | 56.6 | 648 | 9.7 | 648 | 56.3 |
| ends_space_qmark | True | 155 | 2.6 | 155 | 23.9 | 155 | 7.2 |
