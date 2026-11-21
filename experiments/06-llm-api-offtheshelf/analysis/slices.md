# F1 by slice (seed-averaged)

| slice | value | n (gemini_35flash_id1_0shot) | F1 gemini_35flash_id1_0shot | n (nemotron35_super_id1_0shot) | F1 nemotron35_super_id1_0shot |
|---|---|---|---|---|---|
| q_type | entity_or_def | 8 | 87.5 | 9 | 100.0 |
| q_type | location | 1 | 0.0 | 2 | 50.0 |
| q_type | number | 2 | 50.0 | 3 | 55.6 |
| q_type | other | 0 | nan | 0 | nan |
| q_type | person | 3 | 33.3 | 3 | 93.3 |
| q_type | time | 8 | 37.5 | 8 | 70.8 |
| ans_type | date_year | 4 | 25.0 | 4 | 41.7 |
| ans_type | lowercase_phrase | 3 | 33.3 | 3 | 66.7 |
| ans_type | none | 7 | 100.0 | 8 | 100.0 |
| ans_type | number | 1 | 0.0 | 2 | 83.3 |
| ans_type | other_cap | 4 | 50.0 | 5 | 80.0 |
| ans_type | person_like_cap | 3 | 33.3 | 3 | 93.3 |
| ans_len_bucket | 1 | 6 | 16.7 | 7 | 66.7 |
| ans_len_bucket | 2-3 | 6 | 50.0 | 7 | 78.1 |
| ans_len_bucket | 4-6 | 2 | 50.0 | 2 | 100.0 |
| ans_len_bucket | >6 | 1 | 0.0 | 1 | 0.0 |
| ans_len_bucket | none | 7 | 100.0 | 8 | 100.0 |
| lex_overlap_bucket | high | 3 | 100.0 | 4 | 91.7 |
| lex_overlap_bucket | low | 9 | 33.3 | 10 | 68.0 |
| lex_overlap_bucket | mid | 10 | 60.0 | 11 | 87.9 |
| ctx_len_bucket | 120-250 | 1 | 0.0 | 1 | 0.0 |
| ctx_len_bucket | 60-120 | 8 | 75.0 | 10 | 96.7 |
| ctx_len_bucket | <60 | 11 | 45.5 | 12 | 72.2 |
| ctx_len_bucket | >250 | 2 | 50.0 | 2 | 90.0 |
| ans_rel_pos_bucket | Q1 | 6 | 33.3 | 6 | 63.3 |
| ans_rel_pos_bucket | Q2 | 2 | 100.0 | 2 | 100.0 |
| ans_rel_pos_bucket | Q3 | 3 | 0.0 | 4 | 66.7 |
| ans_rel_pos_bucket | Q4 | 4 | 25.0 | 5 | 73.3 |
| ans_rel_pos_bucket | none | 7 | 100.0 | 8 | 100.0 |
| ends_space_qmark | False | 19 | 63.2 | 22 | 82.4 |
| ends_space_qmark | True | 3 | 0.0 | 3 | 66.7 |
