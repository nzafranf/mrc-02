# Hard set composition: 85 of 803 dev examples, F1<0.5 for every seed of classical_pipeline, bidaf_noattn, bidaf, plm_indobert_full

| slice | value | n_hard | share_of_hard | share_of_dev | hard_rate_in_slice |
|---|---|---|---|---|---|
| q_type | entity_or_def | 34 | 40.0% | 35.0% | 12.1% |
| q_type | location | 7 | 8.2% | 7.6% | 11.5% |
| q_type | number | 9 | 10.6% | 11.3% | 9.9% |
| q_type | other | 6 | 7.1% | 3.0% | 25.0% |
| q_type | person | 19 | 22.4% | 16.9% | 14.0% |
| q_type | time | 10 | 11.8% | 26.2% | 4.8% |
| ans_type | date_year | 8 | 9.4% | 11.0% | 9.1% |
| ans_type | lowercase_phrase | 28 | 32.9% | 14.8% | 23.5% |
| ans_type | none | 2 | 2.4% | 46.2% | 0.5% |
| ans_type | number | 5 | 5.9% | 4.2% | 14.7% |
| ans_type | other_cap | 31 | 36.5% | 15.1% | 25.6% |
| ans_type | person_like_cap | 11 | 12.9% | 8.7% | 15.7% |
| lex_overlap_bucket | high | 19 | 22.4% | 18.3% | 12.9% |
| lex_overlap_bucket | low | 42 | 49.4% | 39.7% | 13.2% |
| lex_overlap_bucket | mid | 24 | 28.2% | 42.0% | 7.1% |
| ctx_len_bucket | 120-250 | 19 | 22.4% | 12.8% | 18.4% |
| ctx_len_bucket | 60-120 | 43 | 50.6% | 46.0% | 11.7% |
| ctx_len_bucket | <60 | 20 | 23.5% | 40.3% | 6.2% |
| ctx_len_bucket | >250 | 3 | 3.5% | 0.9% | 42.9% |
| ans_len_bucket | 1 | 22 | 25.9% | 16.6% | 16.5% |
| ans_len_bucket | 2-3 | 24 | 28.2% | 20.2% | 14.8% |
| ans_len_bucket | 4-6 | 10 | 11.8% | 5.7% | 21.7% |
| ans_len_bucket | >6 | 27 | 31.8% | 11.3% | 29.7% |
| ans_len_bucket | none | 2 | 2.4% | 46.2% | 0.5% |
| is_impossible | False | 83 | 97.6% | 53.8% | 19.2% |
| is_impossible | True | 2 | 2.4% | 46.2% | 0.5% |
