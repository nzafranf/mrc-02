# Hard set composition: 86 of 803 dev examples, F1<0.5 for every seed of classical_pipeline, bidaf_noattn, bidaf, plm_indobert_full

| slice | value | n_hard | share_of_hard | share_of_dev | hard_rate_in_slice |
|---|---|---|---|---|---|
| q_type | entity_or_def | 35 | 40.7% | 35.0% | 12.5% |
| q_type | location | 7 | 8.1% | 7.6% | 11.5% |
| q_type | number | 8 | 9.3% | 11.3% | 8.8% |
| q_type | other | 6 | 7.0% | 3.0% | 25.0% |
| q_type | person | 20 | 23.3% | 16.9% | 14.7% |
| q_type | time | 10 | 11.6% | 26.2% | 4.8% |
| ans_type | date_year | 7 | 8.1% | 11.0% | 8.0% |
| ans_type | lowercase_phrase | 29 | 33.7% | 14.8% | 24.4% |
| ans_type | none | 2 | 2.3% | 46.2% | 0.5% |
| ans_type | number | 5 | 5.8% | 4.2% | 14.7% |
| ans_type | other_cap | 31 | 36.0% | 15.1% | 25.6% |
| ans_type | person_like_cap | 12 | 14.0% | 8.7% | 17.1% |
| lex_overlap_bucket | high | 19 | 22.1% | 18.3% | 12.9% |
| lex_overlap_bucket | low | 43 | 50.0% | 39.7% | 13.5% |
| lex_overlap_bucket | mid | 24 | 27.9% | 42.0% | 7.1% |
| ctx_len_bucket | 120-250 | 20 | 23.3% | 12.8% | 19.4% |
| ctx_len_bucket | 60-120 | 43 | 50.0% | 46.0% | 11.7% |
| ctx_len_bucket | <60 | 20 | 23.3% | 40.3% | 6.2% |
| ctx_len_bucket | >250 | 3 | 3.5% | 0.9% | 42.9% |
| ans_len_bucket | 1 | 21 | 24.4% | 16.6% | 15.8% |
| ans_len_bucket | 2-3 | 25 | 29.1% | 20.2% | 15.4% |
| ans_len_bucket | 4-6 | 10 | 11.6% | 5.7% | 21.7% |
| ans_len_bucket | >6 | 28 | 32.6% | 11.3% | 30.8% |
| ans_len_bucket | none | 2 | 2.3% | 46.2% | 0.5% |
| is_impossible | False | 84 | 97.7% | 53.8% | 19.4% |
| is_impossible | True | 2 | 2.3% | 46.2% | 0.5% |
