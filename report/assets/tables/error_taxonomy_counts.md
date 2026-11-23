# Automatic error taxonomy counts, seed 13 (n=803). Needs human judgement: gold_issue, multi_hop_miss, right_type_wrong_entity, wrong_type

| model (seed13) | correct | false_no_answer | false_answer_on_unanswerable | boundary_too_long | boundary_too_short | partial_overlap | wrong_span_same_sentence | wrong_sentence |
|---|---|---|---|---|---|---|---|---|
| no_answer | 371 | 432 | 0 | 0 | 0 | 0 | 0 | 0 |
| sliding_window | 371 | 432 | 0 | 0 | 0 | 0 | 0 | 0 |
| sliding_window_thr | 370 | 355 | 14 | 1 | 12 | 1 | 34 | 16 |
| classical_pipeline | 412 | 204 | 63 | 6 | 30 | 10 | 35 | 43 |
| bidaf_noattn | 428 | 229 | 39 | 30 | 12 | 16 | 21 | 28 |
| bidaf | 421 | 248 | 22 | 47 | 13 | 11 | 12 | 29 |
| plm_indobert_full | 558 | 83 | 39 | 29 | 23 | 18 | 26 | 27 |
