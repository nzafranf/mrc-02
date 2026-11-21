# Outcome type by variant

Share of the gold group (answerable or unanswerable), mean over seeds; count in brackets. Each column sums to 100% within a group.

| gold | outcome | qwen0.5b_en1_0shot_n100 | qwen0.5b_id1_0shot_n100 | qwen0.5b_id1_0shot_n60_first_token | qwen0.5b_id1_0shot_n60_seqlogprob | qwen0.5b_id1_3shot_fixed_n100 |
|---|---|---|---|---|---|---|
| answerable | correct_ans | 6.9% (4) | 6.9% (4) | 8.3% (3) | 8.3% (3) | 6.9% (4) |
| answerable | false_no_answer | 63.8% (37) | 63.8% (37) | 61.1% (22) | 88.9% (32) | 79.3% (46) |
| answerable | wrong_sentence | 6.9% (4) | 6.9% (4) | 5.6% (2) | 0.0% (0) | 1.7% (1) |
| answerable | wrong_span_same_sentence | 1.7% (1) | 1.7% (1) | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| answerable | boundary_too_long | 15.5% (9) | 15.5% (9) | 19.4% (7) | 2.8% (1) | 5.2% (3) |
| answerable | boundary_too_short | 1.7% (1) | 1.7% (1) | 2.8% (1) | 0.0% (0) | 0.0% (0) |
| answerable | partial_overlap | 3.4% (2) | 3.4% (2) | 2.8% (1) | 0.0% (0) | 6.9% (4) |
| unanswerable | correct_no_ans | 92.9% (39) | 100.0% (42) | 100.0% (24) | 100.0% (24) | 97.6% (41) |
| unanswerable | false_answer_on_unanswerable | 7.1% (3) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 2.4% (1) |
