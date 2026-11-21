# 01 · baseline · trivial

**Scope** baseline · **Name** trivial · **Status** done (deterministic, CPU, seconds)

**Differentiating point.** No learning at all. These rules set the floor every other experiment must beat, and show
how much of the score comes only from the 46% unanswerable share of dev.

**Verdict.** Always answering "no answer" gives 46.2 F1. The sliding-window baseline cannot separate answerable from
unanswerable questions: with a dev-tuned threshold it collapses back to always abstaining.

## 1. Where it starts
Input: `data/processed/{train,dev}.jsonl` (the grouped split in `experiments/global`). Text utilities come from
`experiments/global/mrc/qa_text.py`. Everything after that is in this folder.

## 2. Pipeline, end to end
| Step | Command | Output |
|---|---|---|
| Predict (all 3 baselines) | `python experiments/01-baseline-trivial/baselines.py` | `runs/<baseline>/seed13/` |
| Predict another split | `python experiments/01-baseline-trivial/baselines.py --predict-only --split-file X.jsonl --out DIR` | `DIR/<baseline>/` |
| Error analysis | `python experiments/01-baseline-trivial/analyze.py` | `analysis/`, section 4 below |

## 3. Design
- `no_answer`: predicts "" for every question.
- `sliding_window`: the sliding-window scorer of Richardson et al. (2013). It slides a window over the passage, scores
  word overlap with the question (IDF-weighted), and answers with the candidate chunk (a capitalized run, number or
  date) inside the best window. It always answers.
- `sliding_window_thr`: the same, with `na_prob = 1 - overlap`, and the no-answer threshold tuned on dev.

## 4. Results and error analysis
<!-- AUTO-RESULTS:BEGIN (written by analyze.py, edits here are overwritten) -->

### Results (dev, NA threshold tuned on dev)

| variant | runs | n | EM | F1 | HasAns F1 | NoAns acc | predicted no-answer % | NA threshold |
|---|---|---|---|---|---|---|---|---|
| no_answer | 1 | 803 | 46.2 | 46.2 | 0.0 | 100.0 | 100.0 | -1e+09 |
| sliding_window | 1 | 803 | 9.2 | 12.5 | 22.9 | 0.3 | 0.7 | untuned |
| sliding_window_thr | 1 | 803 | 46.1 | 46.8 | 4.4 | 96.2 | 88.7 | 0.0939 |

### Outcome types

Share of each gold group; count (mean over seeds) in brackets. Figure: [analysis/outcomes.png](analysis/outcomes.png).

| gold | outcome | no_answer | sliding_window | sliding_window_thr |
|---|---|---|---|---|
| answerable | correct_ans | 0.0% (0) | 16.9% (73) | 3.0% (13) |
| answerable | false_no_answer | 100.0% (432) | 1.2% (5) | 82.2% (355) |
| answerable | wrong_sentence | 0.0% (0) | 35.0% (151) | 3.7% (16) |
| answerable | wrong_span_same_sentence | 0.0% (0) | 32.6% (141) | 7.9% (34) |
| answerable | boundary_too_long | 0.0% (0) | 1.2% (5) | 0.2% (1) |
| answerable | boundary_too_short | 0.0% (0) | 11.6% (50) | 2.8% (12) |
| answerable | partial_overlap | 0.0% (0) | 1.6% (7) | 0.2% (1) |
| unanswerable | correct_no_ans | 100.0% (371) | 0.3% (1) | 96.2% (357) |
| unanswerable | false_answer_on_unanswerable | 0.0% (0) | 99.7% (370) | 3.8% (14) |

### Variants against `no_answer` (paired bootstrap)

| comparison | n | delta F1 | 95% CI | p (paired bootstrap) |
|---|---|---|---|---|
| sliding_window - no_answer | 803 | -33.75 | [-38.22, -28.88] | 0.0010 |
| sliding_window_thr - no_answer | 803 | +0.61 | [-0.84, 1.96] | 0.3570 |

### Seed stability

single run per variant

### Weakest slices for `sliding_window_thr`, the best variant (n >= 30). All slices: [analysis/slices.md](analysis/slices.md)

| slice | value | n | F1 (sliding_window_thr) |
|---|---|---|---|
| ans_type | lowercase_phrase | 119 | 1.2 |
| ans_len_bucket | >6 | 91 | 2.1 |
| ans_type | person_like_cap | 70 | 2.1 |
| ans_len_bucket | 4-6 | 46 | 3.3 |
| ans_rel_pos_bucket | Q1 | 221 | 3.4 |
| ans_len_bucket | 2-3 | 162 | 3.6 |

### Hard questions

327 of 803 questions are wrong (F1 < 0.5) in every run of every variant; 0% of them are unanswerable (dev base rate 46%). List: [analysis/hard_examples.jsonl](analysis/hard_examples.jsonl).

### Analysis files

[error_sample_no_answer.md](analysis/error_sample_no_answer.md), [error_sample_sliding_window.md](analysis/error_sample_sliding_window.md), [error_sample_sliding_window_thr.md](analysis/error_sample_sliding_window_thr.md), [hard_examples.jsonl](analysis/hard_examples.jsonl), [outcomes.md](analysis/outcomes.md), [outcomes.png](analysis/outcomes.png), [per_example.jsonl](analysis/per_example.jsonl), [results.md](analysis/results.md), [seed_stability.md](analysis/seed_stability.md), [significance.md](analysis/significance.md), [slices.md](analysis/slices.md)

<!-- AUTO-RESULTS:END -->

## 5. Findings
- `sliding_window` alone is 12.5 F1 (it always answers, 23 HasAns F1). With a tuned threshold it only beats always
  abstaining by 0.6 F1, which means question-passage word overlap carries almost no answerability signal.
- The "hard questions" count here is meaningless: `no_answer` gets every answerable question wrong by construction.

## 6. Open questions
- None. Keep this folder as the reference row of every table.
