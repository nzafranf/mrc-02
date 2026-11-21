# 03 · nn · bidaf

**Scope** nn · **Name** bidaf · **Variants** `bidaf` (attention flow), `bidaf_noattn` (ablation) · **Status** done
(3 seeds each, CPU, 25 to 60 min per run)

**Differentiating point.** An end-to-end neural reader trained from scratch with no pretraining: its own tokenizer,
vocabulary, random-init embeddings, BiLSTM encoder, attention flow and pointer heads, plus a NULL position for
unanswerable questions. The ablation removes attention flow and changes nothing else.

**Verdict.** 56.6 F1, not significantly better than the classical pipeline (+3.1, p = 0.07). Attention flow has no
measurable effect (+0.1 F1, p = 0.88). The model mostly abstains: it predicts "no answer" for about 65% of questions
while the true rate is 46%.

## 1. Where it starts
Input: `data/processed/{train,dev}.jsonl`. Everything from tokenization onward is in this folder:
`nn_common.py` (regex tokenizer with character offsets, vocabularies, features, batching) and `bidaf.py` (model,
training, decoding).

## 2. Pipeline, end to end
| Step | Command | Output |
|---|---|---|
| Train + predict dev, attention | `python experiments/03-nn-bidaf/bidaf.py --out-dir experiments/03-nn-bidaf/runs/bidaf/seed13 --seed 13 --threads 2 --epochs 12 --patience 3 --save-attn experiments/03-nn-bidaf/analysis/attn_seed13.npz` (also seeds 42, 87) | `runs/bidaf/seed*/` (model.pt, preds, history, log) |
| Train, ablation | same with `--no-attention --out-dir experiments/03-nn-bidaf/runs/bidaf_noattn/seed13` | `runs/bidaf_noattn/seed*/` |
| Attention heatmaps | `python experiments/03-nn-bidaf/plot_attention.py` | `analysis/attn_plots/` |
| Context-length and seed table | `python experiments/03-nn-bidaf/bidaf_report.py` (prints) | section 4 below |
| Error analysis | `python experiments/03-nn-bidaf/analyze.py` | `analysis/`, section 4 below |
| Predict another split | `python experiments/03-nn-bidaf/bidaf.py --predict-only --checkpoint runs/bidaf/seed13/model.pt --split-file X.jsonl --out Y.jsonl` | `Y.jsonl` |

CPU is enough at this size.

## 3. Design
Tokenizer: regex `\w+|[^\w\s]` with char offsets, so a predicted token span maps to an exact context substring.
Vocab: lowercased train words with min_freq 2 (25,047 incl. specials), char vocab min_freq 5. Contexts are truncated to 400 tokens.
A learnable `<NULL>` token is prepended to every context (index 0). Gold start=end=0 for unanswerable questions.

1. Embedding: `x_t = Highway([E_w(w_t); maxpool_k ReLU(Conv1d_5(E_c(chars_t)))])`, emb 100 + 50 char filters (char emb 16), random init.
2. Contextual encoder: one BiLSTM (hidden h=64, output 2h=128) shared by context and question: `C in R^{(T+1) x 2h}`, `Q in R^{J x 2h}`.
3. Attention flow: `S_tj = w^T [c_t; q_j; c_t * q_j]`.
   Context-to-query: `a_t = softmax_j(S_t:)`, `U_t = sum_j a_tj q_j`.
   Query-to-context: `b = softmax_t(max_j S_tj)`, `h = sum_t b_t c_t` (tiled over t).
   `G_t = [c_t; U_t; c_t * U_t; c_t * h]` (dim 8h).
   `--no-attention`: `G_t = [c_t; qbar; c_t * qbar]`, `qbar` = masked mean of Q (dim 6h). Everything else identical.
4. Modeling: `P = ReLU(W G)` (bottleneck to 2h, for CPU budget), `M = BiLSTM(P)`, `M2 = BiLSTM(M)`.
5. Pointers: `l_start = w_s^T [G; M]`, `l_end = w_e^T [G; M2]`, softmax over positions 0..T.
   Loss = CE(start) + CE(end).
6. Decoding: `p0 = ps[0] * pe[0]`; best span `max_{1<=i<=j, j-i<30} ps[i] pe[j]`.
   `na_prob = p0 / (p0 + best_span)`, `score = best_span`. Prediction is "" if `na_prob > 0.5`. The evaluator also reports the dev-tuned threshold (`at_best_threshold` in `metrics_dev.json`).

Recurrent layers are two unidirectional nets (backward one on length-reversed sequences) rather than packed sequences, because the PyTorch CPU packed-RNN backward was about 8x slower.
`--rnn {lstm,gru,rnn}` swaps the cell; `--no-char` removes the char CNN.

### Training
Adam lr 2e-3, batch 32 (length-bucketed), dropout 0.2, grad clip 5, max 12 epochs, early stop patience 3 on dev F1 (default 0.5 NA decision), best-epoch checkpoint restored.
Params: 2,985,465 (attention), 2,968,440 for the no-attention ablation. Epoch time 125 to 300 s on a shared 8-core CPU (1 run = 2 threads), so a run costs 25 to 60 min, longer than the 15 min target because the machine was shared with other jobs and because 3M of the params are word embeddings.
Truncation at 400 tokens loses 4 of 3608 gold train answers (0.11%, those train examples are dropped) and 0 of 432 dev answers.

Sanity check: overfitting one batch of 32 reaches loss 0.0165 in 51 steps (dropout off), logged at the top of every `log.txt`.

## 4. Results and error analysis
<!-- AUTO-RESULTS:BEGIN (written by analyze.py, edits here are overwritten) -->

### Results (dev, NA threshold tuned on dev)

| variant | runs | n | EM | F1 | HasAns F1 | NoAns acc | predicted no-answer % | NA threshold |
|---|---|---|---|---|---|---|---|---|
| bidaf_noattn | 3 | 803 | 52.5 ± 0.7 | 56.5 ± 0.6 | 30.7 ± 1.3 | 86.5 ± 2.2 | 65.0 ± 3.5 | 0.33 ± 0.139 |
| bidaf | 3 | 803 | 52.7 ± 0.7 | 56.6 ± 0.7 | 28.1 ± 3.2 | 89.8 ± 3.4 | 68.0 ± 4.7 | 0.314 ± 0.183 |

### Outcome types

Share of each gold group; count (mean over seeds) in brackets. Figure: [analysis/outcomes.png](analysis/outcomes.png).

| gold | outcome | bidaf_noattn | bidaf |
|---|---|---|---|
| answerable | correct_ans | 23.2% (100) | 20.8% (90) |
| answerable | false_no_answer | 46.5% (201) | 49.3% (213) |
| answerable | wrong_sentence | 9.4% (41) | 9.7% (42) |
| answerable | wrong_span_same_sentence | 5.8% (25) | 4.2% (18) |
| answerable | boundary_too_long | 7.9% (34) | 8.9% (38) |
| answerable | boundary_too_short | 3.6% (16) | 3.5% (15) |
| answerable | partial_overlap | 3.6% (16) | 3.5% (15) |
| unanswerable | correct_no_ans | 86.5% (321) | 89.8% (333) |
| unanswerable | false_answer_on_unanswerable | 13.5% (50) | 10.2% (38) |

### Variants against `bidaf_noattn` (paired bootstrap)

| comparison | n | delta F1 | 95% CI | p (paired bootstrap) |
|---|---|---|---|---|
| bidaf - bidaf_noattn | 803 | +0.14 | [-1.35, 1.57] | 0.9110 |

### Seed stability

| variant | seeds | same outcome type (mean over seed pairs) | questions that flip right/wrong across seeds |
|---|---|---|---|
| bidaf_noattn | 3 | 78.7% | 21.0% |
| bidaf | 3 | 76.9% | 22.3% |

### Weakest slices for `bidaf`, the best variant (n >= 30). All slices: [analysis/slices.md](analysis/slices.md)

| slice | value | n | F1 (bidaf) |
|---|---|---|---|
| ans_type | other_cap | 121 | 15.6 |
| ans_rel_pos_bucket | Q4 | 51 | 20.5 |
| ans_len_bucket | 4-6 | 46 | 21.0 |
| ans_rel_pos_bucket | Q3 | 48 | 21.4 |
| ans_len_bucket | 1 | 133 | 24.5 |
| ans_type | lowercase_phrase | 119 | 27.2 |

### Hard questions

216 of 803 questions are wrong (F1 < 0.5) in every run of every variant; 4% of them are unanswerable (dev base rate 46%). List: [analysis/hard_examples.jsonl](analysis/hard_examples.jsonl).

### Analysis files

[attn_seed13.npz](analysis/attn_seed13.npz), [error_sample_bidaf.md](analysis/error_sample_bidaf.md), [error_sample_bidaf_noattn.md](analysis/error_sample_bidaf_noattn.md), [hard_examples.jsonl](analysis/hard_examples.jsonl), [outcomes.md](analysis/outcomes.md), [outcomes.png](analysis/outcomes.png), [per_example.jsonl](analysis/per_example.jsonl), [results.md](analysis/results.md), [seed_stability.md](analysis/seed_stability.md), [significance.md](analysis/significance.md), [slices.md](analysis/slices.md)

<!-- AUTO-RESULTS:END -->

### F1 by context length (words; default 0.5 no-answer decision, not the tuned threshold)
| bucket | n | attention | no attention |
|---|---|---|---|
| <60 | 350 | 60.99 +- 1.10 | 60.29 +- 0.51 |
| 60-120 | 358 | 52.40 +- 2.20 | 52.77 +- 0.73 |
| 120-250 | 88 | 52.21 +- 0.78 | 53.41 +- 0.30 |
| >250 | 7 | 42.86 +- 0.00 | 36.51 +- 11.22 |

## 5. Findings
- Attention flow did not help here. The difference (+0.07 F1) is far inside the seed std (1.15). The >250 bucket has 7 questions, ignore it. Both models fall with context length, as expected, with no visible advantage for attention on long contexts.
  Likely reason: with 3.6k answerable and ~6.4k total training questions and a 25k-word random-init embedding, neither model learns reliable question-context alignment. The no-attention model still sees the question through `qbar` and gets the same F1, which says that the model mostly exploits question-type and lexical-overlap cues, not fine alignment. This is a data-scale limit, not proof that attention is useless.
- Learning curves (`history.json`, `log.txt`): train loss falls monotonically 5.5 -> ~1.1, dev F1 climbs until epoch 5 to 9 and then oscillates between 52 and 57 while the loss keeps dropping. That is overfitting (train/dev gap grows, dev noisy at 803 examples), with an under-trained start (epochs 1 to 3 are near the all-NA baseline). Early stopping picked epochs 8, 9, 11 (attention) and 12, 9, 5 (no attention); the cap of 12 epochs truncated no-attention seed 13 while it was still creeping up, so differences of a point are noise.
- No-answer: it did collapse at first. After epoch 1 every run predicts NA for 99 to 100% of questions (F1 = 46.2 baseline), and it releases slowly (epoch 3: 70 to 82%). At the best epoch it predicts NA for 57 to 66% of questions versus a true rate of 46.2%, so it is biased toward abstaining and HasAns_f1 is low (about 31). NoAns_acc of 84 is largely paid for with missed answerable questions. A dev-tuned threshold gains only about 0.5 F1.
- Gradient norms (per layer group, pre-clip, `log.txt`): output layer dominates early (2.4 at epoch 1) and shrinks to ~0.9; embedding, encoder and attention rise from 0.2 to 0.7 to ~1.3 to 0.7 as the pointer head learns. No vanishing: all groups stay at 0.2 to 2.4, and clipping at 5 never binds.
- Attention maps (`attn_seed13.npz`, `attn_plots/`): rows are context tokens, each row is a softmax over question tokens. Some lexical alignments appear (question "kota" aligned with context "kota", "Iran" with "Iran"), but many rows are diffuse and the NULL row mass is about 0.1 to 0.2 per question token.
- Errors depend on the seed more than for the other models: only 77 to 79% of questions get the same outcome type
  across two seeds, and about 22% flip between right and wrong. Read single-seed error samples with care.
- Lists of numbers break the model: it often returns a single number from a list where the gold answer is the whole list.

## 6. Open questions
- Pretrained Indonesian word vectors (fastText) instead of random init: does attention start to matter once
  alignment is learnable? This would be a new experiment, e.g. `07-nn-bidaf-fasttext`.
- A train-size learning curve, to separate the data-scale limit from the architecture.
