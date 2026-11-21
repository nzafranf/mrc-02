# 04 · plm · indobert

**Scope** plm · **Name** indobert · **Variants** `plm_indobert_full` (full fine-tuning; `runs/plm_smoke` is a CPU
pipeline check, not a result) · **Status** done (3 seeds, Kaggle T4)

**Differentiating point.** A pretrained Indonesian encoder (`indobenchmark/indobert-base-p1`, 124M parameters) with its
own subword tokenizer, fine-tuned end to end with a start/end span head. Unanswerable questions point to [CLS].

**Verdict.** Best model: 73.5 F1, significantly ahead of BiDAF (+16.9). It halves the false no-answers of the
non-pretrained models, but wrongly abstains on 22% of answerable questions. Its remaining errors are mostly
boundaries and wrong spans.

## 1. Where it starts
Input: `data/processed/{train,dev}.jsonl`. Everything from subword tokenization onward is in `plm_qa.py`: windows,
labels, training, span decoding and the no-answer score.

## 2. Pipeline, end to end
| Step | Command | Output |
|---|---|---|
| Offset check (CPU) | `python experiments/04-plm-indobert/plm_qa.py --check-offsets --model-name indobenchmark/indobert-base-p1` | printed table (section 3) |
| Train + predict dev (GPU) | `python experiments/04-plm-indobert/plm_qa.py --device cuda --fp16 --run-name plm_indobert_full --model-name indobenchmark/indobert-base-p1 --seed 13 --epochs 3 --lr 3e-5 --batch-size 16` (also 42, 87) | `runs/plm_indobert_full/seed*/` |
| Same on Kaggle | [kaggle/README.md](kaggle/README.md) | `runs/` |
| Error analysis | `python experiments/04-plm-indobert/analyze.py` | `analysis/`, section 4 below |
| Predict another split | `... plm_qa.py --predict-only --checkpoint X.pt --split-file Y.jsonl --out preds.jsonl` (needs `--save-checkpoint` at training) | `preds.jsonl` |

More GPU configurations (XLM-R, IndoLEM, frozen encoder, LoRA, train fractions): [gpu_runs.md](gpu_runs.md). Under the
naming convention, another encoder or training regime is a new experiment (e.g. `07-plm-xlmr`) that calls this
`plm_qa.py` with a different `--model-name` and writes its own `runs/`.

## 3. Design
`AutoModelForQuestionAnswering`: encoder + one linear layer (`qa_outputs`, hidden -> 2) giving a start and an end logit per token. Default encoder `indobenchmark/indobert-base-p1` (124M params, uncased); also `FacebookAI/xlm-roberta-base` and `indolem/indobert-base-uncased`. Regimes: full fine-tuning (default), `--freeze-encoder` (only the 1.5k-parameter head trains), `--lora` (peft, r=8, alpha=16, dropout 0.1, targets `query`,`value`, head trained too; 296k trainable params).

### Input representation
`[CLS] question [SEP] context [SEP]`, truncation only on the context, `max_len=384`, `doc_stride=128` (overlapping windows). Dynamic padding per batch. Training label: start/end token of the first gold answer if it lies fully inside the window, otherwise index 0 (CLS). Unanswerable examples always label CLS in every window.

### Prediction and no-answer score
Per window: top 20 start x top 20 end logits, keep spans inside the context part, end >= start, length <= 30 tokens, score = start + end logit. Best span over all windows of an example. The span text is cut from the ORIGINAL context with the tokenizer character offsets (so the lowercasing of the IndoBERT tokenizer does not matter). `null_score` = min over windows of (start[CLS] + end[CLS]). `na_prob = null_score - best_span_score` (raw logit difference, not a probability; higher = more likely unanswerable; evaluator only needs the ordering). `pred` in `preds_dev.jsonl` is always the best span, so `na_threshold` can be applied afterwards; `pred_argmax` is the decision at threshold 0. The reported metric is `at_best_threshold` in `metrics_dev.json` (`mrc.evaluate.best_na_threshold`, tuned on dev). Best epoch chosen by dev F1 at its best threshold.

### Training
AdamW, lr 3e-5 (full), weight decay 0.01, linear warmup 10% then linear decay, batch 16, 3 epochs, grad clip 1.0, fp16 autocast with GradScaler on CUDA only. Probe/LoRA need larger lr (1e-3 / 3e-4, 5 epochs; see colab file). Seeds 13, 42, 87. `--train-frac` samples whole context groups with the run seed.

### Offset check (train, 3608 answerable examples)
Command: `python experiments/04-plm-indobert/plm_qa.py --check-offsets --model-name <name>`. Test: does `context[offsets of the gold token span]` equal the gold text exactly.

| Tokenizer | exact | mismatch | rate | no window contains answer |
|---|---|---|---|---|
| indobert-base-p1 (BertTokenizerFast, lowercasing) | 3577 | 31 | 0.86% | 0 |
| indolem/indobert-base-uncased | 3577 | 31 | 0.86% | 0 |
| xlm-roberta-base | 3386 | 222 | 6.15% | 0 |

Gold text always matches `context[answer_start:...]` (0 inconsistent). Mismatches are annotation boundaries inside a token (gold "Pulau Hawai" in "Pulau Hawaii", gold "lima" inside "kelima", gold ending before a trailing "."), so the token-aligned span necessarily extends the gold. For XLM-R the extra 190 or so cases are mostly a trailing "." attached to the last word piece; 195 of the 222 become equal after `normalize_answer`, so the metric impact is small. For IndoBERT 0 mismatches are normalization-equal, so those 31 cost partial F1 (an upper bound of about 0.9% of answerable examples). Lowercasing is harmless because spans come from original-text offsets.

## 4. Results and error analysis
<!-- AUTO-RESULTS:BEGIN (written by analyze.py, edits here are overwritten) -->

### Results (dev, NA threshold tuned on dev)

| variant | runs | n | EM | F1 | HasAns F1 | NoAns acc | predicted no-answer % | NA threshold |
|---|---|---|---|---|---|---|---|---|
| plm_indobert_full | 3 | 803 | 69.6 ± 0.4 | 73.5 ± 0.4 | 58.8 ± 1.2 | 90.7 ± 0.6 | 53.8 ± 1.0 | -2.15 ± 0.84 |

### Outcome types

Share of each gold group; count (mean over seeds) in brackets. Figure: [analysis/outcomes.png](analysis/outcomes.png).

| gold | outcome | plm_indobert_full |
|---|---|---|
| answerable | correct_ans | 51.5% (222) |
| answerable | false_no_answer | 22.1% (95) |
| answerable | wrong_sentence | 6.5% (28) |
| answerable | wrong_span_same_sentence | 4.2% (18) |
| answerable | boundary_too_long | 6.6% (29) |
| answerable | boundary_too_short | 6.0% (26) |
| answerable | partial_overlap | 3.2% (14) |
| unanswerable | correct_no_ans | 90.7% (336) |
| unanswerable | false_answer_on_unanswerable | 9.3% (35) |

### Seed stability

| variant | seeds | same outcome type (mean over seed pairs) | questions that flip right/wrong across seeds |
|---|---|---|---|
| plm_indobert_full | 3 | 87.1% | 13.9% |

### Weakest slices for `plm_indobert_full`, the best variant (n >= 30). All slices: [analysis/slices.md](analysis/slices.md)

| slice | value | n | F1 (plm_indobert_full) |
|---|---|---|---|
| ans_rel_pos_bucket | Q3 | 48 | 48.7 |
| ans_type | other_cap | 121 | 51.6 |
| ans_len_bucket | >6 | 91 | 52.4 |
| ans_rel_pos_bucket | Q4 | 51 | 54.8 |
| ans_len_bucket | 4-6 | 46 | 56.2 |
| ans_type | lowercase_phrase | 119 | 56.9 |

### Hard questions

159 of 803 questions are wrong (F1 < 0.5) in every run of every variant; 12% of them are unanswerable (dev base rate 46%). List: [analysis/hard_examples.jsonl](analysis/hard_examples.jsonl).

### Calibration

- plm_indobert_full seed13: ECE 0.1305, mean sigmoid(na_prob) 0.520, unanswerable rate 0.462, tuned threshold on the raw logit -1.098
- plm_indobert_full seed42: ECE 0.1318, mean sigmoid(na_prob) 0.483, unanswerable rate 0.462, tuned threshold on the raw logit -2.191
- plm_indobert_full seed87: ECE 0.1313, mean sigmoid(na_prob) 0.485, unanswerable rate 0.462, tuned threshold on the raw logit -3.154

### Analysis files

[calibration.md](analysis/calibration.md), [error_sample_plm_indobert_full.md](analysis/error_sample_plm_indobert_full.md), [hard_examples.jsonl](analysis/hard_examples.jsonl), [outcomes.md](analysis/outcomes.md), [outcomes.png](analysis/outcomes.png), [per_example.jsonl](analysis/per_example.jsonl), [reliability_plm_indobert_full.png](analysis/reliability_plm_indobert_full.png), [results.md](analysis/results.md), [seed_stability.md](analysis/seed_stability.md), [slices.md](analysis/slices.md)

<!-- AUTO-RESULTS:END -->

## 5. Findings
- Pretraining mostly fixes false no-answers: 103 questions that BiDAF wrongly abstains on are answered correctly here
  (the cross-model confusion tables are in `eda/notebooks/model_errors.ipynb`).
- In the other direction, it answers 28 unanswerable questions that BiDAF correctly declines. Minimal-edit
  unanswerable questions (a changed entity or date) fool it more than the non-pretrained models.
- The hard questions skew to long, lowercase, definition-style answers ("Apa yang dimaksud dengan ..."), and to
  gold-alias problems (Osing/Using) that are gold issues rather than model errors.
- The raw na_prob logit is far from a probability (see Calibration); only its ordering is used.
- Smoke test (300 train examples, 1 epoch, 200 dev, CPU, seed 13; `experiments/04-plm-indobert/runs/plm_smoke/seed13`): dev F1 42.8, EM 42.5, HasAns_f1 3.0, NoAns_acc 98.8. This only proves the pipeline works (20 optimizer steps, the model predicts almost everything as unanswerable). Not a real result.

CPU speed: the machine was fully loaded by other jobs (CPU load 100%), giving about 35 s/step at batch 16, length up to 384 (a lone run would be perhaps 3x faster, still around 10 s/step). Full train is about 6.5k windows = about 410 steps/epoch, which projects to about 4 h/epoch under load and well over the 50 minute budget even when idle. The real CPU run was therefore skipped; real numbers must come from the T4 runs.

## 6. Open questions
- XLM-R and IndoLEM with the same harness (new experiments), frozen encoder vs LoRA vs full fine-tuning, and a
  train-fraction learning curve.
- Does the question-only artifact (AUC 0.90 from the EDA) explain part of the NoAns accuracy?

## Notes
- transformers 5.x: pass `num_labels=2` explicitly; the IndoBERT p1 config carries a 5-label classification head config and the QA head would otherwise emit 5 logits (`logits.split(1, -1)` fails).
- Importing peft failed because the installed torchvision did not match torch (`operator torchvision::nms does not exist`); uninstalling torchvision fixed it (unused here).
- Checkpoints store only trainable parameters (about 500 MB for full fine-tuning); saving is off unless `--save-checkpoint`.
