# 08 · plm · encoder · screen

**Scope** plm · **Name** encoder screen · **Variants** one run folder per encoder in `runs/<run>/seed13/` ·
**Status** screen done (seed 13, Kaggle T4); seeds 42 and 87 deferred

**Differentiating point.** The 04 harness (`experiments/04-plm-indobert/plm_qa.py`, span head, CLS no-answer score)
unchanged, with other pretrained encoders swapped in: multilingual (m-BERT, mDeBERTa-v3, XLM-R-large), Indonesian
(IndoBERT, NusaBERT-large, the ALBERT IndoBERT-lite-large), one fixed configuration per size class. Unlike 04, every
run trains on the full raw train file (8042 questions) and picks its epoch and NA threshold on `official_val`.
Hypotheses were fixed before any run in [PREREG.md](PREREG.md).

**Verdict.** XLM-R-large is the best model: 82.4 F1 on official_val, significantly ahead of mDeBERTa-v3-base (+2.1)
and IndoBERT-base (+10.7). Multilingual encoders beat the Indonesian ones of the same size. Most of the gain is on
answerable questions. The ALBERT checkpoint did not learn the task under the fixed configuration.

## 1. Where it starts
- Train: `data/raw/train.jsonl` (8042, the whole course train file; the internal dev and itest of 01 to 07 are in it).
- Selection and scoring: `data/processed/official_val.jsonl` (2003; 1282 answerable, 721 unanswerable).
- Code: `plm_qa.py` from 04 (this experiment added `--predict-file` and the float32 load, see section 3) and the shared
  `experiments/global/mrc`. No IDK-MRC or TyDiQA-ID data and no checkpoint fine-tuned on them (the course data is a
  re-split of IDK-MRC; see `miscellaneous/docs/decisions.md`).

## 2. Pipeline, end to end
| Step | Command | Output |
|---|---|---|
| Stage the Kaggle dataset | `python experiments/04-plm-indobert/kaggle/make_dataset.py`, then upload as `pbm-mrc-proj` | code + train + official_val, never test |
| Build and push the kernels | `python experiments/08-plm-encoder-screen/kaggle/push.py [--only <kernel> ...]` | one private T4 kernel per entry of `KERNELS` |
| Pull a finished kernel | `python -m kaggle kernels output <owner>/pbm-mrc-08-encoders-<kernel> -p experiments/08-plm-encoder-screen/kaggle/out/<kernel>` | copy `runs/08-plm-encoder-screen/<run>/seed13/` to `runs/` |
| Results and significance | `python experiments/08-plm-encoder-screen/compare.py` | [analysis/results.md](analysis/results.md) |
| Leaderboard file (local only) | `python experiments/global/run.py mrc.make_submission --name 08_<run>_seed13 --preds experiments/08-plm-encoder-screen/runs/<run>/seed13/preds_official_val.jsonl --tune-on-split` | `experiments/global/submissions/official_val/<name>/` |

Predictions (`preds_official_val.jsonl`) and submissions are leaderboard material and are gitignored; configs, logs
and metrics are committed. `colab/` is an outdated Colab path from the first design (6k train, dev selection).

## 3. Design
Fixed before any run (PREREG.md): 10 epochs, effective batch 16, warmup 10% then linear decay, weight decay 0.01,
max_len 384, stride 128, fp16, seed 13. Learning rate by size class: base 3e-5 (as 04), large 1.5e-5 (batch 8 x 2
accumulation). The best epoch and the NA threshold are both chosen on official_val, so every absolute score here is
optimistic, by the same mechanism for each model; differences between models are what this experiment measures.

| run | checkpoint | params |
|---|---|---|
| `indobert_base` | `indobenchmark/indobert-base-p1` | 124M |
| `mbert_base` | `google-bert/bert-base-multilingual-cased` | 177M |
| `mdeberta_v3_base` | `microsoft/mdeberta-v3-base` | 278M |
| `nusabert_large` | `LazarusNLP/NusaBERT-large` | 336M |
| `xlmr_large` | `FacebookAI/xlm-roberta-large` | 559M |
| `albert_indolite_large` | `indobenchmark/indobert-lite-large-p2` | 17M (shared layers) |

Changes to `plm_qa.py` made for this experiment:
- `--predict-file`: after training, predict extra splits with the best-epoch weights.
- Weights load in float32 (`dtype=torch.float32`). transformers 5 keeps a checkpoint's stored dtype, and
  mDeBERTa-v3-base is stored in fp16, which made GradScaler fail at step 1 ("Attempting to unscale FP16 gradients").

Dropped before results: `ChristopherA08/IndoELECTRA`, the only Indonesian ELECTRA found. Its tokenizer emits ids up
to 35000 but the model has 30522 embeddings, so training crashes (CUDA device-side assert). Recorded in `push.py`.

## 4. Results
Full table and all pairs: [analysis/results.md](analysis/results.md). official_val, seed 13, threshold tuned on
official_val.

| run | EM | F1 | HasAns F1 | NoAns acc | best epoch | T4 minutes |
|---|---|---|---|---|---|---|
| xlmr_large | 74.4 | 82.4 | 76.5 | 92.9 | 6 | 82 |
| mdeberta_v3_base | 72.6 | 80.3 | 73.0 | 93.3 | 6 | 57 |
| mbert_base | 69.9 | 77.3 | 71.4 | 87.8 | 8 | 31 |
| nusabert_large | 68.4 | 76.4 | 70.0 | 87.8 | 10 | 71 |
| indobert_base | 64.3 | 71.7 | 62.5 | 88.1 | 9 | 26 |
| albert_indolite_large | 36.0 | 36.1 | 0.2 | 100.0 | 1 | 37 |

Paired bootstrap (10000 resamples over questions):

| difference | dF1 | 95% CI | answerable | unanswerable |
|---|---|---|---|---|
| xlmr_large - mdeberta_v3_base | +2.07 | [0.77, 3.37] | +3.48 | -0.42 |
| xlmr_large - indobert_base | +10.68 | [8.98, 12.40] | +13.95 | +4.85 |
| mdeberta_v3_base - mbert_base | +3.03 | [1.50, 4.59] | +1.62 | +5.55 |
| mbert_base - indobert_base | +5.57 | [3.81, 7.30] | +8.86 | -0.28 |
| nusabert_large - mbert_base | -0.88 | [-2.53, 0.79] | -1.38 | +0.00 |

Validation F1 per epoch (threshold tuned per epoch):

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| xlmr_large | 72.8 | 79.1 | 80.8 | 81.9 | 81.9 | **82.4** | 81.8 | 82.4 | 82.1 | 81.7 |
| mdeberta_v3_base | 70.1 | 77.7 | 79.4 | 79.9 | 79.5 | **80.3** | 79.3 | 79.4 | 79.4 | 79.4 |
| mbert_base | 69.2 | 73.5 | 76.1 | 76.4 | 76.0 | 75.8 | 76.5 | **77.3** | 77.0 | 77.0 |
| nusabert_large | 68.0 | 74.3 | 75.4 | 75.8 | 75.9 | 76.3 | 76.4 | 76.2 | 76.2 | **76.4** |
| indobert_base | 60.2 | 69.2 | 70.1 | 70.1 | 69.0 | 70.7 | 71.4 | 70.9 | **71.7** | 71.5 |
| albert_indolite_large | **36.1** | 36.1 | 36.0 | 36.1 | 36.0 | 36.0 | 36.1 | 36.1 | 36.0 | 36.1 |

### Reading
- Multilingual pretraining wins at equal size: m-BERT beats IndoBERT by 5.6 F1, as in the IDK-MRC paper (82.2 vs
  77.5 on its own test). NusaBERT-large, IndoBERT continued with a larger vocabulary, is no better than m-BERT-base.
- Most gains come from answerable questions (fewer false no-answers and better spans). mDeBERTa is the exception:
  over m-BERT its gain is mostly correct abstention (+5.6 on unanswerable).
- Epochs: every model is within about 1 F1 of its best by epoch 4. Past epoch 6 the curves are flat or slightly
  down for the two best models, so 10 epochs is more than needed.
- ALBERT: training loss stayed at about 2.9 (others fall below 0.1) and the model predicts no-answer for 1998 of
  2003 questions. Under the pre-registered rule it is reported as failed, not retuned. H3 holds only in this trivial
  sense.
- Run-to-run noise: NusaBERT-large was run twice with the same seed and code by mistake (a stray resubmission). The
  two runs scored 76.97 (best epoch 7) and 76.42 (best epoch 10), so GPU nondeterminism alone moves F1 by about
  0.5; only the second run's files exist. XLM-R-large's first run was cut off by the same resubmission; the result
  above is its complete second run.

### Hypotheses (PREREG.md)
| | prediction | outcome |
|---|---|---|
| H1 | mDeBERTa-v3-base beats IndoBERT-base | holds: +8.6 [6.9, 10.3] |
| H2 | a large model beats IndoBERT-base by at least 2 F1 | holds: XLM-R-large +10.7, NusaBERT-large +4.7 |
| H3 | the ALBERT checkpoint does not beat IndoBERT-base | holds, but only because the run failed to learn |
| H4 | gains come mainly from answerable questions | holds against IndoBERT for every model; mDeBERTa over m-BERT is the exception |
| H5 | m-BERT beats IndoBERT-base | holds: +5.6 [3.8, 7.3] |

## 5. Limits and next steps
- One seed per model. Differences under about 1 F1 are not interpretable (see the duplicate NusaBERT run).
- Selection on official_val inflates every score. The leaderboard test (gold after 23 Dec) is the unbiased check.
- Next: seeds 42 and 87 for XLM-R-large, a seed and model ensemble (needs n-best span scores saved by `plm_qa.py`),
  and two further XLM-R-large derivatives (mPMR-large, InfoXLM-large), to be pre-registered before they run.
