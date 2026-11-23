# Status (2026-11-23)

## Done
- M0: requirements checklist, contract, decisions log, grouped split with hashes, shared evaluator.
- M1: data card, slices, leakage audit (2 itest near-duplicates excluded), artifact probe, tokenizer stats, error
  taxonomy v1. Summary: `eda/README.md`.
- Experiments, each with code, runs, `analyze.py` and `summary.md`:
  - `01-baseline-trivial`
  - `02-classical-pipeline-lgbm` (3 seeds)
  - `03-nn-bidaf` with the no-attention ablation (3 seeds each, CPU)
  - `04-plm-indobert` full fine-tuning (3 seeds, Kaggle T4)
  - `05-llm-local-qwen`: 0.5B smoke runs only
  - `06-llm-api-offtheshelf`: partial, 22 to 25 dev questions, stopped by free-tier caps
- Cross-model comparison: `python experiments/global/run.py mrc.analysis.compare` -> `report/assets/`. Per-question
  browser: `eda/notebooks/model_errors.ipynb`.
- 2026-11-21: repository reorganized into one folder per experiment (see `decisions.md`). LaTeX report skeleton in
  `report/latex`, rendered with `miscellaneous/scripts/render_latex.py`.
- 2026-11-23: 01-03 predicted on the official validation set (`preds_official_val.jsonl` in every run dir).
  `mrc.analysis.official_val` compares thresholds; `mrc.make_submission` writes the `id,answer` CSV (checked against the
  sample submission: same ids, same order). 04 was rerun on Kaggle (same config; dev F1 within 0.2 of the original
  runs, which it replaces) to predict official_val and save `checkpoint.pt` for the test set. The checkpoints are
  gitignored (about 470 MB each): they live in the Kaggle output of `aldofhrzy/pbm-mrc-indobert-full` and locally
  with whoever pulled it.

## Official validation results (2003 questions, 36% unanswerable)
| model | F1, dev threshold | F1, val threshold | dev F1 (for reference) |
|---|---|---|---|
| no_answer | 36.0 | 36.0 | 46.2 |
| classical_pipeline | 47.5 ± 0.3 | 47.9 ± 0.2 | 53.4 |
| bidaf_noattn | 46.6 ± 0.9 | 47.1 ± 0.7 | 56.5 |
| bidaf | 46.5 ± 1.1 | 47.2 ± 0.3 | 56.6 |
| plm_indobert_full | 68.7 ± 0.2 | 69.9 ± 0.4 | 73.6 |

Every model drops 6-10 F1 from dev, partly because the no-answer floor drops 10 points (36% vs 46% unanswerable).
Measured above that floor (dev thresholds), classical improves (+11.5 vs +7.2 on dev) while BiDAF holds (+10.5 vs
+10.4), so the BiDAF > classical gap from dev is gone (47.2 vs 47.9 with val thresholds). Dev-tuned thresholds cost under 1 F1: BiDAF abstains on 64% of questions with them vs 58% with
val-tuned ones. IndoBERT drops least (73.6 -> 68.7) and is the only model where the threshold matters: dev thresholds
abstain on 46% of questions, val thresholds on 39%, worth 1.1 F1. Tuning the threshold for EM instead of F1 (the
leaderboard ranks by EM first) gives +0 to +0.45 EM for -0.3 F1; `mrc.make_submission` tunes for EM by default.
Full table: `report/assets/tables/official_val_table.md`.

## Dev results (threshold tuned on dev, optimistic)
| model | EM | F1 | HasAns F1 | NoAns acc |
|---|---|---|---|---|
| no_answer | 46.2 | 46.2 | 0 | 100 |
| classical_pipeline | 51.0 ± 0.2 | 53.4 ± 0.2 | 29.1 | 81.6 |
| bidaf_noattn | 52.5 ± 0.7 | 56.5 ± 0.6 | 30.7 | 86.5 |
| bidaf | 52.7 ± 0.7 | 56.6 ± 0.7 | 28.2 | 89.8 |
| plm_indobert_full | 69.5 ± 0.5 | 73.6 ± 0.4 | 59.2 | 90.3 |

Significant: classical > no_answer (+7.2), IndoBERT > BiDAF (+16.9). Not significant: BiDAF vs classical (+3.1,
p = 0.07), attention vs no attention (+0.1, p = 0.88).

## Open issues
- LLM `na_prob`: `seqlogprob` vs `first_token` undecided (60 questions, 0.5B model). Compare on full dev with 7B.
- Every LLM so far over-abstains under prompt `id1`. Test the prompt before comparing models.
- BiDAF and the classical pipeline barely beat the no-answer floor on HasAns: both over-abstain.
- `torchvision` was uninstalled from the global Python 3.12 (peft import conflict). Not needed here; other global
  packages (ultralytics, easyocr) need it. Pending user decision.
- Leaderboard metric and submission format not yet confirmed (SCeLE).
- Official validation set (2003 questions, 36% unanswerable, 968 passages) is in `data/raw/validation.jsonl`, copied to
  `data/processed/official_val.jsonl` with slice tags. Leakage audit: no passage overlap with train/dev/itest (max
  char 5-gram similarity 0.59); 10 generic questions repeat train wording with different passages.
- Submission format taken from `contoh_submission_validation.csv` (course sample): `id,answer`, empty answer = no answer.

## Next (needs approval for GPU or API spend)
1. LLM on full dev: Qwen2.5-7B-Instruct 4-bit and Sailor2-8B-Chat, with 0-shot, fixed 3-shot and retrieved 3-shot
   prompting. Then 5 prompts x 3 demo orders on the better model.
2. More PLM experiments: XLM-R (3 seeds), IndoBERT frozen vs LoRA, train fractions 0.1, 0.25 and 0.5.
3. Paid `google/gemma-4-31b-it` via OpenRouter (about USD 0.10, returns logprobs) for `06`.
4. Official validation: done for 01-04. On 3 Dec: predict official_test with each model (`--predict-only`; 04 from
   its checkpoints), then `mrc.make_submission --threshold-from official_val` for each.
6. `07-llm-offtheshelf-fewshot`: review `prompt_review.md` (instruction + 7 frozen demos), then run on Kaggle/Colab with
   `gpu/fewshot_gpu.ipynb` (fixed question order, resumable).
5. Submission antara (25 Nov): short report built from `report/latex` and `report/assets/tables/main_table.md`.
