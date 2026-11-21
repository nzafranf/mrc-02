# Status (2026-11-21)

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

## Dev results (threshold tuned on dev, optimistic)
| model | EM | F1 | HasAns F1 | NoAns acc |
|---|---|---|---|---|
| no_answer | 46.2 | 46.2 | 0 | 100 |
| classical_pipeline | 51.0 ± 0.2 | 53.4 ± 0.2 | 29.1 | 81.6 |
| bidaf_noattn | 52.5 ± 0.7 | 56.5 ± 0.6 | 30.7 | 86.5 |
| bidaf | 52.7 ± 0.7 | 56.6 ± 0.7 | 28.2 | 89.8 |
| plm_indobert_full | 69.6 ± 0.4 | 73.5 ± 0.4 | 58.8 | 90.7 |

Significant: classical > no_answer (+7.2), IndoBERT > BiDAF (+16.9). Not significant: BiDAF vs classical (+3.1,
p = 0.07), attention vs no attention (+0.1, p = 0.88).

## Open issues
- LLM `na_prob`: `seqlogprob` vs `first_token` undecided (60 questions, 0.5B model). Compare on full dev with 7B.
- Every LLM so far over-abstains under prompt `id1`. Test the prompt before comparing models.
- BiDAF and the classical pipeline barely beat the no-answer floor on HasAns: both over-abstain.
- `torchvision` was uninstalled from the global Python 3.12 (peft import conflict). Not needed here; other global
  packages (ultralytics, easyocr) need it. Pending user decision.
- Leaderboard metric and submission format not yet confirmed (SCeLE). Official validation set not yet in `data/raw/`.

## Next (needs approval for GPU or API spend)
1. LLM on full dev: Qwen2.5-7B-Instruct 4-bit and Sailor2-8B-Chat, with 0-shot, fixed 3-shot and retrieved 3-shot
   prompting. Then 5 prompts x 3 demo orders on the better model.
2. More PLM experiments: XLM-R (3 seeds), IndoBERT frozen vs LoRA, train fractions 0.1, 0.25 and 0.5.
3. Paid `google/gemma-4-31b-it` via OpenRouter (about USD 0.10, returns logprobs) for `06`.
4. When the official validation set arrives: leakage audit against train, switch model selection to it, and write a
   submission script in `experiments/global/mrc`.
5. Submission antara (25 Nov): short report built from `report/latex` and `report/assets/tables/main_table.md`.
