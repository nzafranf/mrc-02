# Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-11-13 | Internal split 80/10/10 of train into train/dev/itest, grouped by normalized context plus a 120-char prefix key, seed 13 | 3271 contexts for 8042 questions (up to 35 q per context); a random split would leak passages |
| 2026-11-13 | `itest` sealed until M6; official validation replaces dev as main selection set once released | brief releases test gold only after leaderboard closes |
| 2026-11-13 | Metric assumed SQuAD v2 EM/F1; normalization = lowercase, strip `[n]` citations and punctuation, no article removal | Indonesian has no articles; citation marks appear in 19% of contexts. Revisit when the official metric is announced |
| 2026-11-13 | Multiple gold answers: max over golds | standard SQuAD; 64 train questions have 2 golds |
| 2026-11-13 | Seq2seq rung replaced by BiDAF pointer model | extractive QA is pointing, not generation |
| 2026-11-13 | Question-only answerability classifier added as negative control | impossible rate varies from 0.04 to 0.89 by first question word, suggesting dataset-construction artifacts |
| 2026-11-13 | Local CPU for classical/BiDAF; PLM/LLM full runs on Colab/Kaggle T4 | no local GPU |
| 2026-11-13 | Leakage audit found 2 itest contexts near-duplicating train/dev (Abu Bakar 0.998, Kimi no Shiranai Monogatari 0.986; prefix grouping missed them due to a leading quote). Their 2 itest questions are excluded in `experiments/global/mrc/final_test_eval.py` via `data/splits/itest_exclude_ids.txt` instead of re-splitting | models were already training on the frozen train/dev; exclusion removes the leak without changing train/dev hashes |
| 2026-11-13 | For official val/test: run the near-dup audit (`eda/leakage_audit.py`) against train before use | same failure mode can recur |
| 2026-11-21 | Repository reorganized: one folder per experiment `experiments/NN-<scope>-<name>/` owning code, runs, analysis and `summary.md`; shared code in `experiments/global/mrc` (package `mrc`); EDA in `eda/`; LaTeX report in `report/latex`; docs in `miscellaneous/docs`. Old `src/`, `scripts/`, `configs/`, `docs/` removed | each experiment is readable end to end, from its differentiating point to its error analysis; names extend by scope as new experiments (tokenizer, embedding, ...) are added |
| 2026-11-22 | Official validation set copied to `data/processed/official_val.jsonl` and added to the leakage audit; internal train/dev/itest unchanged (hashes identical) | released 2026-11-18; it becomes the main selection set |
