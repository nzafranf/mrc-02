# Orchestrator Prompt, adapted to Proyek Akhir NLP Gasal 2026: MRC

This is `orchestrator_prompt.md` with Section 0 filled in and Sections 7 and 8 rewritten for extractive MRC. Sections 1 to 6 and 9 to 13 of the original apply unchanged except where overridden below. If the brief and this file conflict, the brief wins.

## 0. Assignment inputs

| Field | Value |
|---|---|
| Assignment brief | `miscellaneous/docs/Proyek_Akhir_NLP_Gasal_2026__MRC.pdf` |
| Task type | Extractive MRC, SQuAD v2 style (span or no-answer). 44% of train is `is_impossible` |
| Dataset(s) and splits | `data/raw/train.jsonl` (8042 q, 3271 contexts). Official validation released 18 Nov, test (no gold) 3 Dec, test gold after 23 Dec. Internal grouped split: train 6435 / dev 803 / itest 804 (`python experiments/global/run.py mrc.prepare_data`, seed 13) |
| Language | Indonesian (Wikipedia-style, some wiki citation marks `[n]`) |
| Primary metric | Leaderboard metric not announced yet (announced 11 Nov on SCeLE, check). Assumed SQuAD v2 F1, with EM |
| Secondary metrics | EM, HasAns F1, NoAns accuracy, answerability AUC, sentence recall (pipeline), grounding rate (LLM), cost |
| Compute | Local: CPU only (8 cores, 25 GB). GPU: Colab/Kaggle T4 for PLM/LLM |
| Deadlines | 18 Nov group registration + val set; 25 Nov submission antara; 26 Nov to 9 Dec mentoring; 3 Dec test + leaderboard; 23 Dec leaderboard close; 30 Dec report + code |
| Deliverables | Test predictions from at least 3 models; source code; final report (technical scientific); per-person log, reflection, peer review |
| Mandatory models | At least 3 models: at least 1 end-to-end, at least 1 pipeline, at least 1 traditional ML with feature engineering, at least 1 NN/PLM/LLM. Each must have architecture, components, input representation, prediction mechanism, configs explained |
| Not allowed | Using or manually annotating test gold answers; exchanging predictions between groups; undeclared AI use; unverifiable results |

## 7. Adaptation for extractive MRC (replaces the original table)

Each rung maps to experiment folders `experiments/NN-<scope>-<name>/` (conventions in `experiments/global/README.md`).

| Ladder rung | Concrete model here | Brief category | Key MRC-specific analysis |
|---|---|---|---|
| R0 trivial (`01-baseline-trivial`) | Always no-answer; sliding-window overlap (Richardson 2013); question-only answerability LR (negative control) | baseline | How much of the score is the no-answer prior and question-form artifacts |
| R1 classical (`02-classical-pipeline-lgbm`) | Pipeline: answerability classifier, then BM25/TF-IDF sentence retriever, then LightGBM span ranker with engineered features | Traditional ML + Pipeline | Stage-wise error attribution (A/B/C), feature ablation, feature importance |
| R2 RNN + attention (`03-nn-bidaf`) | BiDAF-lite from scratch, with `--no-attention` and `--rnn` ablations | NN, End-to-end | Attention vs bottleneck by context length; NULL-position no-answer behavior; attention maps on fixed 20 ids |
| R3 Transformer scratch (optional) | Small encoder from scratch, same QA head | NN, End-to-end | Only if time: isolates architecture vs pretraining |
| R4 PLM (`04-plm-indobert`; other encoders and regimes get their own folders) | IndoBERT-base-p1, XLM-R-base: full FT, frozen probe, LoRA; data-efficiency 10/25/50/100% | PLM, End-to-end | Seed variance, tokenizer offset/fertility effects, null-score calibration, IG with deletion curve |
| R5 LLM (`05-llm-local-qwen`, API reference `06-llm-api-offtheshelf`) | Qwen2.5 instruct zero/few-shot (fixed and retrieved demos), then QLoRA | LLM, End-to-end | Prompt sensitivity (5 prompts x 3 orders), grounding/hallucination rate, contamination probe, cost vs R4 |
| Optional R6 | Hybrid pipeline: LLM or PLM answerability verifier on top of PLM reader, or ensemble | Pipeline | Whether a separate verifier beats a single null score |

The "seq2seq with attention" rung of the original is replaced by BiDAF (pointer network with attention), since span extraction is a pointing task, not generation. Logged in `miscellaneous/docs/decisions.md`.

## 8. Milestones mapped to the brief's timeline

| Milestone | Original | Target date | Notes |
|---|---|---|---|
| M0 framing | M0 | 14 Nov | done: requirements, contract, splits |
| M1 data audit + EDA | M1 | 16 Nov | data card, slices, artifact probe, tokenizer stats |
| M2 baselines + classical pipeline | M2 | 18 Nov | analysis pack v1, taxonomy v1 |
| M3 BiDAF | M3+M4+M5 merged | 22 Nov | gradient/attention analyses for the learning note |
| Submission antara | | 25 Nov | short report of M0 to M3 + PLM first run |
| M4 PLM | M7 | 2 Dec | GPU runs, 3 to 5 seeds; ready before test release |
| M5 LLM | M8 | 9 Dec | prompt sensitivity, QLoRA if time |
| Leaderboard | | 3 to 23 Dec | Submit the best model plus the other two. The leaderboard is a test-set touch: pick submissions from dev and official_val only, never iterate on leaderboard feedback beyond the course's intended use, and log every submission in `miscellaneous/docs/leaderboard_log.md` |
| M6 sealed eval + analysis | M9 | 24 to 28 Dec | After test gold release: one run of `python experiments/global/run.py mrc.final_test_eval --confirm ...`, cross-model figures |
| Report | | 30 Dec | Report structure follows brief Section 4 (EDA, quantitative, qualitative) |

Override of original principle 1 for this brief: the official test set has no gold until 23 Dec; the internal `itest` is our sealed set. The official validation set, once released, becomes the main model-selection set and `dev` becomes secondary.

## Delegation notes

- Subagents get `miscellaneous/docs/CONTRACT.md` plus their task, not this file.
- Brief Section 5 says every member must take part in every component and understand every model. The structure here is scaffolding; the group must reproduce runs, do the error analysis by hand, and write the report.
