# Requirements checklist (from the brief)

| # | Requirement (brief section) | Where satisfied | Status |
|---|---|---|---|
| 1 | At least 3 distinct models (2.3) | 02-classical-pipeline-lgbm, 03-nn-bidaf, 04-plm-indobert, 05/06 LLM | done (dev) |
| 2 | At least 1 end-to-end model (2.3.1a) | 03-nn-bidaf, 04-plm-indobert, 05-llm-local-qwen | done (dev) |
| 3 | At least 1 pipeline model (2.3.1b) | 02-classical-pipeline-lgbm (answerability, retriever, span ranker) | done (dev) |
| 4 | At least 1 traditional ML with feature engineering (2.3.2a) | 02-classical-pipeline-lgbm | done (dev) |
| 5 | At least 1 NN/PLM/LLM (2.3.2b) | 03-nn-bidaf (NN), 04-plm-indobert (PLM), 05/06 (LLM) | done (dev) |
| 6 | Explain architecture, components, input representation, prediction mechanism, config per model (2.3) | `experiments/NN-*/summary.md` section 3, `report/latex/main.tex` section 3 | partial |
| 7 | Submission antara: short report + models (2.4), 25 Nov | `report/latex/` (or a short version of it) | todo |
| 8 | Test predictions from at least 3 models in the official format (2.5) | `--predict-only` in every model (04: `--also-predict` + `checkpoint.pt`), `mrc.make_submission` writes the `id,answer` CSV (format from the sample submission) | partial: script done, 01-04 predicted on official_val, 04 checkpoints saved; test set on 3 Dec |
| 9 | Source code to reproduce (2.5) | README and each `summary.md` section 2 (one command per step) | partial |
| 10 | Final report (2.5, 4.4) | `report/latex/main.tex` (render with `miscellaneous/scripts/render_latex.py`) | skeleton |
| 11 | Per-person log, reflection, peer review (2.5) | each member, outside this repo or `miscellaneous/docs/personal_logs/` | todo, human only |
| 12 | EDA: passage, question, answer stats (4.2.1) | `eda/` (README, notebooks, scripts), `miscellaneous/docs/data_card.md` | done |
| 13 | Model comparison (4.2.2) | `mrc.analysis.compare` -> `report/assets/tables`; `eda/notebooks/model_errors.ipynb` | done (dev) |
| 14 | Feature ablation or hyperparameter tuning on validation (4.2.2) | classical ablation, BiDAF attention ablation, PLM regimes | partial |
| 15 | Additional evaluation beyond leaderboard score (4.2.2) | HasAns/NoAns, slices, calibration, stage attribution, grounding | partial |
| 16 | Qualitative: correct and wrong examples, strengths/weaknesses, error causes, reasoning challenges (4.2.3) | automatic outcome types + `experiments/NN-*/analysis/error_sample_*.md` for manual labels | todo, human |
| 17 | No use of test gold or manual annotation of test (5) | test gold only after 23 Dec, for analysis | enforced by contract |
| 18 | Document external models, data, libraries (5) | `miscellaneous/docs/decisions.md`, report appendix | partial |
| 19 | Declare AI use (5) | report appendix: "AI use declaration" | todo |
