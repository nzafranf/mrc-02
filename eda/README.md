# EDA

Two kinds of analysis live here. Neither trains a model.

1. **Data EDA.** What the passages, questions and answers look like, and which properties of the data a model can
   exploit without reading. Train and dev only. The internal test split (itest) appears only as counts and in the
   leakage audit.
2. **Cross-model error browser.** Every model's prediction on every dev question, side by side. It reads saved
   predictions only.

## Layout
| Folder | Content |
|---|---|
| `notebooks/eda.ipynb` | Data EDA, interactive: statistics and example passages with gold answers highlighted (orange marks question words that do not occur in the passage) |
| `notebooks/model_errors.ipynb` | Cross-model error browser (generated, see below) |
| `scripts/eda.py` | Statistics tables and figures |
| `scripts/leakage_audit.py` | Exact and near-duplicate contexts and questions across splits |
| `scripts/artifact_probe.py` | Can answerability be predicted without reading the passage? |
| `scripts/tokenizer_analysis.py` | Subword fertility of candidate PLM tokenizers on the passages |
| `scripts/build_model_errors_nb.py` | Generates `notebooks/model_errors.ipynb`. If you edit the notebook by hand, do not rerun it |
| `outputs/tables/`, `outputs/figures/` | Everything the scripts write (`eda_*`, `artifact_probe.json`, `leakage_audit.json`, `tokenizer_analysis.json`) |

Run any script from anywhere: `python eda/scripts/eda.py`. The slice tags that every analysis uses
(`data/processed/slices_*.jsonl`) come from the shared `experiments/global/mrc/analysis/slices.py`. Definitions are in
[miscellaneous/docs/data_card.md](../miscellaneous/docs/data_card.md).

## Findings (train + dev)
- **Size and split.** 6435 / 803 / 804 questions (train / dev / itest), grouped by passage so that no passage crosses
  splits. 44% of train and 46% of dev questions are unanswerable. Passages are Indonesian Wikipedia paragraphs
  (mean 83 words). Answers are short (median 3 words).
- **Answers sit early.** 52% start in the first quarter of the passage. The answer types are lowercase phrase 31%,
  other capitalized 24%, date/year 18%, person-like 17% and number 10%.
- **Unanswerable questions are paired edits.** 97.6% of two-question passages have exactly one answerable and one
  unanswerable question. The unanswerable one is a minimal edit of its twin (a changed entity, date or relation), so
  word overlap with the passage barely differs (0.47 vs 0.50).
- **Answerability leaks through the question alone.** A question-only TF-IDF classifier predicts unanswerability with
  AUC 0.90 (dev). Typography is a cue: questions ending in " ?" are rarely unanswerable, and the unanswerable rate is
  5% for "dimana" but 88% for "di ...". The passage alone gives AUC 0.51. Every model can learn these shortcuts, so
  read NoAns accuracy with that in mind.
- **Leakage is negligible.** Dev vs train is clean. 2 of 347 itest passages are near-duplicates of train/dev
  passages; their ids are excluded in `data/splits/itest_exclude_ids.txt`.

## Scope of the error analysis: what lives where
The project analyses errors at three levels. This folder covers only the level that needs every model at once.

| Level | Question it answers | Where |
|---|---|---|
| Within one experiment | How do this experiment's variants and seeds fail? Outcome types, slices, seed stability, calibration, error samples for manual labels | `experiments/NN-*/analyze.py` -> `analysis/` and section 4 of each `summary.md` |
| Across experiments, aggregated | Which model is better, is it significant, on which slices | `python experiments/global/run.py mrc.analysis.compare` -> `report/assets/tables`, `report/assets/figures` |
| Across experiments, per question | On this question, what did each model answer and why do they differ? | `notebooks/model_errors.ipynb` (this folder) |

`model_errors.ipynb` reads `report/assets/tables/per_example_all_models.jsonl`, built by
`python experiments/global/run.py mrc.analysis.per_example`. That file covers every model in
`experiments/global/configs/models_registry.yaml` plus the partial API runs. It contains:
- a side-by-side table of all models' answers, prioritising questions every model answered, plus the disagreements
  among them
- outcome type (rows) by model (columns), and the outcome-type confusion between pairs of models (what one model's
  errors become in another)
- seed-vs-seed stability, the shared-error rate between models, and the answerable/unanswerable confusion
- a full-passage view with every model's span marked, and ready-made queries (hard set, "A right, B wrong",
  one error type, slices, export to CSV for labelling)

To add a new experiment to the browser: add its runs to `models_registry.yaml`, rerun `mrc.analysis.per_example`, then
rerun the notebook.
