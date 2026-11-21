# Shared contract (every experiment and agent follows this)

Task: Indonesian extractive MRC, SQuAD v2 style (44% of questions unanswerable). Answer = a literal span of `context`,
or "" for no answer. Repository conventions: `experiments/global/README.md`.

## Data (already built, do not regenerate or edit)
- `data/processed/{train,dev,itest}.jsonl`: fields `id, context, question, answers[{text, answer_start}], is_impossible`.
  Grouped by context, 6435 / 803 / 804.
- `itest` = internal sealed test. **Never read, train on, tune on, or evaluate on `itest`.** Only
  `experiments/global/mrc/final_test_eval.py` touches it, later, once.
- Tune, early-stop and pick thresholds on `dev` only.
- Later: `data/processed/official_val.jsonl` and `official_test.jsonl` (no gold) may appear. Scripts must accept any
  split path via `--split-file`.

## Evaluator (shared, do not reimplement metrics)
`from mrc.evaluate import evaluate, per_example_scores, aggregate, best_na_threshold, normalize_answer, f1, exact_match, load_jsonl`
CLI: `python experiments/global/run.py mrc.evaluate --gold data/processed/dev.jsonl --pred <preds.jsonl> [--na-threshold-search] [--out metrics.json]`
Primary metric: F1 (also EM, HasAns_f1, NoAns_acc). Leaderboard metric not yet announced; assume SQuAD-v2 EM/F1.

## Where an experiment lives
`experiments/NN-<scope>-<name>/` (e.g. `03-nn-bidaf`), owning everything from its differentiating point to its error
analysis: its code, `analyze.py`, `runs/`, `analysis/` and `summary.md`. Shared code goes in `experiments/global/mrc`,
never copied into an experiment.

## Outputs per run
`experiments/NN-<scope>-<name>/runs/<variant>/seed<k>/` containing:
- `config.json` (all hyperparameters, seed, timestamp)
- `preds_dev.jsonl`: one line per dev example: `{"id", "pred", "na_prob" (float, higher = more likely unanswerable),
  "score" (span confidence), "start_char" (int or -1)}`
- `metrics_dev.json` (from `mrc.evaluate.evaluate(..., search=True)`)
- `log.txt` (training log: loss per epoch or step, time, params)

Then run the experiment's `analyze.py` and update the hand-written sections of `summary.md` (differentiating point,
pipeline and commands, design, findings, open questions). Section 4 of `summary.md` is generated.

## Rules
- Python 3.12, CPU-only machine here (8 cores, 25 GB RAM, torch 2.14 CPU, transformers 5.4). GPU models run on
  Colab or Kaggle T4 via the same script with `--device cuda`; locally run a short smoke test only.
- Fixed seeds; every run reproducible with one command. Default seeds: 13, 42, 87.
- No new metric code; no edits to `experiments/global/mrc/evaluate.py` or `data/`. If you think the evaluator is wrong,
  say so in your final message.
- Imports: `from mrc.<module> import ...`. Experiment scripts start with the two-line bootstrap that puts
  `experiments/global` on `sys.path` (copy it from any existing script). Shared modules run via
  `python experiments/global/run.py mrc.<module>`.
- Keep code readable, typed where cheap, no notebooks for results (notebooks only browse saved outputs). Write concise
  docs, no em dashes, no hype words.
- Final message: files created, exact commands run, dev metrics (mean ± std over seeds if run), anything broken or
  suspicious, and 3 to 5 interesting errors you saw in the predictions (question, gold, pred, likely cause).
