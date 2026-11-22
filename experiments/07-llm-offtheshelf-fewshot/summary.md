# 07 · llm · offtheshelf · fewshot

**Scope** llm · **Subscope** offtheshelf · **Name** fewshot · **Variants** one per run folder in `runs/<split>/`
(one model with one prompt variant) · **Status** prompt and demos waiting for review; harness tested on CPU with
Qwen2.5-0.5B (5 dev questions). No GPU or API run yet.

**Differentiating point.** Compared with 05 and 06 (one-line instruction `id1`, mostly 0-shot), this experiment
changes only the prompt. It uses a task-specific instruction (span MRC; find the related sentence first, check it
answers the question, copy the span, otherwise `unanswerable`, with the dataset's usual unanswerable traps spelled out). It adds 7
frozen train demos (4 answerable, 3 unanswerable). The model, parser and grounding are the same off-the-shelf
setup. Runs go in a fixed, resumable question order, so partial runs are still comparable.

**Verdict so far.** None yet.

## 1. Where it starts
Inputs:
- `data/processed/dev.jsonl` (803) and `data/processed/official_val.jsonl` (2003, the official validation set, released
  2026-11-18, copied unchanged by `mrc.prepare_data`).
- The fixed orders `data/splits/order_<split>.txt`, written by `mrc.eval_order` with the global seed 13.
- Demos from `data/processed/train.jsonl` only.

Shared code: `mrc.llm_qa` (model loading, greedy decoding, parser, grounding, `seq_na_scores`) and `mrc.llm_api` (API
backends).

This folder owns:
- the prompts (`prompts.yaml`)
- the demo choice (`select_demos.py` → `demos.json`, `prompt_review.md`)
- the runner (`run_fewshot.py`) and the GPU notebook (`gpu/`)
- the runs and the analysis

## 2. Pipeline, end to end
| Step | Command | Output |
|---|---|---|
| 0. Fixed orders (once; already done) | `python experiments/global/run.py mrc.eval_order` (`--check` verifies) | `data/splits/order_*.txt`, `order.json` |
| 1. Demos (once; already done) | `python experiments/07-llm-offtheshelf-fewshot/select_demos.py` (`--pin ID ...` to choose by hand) | `demos.json`, `prompt_review.md` |
| 2. Review | read `prompt_review.md`; edit `prompts.yaml` or re-pin demos | |
| 3. Bundle for the GPU | `python experiments/07-llm-offtheshelf-fewshot/gpu/make_bundle.py`; upload `gpu/build/mrc07_bundle.zip` to Drive, share "anyone with the link: viewer"; then `python experiments/07-llm-offtheshelf-fewshot/gpu/build_notebook.py <file id>` (bakes the id and SHA-256 into the notebook) | `gpu/build/`, `gpu/fewshot_gpu.ipynb` |
| 4. Run on a GPU | `gpu/fewshot_gpu.ipynb` on Colab or Kaggle, any account: it downloads the bundle with gdown and checks its hash; progress bar + ETA per batch, `progress.json`, each prediction saved immediately; runs go to the runner's own Drive (`MyDrive/mrc07_runs`) or the session disk, then a zip | `runs/<split>/<run>/` |
| 4b. Per-run error report | `python experiments/07-llm-offtheshelf-fewshot/report_fewshot.py <run dir>` (the notebook runs it and saves a timestamped zip of the run to the Drive folder `proyek_akhir/mrc07/results`) | `<run dir>/report_<split>.md` |
| 5. Error analysis | `python experiments/07-llm-offtheshelf-fewshot/analyze.py [--split official_val]` | `analysis/` (dev), `analysis/official_val/`, section 4 |

Direct use:
```
python experiments/07-llm-offtheshelf-fewshot/run_fewshot.py --model Qwen/Qwen2.5-7B-Instruct --load-4bit \
    --split dev --variant fs_id [--limit 200] [--batch-size 4] [--out-root <dir>]
python experiments/07-llm-offtheshelf-fewshot/run_fewshot.py --backend openrouter --model <id> --split dev   # API, untuned
```
Running the same command again resumes the run. `--limit N` stops after the first N questions of the fixed order.

## 3. Design
- **Prompt** (`prompts.yaml`). The instruction goes in the system turn. Each demo is a user turn
  (`Paragraf: ...` / `Pertanyaan: ...`) followed by an assistant turn (the gold span, or `unanswerable`). The query is
  the last user turn.
  - `fs_id` is the instruction in Indonesian; `fs_en` is the same instruction in English.
  - The no-answer string is `unanswerable`: English never occurs in the passages, and the shared parser already maps it
    to no-answer.
  - Chat templates without a system role (Gemma) get the instruction at the top of the first user turn. This is
    detected per model and recorded as `system_role` in config.json.
- **Demos** (`demos.json`, chosen by `select_demos.py`). There are 7 demos from train, in the order A N A N A N A.
  - The 3 of 7 unanswerable share (43%) is close to train's 44%.
  - The answerable slots cover siapa, kapan and berapa (short answers) and apa (a 3–12 word phrase).
  - The unanswerable slots cover kapan, siapa and apa, and each question shares at least 2 content words with its
    passage.
  - Filters for every demo:
    - passage of 250–650 characters, each demo from a different passage
    - no " ?" question artifact
    - unambiguous, clean gold span: it occurs once in the passage, all gold answers agree, and its brackets and
      quotes are balanced
  - The draw is seeded with the global seed. `demos.json` stores a hash, and every run re-checks each demo against
    train. So all runs see the same 7 demos in the same order, and a hand edit is refused.
- **Fixed order** (`mrc.eval_order`). Questions are stratified by answerability, each stratum is shuffled with seed 13,
  and the strata are interleaved. So every prefix keeps the split's unanswerable share (dev 46%, official_val 36%).
  - Partial runs are compared on their common prefix.
  - `order.json` stores the split and order hashes, and a changed split stops the runner.
- **Resumability.**
  - Each prediction is appended to `preds_<split>.partial.jsonl` and fsynced.
  - A rerun skips finished questions and drops a half-written last line.
  - The settings fingerprint (model, prompt hash, demo hash, order hash, decoding) is stored in `config.json`, and a
    rerun with different settings in the same folder is refused.
  - Each session's library versions and GPU are recorded in `config.json`.
  - Keep `--batch-size` the same within a run, because padding can change greedy outputs slightly.
- **na_prob** (local models). This is the shared sequence score from 05: sigmoid(log p(`unanswerable`) − log p(best
  span)), by teacher forcing on the same prompt. The no-answer threshold is tuned on the evaluated split, as for every
  other model. API runs have no log-probs, so they are untuned.
- **Model candidates** for a 16 GB T4/P100 in 4-bit. Check each id on the Hub before running.
  - `Qwen/Qwen2.5-7B-Instruct` (continuity with 05, ungated)
  - a Qwen3 8B instruct model (`TEMPLATE_KWARGS = {"enable_thinking": False}`)
  - `meta-llama/Llama-3.1-8B-Instruct` and `google/gemma-2-9b-it` (gated: HF token as a secret)
  - Sailor2 8B chat (Southeast-Asian languages)
- **Cost.** Qwen2.5-0.5B on a laptop CPU took about 25 s per question with na_prob scoring, because the 7 demos make
  the prompt about 1,500 tokens long. On a GPU, measure first with `LIMIT = 20`, then set the batch size.

## 4. Results and error analysis
Not run yet. `analyze.py` writes the dev block here and an official_val block below it.

## 5. Findings
None yet.

## 6. Open questions
- Does the instruction help, or the demos? The ablation is `--zero-shot` (run name suffix `_0shot`) against the
  7-shot default, with the same model and variant.
- `fs_id` against `fs_en`: the 05 smoke runs hinted that English instructions do no worse.
- Comparisons on official_val need the other models' predictions on official_val too. They exist only for dev so far.
  The notebook defaults to official_val anyway, since that is where every model is compared from now on.
- The NA threshold for official_val should come from dev (tuned there, applied unchanged), not from official_val
  itself. That needs a dev run of the same model; until then report official_val "as generated".
