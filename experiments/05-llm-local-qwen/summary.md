# 05 · llm · local · qwen

**Scope** llm · **Subscope** local · **Name** qwen · **Variants** one per run folder in `runs/` · **Status** smoke tests
only (Qwen2.5-0.5B-Instruct on CPU, 60 to 100 dev questions). The 7B runs are planned on a GPU.

**Differentiating point.** A local instruct LLM, run through the shared prompt, parse and ground harness. This folder
owns model loading and inference settings, few-shot choice, the contamination probe and LoRA fine-tuning.

**Verdict so far.** The 0.5B model is a pipeline check, not a result. Its no-answer signal is weak (tuned thresholds
make it abstain on about 80% of questions), and 100 questions are too few to rank prompts.

## 1. Where it starts
Input: `data/processed/{train,dev}.jsonl`. The shared harness is `experiments/global/mrc/llm_qa.py` with prompts in
`experiments/global/configs/llm_prompts.yaml`. It is shared with `06-llm-api-offtheshelf`, so the two differ only in
where the model runs. This folder owns `llm_contamination.py` (memorization probe), `llm_lora.py` (QLoRA SFT) and the runs.

## 2. Pipeline, end to end
| Step | Command | Output |
|---|---|---|
| Scorer self-test | `python experiments/global/run.py mrc.llm_qa --self-test --model-name <M>` | assertion pass/fail |
| Inference on dev | `python experiments/global/run.py mrc.llm_qa --model-name <M> --prompt-id id1 --shots 0 --run-name <name> [--limit N]` (GPU: `--device cuda --load-4bit --batch-size 8`) | `runs/<name>/` |
| Contamination probe | `python experiments/05-llm-local-qwen/llm_contamination.py --n 50 --model-name <M>` | `contamination/<model>.json` |
| LoRA SFT | `python experiments/05-llm-local-qwen/llm_lora.py --model-name <M> ...`, then inference with `--adapter-dir` | `lora_smoke/adapter` (smoke) |
| Error analysis | `python experiments/05-llm-local-qwen/analyze.py` | `analysis/`, section 4 below |

GPU grid (prompt sensitivity 5 prompts x 3 demo orders, model comparison, QLoRA): [gpu_runs.md](gpu_runs.md).

### Commands used for the smoke runs
```bash
# local smoke (CPU), first 100 dev examples
python experiments/global/run.py mrc.llm_qa --limit 100 --prompt-id id1 --run-name qwen0.5b_id1_0shot_n100 --batch-size 4
python experiments/global/run.py mrc.llm_qa --limit 100 --prompt-id en1 --run-name qwen0.5b_en1_0shot_n100 --batch-size 4
python experiments/global/run.py mrc.llm_qa --limit 100 --prompt-id id1 --shots 3 --run-name qwen0.5b_id1_3shot_fixed_n100 --batch-size 4
python experiments/05-llm-local-qwen/llm_contamination.py --n 50 --max-new-tokens 64
python experiments/05-llm-local-qwen/llm_lora.py --max-steps 5 --batch-size 2 --max-len 512
```
GPU: add `--device cuda --load-4bit --batch-size 8` and a larger `--model-name`; see `experiments/05-llm-local-qwen/gpu_runs.md`
(prompt sensitivity grid 5 prompts x 3 demo orders, model comparison, QLoRA).

## 3. Design
- Modes: `--shots 0`; `--shots k --demo-mode fixed` (balanced answerable/unanswerable demos from train, chosen by
  `--demo-seed`, short contexts only, ids written to `config.json`); `--demo-mode retrieved` (TF-IDF over train
  questions, nearest answerable and unanswerable alternately, most similar demo last). `--demo-order-seed` permutes demos.
  Demos are multi-turn chat messages (user: instruction + context + question, assistant: gold span or `TIDAK ADA JAWABAN`).
- Greedy decoding, `--max-new-tokens 24`.
- Parsing: first non-empty line, drop `Jawaban:`/`Answer:` prefixes, strip quotes and trailing period; no-answer phrases
  (`TIDAK ADA JAWABAN`, `tidak disebutkan`, `no answer`, ...) map to `""`.
- Grounding, stored as `grounding`: `exact` (case-insensitive substring, mapped back to original casing), `snapped`
  (best word n-gram by token F1 >= 0.5), `ungrounded` (F1 < 0.5, the cleaned text is kept as pred and counted as
  hallucination), `noans`. `format_failure_rate` = non-no-answer outputs longer than 25 words (verbose, not a span).
- `na_prob`, chosen with `--na-method {seqlogprob,first_token}` (default `seqlogprob`).
  - `seqlogprob`: teacher forcing on the same chat prompt. `lp_noans` = total log-prob of `TIDAK ADA JAWABAN` plus the
    end-of-turn token; `lp_ans` = total log-prob of the answer string plus the end-of-turn token. The answer string is the
    final grounded `pred`, i.e. exactly what gets thresholded. If greedy output is no-answer there is no pred, so
    `lp_ans` is the best total log-prob among up to 3 context candidates (`candidate_spans`: sentences ranked by
    content-word overlap with the question, candidates are runs of capitalized words or numbers inside them, excluding
    words that are all in the question; if none exist, `lp_ans = lp_noans - 20`). `na_prob = sigmoid(lp_noans - lp_ans)`.
    Stored in preds: `lp_noans`, `lp_ans`, `lpn_noans`, `lpn_ans` (length-normalized, not used), `ans_scored`.
    Cost: one extra forward pass over prompt plus up to 4 short continuations per example (right-padded batch).
  - `first_token` (old): at the first decoding step `na_prob = p_no / (p_no + p_other)` for the first token of
    `TIDAK ADA JAWABAN` (Qwen: the letter `T`). Ambiguous with answers that begin with `T` (Tahun, Tanzania).
    It is always stored as `na_prob_first`; `metrics_dev.json` has `auc_na_prob` and `auc_na_prob_first_token`.
  - `python experiments/global/run.py mrc.llm_qa --self-test` asserts the scored slice decodes to exactly the continuation plus the
    end token (no off-by-one with the chat template) and matches an unpadded reference forward pass.
  - `score` = mean log-prob of generated tokens. Tune the threshold on dev:
    `python experiments/global/run.py mrc.evaluate --gold data/processed/dev.jsonl --pred <preds> --na-threshold-search`
    (also stored in `metrics_dev.json` as `best_na_threshold` / `at_best_threshold`).
  - Runs before this change (table below) used `first_token`.

### Candidate models (names checked on the Hub, 2026-11-13)
| model | note |
|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | local smoke only |
| `Qwen/Qwen2.5-7B-Instruct` | main GPU candidate, 4-bit fits a T4 |
| `sail/Sailor2-8B-Chat` (also `-1B-`, `-20B-Chat`) | SEA-focused, covers Indonesian |
| `SeaLLMs/SeaLLMs-v3-7B-Chat` | SEA-focused, Indonesian |
| `GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct` | Indonesian-focused continued pretraining |
| `aisingapore/gemma2-9b-cpt-sea-lionv3-instruct` | SEA-LION, Indonesian |
| `Qwen/Qwen3-8B` | exists; has a thinking mode, disable it for this task |
Gated (HF token needed): `google/gemma-2-9b-it`, `meta-llama/Llama-3.1-8B-Instruct`. Existence was checked with
`HfApi().model_info`; Indonesian strength is the models' stated focus, not something measured here.

## 4. Results and error analysis
Variants differ in prompt, shots, na_prob method and the number of questions (n), so compare rows with the same n.
<!-- AUTO-RESULTS:BEGIN (written by analyze.py, edits here are overwritten) -->

### Results (dev, NA threshold tuned on dev)

| variant | runs | n | EM | F1 | HasAns F1 | NoAns acc | predicted no-answer % | NA threshold |
|---|---|---|---|---|---|---|---|---|
| qwen0.5b_en1_0shot_n100 | 1 | 100 | 43.0 | 49.4 | 17.9 | 92.9 | 76.0 | 0.0515 |
| qwen0.5b_id1_0shot_n100 | 1 | 100 | 46.0 | 51.9 | 17.1 | 100.0 | 79.0 | 0.0583 |
| qwen0.5b_id1_0shot_n60_first_token | 1 | 60 | 45.0 | 52.1 | 20.2 | 100.0 | 76.7 | 0.0583 |
| qwen0.5b_id1_0shot_n60_seqlogprob | 1 | 60 | 45.0 | 46.1 | 10.2 | 100.0 | 93.3 | 0.0321 |
| qwen0.5b_id1_3shot_fixed_n100 | 1 | 100 | 45.0 | 48.5 | 12.9 | 97.6 | 87.0 | 0.000506 |

### Outcome types

Share of each gold group; count (mean over seeds) in brackets. Figure: [analysis/outcomes.png](analysis/outcomes.png).

| gold | outcome | qwen0.5b_en1_0shot_n100 | qwen0.5b_id1_0shot_n100 | qwen0.5b_id1_0shot_n60_first_token | qwen0.5b_id1_0shot_n60_seqlogprob | qwen0.5b_id1_3shot_fixed_n100 |
|---|---|---|---|---|---|---|
| answerable | correct_ans | 6.9% (4) | 6.9% (4) | 8.3% (3) | 8.3% (3) | 6.9% (4) |
| answerable | false_no_answer | 63.8% (37) | 63.8% (37) | 61.1% (22) | 88.9% (32) | 79.3% (46) |
| answerable | wrong_sentence | 6.9% (4) | 6.9% (4) | 5.6% (2) | 0.0% (0) | 1.7% (1) |
| answerable | wrong_span_same_sentence | 1.7% (1) | 1.7% (1) | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| answerable | boundary_too_long | 15.5% (9) | 15.5% (9) | 19.4% (7) | 2.8% (1) | 5.2% (3) |
| answerable | boundary_too_short | 1.7% (1) | 1.7% (1) | 2.8% (1) | 0.0% (0) | 0.0% (0) |
| answerable | partial_overlap | 3.4% (2) | 3.4% (2) | 2.8% (1) | 0.0% (0) | 6.9% (4) |
| unanswerable | correct_no_ans | 92.9% (39) | 100.0% (42) | 100.0% (24) | 100.0% (24) | 97.6% (41) |
| unanswerable | false_answer_on_unanswerable | 7.1% (3) | 0.0% (0) | 0.0% (0) | 0.0% (0) | 2.4% (1) |

### Variants against `qwen0.5b_en1_0shot_n100` (paired bootstrap)

| comparison | n | delta F1 | 95% CI | p (paired bootstrap) |
|---|---|---|---|---|
| qwen0.5b_id1_0shot_n100 - qwen0.5b_en1_0shot_n100 | 100 | +2.53 | [-1.48, 7.08] | 0.2290 |
| qwen0.5b_id1_0shot_n60_first_token - qwen0.5b_en1_0shot_n100 | 60 | +1.92 | [-4.19, 8.19] | 0.5070 |
| qwen0.5b_id1_0shot_n60_seqlogprob - qwen0.5b_en1_0shot_n100 | 60 | -4.06 | [-11.48, 3.53] | 0.2910 |
| qwen0.5b_id1_3shot_fixed_n100 - qwen0.5b_en1_0shot_n100 | 100 | -0.90 | [-8.06, 5.65] | 0.7690 |

### Seed stability

single run per variant

### Weakest slices for `qwen0.5b_id1_0shot_n60_first_token`, the best variant (n >= 30). All slices: [analysis/slices.md](analysis/slices.md)

| slice | value | n | F1 (qwen0.5b_id1_0shot_n60_first_token) |
|---|---|---|---|
| lex_overlap_bucket | low | 30 | 43.9 |
| ends_space_qmark | False | 51 | 58.3 |

### Hard questions

27 of 60 questions are wrong (F1 < 0.5) in every run of every variant; 0% of them are unanswerable (dev base rate 40%). List: [analysis/hard_examples.jsonl](analysis/hard_examples.jsonl).

### Analysis files

[error_sample_qwen0.5b_en1_0shot_n100.md](analysis/error_sample_qwen0.5b_en1_0shot_n100.md), [error_sample_qwen0.5b_id1_0shot_n100.md](analysis/error_sample_qwen0.5b_id1_0shot_n100.md), [error_sample_qwen0.5b_id1_0shot_n60_first_token.md](analysis/error_sample_qwen0.5b_id1_0shot_n60_first_token.md), [error_sample_qwen0.5b_id1_0shot_n60_seqlogprob.md](analysis/error_sample_qwen0.5b_id1_0shot_n60_seqlogprob.md), [error_sample_qwen0.5b_id1_3shot_fixed_n100.md](analysis/error_sample_qwen0.5b_id1_3shot_fixed_n100.md), [hard_examples.jsonl](analysis/hard_examples.jsonl), [outcomes.md](analysis/outcomes.md), [outcomes.png](analysis/outcomes.png), [per_example.jsonl](analysis/per_example.jsonl), [results.md](analysis/results.md), [seed_stability.md](analysis/seed_stability.md), [significance.md](analysis/significance.md), [slices.md](analysis/slices.md)

<!-- AUTO-RESULTS:END -->

### Smoke results as logged (Qwen2.5-0.5B-Instruct, first 100 dev: 58 answerable, 42 not)
Timing was measured while 5 to 6 other jobs saturated the 8 CPU cores, so sec/example is inflated several-fold
(batch size 4, fp32). Treat as an upper bound.

| config | EM | F1 | HasAns_f1 | NoAns_acc | pred_noans | exact/snapped/ungrounded/noans | s/ex |
|---|---|---|---|---|---|---|---|
| id1 0-shot | 16.0 | 26.3 | 26.3 | 26.2 | 28% | .14/.40/.18/.28 | 17.3 |
| en1 0-shot | 22.0 | 32.8 | 34.1 | 31.0 | 20% | .28/.38/.14/.20 | 9.4 |
| id1 3-shot fixed | 21.0 | 32.3 | 55.6 | 0.0 | 1% | .58/.33/.08/.01 | 13.2 |

With the dev-tuned `na_prob` threshold F1 rises to 51.9 (id1) and 49.4 (en1), but the tuned threshold abstains on
about 80% of examples (NoAns_acc 100, HasAns_f1 17), which is about what "always no answer" gives (44% NoAns).
So the 0.5B model's no-answer signal is weak; 100 examples is also too few to rank the prompts.
The 3-shot run shows the main failure of the small model: demos lift HasAns_f1 from 26 to 56 but the model almost never
abstains (NoAns_acc 0), so F1 is capped near 32 unless the `na_prob` threshold recovers abstention (best-threshold F1 48.5).

### na_prob method comparison (Qwen2.5-0.5B-Instruct, id1 0-shot, first 60 dev, 36 answerable / 24 not)
Runs `runs/qwen0.5b_id1_0shot_n60_{seqlogprob,first_token}` (same greedy outputs, only `na_prob` differs).
Untuned: EM 15.0, F1 26.6, HasAns_f1 27.7, NoAns_acc 25.0 for both. At the dev-tuned threshold:

| method | AUC vs is_impossible | threshold | F1 | HasAns_f1 | NoAns_acc | pred_noans | s/ex |
|---|---|---|---|---|---|---|---|
| first_token | 0.641 | 0.058 | 52.1 | 20.2 | 100 | 77% | 3.6 |
| seqlogprob | 0.552 | 0.032 | 46.1 | 10.2 | 100 | 93% | 7.9 |

With 60 examples (24 negatives) the AUC difference is within noise; on this 0.5B model the sequence score is not better.
Timing is under CPU contention. Re-compare on full dev with the 7B models before choosing.

## 5. Findings
- Over-abstention on easy answerable questions (`TIDAK ADA JAWABAN` for "kapankah Uni Eropa didirikan?").
- Verbose sentences instead of spans (ungrounded or snapped to a long n-gram).
- Answering unanswerable questions by rewriting nearby text (snapped onto an unrelated clause).
- Right entity-type, wrong entity (picks another name from the context).

**Contamination.** ROUGE-L recall of the continuation vs the true second half: original 0.114, sentence-shuffled control 0.122;
0 of 50 above 0.5 in both. No sign of verbatim memorization for this model (a 0.5B model would be a weak test).
Re-run with the 7B models on the GPU. Details per context: `experiments/05-llm-local-qwen/contamination/*.json`.

**LoRA.** `python experiments/05-llm-local-qwen/llm_lora.py --max-steps 5 --batch-size 2 --max-len 512` ran on CPU: 8.8M trainable params
(1.75%), 5 steps in about 100 s (loss 0.78, 0.18, 0.19, 0.90, 1.19, noisy with batch 2), adapter saved to
`experiments/05-llm-local-qwen/lora_smoke/adapter`. Evaluate adapters with `llm_qa --adapter-dir`. Not a real training run.

## 6. Open questions
- Qwen2.5-7B-Instruct and Sailor2-8B-Chat on full dev: zero-shot, fixed 3-shot and retrieved 3-shot.
- All LLMs so far over-abstain under prompt `id1`, and 06 shows the same. Test whether the prompt causes it before
  comparing models.
- `seqlogprob` vs `first_token` na_prob on full dev.
