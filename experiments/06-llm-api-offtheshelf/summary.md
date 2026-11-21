# 06 · llm · api · offtheshelf

**Scope** llm · **Subscope** api · **Name** offtheshelf · **Variants** one per run folder in `runs/` (`mock_*` are
offline tests and are excluded) · **Status** partial: free-tier daily caps stopped the runs at 22 (Gemini) and
25 (Nemotron) dev questions

**Differentiating point.** Strong off-the-shelf models behind an API, run through the same prompts, parser and grounding
as 05. This gives a reference point for what a strong model reaches with no training. It also checks the harness: if
a strong model scores well through the same parser, the parser and prompt are not what limits the small local models.

**Verdict so far.** Too few questions to conclude. Nemotron 3.5 Super answers most answerable questions on its 25.
Gemini 3.5 Flash abstains on 16 of 22, including easy date and name questions, which again points at prompt `id1`.

## 1. Where it starts
Input: `data/processed/dev.jsonl` only (never a test split: the harness refuses files with "test" in the name unless
`--allow-test` is given). The harness is `experiments/global/mrc/llm_qa.py --backend openrouter|gemini`. The API client
(cache, retries, rate limit, daily-quota abort) is `experiments/global/mrc/llm_api.py`. This folder owns the staged
runner, the response cache and the runs.

## 2. Pipeline, end to end
| Step | Command | Output |
|---|---|---|
| Staged run (5, 25, 100, 803 questions with gates) | `python experiments/06-llm-api-offtheshelf/run_llm_api_staged.py --backend gemini --api-model <id> --prompt-id id1 --shots 0 --run-name <name> -- --reasoning-effort none` | `runs/<name>/`, `cache/<model>.jsonl` |
| Error analysis | `python experiments/06-llm-api-offtheshelf/analyze.py` | `analysis/`, section 4 below |

Each stage reruns the harness with a larger `--limit`, and cached responses are reused, so a stage only pays for new
questions. A stage stops the run if the share of empty outputs, ungrounded outputs or no-answer predictions is
outside the gates.

### Direct commands
Keys come from the environment only. Never write them to a file in the repo.

```
export OPENROUTER_API_KEY=...        # PowerShell: $env:OPENROUTER_API_KEY="..."
python experiments/global/run.py mrc.llm_qa --backend openrouter --api-model google/gemma-4-31b-it:free --prompt-id id1 --shots 0 --rpm 15

export GEMINI_API_KEY=...
python experiments/global/run.py mrc.llm_qa --backend gemini --api-model gemini-2.5-flash --prompt-id id1 --shots 0 --rpm 10 --reasoning-effort none
```
- Check the model id against the provider's current model list before running.
- `--limit 50` first as a smoke test. Reruns and re-scoring are free: every raw response is cached in
  `experiments/06-llm-api-offtheshelf/cache/<model_slug>.jsonl` (key = sha256 of model + messages + params), and runs resume
  by skipping cached requests.
- `--logprobs` requests top-5 logprobs (many providers refuse or ignore it). If returned, `na_prob` is crude:
  exp(logprob of first generated token) if the output is no-answer, else 1 minus that. Without it `na_prob` is
  absent and metrics are computed without threshold search.
- `--base-url` overrides the endpoint (Gemini default: `https://generativelanguage.googleapis.com/v1beta/openai/`,
  requests go to `<base>/chat/completions`).
- Offline test with no key: add `--mock` (fake responder: exact span, near-copy, no-answer, junk).
- Split files with "test" in the name are refused unless `--allow-test` is given.

## 3. Design
- Same prompts, parsing and grounding as 05 (see `05-llm-local-qwen/summary.md`, section 3).
- No logprobs on the free tiers, so `na_prob` is absent and the no-answer threshold cannot be tuned. Results are
  untuned and not directly comparable to the dev-tuned rows of the other experiments.

## 4. Results and error analysis
Each run covers only the first n dev questions. Compare with other models only on the same ids (the per-question view
in `eda/notebooks/model_errors.ipynb` does this).
<!-- AUTO-RESULTS:BEGIN (written by analyze.py, edits here are overwritten) -->

### Results (dev, NA threshold tuned on dev)

| variant | runs | n | EM | F1 | HasAns F1 | NoAns acc | predicted no-answer % | NA threshold |
|---|---|---|---|---|---|---|---|---|
| gemini_35flash_id1_0shot | 1 | 22 | 54.5 | 54.5 | 33.3 | 100.0 | 72.7 | untuned |
| nemotron35_super_id1_0shot | 1 | 25 | 72.0 | 80.5 | 71.4 | 100.0 | 44.0 | untuned |

### Outcome types

Share of each gold group; count (mean over seeds) in brackets. Figure: [analysis/outcomes.png](analysis/outcomes.png).

| gold | outcome | gemini_35flash_id1_0shot | nemotron35_super_id1_0shot |
|---|---|---|---|
| answerable | correct_ans | 33.3% (5) | 58.8% (10) |
| answerable | false_no_answer | 60.0% (9) | 17.6% (3) |
| answerable | wrong_sentence | 6.7% (1) | 5.9% (1) |
| answerable | wrong_span_same_sentence | 0.0% (0) | 0.0% (0) |
| answerable | boundary_too_long | 0.0% (0) | 17.6% (3) |
| answerable | boundary_too_short | 0.0% (0) | 0.0% (0) |
| answerable | partial_overlap | 0.0% (0) | 0.0% (0) |
| unanswerable | correct_no_ans | 100.0% (7) | 100.0% (8) |
| unanswerable | false_answer_on_unanswerable | 0.0% (0) | 0.0% (0) |

### Variants against `gemini_35flash_id1_0shot` (paired bootstrap)

| comparison | n | delta F1 | 95% CI | p (paired bootstrap) |
|---|---|---|---|---|
| nemotron35_super_id1_0shot - gemini_35flash_id1_0shot | 22 | +24.85 | [9.09, 41.52] | 0.0010 |

### Seed stability

single run per variant

### Weakest slices for `nemotron35_super_id1_0shot`, the best variant (n >= 30). All slices: [analysis/slices.md](analysis/slices.md)

| slice | value | n | F1 (nemotron35_super_id1_0shot) |
|---|---|---|---|

### Hard questions

4 of 22 questions are wrong (F1 < 0.5) in every run of every variant; 0% of them are unanswerable (dev base rate 32%). List: [analysis/hard_examples.jsonl](analysis/hard_examples.jsonl).

### Analysis files

[error_sample_gemini_35flash_id1_0shot.md](analysis/error_sample_gemini_35flash_id1_0shot.md), [error_sample_nemotron35_super_id1_0shot.md](analysis/error_sample_nemotron35_super_id1_0shot.md), [hard_examples.jsonl](analysis/hard_examples.jsonl), [outcomes.md](analysis/outcomes.md), [outcomes.png](analysis/outcomes.png), [per_example.jsonl](analysis/per_example.jsonl), [results.md](analysis/results.md), [seed_stability.md](analysis/seed_stability.md), [significance.md](analysis/significance.md), [slices.md](analysis/slices.md)

<!-- AUTO-RESULTS:END -->

On the same 22 ids, IndoBERT (mean of 3 seeds, dev-tuned threshold) scores 59.1 EM, 64.5 F1, 52.4 HasAns F1 and
90.5 NoAns accuracy.

## 5. Caveats (read before interpreting results)
- Allowed by the brief (Section 5: external or pretrained models are allowed if documented). Declare these
  runs in the report's AI-use and model sections; confirm on SCeLE if in doubt.
- Contamination risk is higher than for local models (Indonesian Wikipedia passages, questions resembling
  public datasets such as TyDiQA-ID and IDK-MRC). The prefix-completion probe cannot run without logprobs, so
  this is a stated, unmeasured limitation.
- The no-answer threshold cannot be tuned without logprobs. Report untuned results; they are not directly
  comparable to IndoBERT's dev-tuned threshold. Report both where possible.
- Free tiers may log or train on prompts. Dev passages are low-sensitivity Wikipedia text, but do not send the
  official test set through a free tier until the course confirms it is acceptable (the script guards this with
  `--allow-test`).
- Rate limits: run one or two configurations on full dev via API. The prompt-sensitivity grid stays on Kaggle.
- Silent model updates: pin the model id, temperature 0, and log the provider-returned model id and date
  (both in `config.json`). All raw responses are cached for re-scoring.
- Reasoning models may spend the small `max_tokens` budget on hidden thinking and return empty output; lower
  reasoning (`--reasoning-effort none`) or raise `--max-new-tokens` and check `format_failure_rate` and empty
  `raw_output` rows.
- Gemini free tier (checked 2026-11-14): limit is 20 requests per day per model (quota id
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier`), so a full 803-example dev run is not possible in
  under about 40 days on one model. Use a paid key, or another provider. `gemini-2.5-flash` returns 404 for
  new users; the `gemini-3.5-flash` id was served and billed against the `gemini-3.6-flash` quota.
- Gemini 3.x: `reasoning_effort none` works on gemini-3.5-flash (8 completion tokens, non-empty content);
  `low` returned empty content with `max_tokens` 24. `gemini-3.5-flash-lite` rejected `none` with HTTP 400.
  `gemini-3.8-flash` timed out repeatedly at probe time.
- With id1 zero-shot, gemini-3.5-flash abstains on 73% of the first 22 examples (16 of 22), including
  answerable date and name questions. Sample too small to conclude; look at this first on a paid run.

## 6. Run log
| Date | Model | Stage reached | Outcome |
|---|---|---|---|
| 2026-11-14 | google/gemma-4-31b-it:free | 0 of 5 | HTTP 429 "rate-limited upstream" (shared Google AI Studio free pool), not an account limit |
| 2026-11-14 | nvidia/nemotron-3.5-super-vl:free, reasoning none | 25 passed, stopped in stage 100 | OpenRouter free tier daily cap: 50 requests/day without 10 USD of credits (1000/day with). 51 responses cached |
| 2026-11-14 | gemini-3.5-flash, reasoning none | 5 passed, stopped in stage 25 | Free tier daily cap 20 requests per model; 22 responses cached, resets in about 4 h from 11:00 local. Rescored on 22 |

Operational notes:
- Nemotron 3.5 is a reasoning model: with default reasoning it spends the whole `max_tokens` budget on hidden reasoning and returns empty content. Always pass `--reasoning-effort none`.
- Run through `experiments/06-llm-api-offtheshelf/run_llm_api_staged.py` (stages 5, 25, 100, 803 with gates on empty, ungrounded and no-answer rates). A daily-quota 429 now aborts immediately instead of retrying.
- Paid `google/gemma-4-31b-it` (not `:free`) returns logprobs on OpenRouter, which would allow a dev-tuned no-answer threshold.

## 7. Open questions
- A paid run of `google/gemma-4-31b-it` (about USD 0.10 for full dev, and it returns logprobs, so the threshold can be
  tuned): first the over-abstained questions across all 6 prompts, then full dev.
