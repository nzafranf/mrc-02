# 02 · classical · pipeline · lgbm

**Scope** classical · **Subscope** pipeline · **Name** lgbm · **Status** done (3 seeds, CPU, about 2 min per seed)

**Differentiating point.** Traditional ML with hand-built features and no neural network, split into three stages that
are trained separately: answerability, sentence retrieval and span ranking. It covers the brief's "pipeline" and
"traditional ML with feature engineering" requirements.

**Verdict.** 53.4 F1, a significant +7.2 over always abstaining, but it misses half the answerable questions
(false no-answer is the main error type). Stage attribution puts half of all errors on the answerability stage.

## 1. Where it starts
Input: `data/processed/{train,dev}.jsonl`. The tokenizer, light stemmer and sentence splitter are shared
(`experiments/global/mrc/qa_text.py`, also used by 01). Everything from feature extraction onward is here.

## 2. Pipeline, end to end
| Step | Command | Output |
|---|---|---|
| Train + predict dev (per seed) | `python experiments/02-classical-pipeline-lgbm/classical_pipeline.py --seed 13 --split-file data/processed/dev.jsonl` (also 42, 87) | `runs/classical_pipeline/seed*/` |
| Feature-group ablation | `python experiments/02-classical-pipeline-lgbm/classical_pipeline.py --ablation --seed 42` | `analysis/ablation.md`, `analysis/importance.md` |
| Stage-wise analysis | `python experiments/02-classical-pipeline-lgbm/classical_report.py` | `analysis/stages.md` |
| Error analysis | `python experiments/02-classical-pipeline-lgbm/analyze.py` | `analysis/`, section 4 below |
| Predict another split | `... classical_pipeline.py --predict-only --split-file X.jsonl --out Y.jsonl` (fit on train, threshold tuned on dev) | `Y.jsonl` |

`--predict-only` writes `pred` (after the threshold), `na_prob`, `score`, `start_char` and `span_pred` (the span before
the threshold). Everything is tuned on dev only.

## 3. Design
```
 question, context
        |
        |  tokenizer + light stemmer + sentence splitter (experiments/global/mrc/qa_text.py)
        v
 +-----------------------------+      top-2 sentences      +----------------------------------+
 | Stage B  sentence retriever |-------------------------->| Stage C  span ranker             |
 | TF-IDF, BM25, overlap, pos  |                           | n-gram candidates (<= 8 words)   |
 | -> standardized LogReg      |                           | 56 features -> LightGBM lambdarank|
 +-----------------------------+                           +----------------+-----------------+
                                                                            | best span + score
 +-----------------------------+                                            v
 | Stage A  answerability      |  na_prob      +--------------------------------------------+
 | 27 features -> LightGBM     |-------------->| final: "" if na_prob > thr (tuned on dev)  |
 | (no dependency on B and C)  |               |        else best span                      |
 +-----------------------------+               +--------------------------------------------+
```

Code: `experiments/02-classical-pipeline-lgbm/classical_pipeline.py` (classes `SentenceRetriever`, `AnswerabilityClassifier`, `SpanRanker`,
`ClassicalPipeline`), shared text utilities in `experiments/global/mrc/qa_text.py`. BM25 and IDF are implemented directly with
sentence-level IDF fitted on training contexts (rank_bm25 would build its IDF from only the 3 to 5 sentences of one
context, which is too noisy). The trivial baselines are in `experiments/01-baseline-trivial/baselines.py`.

### Features
Stage B (16 per sentence): TF-IDF cosine, BM25, IDF-weighted overlap and its ratio, overlap count, stem-bigram overlap,
sentence position, length, number of expected-type tokens, and the three scores relative to the best sentence in the context.

Stage A (27, 6 groups): `question` (question type from the wh-word, length, "tahun", content words, capitalized words),
`retrieval` (max / second / margin TF-IDF, max BM25, whole-context TF-IDF), `overlap` (best-sentence idf-weighted overlap
ratio, fraction of question content stems missing from the context, bigram overlap), `entity_type` (share of question
proper nouns / numbers found in the context, expected-type tokens in best sentence and context), `window` (best 5 and
10 token idf-weighted overlap windows), `context` (length, sentence count).

Stage C (56, 7 groups): `type_match` (question type vs candidate year / number / month / capitalization / unit / in
parentheses, plus a hand-written compatibility score), `overlap` (candidate words that occur in the question),
`proximity` (token distance to the nearest question-word match left and right, number of matches within 3 and 6 tokens,
idf mass within 5), `context_words` (identity of the 1 to 2 words left and right from a 36-symbol list, whether they occur in
the question), `shape_boundary` (length, position, capitalized or numeric neighbours, maximal capitalized run),
`pos_heuristics` (stopword share, verb-like prefix me-/ber-/ter-/di- share, neighbours verb-like, internal punctuation),
`sentence` (rank, stage B score, relative score, TF-IDF, BM25, overlap ratio of the sentence).
Candidates: every word n-gram of at most 8 words in the top-2 sentences that does not start or end with a lowercase
stopword or punctuation and has balanced parentheses (about 152 per question). Label: normalized text equals a gold answer.
Stage C is trained with LightGBM lambdarank, one group per question (`--objective binary` is also available; it gave about 1 F1 less on seed 13).

## 4. Results and error analysis
<!-- AUTO-RESULTS:BEGIN (written by analyze.py, edits here are overwritten) -->

### Results (dev, NA threshold tuned on dev)

| variant | runs | n | EM | F1 | HasAns F1 | NoAns acc | predicted no-answer % | NA threshold |
|---|---|---|---|---|---|---|---|---|
| classical_pipeline | 3 | 803 | 51.0 ± 0.2 | 53.4 ± 0.2 | 29.1 ± 1.1 | 81.6 ± 1.3 | 62.2 ± 1.6 | 0.397 ± 0.0162 |

### Outcome types

Share of each gold group; count (mean over seeds) in brackets. Figure: [analysis/outcomes.png](analysis/outcomes.png).

| gold | outcome | classical_pipeline |
|---|---|---|
| answerable | correct_ans | 24.7% (107) |
| answerable | false_no_answer | 45.5% (197) |
| answerable | wrong_sentence | 11.5% (50) |
| answerable | wrong_span_same_sentence | 7.3% (32) |
| answerable | boundary_too_long | 1.9% (8) |
| answerable | boundary_too_short | 6.7% (29) |
| answerable | partial_overlap | 2.4% (10) |
| unanswerable | correct_no_ans | 81.6% (303) |
| unanswerable | false_answer_on_unanswerable | 18.4% (68) |

### Seed stability

| variant | seeds | same outcome type (mean over seed pairs) | questions that flip right/wrong across seeds |
|---|---|---|---|
| classical_pipeline | 3 | 92.7% | 8.2% |

### Weakest slices for `classical_pipeline`, the best variant (n >= 30). All slices: [analysis/slices.md](analysis/slices.md)

| slice | value | n | F1 (classical_pipeline) |
|---|---|---|---|
| ans_len_bucket | >6 | 91 | 13.0 |
| ans_type | lowercase_phrase | 119 | 17.2 |
| ans_rel_pos_bucket | Q1 | 221 | 21.0 |
| ans_type | other_cap | 121 | 24.4 |
| ans_type | date_year | 88 | 29.7 |
| ans_len_bucket | 1 | 133 | 30.1 |

### Hard questions

344 of 803 questions are wrong (F1 < 0.5) in every run of every variant; 16% of them are unanswerable (dev base rate 46%). List: [analysis/hard_examples.jsonl](analysis/hard_examples.jsonl).

### Analysis files

[ablation.md](analysis/ablation.md), [error_sample_classical_pipeline.md](analysis/error_sample_classical_pipeline.md), [hard_examples.jsonl](analysis/hard_examples.jsonl), [importance.md](analysis/importance.md), [outcomes.md](analysis/outcomes.md), [outcomes.png](analysis/outcomes.png), [per_example.jsonl](analysis/per_example.jsonl), [results.md](analysis/results.md), [seed_stability.md](analysis/seed_stability.md), [slices.md](analysis/slices.md), [stages.md](analysis/stages.md)

<!-- AUTO-RESULTS:END -->

### Pipeline-specific analysis
- Stage metrics and per-stage error attribution: [analysis/stages.md](analysis/stages.md). Each error is assigned to
  the earliest stage that failed: answerability 67% (false no-answer 50%, false answer 17%), retrieval 6%,
  candidate generation 12%, ranking 15%.
- Feature-group ablation: [analysis/ablation.md](analysis/ablation.md). Feature importance:
  [analysis/importance.md](analysis/importance.md).

### Sample errors (seed 42)

- [A_false_answer] Q: Apakah pendidikan pertama Gladys Emerson'? | gold: (none) | pred: gelar Ph | na_prob 0.16
- [A_false_answer] Q: Di manakah penemu aljabar? | gold: (none) | pred: Arab | na_prob 0.20
- [A_false_answer] Q: Kapan anak Parlindoengan Loebis lahir? | gold: (none) | pred: 31 Desember 1994 | na_prob 0.30
- [A_false_no_answer] Q: kapankah papua nugini melepaskan diri dari indonesia? | gold: 1 Juli 1971 | pred: (none) | na_prob 0.62
- [A_false_no_answer] Q: Siapa presiden pertama Nauru? | gold: Hammer DeRoburt | pred: (none) | na_prob 0.76
- [A_false_no_answer] Q: Apakah ibukota Papua Nugini? | gold: Port Moresby | pred: (none) | na_prob 0.59
- [B] Q: Kapan Yan Xishan diangkat menjadi panglima perang Tiongkok ? | gold: 1911 | pred: pemerintah Republik Tiongkok | na_prob 0.38
- [B] Q: apakah ibukota provinsi  Kalimantan Selatan? | gold: Banjarmasin | pred: pulau Kalimantan | na_prob 0.31
- [B] Q: Apa yang tidak ada di kereta ekonomi sebelum tahun 2010? | gold: AC | pred: PT KA | na_prob 0.37
- [C_gen] Q: Siapakah yang disebut dengan Paus ? | gold: Uskup Roma dan pemimpin Gereja Katolik di seluruh dunia | pred: Paus Fransiskus | na_prob 0.07
- [C_gen] Q: Apa yang dimaksud dengan neokonservatif ? | gold: aliran politik yang lahir di Amerika Serikat pada tahun 1960-an di kalangan Demokrat yang tidak sepakat dengan kebijakan dalam dan luar negeri partainya | pred: Neokonservatisme | na_prob 0.21
- [C_gen] Q: Apa yang dimaksud dengan ELO? | gold: metode untuk menghitung tingkat keterampilan (skill) relatif pemain pada permainan antara dua pemain seperti catur dan igo | pred: Árpád Élő | na_prob 0.19
- [C_rank] Q: Apakah ibu kota Tanzania? | gold: Dar-es-Salam | pred: United Republic | na_prob 0.36
- [C_rank] Q: Siapakah Menteri Transmigrasi dan Pemukiman Proyek lahan gambut satu juta hektar? | gold: Siswono Yudo Husodo | pred: Kalimantan Tengah | na_prob 0.29
- [C_rank] Q: Dimana letak Düsseldorf? | gold: Jerman | pred: Sungai Rhein | na_prob 0.06

## 5. Findings
- The answerability stage (AUC 0.75) is the bottleneck, not span extraction. With the gold sentence among the top 2,
  span EM is 48.
- Candidate generation caps recall at 73%: long definition-style answers ("Apa yang dimaksud dengan ...") exceed the
  8-word limit.
- Errors are very stable across seeds (93% same outcome type), so per-seed error samples are representative.

## 6. Open questions
- Use the question-form artifacts found in the EDA (the question-only classifier reaches AUC 0.90) as answerability
  features, and check how much of the gain is the shortcut.
- Raise the candidate length limit for definition questions.
