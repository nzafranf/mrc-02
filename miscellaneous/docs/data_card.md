# Data card: Indonesian extractive MRC (SQuAD v2 style)

Analysis uses `train` and `dev` only. `itest` appears only as counts and in the leakage audit. Code: `experiments/global/mrc/analysis/slices.py`, `eda/scripts/{eda,leakage_audit,artifact_probe,tokenizer_analysis}.py`. Raw numbers: `eda/outputs/tables/`, figures: `eda/outputs/figures/eda_*.png`. Summary: `eda/README.md`.

## Source and sizes

Source guess: Indonesian Wikipedia paragraphs (many `[n]` citation markers, list-style intros, Japanese and Arabic script in contexts), questions written by crowd annotators with uneven typography (`" ?"`, double spaces, lowercase starts). Unanswerable questions look like human-written minimal edits of an answerable question on the same paragraph (see below).

| Split | Examples | Unique contexts | Unanswerable | Multi-gold | Context has [n] |
|---|---|---|---|---|---|
| train | 6435 | 2633 | 43.9% | 0.8% | 36.8% |
| dev | 803 | 291 | 46.2% | 0.7% | 39.0% |
| itest | 804 | 347 | (not analysed) | | |

Split method: `experiments/global/mrc/prepare_data.py`, seed 13, 80/10/10 by group, group = normalized context (lowercase, citations removed, whitespace collapsed) union identical 120-char normalized prefix. All gold offsets verify (`context[start:start+len]==text`, 0 mismatches). 4 train answers have leading or trailing whitespace.

Lengths (train+dev, words): context mean 82.6, median 71, max 2798; question mean 5.6; answer mean 5.2, median 3, max 67. Typical context has 2 questions, long tail up to 35 (`eda_q_per_context_hist.md`).

## Structure finding: paired questions

Of contexts with exactly 2 questions (1631 in train+dev), 1592 (97.6%) have exactly one unanswerable and one answerable question. Over all contexts with 2+ questions: 97.6% mixed, 2.3% all answerable, 0% all unanswerable. The unanswerable question is a minimal edit of its twin. Whether to exploit this (e.g. per-context ranking of NA scores) is a team decision; it is a property of the data, not of reading comprehension.

## Leakage audit (itest: counts only)

Method: exact match on normalized text, plus char 5-gram TF-IDF cosine (datasketch not installed, sklearn fallback) over unique contexts and unique questions. Near-duplicate = cosine > 0.8.

| Field | Pair | Unique in first | Exact | Near > 0.8 | Near > 0.6 |
|---|---|---|---|---|---|
| context | dev vs train | 291 | 0 | 0 | 0 |
| context | itest vs train | 347 | 0 | 1 | 1 |
| context | itest vs dev | 347 | 0 | 1 | 1 |
| question | dev vs train | 803 | 2 | 16 | 92 |
| question | itest vs train | 803 | 9 | 21 | 85 |
| question | itest vs dev | 803 | 2 | 2 | 13 |

- No shared ids and no shared (context, question) pairs across splits.
- Context near-duplicates: 1 itest context vs train is the same Abu Bakar article (cos 0.998, differs by a leading quote mark); 1 itest context vs dev is the same Kimi no Shiranai Monogatari article (0.986). The 120-char prefix rule missed both because of a leading `'` or `"`. Impact is small (2 of 347). Dev vs train is clean.
- Question near-duplicates are mostly generic questions on different pages ("apa itu dna?", "berapakah luas jepang ?"). Exact question matches: dev/train 2, itest/train 9, itest/dev 2.
- Recommendation: keep the split frozen; note the 2 itest contexts.

## Slice definitions (`tag_example`, deterministic, input only)

| Tag | Definition |
|---|---|
| q_word, q_word_raw | First interrogative among first 3 tokens, normalized (kapankah to kapan, siapakah to siapa, dimanakah to dimana, berapakah to berapa, apakah to apa, kenapa to mengapa); raw first word kept |
| q_type | Rule based: time (kapan, tahun berapa), person (siapa), location (dimana, "di mana", mana + place noun), number (berapa, jumlah), entity_or_def (apa, mana), yesno_like (apakah/adakah/benarkah pattern, rare), other |
| ctx_len_bucket | Words in context: <60, 60-120, 120-250, >250 |
| ans_len_bucket | Words in first gold: 1, 2-3, 4-6, >6, none |
| ans_type | date_year, number, person_like_cap (2 to 4 capitalized words), other_cap, lowercase_phrase, none |
| ans_rel_pos_bucket | Quartile Q1..Q4 of answer start / context chars, none if unanswerable |
| ans_sent_idx | Index of the regex-split sentence containing the answer start, -1 if unanswerable |
| lex_overlap, lex_overlap_bucket | Fraction of question content words (stopwords removed) found in the answer sentence (best-overlap sentence if unanswerable); low <0.34, mid, high >=0.67 |
| has_citation | Context contains `[n]` |
| n_gold | Number of non-empty gold answers |

Output: `data/processed/slices_{train,dev,itest}.jsonl`. Caveat: q_word for "di mana" resolves to `mana`, so use `q_word_raw` for the "di" vs "dimana" distinction.

## EDA summary (train+dev)

| Item | Result |
|---|---|
| Answer position | Q1 51.8%, Q2 23.1%, Q3 14.5%, Q4 10.5% (strong early bias) |
| Answer type | lowercase phrase 30.8%, other capitalized 24.4%, date/year 17.9%, person-like 16.7%, number 10.3% |
| Answer length | 1 word 23.5%, 2-3 words 36%, median 3 |
| Context length bucket | 60-120 words 46%, <60 39%, 120-250 14%, >250 1% |
| Unanswerable rate by context length, citation | flat (43% to 46%) |
| Overlap, answerable vs unanswerable | mean 0.50 vs 0.47; unanswerable questions overlap the best sentence as much as answerable ones |
| Unanswerable rate by lex_overlap bucket | high 31.7%, low 44.2%, mid 49.7% |

Unanswerable rate by raw first word (n>=100): dimana 0.045, dimanakah 0.049, berapakah 0.178, siapakah 0.239, kapankah 0.278, apakah 0.373, apa 0.444, kapan 0.485, berapa 0.578, siapa 0.601, di 0.880. By normalized q_word: mana 0.76, bagaimana 0.73, mengapa 0.69, dimana 0.05, others 0.42 to 0.46. By q_type: other 0.57, time 0.48, rest 0.41 to 0.44. Tables: `eda_qword_raw.md`, `eda_qtype.md`.

## Artifact probe: is unanswerability predictable without the context?

Logistic regression (C=2) on TF-IDF. Train on train, eval on dev (one run, LR is deterministic), plus 5-fold GroupKFold on train grouped by normalized context prefix. Majority baseline: 53.8% accuracy on dev.

| Variant | Dev acc | Dev AUC | CV acc (train) | CV AUC (train) |
|---|---|---|---|---|
| Majority | 0.538 | 0.500 | 0.561 | 0.500 |
| Question only (word 1-2 + char_wb 2-5) | 0.806 | 0.901 | 0.794 +- 0.010 | 0.879 +- 0.009 |
| Question + surface format cues | 0.816 | 0.906 | 0.798 +- 0.009 | 0.890 +- 0.006 |
| Format cues only (6 numbers) | 0.746 | 0.797 | 0.714 +- 0.010 | 0.784 +- 0.011 |
| Context only | 0.538 | 0.511 | 0.558 +- 0.002 | 0.517 +- 0.006 |
| Question + overlap features | 0.823 | 0.902 | 0.799 +- 0.009 | 0.883 +- 0.007 |

Reading: a model that never sees the context gets 80% accuracy and 0.90 AUC. Context alone is at chance (no context-level leakage). Lexical overlap adds nothing. Typography alone gives 0.75 accuracy:

| Cue | Rate | Unanswerable if present | if absent |
|---|---|---|---|
| space before "?" (" ?") | 17.7% | 2.6% | 53.1% |
| double space inside question | 1.6% | 6.1% | 44.8% |
| starts lowercase | 16.0% | 11.5% | 50.4% |

Likely cause: answerable questions are original crowd questions (messy typography), unanswerable ones were rewritten more cleanly. Tokens predicting unanswerable: terakhir, inggris, menurut, dilarang, ditutup, dihancurkan, dibubarkan, "yang bukan", "di mana". Predicting answerable: dimana, berapakah, dimanakah, siapakah, pertama, terbesar, didirikan, dibentuk. Consequence: report a question-only baseline (0.81 acc) as the floor for NoAns accuracy, and consider a typography-normalized dev variant (strip spacing and case) to check that models do not rely on style.

### How unanswerable questions were made (40 random train unanswerable, each read next to its answerable twin)

| Subtype | n/40 | Example (unanswerable vs twin) |
|---|---|---|
| Entity swap to a similar entity | 11 | "Berapa luas Kota Paris?" vs "berapakah luas Kota Pariaman?"; Austria vs Jerman |
| Event or attribute swap (verb changed) | 9 | "Kapan Kaisar Tang Gaozu diselamatkan?" vs "...meninggal?"; "Di manakah Hitler meninggal?" vs "...lahir?" |
| Added or unsupported modifier | 6 | "tarian Angin Mamiri Modern", "mata uang ... yang tidak resmi", "habitat gajah betina" |
| Antonym or opposite word | 6 | "ibu ekonomi" vs "bapak ekonomi"; "kantor cabang" vs "kantor pusat"; "terakhir" vs "pertama" |
| Fact absent, topic related | 4 | "Siapa anggota keluarga Habash yang selamat..." (no such fact in text) |
| Name corrupted or misspelled | 3 | "Wu Zutian" vs "Wu Zetian"; "Djal" vs "Dajjal"; "Raden Palaji Marsogo" |
| Negation | 1 | "Siapa yang bukan pendiri kerajaan Kahuripan?" |

All 40 had an answerable twin on the same paragraph sharing most tokens; none was an unrelated question. Subtype labels are one reviewer's judgment (counts +-3). Implication: unanswerable questions are adversarial near-misses, so the NA decision must check that the exact entity or relation of the question is in the context, not just the topic.

## Tokenizer analysis (train+dev unique contexts, n=2924)

Fertility = tokens per whitespace word on the first 600 contexts, no special tokens. For XLM-R and Qwen it is slightly inflated because words are tokenized without a leading space; read as an upper bound.

| Tokenizer | Tokens/word | % words split | Mean ctx tokens | p95 | Max | % ctx > 384 | % ctx > 512 |
|---|---|---|---|---|---|---|---|
| indobenchmark/indobert-base-p1 | 1.46 | 25.1 | 114.5 | 239 | 3660 | 0.8 | 0.3 |
| FacebookAI/xlm-roberta-base | 1.58 | 34.7 | 124.8 | 258 | 4199 | 1.2 | 0.4 |
| Qwen/Qwen2.5-1.5B-Instruct | 2.52 | 81.9 | 181.1 | 371 | 6051 | 4.6 | 1.6 |

Max length 384 with stride truncates about 1% of contexts for the encoders; Qwen needs about 1.6x the tokens. Windowing matters only for a handful of long contexts.

## Label quality (60 random answerable train examples, seed 7, one reviewer)

| Issue | n | Examples |
|---|---|---|
| Wrong answer (entity or sentence mismatch) | 2 | "populasi kota Bekasi": gold is Kabupaten Bekasi; "istilah digital pertama kali": gold 1950 is the Simon computer |
| Boundary too long, mixed or junk | 4 | Abu Bakar birth gold is a junk char plus "; 572"; "berapa lama": "1942 dan berakhir pada tanggal 17 Agustus 1945"; Kolaka area merges land and sea area; "45.764,53" lacks "km2" while other area golds include it |
| Multiple valid spans, one given | 1 | "Dari mana Samaun Bakri berasal": Padang Pariaman vs Nagari Kurai Taji |
| Borderline (not counted) | 3 | Umar position reason span, Tanimbar location, Kamen Rider "karakter utama" |

Definite issues: 7 of 60 = 11.7% (Wilson 95% interval about 6% to 22%); with borderline 10 of 60 = 16.7%. Only 0.8% of answerable examples have 2 golds, so true ambiguity exceeds what the gold shows. Unit and trailing-clause inconsistency caps achievable EM; expect an F1 to EM gap. One reviewer, small n.

## Proposed error taxonomy

Mutually exclusive, first matching rule wins; full rules in `miscellaneous/docs/error_taxonomy.md`.

| # | Category |
|---|---|
| 1 | gold issue |
| 2 | false no-answer |
| 3 | false answer on unanswerable |
| 4 | boundary too long |
| 5 | boundary too short |
| 6 | multi-hop miss |
| 7 | wrong sentence or distractor |
| 8 | right type, wrong entity |
| 9 | wrong type |
