# Classical pipeline: stage-wise analysis

## Final dev results (mean ± std over seeds 13, 42, 87; threshold tuned on dev, so slightly optimistic)

| model | EM | F1 | HasAns F1 | NoAns acc |
|---|---|---|---|---|
| baseline no_answer | 46.20 | 46.20 | 0.00 | 100.00 |
| baseline sliding_window | 9.22 | 12.45 | 22.91 | 0.27 |
| baseline sliding_window_thr | 46.08 | 46.81 | 4.37 | 96.23 |
| **classical pipeline** | 50.98 ± 0.29 | 53.36 ± 0.26 | 29.13 ± 1.40 | 81.58 ± 1.65 |

Per seed F1: seed 13 53.60, seed 42 53.09, seed 87 53.40. The sliding-window baseline row is the always-answer version (no threshold); the thresholded version degenerates to almost always no-answer because question/sentence overlap does not separate answerable from unanswerable questions.

## Stage-wise metrics (dev, mean ± std)

| stage | metric | value |
|---|---|---|
| A answerability | AUC | 0.75 ± 0.00 |
| A | accuracy at tuned threshold | 67.00 ± 0.43 |
| B sentence retrieval | recall@1 (LR ranker) | 62.04 ± 0.00 |
| B | recall@1 TF-IDF only / BM25 only | 51.85 ± 0.00 / 53.70 ± 0.00 |
| B | recall@2 (used by stage C) | 88.66 ± 0.00 |
| B | recall@3 | 96.06 ± 0.00 |
| C span ranker | candidate recall (gold in candidate set, top-2 sentences) | 73.38 ± 0.00 |
| C | avg candidates per question | 152 |
| C | span EM given gold sentence in top-2 | 47.78 ± 1.04 |
| C | span F1 given gold sentence in top-2 | 57.09 ± 0.47 |
| C | span EM given gold sentence ranked 1st | 44.28 ± 1.51 |
| C | span EM on all answerable (no stage A) | 43.29 ± 0.93 |

All answerable-only numbers are over the 432 answerable dev questions (stage A numbers over all 803).

## Error attribution (dev, EM = 0, at the tuned threshold)

Each error is assigned to the earliest failing stage: A (answerability decision wrong), B (gold sentence not in top-2), C-gen (gold span not among candidates), C-rank (gold among candidates, wrong one ranked first).

| cause | count (mean over seeds) | share of errors |
|---|---|---|
| A_false_answer | 68.3 | 17.4% |
| A_false_no_answer | 196.7 | 50.0% |
| B | 23.0 | 5.8% |
| C_gen | 47.0 | 11.9% |
| C_rank | 58.7 | 14.9% |
| total errors | 393.7 of 803 | |

A_false_answer = question unanswerable but a span was returned; A_false_no_answer = answerable but "" returned.

## Sample errors (seed 42)

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

## Ablation and importance

See [ablation.md](ablation.md) (drop one feature group per stage, delta F1) and [importance.md](importance.md) (LightGBM gain, LR coefficients).
