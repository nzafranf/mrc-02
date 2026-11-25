# 08 · plm · encoder · screen: pre-registration

Written 2026-11-24, before any run of this experiment. Revised the same day, still before any run: full train,
selection on official_val, 10 epochs, seed 13 only, m-BERT added (see "Setup").

## Question
Do encoders that led SQuAD 2.0 in English (ALBERT, ELECTRA-style replaced-token-detection pretraining) or that are
larger or newer (XLM-R-large, NusaBERT-large) beat IndoBERT-base-p1 (04, dev F1 73.5 ± 0.4) on this data, with the
same harness, the same data and a comparable tuning budget?

## Setup
- Train on the full raw train file `data/raw/train.jsonl` (8042 questions). The internal dev and itest are no longer
  held out, so the internal sealed test is given up; the leaderboard test (gold after 23 Dec) is the unbiased check.
- Best epoch and no-answer threshold chosen on `official_val` (2003). The reported official_val score is therefore
  optimistic, equally for every model; comparisons between models stay fair, absolute numbers do not.
- 10 epochs (as the IDK-MRC paper), seed 13 only. Seeds 42 and 87 are deferred; until then no difference smaller
  than about 1 F1 (the seed spread of 04) is called real, and significance uses paired bootstrap over questions only.

## Candidates
| run name | checkpoint | family | why |
|---|---|---|---|
| `albert_indolite_large` | `indobenchmark/indobert-lite-large-p2` | ALBERT (factorized embeddings, shared layers, SOP) | Indonesian ALBERT; ALBERT is the reader of Retro-Reader and other SQuAD 2.0 leaders |
| `electra_indo_base` | `ChristopherA08/IndoELECTRA` | ELECTRA base | the only Indonesian ELECTRA checkpoint found; community upload |
| `mdeberta_v3_base` | `microsoft/mdeberta-v3-base` | DeBERTa-v3 (ELECTRA-style RTD pretraining + disentangled attention) | strongest base-size multilingual encoder; 84.0 F1 on English SQuAD 2.0 dev in public fine-tunes |
| `xlmr_large` | `FacebookAI/xlm-roberta-large` | RoBERTa, multilingual, 560M | scale baseline |
| `nusabert_large` | `LazarusNLP/NusaBERT-large` | BERT, IndoBERT continued pretraining with extended vocab, 337M | newest Indonesian encoder |
| `indobert_base` | `indobenchmark/indobert-base-p1` | BERT | reference: 04's model in this setup |
| `mbert_base` | `google-bert/bert-base-multilingual-cased` | BERT, multilingual | best model of the IDK-MRC paper (82.23 F1 vs IndoBERT 77.45, Table 5) |

## Budget (fixed now, identical in kind to what 04 got)
One configuration per model, no search: 10 epochs, effective batch 16, warmup 10%, linear decay, weight decay 0.01,
max_len 384, stride 128, fp16. Learning rate by size class, taken from common practice, not from our dev set:
base 3e-5 (as 04), large 1.5e-5. Seed 13. Best epoch and NA threshold on official_val.

## Hypotheses
- H1. mDeBERTa-v3-base beats IndoBERT-base (`indobert_base`) on official_val F1 (paired bootstrap, p < 0.05). Falsified if the difference is
  not significant or negative.
- H2. At least one large model (XLM-R-large or NusaBERT-large) beats `indobert_base` by at least 2 F1 on
  official_val, significantly. Falsified otherwise.
- H3. The ALBERT checkpoint does not beat `indobert_base`. Reason: its sentencepiece offsets glue punctuation to
  words (56% of training gold spans are not reproduced exactly by token offsets, against 0.9% for IndoBERT; 121
  of these differ even after normalization), and IndoNLU reports lite models at or below IndoBERT-base.
- H5. m-BERT beats `indobert_base`, as in the IDK-MRC paper. Falsified if not significant.
- H4. The main gain of any better model is on answerable questions (fewer false no-answers, the largest error type
  of 04, 22% of answerable questions), not on unanswerable ones.

## Known risks
- ELECTRA discriminators usually want higher learning rates (around 1e-4 with layer-wise decay). 3e-5 may
  undertrain `electra_indo_base`; if so it is reported as such, not retuned.
- DeBERTa-v3 can overflow in fp16; a NaN loss is logged and the run reported as failed.
- The data is IDK-MRC (Putri and Oh, 2022) re-split (about 94% of questions match exactly; see
  `miscellaneous/docs/decisions.md`). No checkpoint already fine-tuned on IDK-MRC or TyDiQA-ID is used here.

## Outcome (added 2026-11-25, after all runs; nothing above was changed)
Seed 13, official_val, threshold and epoch chosen on official_val. Details: `summary.md`, `analysis/results.md`.
- H1 holds: mDeBERTa-v3-base 80.3 vs IndoBERT-base 71.7, +8.6 F1 [6.9, 10.3].
- H2 holds: XLM-R-large 82.4 (+10.7 [9.0, 12.4]); NusaBERT-large 76.4 (+4.7 [3.1, 6.3]).
- H3 holds trivially: the ALBERT run did not learn (loss about 2.9 throughout, no-answer on 1998 of 2003); reported
  as failed, not retuned, as fixed above.
- H4 holds against IndoBERT for every model (answerable +7.5 to +14.0); over m-BERT, mDeBERTa's gain is mostly on
  unanswerable questions (+5.6).
- H5 holds: m-BERT-base 77.3, +5.6 F1 [3.8, 7.3].
- `electra_indo_base` was not run: the checkpoint is broken (tokenizer ids up to 35000, embeddings 30522).
