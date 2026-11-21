# Proposed initial error taxonomy (extractive MRC, SQuAD v2 style)

Categories are mutually exclusive. Assign the first matching rule, top to bottom. Applies to dev errors only (F1 < 1 after normalization, or wrong answerability decision).

| # | Category | Rule | Typical cause |
|---|---|---|---|
| 1 | gold issue | Gold is wrong, truncated, or misses an equally valid span (checked by hand before any other label) | Label noise, unit or boundary inconsistency (about 12% of answerable gold in a 60-example audit) |
| 2 | false no-answer | Gold is a span, model predicts "" | NA threshold too high, question shaped like a minimal-edit negative |
| 3 | false answer on unanswerable | Gold is "", model predicts a span | Question differs from the answerable twin by one entity, verb, antonym or modifier; model matches the template |
| 4 | boundary too long | Pred contains gold, pred has extra tokens | Clause or list over-extended, units or qualifiers added |
| 5 | boundary too short | Pred is a strict part of gold | Stops at first item or comma in a long gold span |
| 6 | multi-hop miss | Gold needs two or more sentences, pred covers one | Answer requires combining facts or resolving a pronoun |
| 7 | wrong sentence or distractor | Pred lies in a sentence other than the gold sentence, same type as gold | Another date, number or entity near question words |
| 8 | right type, wrong entity | Pred is in the gold sentence with the right answer type (date, number, person, place) but a different entity | Two candidates in one sentence (birth vs death date) |
| 9 | wrong type | Pred type differs from what the question asks (e.g. a name for "kapan") | Question word ignored, model follows lexical match only |

Notes
- Tag every error with the slice fields from `experiments/global/mrc/analysis/slices.py` (q_word, q_type, ans_type, ctx_len_bucket, ans_sent_idx, lex_overlap_bucket) so category counts can be cut by slice.
- Categories 2 and 3 are the answerability decision; report them with the NA threshold fixed on dev.
- Rules 7, 8, 9 apply only when both gold and pred are non-empty and overlap is partial or zero; use sentence index to separate 7 from 8.
