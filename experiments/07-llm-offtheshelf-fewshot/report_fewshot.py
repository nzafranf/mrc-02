"""Error report for one run (also partial ones), counts from general to specific, in the project's error taxonomy.

  python experiments/07-llm-offtheshelf-fewshot/report_fewshot.py <run dir> [--tuned] [--examples 3]
-> prints the report and writes <run dir>/report_<split>.md

Decision: by default the model's own output decides answer vs abstain ("as generated"). --tuned applies the
no-answer threshold tuned on the same questions (optimistic, and meaningless on a few dozen questions).
Outcome rules are mrc.analysis.compare.classify, the same as every analyze.py (miscellaneous/docs/error_taxonomy.md):

  all questions
  ├─ correct (exact match)
  │  ├─ correct answer            answerable, span matches gold
  │  └─ correct no-answer         unanswerable, model abstains
  └─ wrong
     ├─ answerability error
     │  ├─ false no-answer        answerable, model abstains
     │  └─ false answer           unanswerable, model gives a span
     └─ span error                answerable, model gives a span that is not an exact match
        ├─ overlaps gold          partial credit
        │  ├─ boundary too long   pred contains gold
        │  ├─ boundary too short  gold contains pred
        │  └─ partial overlap     share tokens, neither contains the other
        └─ no overlap             zero F1
           ├─ wrong span, same sentence
           ├─ wrong sentence
           └─ not in the passage  ungrounded output (hallucinated or reformatted), LLM-specific; the shared
                                  classify() would pin it to an arbitrary sentence
Then: grounding of the spans the model gave (exact copy, snapped to the passage, ungrounded = not in the passage)
by outcome, errors by question type, and a few examples per leaf.
"""
import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import argparse
import json
from collections import Counter
from pathlib import Path

from mrc.analysis.compare import classify
from mrc.analysis.slices import sentences
from mrc.eval_order import SPLIT_FILES
from mrc.evaluate import aggregate, best_na_threshold, load_jsonl, per_example_scores
from mrc.paths import PROC

TREE = [  # (depth, label, leaf outcomes counted in this node)
    (0, "all questions", None),
    (1, "correct (exact match)", ["correct_ans", "correct_no_ans"]),
    (2, "correct answer", ["correct_ans"]),
    (2, "correct no-answer", ["correct_no_ans"]),
    (1, "wrong", "WRONG"),
    (2, "answerability error", ["false_no_answer", "false_answer_on_unanswerable"]),
    (3, "false no-answer (answerable, abstained)", ["false_no_answer"]),
    (3, "false answer (unanswerable, gave a span)", ["false_answer_on_unanswerable"]),
    (2, "span error (answerable, wrong span)", ["boundary_too_long", "boundary_too_short", "partial_overlap",
                                              "wrong_span_same_sentence", "wrong_sentence", "not_in_passage"]),
    (3, "overlaps gold (partial credit)", ["boundary_too_long", "boundary_too_short", "partial_overlap"]),
    (4, "boundary too long", ["boundary_too_long"]),
    (4, "boundary too short", ["boundary_too_short"]),
    (4, "partial overlap", ["partial_overlap"]),
    (3, "no overlap (F1 = 0)", ["wrong_span_same_sentence", "wrong_sentence", "not_in_passage"]),
    (4, "wrong span, same sentence", ["wrong_span_same_sentence"]),
    (4, "wrong sentence", ["wrong_sentence"]),
    (4, "not in the passage", ["not_in_passage"]),
]
WRONG = ["false_no_answer", "false_answer_on_unanswerable", "boundary_too_long", "boundary_too_short",
         "partial_overlap", "wrong_span_same_sentence", "wrong_sentence", "not_in_passage"]
LEAVES = ["correct_ans", "correct_no_ans"] + WRONG


def load_preds(run: Path, split: str) -> list[dict]:
    for name in (f"preds_{split}.partial.jsonl", f"preds_{split}.jsonl"):
        if (run / name).exists():
            return [json.loads(l) for l in (run / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    raise SystemExit(f"no predictions in {run}")


def report(run: Path, tuned: bool = False, n_examples: int = 3) -> str:
    conf = json.loads((run / "config.json").read_text(encoding="utf-8"))
    split = conf["fingerprint"]["split"]
    gold_by = {e["id"]: e for e in load_jsonl(SPLIT_FILES[split])}
    slices = {s["id"]: s for s in load_jsonl(PROC / f"slices_{split}.jsonl")} \
        if (PROC / f"slices_{split}.jsonl").exists() else {}
    preds = {p["id"]: p for p in load_preds(run, split)}
    gold = [gold_by[i] for i in preds]
    thr = best_na_threshold(gold, preds)[0] if tuned and all("na_prob" in p for p in preds.values()) else None
    sc = per_example_scores(gold, preds, thr)
    out = {}
    for ex in gold:
        v = sc[ex["id"]] | {"start": preds[ex["id"]].get("start_char", -1)}
        c = classify(ex, v, sentences(ex["context"]))
        if c in ("wrong_span_same_sentence", "wrong_sentence") and preds[ex["id"]].get("grounding") == "ungrounded":
            c = "not_in_passage"
        out[ex["id"]] = ("correct_no_ans" if ex["is_impossible"] else "correct_ans") if c == "correct" else c
    n, cnt, agg = len(gold), Counter(out.values()), aggregate(sc)
    n_ans = sum(not e["is_impossible"] for e in gold)
    pct = lambda k, d: f"{100 * k / d:.1f}%" if d else "n/a"

    L = [f"# Error report: {run.name} on {split}", "",
         f"Model `{conf['fingerprint']['model']}`, prompt `{conf['fingerprint']['variant']}`, "
         f"{len(conf.get('demo_ids', []))} demos, decision {'tuned NA threshold ' + format(thr, '.3g') if thr is not None else 'as generated'}. "
         f"{n} questions = the first {n} of the fixed order ({n_ans} answerable, {n - n_ans} unanswerable).", "",
         f"EM {agg['exact']:.1f} · F1 {agg['f1']:.1f} · HasAns F1 {agg['HasAns_f1']:.1f} · NoAns acc "
         f"{agg['NoAns_acc']:.1f} · predicted no-answer {agg['pred_noans_rate']:.1f}%", ""]
    if n < 100:
        L += [f"With n = {n}, one question moves a rate by {100 / n:.1f} points; treat the shares as rough.", ""]
    L += ["## Outcomes, general to specific", "", "```", f"{'outcome':<52}{'n':>5}{'% all':>9}{'% parent':>10}"]
    parents = {}
    for depth, label, keys in TREE:
        k = n if keys is None else sum(cnt[x] for x in (WRONG if keys == "WRONG" else keys))
        parents[depth] = k
        par = parents.get(depth - 1, n)
        name = ("   " * (depth - 1) + "└─ " if depth else "") + label
        L.append(f"{name:<52}{k:>5}{pct(k, n):>9}{(pct(k, par) if depth and par else ''):>10}")
    L.append("```")

    answered = [i for i in out if sc[i]["pred"]]
    L += ["", "## Grounding of the spans the model gave", "",
          "exact = copied from the passage; snapped = mapped to the closest passage span; ungrounded = not in the "
          "passage (hallucinated or reformatted).", "", "| outcome | exact | snapped | ungrounded |", "|---|---|---|---|"]
    for o in LEAVES:
        ids = [i for i in answered if out[i] == o]
        if ids:
            g = Counter(preds[i]["grounding"] for i in ids)
            L.append(f"| {o} | {g['exact']} | {g['snapped']} | {g['ungrounded']} |")

    if slices:
        qt = sorted({slices.get(i, {}).get("q_type", "?") for i in out})
        L += ["", "## Errors by question type", "", "| q_type | n | wrong | " + " | ".join(WRONG) + " |",
              "|---|---|---|" + "---|" * len(WRONG)]
        for t in qt:
            ids = [i for i in out if slices.get(i, {}).get("q_type", "?") == t]
            c = Counter(out[i] for i in ids)
            L.append(f"| {t} | {len(ids)} | {sum(c[w] for w in WRONG)} | " + " | ".join(str(c[w]) for w in WRONG) + " |")

    L += ["", f"## Examples (up to {n_examples} per error type, in the fixed order)", ""]
    for o in WRONG:
        ids = [i for i in out if out[i] == o][:n_examples]
        if not ids:
            continue
        L += [f"### {o} ({cnt[o]})", ""]
        for i in ids:
            ex, p = gold_by[i], preds[i]
            golds = " / ".join(sorted({a["text"] for a in ex["answers"]})) or "(unanswerable)"
            L += [f"- `{i}` **Q:** {ex['question']}  ", f"  **gold:** {golds} · **pred:** {sc[i]['pred'] or '(no answer)'}"
                  f" · raw `{p['raw_output'].strip()[:120]}` · {p['grounding']} · F1 {sc[i]['f1']:.2f}"]
        L.append("")
    text = "\n".join(L)
    (run / f"report_{split}.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--tuned", action="store_true")
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()
    for r in a.run_dirs:
        print(report(r.resolve(), a.tuned, a.examples), "\n")
