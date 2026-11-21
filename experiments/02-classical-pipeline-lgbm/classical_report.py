"""Pipeline-specific analysis: stage-wise metrics and per-stage error attribution -> analysis/stages.md.
Run after training; analyze.py covers the analysis shared by all experiments.

  python experiments/02-classical-pipeline-lgbm/classical_report.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import json
from pathlib import Path

import numpy as np

EXP = ROOT / "experiments/02-classical-pipeline-lgbm"
SEEDS = (13, 42, 87)

def ms(vals, pct=True):
    v = np.array(vals) * (100 if pct else 1)
    return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}"


def main():
    runs = {s: json.loads((EXP / f"runs/classical_pipeline/seed{s}/stage_metrics.json").read_text(encoding="utf-8")) for s in SEEDS}
    met = {s: json.loads((EXP / f"runs/classical_pipeline/seed{s}/metrics_dev.json").read_text(encoding="utf-8")) for s in SEEDS}
    fin = [runs[s]["final_at_dev_thr"] for s in SEEDS]
    st = [runs[s]["stages"] for s in SEEDS]
    base = {}
    for n in ("no_answer", "sliding_window", "sliding_window_thr"):
        base[n] = json.loads((ROOT / f"experiments/01-baseline-trivial/runs/{n}/seed13/metrics_dev.json").read_text(encoding="utf-8"))
    t = ["# Classical pipeline: stage-wise analysis\n"]
    t.append("## Final dev results (mean ± std over seeds 13, 42, 87; threshold tuned on dev, so slightly optimistic)\n")
    t.append("| model | EM | F1 | HasAns F1 | NoAns acc |\n|---|---|---|---|---|")
    for n, m in base.items():
        a = m["at_best_threshold"] if n != "sliding_window" else m
        t.append(f"| baseline {n} | {a['exact']:.2f} | {a['f1']:.2f} | {a['HasAns_f1']:.2f} | {a['NoAns_acc']:.2f} |")
    t.append("| **classical pipeline** | " + " | ".join(ms([f[k] / 100 for f in fin]) for k in ("exact", "f1", "HasAns_f1", "NoAns_acc")) + " |")
    t.append("\nPer seed F1: " + ", ".join(f"seed {s} {runs[s]['final_at_dev_thr']['f1']:.2f}" for s in SEEDS)
             + ". The sliding-window baseline row is the always-answer version (no threshold); the thresholded version "
             "degenerates to almost always no-answer because question/sentence overlap does not separate answerable from unanswerable questions.\n")
    t.append("## Stage-wise metrics (dev, mean ± std)\n")
    t.append("| stage | metric | value |\n|---|---|---|")
    t.append(f"| A answerability | AUC | {ms([x['A']['auc'] for x in st], False)} |")
    t.append(f"| A | accuracy at tuned threshold | {ms([x['A']['acc_at_thr'] for x in st])} |")
    t.append(f"| B sentence retrieval | recall@1 (LR ranker) | {ms([x['B']['recall@1'] for x in st])} |")
    t.append(f"| B | recall@1 TF-IDF only / BM25 only | {ms([x['B']['recall@1_tfidf_only'] for x in st])} / {ms([x['B']['recall@1_bm25_only'] for x in st])} |")
    t.append(f"| B | recall@2 (used by stage C) | {ms([x['B']['recall@2'] for x in st])} |")
    t.append(f"| B | recall@3 | {ms([x['B']['recall@3'] for x in st])} |")
    t.append(f"| C span ranker | candidate recall (gold in candidate set, top-2 sentences) | {ms([x['C']['candidate_recall'] for x in st])} |")
    t.append(f"| C | avg candidates per question | {np.mean([x['C']['avg_candidates_per_question'] for x in st]):.0f} |")
    t.append(f"| C | span EM given gold sentence in top-2 | {ms([x['C']['span_em_given_gold_sentence_in_topk'] for x in st])} |")
    t.append(f"| C | span F1 given gold sentence in top-2 | {ms([x['C']['span_f1_given_gold_sentence_in_topk'] for x in st])} |")
    t.append(f"| C | span EM given gold sentence ranked 1st | {ms([x['C']['span_em_given_gold_sentence_rank1'] for x in st])} |")
    t.append(f"| C | span EM on all answerable (no stage A) | {ms([x['C']['span_em_on_answerable_unconditional'] for x in st])} |")
    t.append("\nAll answerable-only numbers are over the 432 answerable dev questions (stage A numbers over all 803).\n")
    t.append("## Error attribution (dev, EM = 0, at the tuned threshold)\n")
    t.append("Each error is assigned to the earliest failing stage: A (answerability decision wrong), B (gold sentence not in top-2), "
             "C-gen (gold span not among candidates), C-rank (gold among candidates, wrong one ranked first).\n")
    keys = ["A_false_answer", "A_false_no_answer", "B", "C_gen", "C_rank"]
    t.append("| cause | count (mean over seeds) | share of errors |\n|---|---|---|")
    tot = np.mean([runs[s]["errors"]["total_errors"] for s in SEEDS])
    for k in keys:
        c = np.mean([runs[s]["errors"][k] for s in SEEDS])
        t.append(f"| {k} | {c:.1f} | {100*c/tot:.1f}% |")
    t.append(f"| total errors | {tot:.1f} of 803 | |\n")
    t.append("A_false_answer = question unanswerable but a span was returned; A_false_no_answer = answerable but \"\" returned.\n")
    t.append("## Sample errors (seed 42)\n")
    errs = json.loads((EXP / "runs/classical_pipeline/seed42/errors_dev.json").read_text(encoding="utf-8"))
    for k in keys:
        ex = [e for e in errs if e["stage"] == k and e["f1"] == 0][:3]
        for e in ex:
            t.append(f"- [{k}] Q: {e['question']} | gold: {e['gold'] or '(none)'} | pred: {e['pred'] or '(none)'} | na_prob {e['na_prob']:.2f}")
    t.append("\n## Ablation and importance\n\nSee [ablation.md](ablation.md) (drop one feature group per stage, delta F1) and [importance.md](importance.md) (LightGBM gain, LR coefficients).\n")
    (EXP / "analysis/stages.md").write_text("\n".join(t), encoding="utf-8")
    print("\n".join(t[-30:]))


if __name__ == "__main__":
    main()
