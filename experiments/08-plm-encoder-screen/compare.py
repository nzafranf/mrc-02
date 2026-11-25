"""Results table and paired bootstrap for 08, from runs/<run>/seed13/{preds,metrics}_official_val.* (no inference).

  python experiments/08-plm-encoder-screen/compare.py [--n-boot 10000]
Each run is scored at its own NA threshold, the one plm_qa.py tuned on official_val (metrics_official_val.json,
best_na_threshold), so the absolute numbers are optimistic for every model alike. Writes analysis/results.md.
The predictions are leaderboard material and stay local (gitignored); this script and its output are committed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/global"))
from mrc.evaluate import aggregate, load_jsonl, per_example_scores  # noqa: E402

RUNS = ["xlmr_large", "mdeberta_v3_base", "mbert_base", "nusabert_large", "indobert_base", "albert_indolite_large"]
PAIRS = [("mbert_base", "indobert_base"), ("mdeberta_v3_base", "indobert_base"), ("mdeberta_v3_base", "mbert_base"),
         ("xlmr_large", "indobert_base"), ("xlmr_large", "mbert_base"), ("xlmr_large", "mdeberta_v3_base"),
         ("nusabert_large", "indobert_base"), ("nusabert_large", "mbert_base"), ("nusabert_large", "mdeberta_v3_base"),
         ("albert_indolite_large", "indobert_base")]

ap = argparse.ArgumentParser()
ap.add_argument("--n-boot", type=int, default=10000)
a = ap.parse_args()
gold = load_jsonl(ROOT / "data/processed/official_val.jsonl")
ids = [g["id"] for g in gold]

f1s, rows = {}, []
for r in RUNS:
    d = HERE / "runs" / r / "seed13"
    met = json.load(open(d / "metrics_official_val.json"))
    thr = met["best_na_threshold"]
    preds = {p["id"]: p for p in load_jsonl(d / "preds_official_val.jsonl")}
    sc = per_example_scores(gold, preds, thr)
    f1s[r] = np.array([sc[i]["f1"] for i in ids]) * 100
    m = aggregate(sc)
    best_ep = next(int(l.split()[2]) for l in open(d / "log.txt", encoding="utf-8") if l.startswith("best epoch"))
    rows.append(f"| {r} | {m['exact']:.1f} | {m['f1']:.1f} | {m['HasAns_f1']:.1f} | {m['NoAns_acc']:.1f} | "
                f"{m['pred_noans_rate']:.1f} | {best_ep} | {thr:.2f} |")
has = np.array([sc[i]["has_ans"] for i in ids])  # the evaluator's own answerability flag

rng = np.random.default_rng(0)
idx = rng.integers(0, len(ids), size=(a.n_boot, len(ids)))
sig = []
for x, y in PAIRS:
    d = f1s[x] - f1s[y]
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / a.n_boot)
    sig.append(f"| {x} - {y} | {d.mean():+.2f} | [{lo:.2f}, {hi:.2f}] | {p:.4f} | {d[has].mean():+.2f} | "
               f"{d[~has].mean():+.2f} |")

out = HERE / "analysis/results.md"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join([
    "# 08 results on official_val (written by compare.py)", "",
    f"n = {len(ids)} ({has.sum()} answerable, {(~has).sum()} unanswerable). Seed 13, full train (8042), "
    "epoch and NA threshold chosen on official_val.", "",
    "| run | EM | F1 | HasAns F1 | NoAns acc | predicted no-answer % | best epoch | NA threshold |",
    "|---|---|---|---|---|---|---|---|", *rows, "",
    f"Paired bootstrap over questions ({a.n_boot} resamples, F1 points; two-sided p).", "",
    "| difference | dF1 | 95% CI | p | answerable | unanswerable |", "|---|---|---|---|---|---|", *sig, ""]),
    encoding="utf-8")
print(out.read_text(encoding="utf-8"))
