"""Aggregate bidaf runs: mean/std table, context-length buckets, error samples.

python experiments/03-nn-bidaf/bidaf_report.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import json
import re
from pathlib import Path

import numpy as np

from mrc.evaluate import aggregate, load_jsonl, per_example_scores

DEV = "data/processed/dev.jsonl"
BUCKETS = [("<60", 0, 60), ("60-120", 60, 120), ("120-250", 120, 250), (">250", 250, 10**9)]


def run_metrics(model: str):
    gold = load_jsonl(DEV)
    res = []
    for d in sorted(Path(f"experiments/03-nn-bidaf/runs/{model}").glob("seed*")):
        p = d / "preds_dev.jsonl"
        if not p.exists():
            continue
        preds = {x["id"]: x for x in load_jsonl(p)}
        m = json.loads((d / "metrics_dev.json").read_text())
        sc = per_example_scores(gold, preds)
        b = {}
        for name, lo, hi in BUCKETS:
            sub = {e["id"]: sc[e["id"]] for e in gold if lo <= len(e["context"].split()) < hi}
            b[name] = (len(sub), aggregate(sub)["f1"], aggregate(sub)["exact"])
        res.append((d.name, m, b))
    return res


def ms(xs):
    return f"{np.mean(xs):.2f} +- {np.std(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"


def main():
    for model in ("bidaf", "bidaf_noattn"):
        r = run_metrics(model)
        if not r:
            continue
        print(f"## {model} ({len(r)} seeds: {[x[0] for x in r]})")
        for k in ("exact", "f1", "HasAns_f1", "NoAns_acc", "pred_noans_rate"):
            print(f"{k}: {ms([x[1][k] for x in r])}")
        print("F1 at best dev threshold:", ms([x[1]["at_best_threshold"]["f1"] for x in r]))
        for name, _, _ in BUCKETS:
            print(f"bucket {name}: n={r[0][2][name][0]} F1 {ms([x[2][name][1] for x in r])}")
        for n, m, _ in r:
            print(n, {k: round(m[k], 2) for k in ("exact", "f1", "HasAns_f1", "NoAns_acc", "pred_noans_rate")})
    # error samples from bidaf seed13
    gold = {e["id"]: e for e in load_jsonl(DEV)}
    d = Path("experiments/03-nn-bidaf/runs/bidaf/seed13")
    if (d / "preds_dev.jsonl").exists():
        preds = {x["id"]: x for x in load_jsonl(d / "preds_dev.jsonl")}
        sc = per_example_scores(list(gold.values()), preds)
        bad = [i for i, v in sc.items() if v["f1"] < 0.5]
        import random
        random.Random(0).shuffle(bad)
        print("\n## error samples")
        for i in bad[:14]:
            g = gold[i]
            print(i, "|Q:", g["question"], "|GOLD:", g["answers"][0]["text"] if g["answers"] and not g["is_impossible"] else "<NA>",
                  "|PRED:", sc[i]["pred"] or "<NA>", "|na_prob %.2f" % preds[i]["na_prob"])


if __name__ == "__main__":
    main()
