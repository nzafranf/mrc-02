"""Official validation set: how far dev-tuned no-answer thresholds transfer, model by model.

  python experiments/global/run.py mrc.analysis.official_val

For every model in models_registry.yaml whose run dirs hold preds_official_val.jsonl, each seed is scored three ways:
  as predicted    the model's own decision (`pred`)
  dev threshold   no-answer threshold tuned on preds_dev.jsonl, applied to official_val (what a blind submission gets)
  val threshold   threshold tuned on official_val itself (optimistic here; it is the threshold a test submission uses)
The gap between the last two is what the dev/official_val shift costs. The internal dev split is 46% unanswerable and
official_val 36%, so dev thresholds tend to abstain too often. Writes report/assets/tables/official_val_table.{md,csv}.

Spans come from `span_pred` when a run stores it (02 thresholds `pred` at prediction time), otherwise from `pred`.
BiDAF writes "" as `pred` when NULL wins, so for it a threshold can only add abstentions, never answers.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import yaml

from mrc.analysis.compare import write_csv, write_md
from mrc.evaluate import aggregate, best_na_threshold, load_jsonl, per_example_scores
from mrc.make_submission import load_preds, raw_span
from mrc.paths import CONFIGS, PROC, ROOT, TABLES

KEYS = ["exact", "f1", "HasAns_f1", "NoAns_acc", "pred_noans_rate"]


def score_run(run_dir: Path, gold_dev, gold_val):
    dev = load_preds(run_dir / "preds_dev.jsonl")
    val = load_preds(run_dir / "preds_official_val.jsonl")
    as_pred = aggregate(per_example_scores(gold_val, val))
    for p in list(dev.values()) + list(val.values()):
        p["pred"] = raw_span(p)
    has_na = any("na_prob" in p for p in val.values())
    if not has_na:
        return {"as predicted": as_pred, "dev threshold": as_pred, "val threshold": as_pred}, None, None
    t_dev = best_na_threshold([g for g in gold_dev if g["id"] in dev], dev)[0]
    t_val = best_na_threshold(gold_val, val)[0]
    return ({"as predicted": as_pred,
             "dev threshold": aggregate(per_example_scores(gold_val, val, t_dev)),
             "val threshold": aggregate(per_example_scores(gold_val, val, t_val))}, t_dev, t_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(CONFIGS / "models_registry.yaml"))
    a = ap.parse_args()
    reg = yaml.safe_load(open(a.registry, encoding="utf-8"))
    gold_dev, gold_val = load_jsonl(PROC / "dev.jsonl"), load_jsonl(PROC / "official_val.jsonl")

    rows, skipped = [], []
    for name in reg["order"]:
        dirs = [Path(d) for d in sorted(glob.glob(str(ROOT / reg["models"][name])))
                if (Path(d) / "preds_official_val.jsonl").exists()]
        if not dirs:
            skipped.append(name)
            continue
        res = [score_run(d, gold_dev, gold_val) for d in dirs]
        for mode in ("as predicted", "dev threshold", "val threshold"):
            row = [name, len(dirs), mode]
            for k in KEYS:
                v = [r[0][mode][k] for r in res]
                row.append(f"{np.mean(v):.2f} ± {np.std(v):.2f}")
            thr = [r[1] if mode == "dev threshold" else r[2] for r in res] if mode != "as predicted" else []
            thr = [t for t in thr if t is not None]
            row.append(f"{np.mean(thr):.3f}" if thr else "")
            rows.append(row)

    hdr = ["model", "seeds", "threshold", "EM", "F1", "HasAns_F1", "NoAns_acc", "pred_noans_%", "thr (mean)"]
    title = ("Official validation (2003 questions, 36% unanswerable), mean ± std over seeds. 'dev threshold' is "
             "tuned on the internal dev split; 'val threshold' is tuned on official_val itself (optimistic)")
    write_md(TABLES / "official_val_table.md", hdr, rows, title)
    write_csv(TABLES / "official_val_table.csv", hdr, rows)
    print((TABLES / "official_val_table.md").read_text(encoding="utf-8"))
    if skipped:
        print("no preds_official_val.jsonl yet:", ", ".join(skipped))


if __name__ == "__main__":
    main()
