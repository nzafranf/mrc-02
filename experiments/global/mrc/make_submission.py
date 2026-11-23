"""Write a leaderboard submission CSV (`id,answer`, empty answer = no answer) from one run's predictions.

  python experiments/global/run.py mrc.make_submission \
      --run experiments/04-plm-indobert/runs/plm_indobert_full/seed13 --split-file data/processed/official_val.jsonl \
      --threshold-from official_val --out submissions/plm_indobert_full_seed13_official_val.csv

Reads <run>/preds_<split stem>.jsonl (or --pred). The no-answer threshold:
  --threshold-from SPLIT   tune it on <run>/preds_<SPLIT>.jsonl against data/processed/<SPLIT>.jsonl
                           (official_val is the selection set once it exists; dev before that)
  --na-threshold T         use T as given
  neither                  write `pred` as predicted (the model's own decision)
Tuning maximises EM by default (--tune-metric exact), because the leaderboard ranks by EM first and F1 second.
Rows follow the split file's order and cover every id in it; the script stops if a prediction is missing. If the split
has gold answers, the CSV is scored back with the shared evaluator as a check. Thresholds are never tuned on a test
split.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mrc.evaluate import aggregate, best_na_threshold, load_jsonl, per_example_scores
from mrc.paths import PROC


def raw_span(p: dict) -> str:
    """The span before any no-answer decision. 02 writes it as span_pred; the others keep it in pred."""
    return p.get("span_pred", p.get("pred", "")) or ""


def load_preds(path) -> dict[str, dict]:
    preds = {}
    for p in load_jsonl(path):
        if p.get("na_prob") is None:
            p.pop("na_prob", None)
        preds[p["id"]] = p
    return preds


def tune_threshold(run_dir: Path, split: str, metric: str = "exact") -> float | None:
    if "test" in split:
        raise SystemExit("Refusing to tune a threshold on a test split.")
    gold = load_jsonl(PROC / f"{split}.jsonl")
    preds = load_preds(run_dir / f"preds_{split}.jsonl")
    for p in preds.values():
        p["pred"] = raw_span(p)
    return best_na_threshold([g for g in gold if g["id"] in preds], preds, metric=metric)[0]


def answers(rows, preds, thr):
    out = []
    for ex in rows:
        p = preds[ex["id"]]
        if thr is None:
            a = p.get("pred", "") or ""
        else:
            a = "" if p.get("na_prob", float("-inf")) > thr else raw_span(p)
        out.append((ex["id"], " ".join(a.split())))  # one line per answer
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", required=True, help="questions to answer (official_val.jsonl, official_test.jsonl)")
    ap.add_argument("--run", help="run directory holding preds_<split>.jsonl")
    ap.add_argument("--pred", help="prediction file (default <run>/preds_<split stem>.jsonl)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--threshold-from", help="split to tune the no-answer threshold on, e.g. official_val or dev")
    g.add_argument("--na-threshold", type=float)
    ap.add_argument("--tune-metric", choices=["exact", "f1"], default="exact",
                    help="what --threshold-from maximises (leaderboard: EM first, then F1)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    split_file = Path(a.split_file)
    run_dir = Path(a.run) if a.run else None
    pred_path = Path(a.pred) if a.pred else (run_dir / f"preds_{split_file.stem}.jsonl" if run_dir else None)
    if pred_path is None:
        raise SystemExit("give --run or --pred")
    if a.threshold_from and run_dir is None:
        raise SystemExit("--threshold-from needs --run")

    rows = load_jsonl(split_file)
    preds = load_preds(pred_path)
    missing = [ex["id"] for ex in rows if ex["id"] not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} questions have no prediction in {pred_path}, e.g. {missing[:3]}")
    if len({ex['id'] for ex in rows}) != len(rows):
        raise SystemExit(f"duplicate ids in {split_file}")
    thr = tune_threshold(run_dir, a.threshold_from, a.tune_metric) if a.threshold_from else a.na_threshold

    out_rows = answers(rows, preds, thr)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "answer"])
        w.writerows(out_rows)

    report = {"out": str(out), "pred": str(pred_path), "n": len(out_rows),
              "threshold": thr, "threshold_from": a.threshold_from or ("given" if thr is not None else "model"),
              "tune_metric": a.tune_metric if a.threshold_from else None,
              "no_answer_rate": round(100 * sum(1 for _, x in out_rows if not x) / len(out_rows), 2)}
    if "answers" in rows[0]:
        sub = {i: {"id": i, "pred": x} for i, x in out_rows}
        m = aggregate(per_example_scores(rows, sub))
        report["check"] = {k: round(m[k], 2) for k in ("exact", "f1", "HasAns_f1", "NoAns_acc")}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
