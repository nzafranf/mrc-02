"""Write a leaderboard submission (`id,answer`, empty answer = no answer) from one run's predictions.

Two output modes (merged from the two versions of this script, 2026-11-25; both command styles still work):
  --out FILE.csv    write only that CSV
  --name NAME       write experiments/global/submissions/<split stem>/<NAME>/{submission.csv, answers.json,
                    answers.jsonl, meta.json} (gitignored: submissions stay local)

  python experiments/global/run.py mrc.make_submission \
      --run experiments/04-plm-indobert/runs/plm_indobert_full/seed13 --split-file data/processed/official_val.jsonl \
      --threshold-from official_val --out submissions/plm_indobert_full_seed13_official_val.csv
  python experiments/global/run.py mrc.make_submission --name 08_xlmr_large_seed13 \
      --preds experiments/08-plm-encoder-screen/runs/xlmr_large/seed13/preds_official_val.jsonl --tune-on-split

Predictions: --pred/--preds FILE, or --run DIR (reads DIR/preds_<split stem>.jsonl). The no-answer threshold:
  --threshold-from SPLIT   tune it on <run>/preds_<SPLIT>.jsonl against data/processed/<SPLIT>.jsonl
  --tune-on-split          tune it on the submitted split itself (needs gold; the score is then optimistic)
  --dev-metrics FILE       use best_na_threshold from a run's metrics_dev.json
  --na-threshold/--threshold T   use T as given
  none of these (--as-is)  write `pred` as predicted (the model's own decision)
Tuning maximises EM by default (--tune-metric exact), because the leaderboard ranks by EM first and F1 second.
Thresholds are never tuned on a test split. Rows follow the split file's order (for official_val this equals the
course sample file data/val/contoh_submission_validation.csv, which is checked when present). Every id must have a
prediction. If the split has gold answers, the written answers are scored back with the shared evaluator.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

from mrc.evaluate import aggregate, best_na_threshold, load_jsonl, per_example_scores
from mrc.paths import EXP, PROC, ROOT

SAMPLE = ROOT / "data/val/contoh_submission_validation.csv"


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


def tune(gold: list[dict], preds: dict[str, dict], split_name: str, metric: str) -> float | None:
    if "test" in split_name:
        raise SystemExit("Refusing to tune a threshold on a test split.")
    if not any("na_prob" in p for p in preds.values()):
        raise SystemExit("tuning a threshold needs na_prob in the predictions")
    raw = {i: {**p, "pred": raw_span(p)} for i, p in preds.items()}
    return best_na_threshold([g for g in gold if g["id"] in raw], raw, metric=metric)[0]


def decide(rows, preds, thr):
    out = []
    for ex in rows:
        p = preds[ex["id"]]
        if thr is None:
            a = p.get("pred", "") or ""
        else:
            a = "" if p.get("na_prob", float("-inf")) > thr else raw_span(p)
        out.append((ex["id"], " ".join(a.split())))  # one line per answer, no stray spaces
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", default=str(PROC / "official_val.jsonl"),
                    help="questions to answer (official_val.jsonl, official_test.jsonl)")
    ap.add_argument("--run", help="run directory holding preds_<split>.jsonl")
    ap.add_argument("--pred", "--preds", dest="pred", help="prediction file (default <run>/preds_<split stem>.jsonl)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--threshold-from", help="split to tune the no-answer threshold on, e.g. official_val or dev")
    g.add_argument("--tune-on-split", action="store_true", help="tune on the submitted split itself")
    g.add_argument("--dev-metrics", help="metrics_dev.json whose best_na_threshold is used")
    g.add_argument("--na-threshold", "--threshold", dest="na_threshold", type=float)
    g.add_argument("--as-is", action="store_true", help="keep the model's own decision (the default)")
    ap.add_argument("--tune-metric", choices=["exact", "f1"], default="exact",
                    help="what tuning maximises (leaderboard: EM first, then F1)")
    o = ap.add_mutually_exclusive_group(required=True)
    o.add_argument("--out", help="CSV path to write")
    o.add_argument("--name", help="write experiments/global/submissions/<split stem>/<name>/")
    a = ap.parse_args()

    split_file = Path(a.split_file)
    run_dir = Path(a.run) if a.run else None
    pred_path = Path(a.pred) if a.pred else (run_dir / f"preds_{split_file.stem}.jsonl" if run_dir else None)
    if pred_path is None:
        raise SystemExit("give --run or --pred")
    if a.threshold_from and run_dir is None:
        raise SystemExit("--threshold-from needs --run")

    rows = load_jsonl(split_file)
    if len({ex["id"] for ex in rows}) != len(rows):
        raise SystemExit(f"duplicate ids in {split_file}")
    preds = load_preds(pred_path)
    missing = [ex["id"] for ex in rows if ex["id"] not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} questions have no prediction in {pred_path}, e.g. {missing[:3]}")

    if a.threshold_from:
        thr = tune(load_jsonl(PROC / f"{a.threshold_from}.jsonl"),
                   load_preds(run_dir / f"preds_{a.threshold_from}.jsonl"), a.threshold_from, a.tune_metric)
        source = f"tuned on {a.threshold_from} ({a.tune_metric})"
    elif a.tune_on_split:
        thr = tune(rows, preds, split_file.stem, a.tune_metric)
        source = f"tuned on this split ({a.tune_metric}; optimistic score)"
    elif a.dev_metrics:
        thr = json.loads(Path(a.dev_metrics).read_text(encoding="utf-8"))["best_na_threshold"]
        source = str(Path(a.dev_metrics))
    else:
        thr = a.na_threshold
        source = "given" if thr is not None else "model"

    out_rows = decide(rows, preds, thr)
    if split_file.stem == "official_val" and SAMPLE.exists():
        with open(SAMPLE, encoding="utf-8", newline="") as f:
            if [r["id"] for r in csv.DictReader(f)] != [i for i, _ in out_rows]:
                raise SystemExit(f"id order differs from {SAMPLE}")

    out_csv = Path(a.out) if a.out else EXP / "global/submissions" / split_file.stem / a.name / "submission.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "answer"])
        w.writerows(out_rows)

    report = {"out": str(out_csv), "pred": str(pred_path), "n": len(out_rows), "threshold": thr,
              "threshold_from": source,
              "no_answer_rate": round(100 * sum(1 for _, x in out_rows if not x) / len(out_rows), 2)}
    if all("answers" in ex for ex in rows):
        m = aggregate(per_example_scores(rows, {i: {"id": i, "pred": x} for i, x in out_rows}))
        report["check"] = {k: round(m[k], 2) for k in ("exact", "f1", "HasAns_f1", "NoAns_acc")}
    if a.name:
        d = out_csv.parent
        answers = dict(out_rows)
        (d / "answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
        with open(d / "answers.jsonl", "w", encoding="utf-8") as f:
            for i, x in out_rows:
                f.write(json.dumps({"id": i, "answer": x}, ensure_ascii=False) + "\n")
        meta = {**report, "name": a.name, "split": split_file.name,
                "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
                "created": datetime.datetime.now().isoformat(timespec="seconds")}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
