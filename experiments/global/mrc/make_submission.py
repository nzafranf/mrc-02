"""Turn one run's predictions on a split into a submission file.

The no-answer threshold comes from the run's own dev metrics
(`--dev-metrics .../metrics_dev.json`, key `best_na_threshold`), is given explicitly (`--threshold`), is tuned
on the submitted split itself (`--tune-on-split`, only for a split with gold such as official_val; its score is then
optimistic and meta.json says so), or the predictions are taken as they are (`--as-is`).

Usage:
  python experiments/global/run.py mrc.make_submission --name bidaf_seed13 \
      --preds experiments/03-nn-bidaf/runs/bidaf/seed13/preds_official_val.jsonl \
      --dev-metrics experiments/03-nn-bidaf/runs/bidaf/seed13/metrics_dev.json

Writes experiments/global/submissions/<split>/<name>/:
  submission.csv id,answer in the order of data/val/contoh_submission_validation.csv (course format, "" = no answer)
  answers.json   {id: answer}, "" = no answer (SQuAD v2 official format)
  answers.jsonl  one {"id", "answer"} per line
  meta.json      sources, threshold, split hash, counts, and the metrics on the split if it has gold answers
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

from mrc.evaluate import aggregate, best_na_threshold, load_jsonl, per_example_scores
from mrc.paths import EXP, PROC, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--split-file", default=str(PROC / "official_val.jsonl"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dev-metrics")
    g.add_argument("--threshold", type=float)
    g.add_argument("--as-is", action="store_true")
    g.add_argument("--tune-on-split", action="store_true")
    ap.add_argument("--order-csv", default=str(ROOT / "data/val/contoh_submission_validation.csv"),
                    help="sample submission whose id order submission.csv follows (ignored if missing)")
    a = ap.parse_args()

    split_path = Path(a.split_file)
    split = load_jsonl(split_path)
    preds = {p["id"]: p for p in load_jsonl(a.preds)}
    missing = [e["id"] for e in split if e["id"] not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} ids of {split_path.name} have no prediction (first: {missing[0]})")

    # runs that already applied a dev threshold keep the raw span in span_pred; re-thresholding needs the raw span
    for p in preds.values():
        if "span_pred" in p and (a.tune_on_split or a.dev_metrics or a.threshold is not None):
            p["pred"] = p["span_pred"]
    thr = None
    if a.tune_on_split:
        if "na_prob" not in next(iter(preds.values())):
            raise SystemExit("--tune-on-split needs na_prob in the predictions")
        thr, _ = best_na_threshold(split, preds)
    elif a.dev_metrics:
        thr = json.loads(Path(a.dev_metrics).read_text(encoding="utf-8"))["best_na_threshold"]
    elif a.threshold is not None:
        thr = a.threshold

    answers = {}
    for e in split:
        p = preds[e["id"]]
        ans = (p.get("pred", "") or "").strip()  # sentencepiece offsets can carry a leading space
        if thr is not None and p.get("na_prob") is not None and p["na_prob"] > thr:
            ans = ""
        answers[e["id"]] = ans

    out = EXP / "global/submissions" / split_path.stem / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
    order = list(answers)
    if Path(a.order_csv).exists():
        with open(a.order_csv, encoding="utf-8", newline="") as f:
            order = [r["id"] for r in csv.DictReader(f)]
        if set(order) != set(answers):
            raise SystemExit(f"{a.order_csv} ids differ from {split_path.name}")
    with open(out / "submission.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "answer"])
        w.writerows([i, answers[i]] for i in order)
    with open(out / "answers.jsonl", "w", encoding="utf-8") as f:
        for i, ans in answers.items():
            f.write(json.dumps({"id": i, "answer": ans}, ensure_ascii=False) + "\n")

    meta = {
        "name": a.name,
        "preds": str(Path(a.preds).resolve().relative_to(ROOT)),
        "threshold_source": a.dev_metrics and str(Path(a.dev_metrics).resolve().relative_to(ROOT))
        or ("tuned on this split (optimistic score)" if a.tune_on_split
            else "explicit" if a.threshold is not None else "as-is"),
        "na_threshold": thr,
        "split": split_path.name,
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "n": len(answers),
        "n_no_answer": sum(v == "" for v in answers.values()),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if all("answers" in e for e in split):
        meta["metrics"] = aggregate(per_example_scores(split, {i: {"pred": v} for i, v in answers.items()}))
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    m = meta.get("metrics", {})
    print(f"{a.name}: n={meta['n']} no-answer={meta['n_no_answer']} thr={thr} "
          f"EM={m.get('exact', float('nan')):.1f} F1={m.get('f1', float('nan')):.1f} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
