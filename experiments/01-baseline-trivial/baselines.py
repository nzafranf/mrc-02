"""Trivial baselines: always-no-answer, sliding-window (Richardson et al. 2013 style), and the
sliding-window with a dev-tuned overlap threshold for no-answer.

  python experiments/01-baseline-trivial/baselines.py --split-file data/processed/dev.jsonl
  python experiments/01-baseline-trivial/baselines.py --predict-only --split-file X.jsonl --out DIR   (threshold tuned on dev)
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import argparse
import datetime
import json
import time
from pathlib import Path

from mrc.evaluate import aggregate, best_na_threshold, evaluate, load_jsonl, per_example_scores
from mrc.qa_text import (MONTHS, QTYPES, IdfStats, content_stems, is_year, qtype, tokenize,
                                tokenize_context)



def entity_spans(toks, a: int, b: int) -> list[tuple[int, int]]:
    """Candidate answer chunks in sentence tokens [a,b): maximal capitalized runs (not just the sentence-initial
    word), numbers, and dates (digits + month names). Returns inclusive token index pairs."""
    spans, i = [], a
    while i < b:
        t = toks[i]
        if t.punct:
            i += 1
            continue
        if t.cap or t.digit or t.low in MONTHS:
            j = i
            while j + 1 < b and (toks[j + 1].cap or toks[j + 1].digit or toks[j + 1].low in MONTHS
                                 or (toks[j + 1].low in ("of", "de", "van", "bin", "al") and j + 2 < b and toks[j + 2].cap)):
                j += 1
            if not (i == a and j == i and not t.digit):  # lone sentence-initial word is not an entity
                spans.append((i, j))
            i = j + 1
        else:
            i += 1
    return spans


class SlidingWindow:
    def __init__(self, idf: IdfStats):
        self.idf = idf

    def predict(self, ex: dict) -> dict:
        ctx = ex["context"]
        toks, _, ranges = tokenize_context(ctx)
        qt = tokenize(ex["question"])
        qs = set(content_stems(qt))
        qlow = {t.low for t in qt}
        qtp = qtype(qt)
        w = {s: self.idf.idf(s) for s in qs}
        tot = sum(w.values()) or 1.0
        best_si, best_sc = 0, -1.0
        for si, (a, b) in enumerate(ranges):
            sc = sum(w[s] for s in qs & {t.stem for t in toks[a:b]})
            if sc > best_sc:
                best_si, best_sc = si, sc
        a, b = ranges[best_si]
        match = [i for i in range(a, b) if toks[i].stem in qs]
        best, best_val = None, -1e9
        for (i, j) in entity_spans(toks, a, b):
            if any(toks[k].low in qlow or toks[k].stem in qs for k in range(i, j + 1)):
                continue
            dist = min((min(abs(i - m), abs(j - m)) for m in match), default=10)
            val = -dist
            span_txt = [toks[k] for k in range(i, j + 1)]
            numeric = any(t.digit for t in span_txt)
            if qtp in (QTYPES["when"], QTYPES["num"]) and numeric:
                val += 5 + 2 * any(is_year(t.text) for t in span_txt)
            if qtp in (QTYPES["who"], QTYPES["where"]) and not numeric:
                val += 5
            if val > best_val:
                best, best_val = (i, j), val
        if best is None:  # fallback: first non-question content token right after the first match
            cand = [k for k in range(a, b) if not toks[k].punct and not toks[k].stop and toks[k].stem not in qs]
            if not cand:
                return {"pred": "", "start_char": -1, "score": 0.0, "overlap": best_sc / tot}
            k = min(cand, key=lambda k: min((abs(k - m) for m in match), default=0))
            best = (k, k)
        s, e = toks[best[0]].s, toks[best[1]].e
        return {"pred": ctx[s:e], "start_char": s, "score": best_val, "overlap": best_sc / tot}


def write_run(out: Path, name: str, preds: list[dict], gold: list[dict] | None, config: dict, log: str, split_name: str):
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "log.txt").write_text(log, encoding="utf-8")
    with open(out / f"preds_{split_name}.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", default="data/processed/dev.jsonl")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--out", default=None, help="output dir (predict-only); preds go to <out>/<model>/preds.jsonl")
    a = ap.parse_args()
    t0 = time.time()
    train = load_jsonl(ROOT / "data/processed/train.jsonl")
    dev = load_jsonl(ROOT / "data/processed/dev.jsonl")
    split = load_jsonl(ROOT / a.split_file) if not Path(a.split_file).is_absolute() else load_jsonl(a.split_file)
    split_name = Path(a.split_file).stem
    idf = IdfStats(list({e["context"] for e in train}))
    sw = SlidingWindow(idf)
    dev_out = {e["id"]: sw.predict(e) for e in dev}
    # threshold on overlap ratio: predict no answer if overlap < thr  (na_prob = 1 - overlap)
    dev_preds = {i: {"pred": p["pred"], "na_prob": 1 - p["overlap"]} for i, p in dev_out.items()}
    thr, _ = best_na_threshold(dev, dev_preds)
    models = {"no_answer": None, "sliding_window": None, "sliding_window_thr": thr}
    for name, th in models.items():
        out_preds = []
        for e in split:
            if name == "no_answer":
                out_preds.append({"id": e["id"], "pred": "", "na_prob": 1.0, "score": 0.0, "start_char": -1})
                continue
            p = sw.predict(e) if split is not dev else dev_out[e["id"]]
            na = 1 - p["overlap"]
            if name == "sliding_window":
                na = 0.0  # always answers
            pred = "" if (th is not None and na > th) else p["pred"]
            out_preds.append({"id": e["id"], "pred": pred, "na_prob": na, "score": p["score"],
                              "start_char": p["start_char"] if pred else -1})
        cfg = {"model": name, "seed": a.seed, "overlap_na_threshold_on_na_prob": th, "split": a.split_file,
               "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "deterministic": True}
        if a.predict_only:
            base = Path(a.out or ROOT / "experiments/01-baseline-trivial/predictions") / name
            base.mkdir(parents=True, exist_ok=True)
            with open(base / "preds.jsonl", "w", encoding="utf-8") as f:
                for p in out_preds:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            continue
        out = ROOT / f"experiments/01-baseline-trivial/runs/{name}/seed{a.seed}"
        write_run(out, name, out_preds, split, cfg, f"{name}: deterministic, {time.time()-t0:.1f}s\n", split_name)
        if "is_impossible" in split[0]:
            pf = out / f"preds_{split_name}.jsonl"
            m = evaluate(ROOT / a.split_file, pf, search=True)
            (out / f"metrics_{split_name}.json").write_text(json.dumps(m, indent=2))
            print(name, "EM %.2f F1 %.2f HasAns_F1 %.2f NoAns_acc %.2f" % (m["exact"], m["f1"], m["HasAns_f1"], m["NoAns_acc"]),
                  "| best-thr F1 %.2f" % m["at_best_threshold"]["f1"])


if __name__ == "__main__":
    main()
