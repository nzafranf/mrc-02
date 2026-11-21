"""Shared evaluator for every model (SQuAD v2 style, Indonesian-aware normalization).

Prediction file format (JSONL, one per example):
  {"id": str, "pred": str, "na_prob": float (optional, higher = more likely no answer),
   "score": float (optional, span confidence)}
An empty string "" as pred means "no answer".

Usage:
  python experiments/global/run.py mrc.evaluate --gold data/processed/dev.jsonl --pred experiments/x/runs/0/preds_dev.jsonl
  python experiments/global/run.py mrc.evaluate ... --out metrics.json --na-threshold-search
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import string
from pathlib import Path

_PUNCT = set(string.punctuation) | {"“", "”", "‘", "’", "–", "—", "«", "»"}
_CITE = re.compile(r"\[\d+\]")


def normalize_answer(s: str) -> str:
    """Lowercase, drop wiki citation marks like [1], punctuation, extra whitespace.
    English articles are NOT removed (Indonesian has none); decision logged in miscellaneous/docs/decisions.md."""
    s = _CITE.sub(" ", s.lower())
    s = "".join(ch for ch in s if ch not in _PUNCT)
    return " ".join(s.split())


def get_tokens(s: str) -> list[str]:
    return normalize_answer(s).split() if s else []


def exact_match(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def f1(gold: str, pred: str) -> float:
    g, p = get_tokens(gold), get_tokens(pred)
    if len(g) == 0 or len(p) == 0:
        return float(g == p)
    common = collections.Counter(g) & collections.Counter(p)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def gold_texts(ex: dict) -> list[str]:
    texts = [a["text"] for a in ex.get("answers", []) if normalize_answer(a["text"])]
    return texts if texts else [""]


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def per_example_scores(gold: list[dict], preds: dict[str, dict], na_threshold: float | None = None):
    """Return {id: {"em", "f1", "has_ans", "pred"}}. If na_threshold given and na_prob present,
    predict "" whenever na_prob > threshold."""
    out = {}
    for ex in gold:
        p = preds.get(ex["id"], {"pred": ""})
        pred = p.get("pred", "") or ""
        if na_threshold is not None and "na_prob" in p and p["na_prob"] > na_threshold:
            pred = ""
        golds = gold_texts(ex)
        out[ex["id"]] = {
            "em": max(exact_match(g, pred) for g in golds),
            "f1": max(f1(g, pred) for g in golds),
            "has_ans": golds != [""],
            "pred": pred,
        }
    return out


def aggregate(scores: dict) -> dict:
    def avg(xs):
        return 100.0 * sum(xs) / len(xs) if xs else float("nan")
    allv = list(scores.values())
    has = [v for v in allv if v["has_ans"]]
    no = [v for v in allv if not v["has_ans"]]
    pred_na = [v for v in allv if v["pred"] == ""]
    return {
        "n": len(allv), "exact": avg([v["em"] for v in allv]), "f1": avg([v["f1"] for v in allv]),
        "HasAns_n": len(has), "HasAns_exact": avg([v["em"] for v in has]), "HasAns_f1": avg([v["f1"] for v in has]),
        "NoAns_n": len(no), "NoAns_acc": avg([v["em"] for v in no]),
        "pred_noans_rate": avg([1.0] * len(pred_na) + [0.0] * (len(allv) - len(pred_na))),
    }


def best_na_threshold(gold, preds, metric="f1"):
    """Search na_prob threshold on THIS split. Only call on train/dev, never on test."""
    cands = sorted({p["na_prob"] for p in preds.values() if "na_prob" in p})
    if not cands:
        return None, None
    best_t, best = None, -1.0
    grid = cands if len(cands) <= 400 else [cands[int(i * (len(cands) - 1) / 399)] for i in range(400)]
    for t in [-1e9] + grid:
        m = aggregate(per_example_scores(gold, preds, t))[metric]
        if m > best:
            best_t, best = t, m
    return best_t, best


def evaluate(gold_path, pred_path, na_threshold=None, search=False):
    gold = load_jsonl(gold_path)
    preds = {p["id"]: p for p in load_jsonl(pred_path)}
    missing = [g["id"] for g in gold if g["id"] not in preds]
    res = aggregate(per_example_scores(gold, preds, na_threshold))
    res["missing_preds"] = len(missing)
    if search:
        t, _ = best_na_threshold(gold, preds)
        res["best_na_threshold"] = t
        res["at_best_threshold"] = aggregate(per_example_scores(gold, preds, t)) if t is not None else None
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--na-threshold", type=float, default=None)
    ap.add_argument("--na-threshold-search", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.na_threshold_search and ("test" in Path(a.gold).name):
        raise SystemExit("Refusing threshold search on a test split.")
    r = evaluate(a.gold, a.pred, a.na_threshold, a.na_threshold_search)
    print(json.dumps(r, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2), encoding="utf-8")
