"""Exact / near-duplicate audit of contexts and questions across train/dev/itest and, once it is in data/processed,
the official validation set (counts + examples only)."""
import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "experiments" / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import json, re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

P = ROOT / "data/processed"
S = ["train", "dev", "itest"] + (["official_val"] if (P / "official_val.jsonl").exists() else [])
PAIRS = [("dev", "train"), ("itest", "train"), ("itest", "dev")] + (
    [("official_val", "train"), ("official_val", "dev"), ("official_val", "itest")] if "official_val" in S else [])
rows = {s: [json.loads(l) for l in open(P / f"{s}.jsonl", encoding="utf-8")] for s in S}
norm = lambda x: re.sub(r"\s+", " ", re.sub(r"\[\d+\]", "", x.lower())).strip()
res, ex = {}, {}
for field in ["context", "question"]:
    uniq = {s: sorted({norm(r[field]) for r in rows[s]}) for s in S}
    allt = [t for s in S for t in uniq[s]]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(5, 5), sublinear_tf=True, min_df=1).fit(allt)
    M = {s: vec.transform(uniq[s]) for s in S}
    for a, b in PAIRS:
        exact = len(set(uniq[a]) & set(uniq[b]))
        sim = (M[a] @ M[b].T).tocsr()
        mx = np.asarray(sim.max(axis=1).todense()).ravel()
        arg = np.asarray(sim.argmax(axis=1)).ravel()
        k = f"{field}:{a}->{b}"
        res[k] = dict(n_unique=len(uniq[a]), exact=exact, near_gt_0_8=int((mx > 0.8).sum()), near_gt_0_6=int((mx > 0.6).sum()),
                      max_sim=round(float(mx.max()), 3))
        order = np.argsort(-mx)[:2]
        ex[k] = [(round(float(mx[i]), 3), uniq[a][i][:160], uniq[b][arg[i]][:160]) for i in order]
# context id overlaps by example id
ids = {s: {r["id"] for r in rows[s]} for s in S}
res["id_overlap"] = {f"{a}&{b}": len(ids[a] & ids[b]) for b, a in PAIRS}
# (question, context) exact pairs
pairs = {s: {(norm(r["context"]), norm(r["question"])) for r in rows[s]} for s in S}
res["pair_overlap"] = {f"{a}&{b}": len(pairs[a] & pairs[b]) for b, a in PAIRS}
json.dump({"counts": res, "examples": ex}, open(ROOT / "eda/outputs/tables/leakage_audit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in res.items(): print(k, v)
for k, v in ex.items(): print(k, v)
