"""Deterministic group split of data/raw/train.jsonl into train / dev / itest (internal sealed test).

Groups: examples sharing a context (after normalization) OR near-duplicate contexts
(same first 120 normalized chars, a cheap proxy for "same Wikipedia paragraph, light edits")
always land in the same split. Writes data/processed/{train,dev,itest}.jsonl and
data/splits/{split}_ids.txt + SHA-256 hashes in data/splits/hashes.json.

When the official validation/test sets are released, put them in data/raw/ and they are
copied to data/processed/official_val.jsonl / official_test.jsonl untouched.
"""
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

from mrc.paths import ROOT
RAW, PROC, SPL = ROOT / "data/raw", ROOT / "data/processed", ROOT / "data/splits"
SEED, FRAC = 13, {"train": 0.8, "dev": 0.1, "itest": 0.1}


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"\[\d+\]", "", s.lower())).strip()


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    rows = [json.loads(l) for l in open(RAW / "train.jsonl", encoding="utf-8")]
    dsu = DSU()
    by_prefix = {}
    for r in rows:
        c = norm(r["context"])
        dsu.find(c)
        k = c[:120]
        if k in by_prefix:
            dsu.union(c, by_prefix[k])
        else:
            by_prefix[k] = c
    groups = {}
    for r in rows:
        groups.setdefault(dsu.find(norm(r["context"])), []).append(r)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    n, out, acc = len(rows), {k: [] for k in FRAC}, 0
    bounds = [("train", FRAC["train"] * n), ("dev", (FRAC["train"] + FRAC["dev"]) * n), ("itest", n + 1)]
    for g in keys:
        split = next(s for s, b in bounds if acc < b)
        out[split].extend(groups[g])
        acc += len(groups[g])
    PROC.mkdir(parents=True, exist_ok=True)
    SPL.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for s, rs in out.items():
        rs.sort(key=lambda r: r["id"])
        txt = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
        (PROC / f"{s}.jsonl").write_text(txt, encoding="utf-8")
        ids = "\n".join(r["id"] for r in rs) + "\n"
        (SPL / f"{s}_ids.txt").write_text(ids, encoding="utf-8")
        hashes[s] = {"n": len(rs), "n_impossible": sum(r["is_impossible"] for r in rs),
                     "sha256_ids": hashlib.sha256(ids.encode()).hexdigest(),
                     "sha256_file": hashlib.sha256(txt.encode("utf-8")).hexdigest()}
    for name in ("validation", "val", "test"):
        for f in RAW.glob(f"*{name}*.jsonl"):
            shutil.copy(f, PROC / ("official_test.jsonl" if name == "test" else "official_val.jsonl"))
    hashes["n_groups"] = len(groups)
    (SPL / "hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
