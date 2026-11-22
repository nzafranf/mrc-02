"""Fixed evaluation order of each split, decided once with the project's global seed.

  python experiments/global/run.py mrc.eval_order            # writes data/splits/order_<split>.txt + order.json
  python experiments/global/run.py mrc.eval_order --check    # verifies the files, writes nothing

Why: runs that cannot cover a whole split (API quotas, GPU time limits, a crash in the middle) all process the
questions in this one order. Any prefix of the order is then a sample every such run shares, so partial runs are
compared on the same questions, and a resumed run continues where it stopped instead of somewhere random.

Order: stratified by answerability. Each stratum is shuffled with GLOBAL_SEED, then the strata are interleaved so
that every prefix keeps the split's unanswerable share (largest deficit first). A prefix of 100 questions of dev
therefore has 46 unanswerable ones, like the full split (46.2%).

order.json records the seed, the method, the SHA-256 of each split file and of each order file. Runners call
`load_order(split)`, which refuses to run if the split file changed since the order was written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random

from mrc.evaluate import load_jsonl
from mrc.paths import PROC, SPLITS

GLOBAL_SEED = 13  # the project seed (also the split seed and the first training seed); decided once, never changed
SPLIT_FILES = {"dev": PROC / "dev.jsonl", "official_val": PROC / "official_val.jsonl"}  # never a test split
META = SPLITS / "order.json"
METHOD = "stratified by is_impossible; strata shuffled with GLOBAL_SEED; interleaved by largest deficit"


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_order(rows: list[dict], seed: int = GLOBAL_SEED) -> list[str]:
    strata = {}
    for r in rows:
        strata.setdefault(bool(r["is_impossible"]), []).append(r["id"])
    keys = sorted(strata)  # False (answerable) first, so ties go to answerable
    for k in keys:
        strata[k].sort()  # independent of file order
        random.Random(f"{seed}-{k}").shuffle(strata[k])
    share = {k: len(strata[k]) / len(rows) for k in keys}
    taken, out = {k: 0 for k in keys}, []
    for i in range(len(rows)):
        k = max((k for k in keys if taken[k] < len(strata[k])), key=lambda k: share[k] * (i + 1) - taken[k])
        out.append(strata[k][taken[k]])
        taken[k] += 1
    return out


def order_path(split: str):
    return SPLITS / f"order_{split}.txt"


def load_order(split: str) -> list[str]:
    """The fixed order of a split, after checking that neither the split file nor the order file changed."""
    if split not in SPLIT_FILES:
        raise SystemExit(f"no fixed order for split {split!r}; choose from {sorted(SPLIT_FILES)}")
    meta = json.loads(META.read_text(encoding="utf-8"))[split]
    if sha256(SPLIT_FILES[split]) != meta["sha256_split_file"]:
        raise SystemExit(f"{SPLIT_FILES[split]} changed since {META} was written; the fixed order is no longer valid")
    if sha256(order_path(split)) != meta["sha256_order_file"]:
        raise SystemExit(f"{order_path(split)} was edited; restore it from git")
    return order_path(split).read_text(encoding="utf-8").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the existing files against the splits")
    a = ap.parse_args()
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    for split, f in SPLIT_FILES.items():
        if not f.exists():
            print(f"skip {split}: {f} not found")
            continue
        rows = load_jsonl(f)
        order = make_order(rows)
        assert sorted(order) == sorted(r["id"] for r in rows)
        text = "\n".join(order) + "\n"
        if a.check:
            same = order_path(split).read_text(encoding="utf-8") == text and \
                meta.get(split, {}).get("sha256_split_file") == sha256(f)
            print(f"{split}: {'ok' if same else 'MISMATCH'}")
            continue
        order_path(split).write_text(text, encoding="utf-8", newline="\n")
        imp = {r["id"]: r["is_impossible"] for r in rows}
        prefix = {n: round(sum(imp[i] for i in order[:n]) / n, 3) for n in (50, 100, 200, 400) if n <= len(order)}
        meta[split] = {"seed": GLOBAL_SEED, "method": METHOD, "n": len(order), "split_file": f.relative_to(PROC.parent.parent).as_posix(),
                       "sha256_split_file": sha256(f), "sha256_order_file": sha256(order_path(split)),
                       "unanswerable_share_full": round(sum(imp.values()) / len(rows), 3),
                       "unanswerable_share_prefix": prefix}
        print(f"{split}: {len(order)} ids -> {order_path(split).name}; unanswerable share full "
              f"{meta[split]['unanswerable_share_full']}, prefixes {prefix}")
    if not a.check:
        META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
