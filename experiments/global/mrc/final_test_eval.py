"""Single sealed evaluation on the internal test split (itest). Run ONCE in milestone M6.

Usage: python experiments/global/run.py mrc.final_test_eval --preds experiments/<model>/runs/<seed>/preds_itest.jsonl [...]
Writes experiments/global/final_test_eval_log.json with split hash check and a timestamp.
Refuses to run if the itest id hash changed since prepare_data.py.
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

from mrc.paths import ROOT
from mrc.evaluate import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--preds", nargs="+", required=True)
ap.add_argument("--confirm", action="store_true", help="required: confirms frozen configs were approved")
a = ap.parse_args()
if not a.confirm:
    raise SystemExit("Pass --confirm after the group has approved the frozen configs.")
hashes = json.loads((ROOT / "data/splits/hashes.json").read_text())
ids = (ROOT / "data/splits/itest_ids.txt").read_text(encoding="utf-8")
if hashlib.sha256(ids.encode()).hexdigest() != hashes["itest"]["sha256_ids"]:
    raise SystemExit("itest id hash mismatch, split was modified.")
log_path = ROOT / "experiments/global/final_test_eval_log.json"
log = json.loads(log_path.read_text()) if log_path.exists() else []
# Exclude itest examples whose context near-duplicates a train/dev context (leakage audit, see decisions.md).
excl = set((ROOT / "data/splits/itest_exclude_ids.txt").read_text().split())
gold_clean = ROOT / "data/processed/itest_clean.jsonl"
with open(ROOT / "data/processed/itest.jsonl", encoding="utf-8") as fin, open(gold_clean, "w", encoding="utf-8") as fout:
    fout.writelines(l for l in fin if json.loads(l)["id"] not in excl)
for p in a.preds:
    r = evaluate(gold_clean, p)
    log.append({"preds": p, "time": datetime.datetime.now().isoformat(), "metrics": r})
    print(p, json.dumps(r))
log_path.write_text(json.dumps(log, indent=2))
