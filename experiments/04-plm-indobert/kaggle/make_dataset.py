"""Stage the private Kaggle dataset for the GPU kernel: shared code, this experiment's code, train/dev only.

  python experiments/04-plm-indobert/kaggle/make_dataset.py [--out DIR]
The internal test split (itest) and the official test set are never copied. See kaggle/README.md for upload steps.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=HERE / "build")
a = ap.parse_args()
proj = a.out / "proj"
if proj.exists():
    shutil.rmtree(proj)
ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
shutil.copytree(ROOT / "experiments/global/mrc", proj / "experiments/global/mrc", ignore=ignore)
shutil.copytree(ROOT / "experiments/global/configs", proj / "experiments/global/configs")
(proj / "experiments/04-plm-indobert").mkdir(parents=True)
shutil.copy(ROOT / "experiments/04-plm-indobert/plm_qa.py", proj / "experiments/04-plm-indobert/")
(proj / "data/processed").mkdir(parents=True)
for f in ("train.jsonl", "dev.jsonl", "official_val.jsonl"):  # official_val: 2026-11-24, for validation predictions
    shutil.copy(ROOT / "data/processed" / f, proj / "data/processed" / f)
(proj / "data/raw").mkdir(parents=True)  # 2026-11-24: full train (8042) for 08, which selects on official_val
shutil.copy(ROOT / "data/raw/train.jsonl", proj / "data/raw/train.jsonl")
shutil.copy(HERE / "dataset-metadata.json", a.out / "dataset-metadata.json")
assert not list(proj.rglob("*itest*")) and not list(proj.rglob("*test*.jsonl")), "test data must not be uploaded"
print(f"staged {sum(1 for _ in proj.rglob('*') if _.is_file())} files in {a.out}")
