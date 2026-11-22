"""Stage what the GPU notebook needs, for Kaggle (dataset folder) and Colab (one zip).

  python experiments/07-llm-offtheshelf-fewshot/gpu/make_bundle.py
-> gpu/build/kaggle/proj/                  the files below at their project paths, plus
   gpu/build/kaggle/dataset-metadata.json    -> upload this folder as a private Kaggle dataset
   gpu/build/mrc07_bundle.zip                the same files zipped (put it in Google Drive for Colab)

Contents: the shared package and configs, this experiment's code, prompts and frozen demos, the train/dev/official
validation splits, and the fixed orders. The internal test split and any official test file are never included
(checked below). Rebuild and re-upload after changing prompts.yaml, demos.json or the code.
"""
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
BUILD = Path(__file__).resolve().parent / "build"
FILES = [
    *[p for p in (ROOT / "experiments/global/mrc").rglob("*.py")],
    *[p for p in (ROOT / "experiments/global/configs").glob("*.yaml")],
    ROOT / "experiments/global/run.py",
    *[EXP / f for f in ("fewshot.py", "run_fewshot.py", "report_fewshot.py", "analyze.py", "select_demos.py", "prompts.yaml", "demos.json")],
    *[ROOT / "data/processed" / f for f in ("train.jsonl", "dev.jsonl", "official_val.jsonl",
                                            "slices_dev.jsonl", "slices_official_val.jsonl")],
    ROOT / "data/splits/order.json", ROOT / "data/splits/order_dev.txt", ROOT / "data/splits/order_official_val.txt",
]
DATASET_ID = "naufalzafranfadil/pbm-mrc-llm-fewshot"  # change the owner prefix to your Kaggle username


def main():
    proj = BUILD / "kaggle" / "proj"
    if proj.exists():
        shutil.rmtree(proj)
    rel = [f.relative_to(ROOT) for f in FILES if "__pycache__" not in f.parts]
    bad = [r for r in rel if "test" in r.name.lower() and r.suffix in (".jsonl", ".json", ".txt", ".csv")]
    assert not bad, f"test data must never be uploaded: {bad}"
    for r in rel:
        (proj / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / r, proj / r)
    with zipfile.ZipFile(BUILD / "mrc07_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for r in rel:
            z.write(proj / r, r.as_posix())
    (BUILD / "kaggle" / "dataset-metadata.json").write_text(json.dumps(
        {"title": "pbm-mrc-llm-fewshot", "id": DATASET_ID, "licenses": [{"name": "other"}], "isPrivate": True}))
    print(f"staged {len(rel)} files in {proj} and {BUILD / 'mrc07_bundle.zip'}")


if __name__ == "__main__":
    main()
