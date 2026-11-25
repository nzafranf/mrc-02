"""Build and push the two 08 Kaggle kernels. Each kernel is one script: runner.py + its job list.

  python experiments/08-plm-encoder-screen/kaggle/push.py [--owner naufalzafranfadil] [--dry-run]
Kernel "base": IndoBERT-base (reference), m-BERT, mDeBERTa-v3-base, IndoELECTRA, about 2 h on a T4.
Kernel "large": IndoBERT-lite-large (ALBERT), XLM-R-large, NusaBERT-large, about 5 to 6 h.
Setup in runner.py: full raw train (8042), epoch and threshold chosen on official_val, 10 epochs, seed 13.
Both use the dataset staged by experiments/04-plm-indobert/kaggle/make_dataset.py (raw train, official_val; never the
leaderboard test). Pull outputs: python -m kaggle kernels output <owner>/<slug> -p experiments/08-plm-encoder-screen/kaggle/out/<name>
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNELS = {
    "base": [
        ("08-plm-encoder-screen", "indobert_base", "indobenchmark/indobert-base-p1", "base"),
        ("08-plm-encoder-screen", "mbert_base", "google-bert/bert-base-multilingual-cased", "base"),
        ("08-plm-encoder-screen", "mdeberta_v3_base", "microsoft/mdeberta-v3-base", "base"),
    ],
    # rerun after the fp16-checkpoint fix in plm_qa.build_model (2026-11-24); the first base run crashed at step 1
    "fix": [
        ("08-plm-encoder-screen", "mdeberta_v3_base", "microsoft/mdeberta-v3-base", "base"),
    ],
    # One large model per kernel (2026-11-24): the 3-model kernel sat QUEUED for 3 h and would have run about 9 h,
    # close to Kaggle's 12 h session cap. Re-pushing "large" replaces that stuck version (now ALBERT only).
    "large": [
        ("08-plm-encoder-screen", "albert_indolite_large", "indobenchmark/indobert-lite-large-p2", "large"),
    ],
    "xlmr": [
        ("08-plm-encoder-screen", "xlmr_large", "FacebookAI/xlm-roberta-large", "large"),
    ],
    "nusabert": [
        ("08-plm-encoder-screen", "nusabert_large", "LazarusNLP/NusaBERT-large", "large"),
    ],
}
# Dropped: ChristopherA08/IndoELECTRA. Its tokenizer emits ids up to 35000 but its config and embeddings have
# vocab_size 30522, so any id above that crashes the embedding lookup (CUDA device-side assert, base run 2026-11-24).
EXCLUDED = {"electra_indo_base": "tokenizer/model vocab mismatch (35000 vs 30522)"}

ap = argparse.ArgumentParser()
ap.add_argument("--owner", default="naufalzafranfadil")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--only", nargs="*", help="kernel names to build/push (default all)")
a = ap.parse_args()
runner = (HERE / "runner.py").read_text(encoding="utf-8")
for name, jobs in KERNELS.items():
    if a.only and name not in a.only:
        continue
    d = HERE / "build" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(f"{runner}\n\nJOBS = {json.dumps(jobs, indent=1)}\nmain([tuple(j) for j in JOBS])\n",
                              encoding="utf-8")
    meta = {"id": f"{a.owner}/pbm-mrc-08-encoders-{name}", "title": f"pbm-mrc-08-encoders-{name}", "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [f"{a.owner}/pbm-mrc-proj"], "competition_sources": [],
            "kernel_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"built {d}")
    if not a.dry_run:
        subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(d)], check=True)
