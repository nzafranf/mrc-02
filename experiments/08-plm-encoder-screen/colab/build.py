"""OUTDATED since 2026-11-24 (08 moved to full train + official_val selection, 10 epochs, seed 13; see kaggle/runner.py).
Build the Colab path for 08: a code+data bundle and a notebook that runs the same jobs as the Kaggle kernels.

  python experiments/08-plm-encoder-screen/colab/build.py
Writes colab/build/mrc08_bundle.zip (shared code, plm_qa.py, train/dev/official_val; never itest or test) and
colab/encoders_colab.ipynb with the bundle's SHA-256 baked in. Put both in the Drive folder proyek_akhir/mrc08 and
open the notebook in Colab (T4). Each finished seed is copied to mrc08/runs/ on Drive and skipped on a rerun, so a
cut session resumes. Jobs and hyperparameters come from kaggle/push.py and kaggle/runner.py, so both paths match.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "kaggle"))
import runner  # noqa: E402  (SIZE, BASE)

# push.py builds on import, so read its job table textually instead of importing it
src = (HERE.parent / "kaggle/push.py").read_text(encoding="utf-8")
ns: dict = {}
exec(src[src.index("KERNELS = {"):src.index("\n}\n") + 3], ns)
# Colab order: the existing reference model first, then the likeliest winners, then the rest
ORDER = ["plm_indobert_full_r2", "mdeberta_v3_base", "xlmr_large", "nusabert_large", "albert_indolite_large",
         "electra_indo_base"]
jobs = {j[1]: j for k in ns["KERNELS"].values() for j in k}
JOBS = [list(jobs[r]) for r in ORDER]

build = HERE / "build"
if build.exists():
    shutil.rmtree(build)
subprocess.run([sys.executable, str(ROOT / "experiments/04-plm-indobert/kaggle/make_dataset.py"), "--out",
                str(build / "stage")], check=True)
zpath = build / "mrc08_bundle.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted((build / "stage/proj").rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(build / "stage"))
    names = z.namelist()
assert not [n for n in names if "itest" in n or n.endswith("test.jsonl")], "test data must not be bundled"
sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
shutil.rmtree(build / "stage")


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip() + "\n"}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip() + "\n"}


cells = [
    md("""# 08 · encoder screen on Colab (T4)
Runs `experiments/04-plm-indobert/plm_qa.py` for each encoder and seed, exactly as the Kaggle kernels do
(`experiments/08-plm-encoder-screen/kaggle/`). Each run fine-tunes on train, picks the epoch and the no-answer
threshold on dev, and predicts `official_val`. Finished seeds are copied to `mrc08/runs/` on Drive and skipped on a
rerun, so after a disconnect just run all cells again. Runtime: base models about 7 min per seed, large about 25.

Put `mrc08_bundle.zip` in the same Drive folder as this notebook (`proyek_akhir/mrc08`)."""),
    code(f"""
# Which jobs to run. Remove entries to run fewer; the order is the run order.
JOBS = {json.dumps(JOBS, indent=1)}
SEEDS = [13, 42, 87]
SIZE = {json.dumps(runner.SIZE)}
BASE = {json.dumps(runner.BASE)}
BUNDLE_SHA256 = "{sha}"
"""),
    code("""
import glob, hashlib, json, os, shutil, subprocess, sys, time, zipfile
from google.colab import drive
drive.mount("/content/drive")
cands = [p for pat in ("/content/drive/MyDrive/mrc08", "/content/drive/MyDrive/*/mrc08", "/content/drive/MyDrive/*/*/mrc08",
                       "/content/drive/MyDrive/*/*/*/mrc08", "/content/drive/MyDrive/*/*/*/*/mrc08")
         for p in glob.glob(pat) if os.path.exists(p + "/mrc08_bundle.zip")]
assert cands, "folder mrc08 with mrc08_bundle.zip not found under MyDrive (add a shortcut to My Drive if it is shared)"
DRV = cands[0]; print("Drive folder:", DRV)
z = DRV + "/mrc08_bundle.zip"
assert hashlib.sha256(open(z, "rb").read()).hexdigest() == BUNDLE_SHA256, "bundle hash differs: rebuild the notebook"
shutil.rmtree("/content/proj", ignore_errors=True)
zipfile.ZipFile(z).extractall("/content")
os.chdir("/content/proj")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.4.0", "sentencepiece", "protobuf"], check=True)
import torch, transformers
print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))
"""),
    code("""
status_path = DRV + "/runs/status.json"
os.makedirs(DRV + "/runs", exist_ok=True)
status = json.load(open(status_path)) if os.path.exists(status_path) else {}
todo = [(j, s) for j in JOBS for s in SEEDS]
for n, ((exp, run, model, size), seed) in enumerate(todo, 1):
    dest = f"{DRV}/runs/{exp}/{run}/seed{seed}"
    if os.path.exists(dest + "/metrics_dev.json") and os.path.exists(dest + "/preds_official_val.jsonl"):
        print(f"[{n}/{len(todo)}] skip {run} seed{seed} (on Drive)"); continue
    hp, out = SIZE[size], f"/content/runs/{exp}/{run}/seed{seed}"
    cmd = [sys.executable, "experiments/04-plm-indobert/plm_qa.py", "--device", "cuda", "--fp16", "--run-name", run,
           "--model-name", model, "--seed", str(seed), "--epochs", str(BASE["epochs"]), "--warmup-ratio", str(BASE["warmup"]),
           "--lr", hp["lr"], "--batch-size", str(hp["bs"]), "--grad-accum", str(hp["accum"]), "--out-dir", out,
           "--predict-file", "data/processed/official_val.jsonl"]
    print(f"[{n}/{len(todo)}]", " ".join(cmd[2:]), flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:  # keep the cell output short: epoch summaries, errors, and every 50th step
        if line.startswith(("==", "best epoch", "wrote", "train examples", "params", "Traceback")) or "Error" in line \\
                or (" step " in line and int(line.split(" step ")[1].split("/")[0]) % 50 == 0):
            print("   ", line.rstrip(), flush=True)
    rc = p.wait()
    m = out + "/metrics_dev.json"
    f1 = json.load(open(m))["at_best_threshold"]["f1"] if os.path.exists(m) else None
    status[f"{run}/seed{seed}"] = {"rc": rc, "dev_f1": f1, "minutes": round((time.time() - t0) / 60, 1),
                                   "gpu": torch.cuda.get_device_name(0)}
    print("    STATUS", status[f"{run}/seed{seed}"], flush=True)
    if rc == 0:
        shutil.copytree(out, dest, dirs_exist_ok=True)
    json.dump(status, open(status_path, "w"), indent=1)
print(json.dumps(status, indent=1))
"""),
    code("""
# Zip everything finished so far (Drive: mrc08/results/), for pulling into experiments/<exp>/runs/ locally
stamp = time.strftime("%Y%m%d-%H%M")
os.makedirs(DRV + "/results", exist_ok=True)
print(shutil.make_archive(f"{DRV}/results/mrc08_runs_{stamp}", "zip", DRV + "/runs"))
"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
(HERE / "encoders_colab.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"bundle {zpath} ({zpath.stat().st_size / 1e6:.1f} MB) sha256 {sha}")
print(f"notebook {HERE / 'encoders_colab.ipynb'}; jobs: {[j[1] for j in JOBS]}")
