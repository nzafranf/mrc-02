# Shared body of the 08 Kaggle kernels (copied next to each kernel's run.py by push.py, because a Kaggle script
# kernel is a single file). Trains every job with experiments/04-plm-indobert/plm_qa.py.
# Setup (2026-11-24): train on the full raw train file (8042 questions, no internal dev/itest held out), choose the
# best epoch and the no-answer threshold on official_val, 10 epochs, seed 13 only (more seeds deferred).
# plm_qa.py names its selection-split outputs *_dev; they are renamed to *_official_val here.
import glob, json, os, shutil, subprocess, sys, time

TRAIN, SELECT = "data/raw/train.jsonl", "data/processed/official_val.jsonl"
BASE = dict(epochs=10, warmup=0.1)
SIZE = {"base": dict(lr="3e-5", bs=16, accum=1), "large": dict(lr="1.5e-5", bs=8, accum=2)}


def main(jobs, seeds=(13,)):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.4.0", "sentencepiece", "protobuf"],
                   check=True)
    hit = glob.glob("/kaggle/input/**/experiments/global/mrc/evaluate.py", recursive=True)
    assert hit, "dataset not mounted: " + str(os.listdir("/kaggle/input"))
    proj = hit[0].split("/experiments/global/")[0]
    work = "/kaggle/working/proj"
    shutil.copytree(proj, work, dirs_exist_ok=True)
    os.chdir(work)
    subprocess.run([sys.executable, "-c", "import torch,transformers;print(torch.__version__,transformers.__version__,"
                    "torch.cuda.get_device_name(0))"], check=True)
    status = {}
    for exp, run, model, size in jobs:
        hp = SIZE[size]
        for seed in seeds:
            out = f"/kaggle/working/runs/{exp}/{run}/seed{seed}"
            cmd = [sys.executable, "experiments/04-plm-indobert/plm_qa.py", "--device", "cuda", "--fp16",
                   "--run-name", run, "--model-name", model, "--seed", str(seed), "--epochs", str(BASE["epochs"]),
                   "--warmup-ratio", str(BASE["warmup"]), "--lr", hp["lr"], "--batch-size", str(hp["bs"]),
                   "--grad-accum", str(hp["accum"]), "--out-dir", out, "--train-file", TRAIN, "--dev-file", SELECT]
            print(" ".join(cmd), flush=True)
            t0 = time.time()
            rc = subprocess.run(cmd).returncode  # one failed job must not stop the others
            for a, b in (("preds_dev.jsonl", "preds_official_val.jsonl"), ("metrics_dev.json", "metrics_official_val.json")):
                if os.path.exists(f"{out}/{a}"):
                    os.replace(f"{out}/{a}", f"{out}/{b}")
            m = f"{out}/metrics_official_val.json"
            met = json.load(open(m)) if os.path.exists(m) else {}
            status[f"{run}/seed{seed}"] = {"rc": rc, "val_f1_at_val_thr": met.get("at_best_threshold", {}).get("f1"),
                                           "best_epoch": met.get("best_epoch"),
                                           "minutes": round((time.time() - t0) / 60, 1)}
            print("STATUS", json.dumps(status[f"{run}/seed{seed}"]), flush=True)
            json.dump(status, open("/kaggle/working/status.json", "w"), indent=1)
    shutil.rmtree(work)
    print(json.dumps(status, indent=1))
