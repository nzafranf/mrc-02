# Kaggle GPU kernel: IndoBERT-base-p1 full fine-tune, seeds 13/42/87. Pushed by kaggle/README.md step 3.
# Each seed also predicts the official validation set (preds_official_val.jsonl) and saves checkpoint.pt so the
# official test set can be predicted later with --predict-only, without retraining.
# The dataset (make_dataset.py) mirrors the project layout: experiments/global, this experiment, data/processed.
import glob, json, os, shutil, subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.4.0"], check=True)
hit = glob.glob("/kaggle/input/**/experiments/global/mrc/evaluate.py", recursive=True)
assert hit, "dataset not mounted: " + str(os.listdir("/kaggle/input"))
proj = hit[0].split("/experiments/global/")[0]
work = "/kaggle/working/proj"
shutil.copytree(proj, work, dirs_exist_ok=True)
os.chdir(work)
subprocess.run([sys.executable, "-c", "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.get_device_name(0))"], check=True)
RUN = "plm_indobert_full"
for seed in (13, 42, 87):
    cmd = [sys.executable, "experiments/04-plm-indobert/plm_qa.py", "--device", "cuda", "--fp16", "--run-name", RUN,
           "--model-name", "indobenchmark/indobert-base-p1", "--seed", str(seed), "--epochs", "3", "--lr", "3e-5", "--batch-size", "16",
           "--save-checkpoint", "--also-predict", "data/processed/official_val.jsonl"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
out = "/kaggle/working/runs"
shutil.copytree(f"{work}/experiments/04-plm-indobert/runs", out, dirs_exist_ok=True)
shutil.rmtree(work)
print(json.dumps({s: json.load(open(f"{out}/{RUN}/seed{s}/metrics_dev.json")).get("at_best_threshold", {}).get("f1") for s in (13, 42, 87)}))
