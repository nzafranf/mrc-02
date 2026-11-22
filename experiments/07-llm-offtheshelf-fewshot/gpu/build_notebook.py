"""Write gpu/fewshot_gpu.ipynb for a given uploaded bundle (any account can run it; it downloads the bundle by id).

  python experiments/07-llm-offtheshelf-fewshot/gpu/make_bundle.py            # 1. build gpu/build/mrc07_bundle.zip
  (upload the zip to Google Drive, share it "anyone with the link: viewer", copy its file id)
  python experiments/07-llm-offtheshelf-fewshot/gpu/build_notebook.py <file id>   # 2. bake id + SHA-256 in
The checksum is read from the local zip, so the notebook refuses any other bundle than the one built here.
"""
import hashlib
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
out = HERE / "fewshot_gpu.ipynb"
bundle_id = sys.argv[1]
bundle_sha = hashlib.sha256((HERE / "build/mrc07_bundle.zip").read_bytes()).hexdigest()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# 07 · few-shot off-the-shelf LLM on a GPU (Colab or Kaggle, any account)

Runs `run_fewshot.py` on dev or the official validation set, in the fixed question order
(`data/splits/order_<split>.txt`, global seed 13), with the frozen demos in `demos.json`.

* **Any account.** The code and data bundle is downloaded with `gdown` from a link-shared Drive file and checked
  against its SHA-256, so every account runs exactly the same bundle. No access to the project's Drive is needed.
* **Saved per question, with progress.** Each prediction is written and fsynced as soon as it is made. The run cell
  prints a progress bar with s/question and ETA after every batch; `progress.json` in the run folder has the same.
* **Resumable.** If the session dies, run all cells again with the same settings: the run continues after the last
  finished question. Colab: runs go to *your own* Drive (`MyDrive/mrc07_runs`) if you allow the mount, otherwise
  to the session disk. Kaggle: `/kaggle/working/mrc07_runs`; to resume, attach the previous version's output as an
  input (cell 3).
* **Results to Google Drive.** The last cell prints an error report per run (outcome counts from general to
  specific) and saves `<run>_<split>_<n>q_<YYYYmmdd-HHMMSS>WIB.zip` to the project's `results` folder on Drive
  (your account needs edit access to it; otherwise the zip goes to your own `MyDrive/mrc07_results`).
* **Comparable.** Every run processes the same order, so partial runs share a prefix. `LIMIT = 200` means
  "the first 200 questions of the fixed order".
* **Safe.** The bundle has train, dev and official validation only; the internal test split is never in it.

Runtime: Colab → Runtime → Change runtime type → T4 GPU. Kaggle → Accelerator GPU T4 x2 or P100, Internet on."""),
code("""# 1. Settings. Keep MODEL, VARIANT, SPLIT and RUN_NAME the same to resume a run.
MODEL = "Qwen/Qwen2.5-14B-Instruct"  # @param ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen3-14B", "meta-llama/Llama-3.1-8B-Instruct"] {allow-input: true}
VARIANT = "fs_id"                    # fs_id (Indonesian instruction) or fs_en (English), see prompts.yaml
# SPLIT: official_val (2003) is where every model is compared; dev (803) is for tuning and older comparisons
SPLIT = "official_val"  # @param ["official_val", "dev"]
LIMIT = 20                           # None = the whole split; 20 = the first 20 of the fixed order (timing run)
BIG = any(t in MODEL for t in ("12B", "13B", "14B", "15B"))  # 4-bit weights of these fill ~10 of a T4's 15 GB
BATCH_SIZE = 1 if BIG else 4         # questions per forward pass; keep it fixed within a run
SCORE_CHUNK = 1 if BIG else None     # na_prob scoring passes: 1 continuation at a time on big models (less memory)
LOAD_4BIT = True                     # bitsandbytes NF4 4-bit quantization (weights ~1/4 of fp16); keep on for 7B+
TEMPLATE_KWARGS = {}                 # set automatically for Qwen3 (thinking off); add others here
RUN_NAME = None                      # default: <model>_<variant>
RESULTS_FOLDER_ID = "1xgm78LZTuFsk35A1rD4aPgdtKWyTgQ1t"  # project Drive folder for the result zips
if "Qwen3" in MODEL and "enable_thinking" not in TEMPLATE_KWARGS:
    TEMPLATE_KWARGS["enable_thinking"] = False  # answer directly, no <think> block
GATED = MODEL.startswith(("meta-llama/", "google/gemma", "mistralai/"))  # need a Hugging Face token
RUN = RUN_NAME or f"{MODEL.split('/')[-1].replace(':', '_')}_{VARIANT}"
SAVE_TO_DRIVE = True                 # Colab: save runs to your own Google Drive (survives disconnects)

# The bundle every account uses (written by gpu/make_bundle.py, uploaded link-shared). Change both together.
BUNDLE_FILE_ID = "%s"
BUNDLE_SHA256 = "%s\"""" % (bundle_id, bundle_sha)),
code("""# 2. Download and check the bundle, choose the output folder
import glob, hashlib, json, os, pathlib, shutil, subprocess, sys, zipfile

ON_KAGGLE = os.path.exists("/kaggle/input")
ON_COLAB = "google.colab" in sys.modules
WORK = "/kaggle/working" if ON_KAGGLE else "/content" if ON_COLAB else os.path.abspath("gpu_work")
os.makedirs(WORK, exist_ok=True)
PROJ, ZIP = f"{WORK}/mrc07", f"{WORK}/mrc07_bundle.zip"

def fetch(file_id, dst):
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown
    try:
        gdown.download(id=file_id, output=dst, quiet=True)
    except Exception as e:  # fallback without gdown
        print("gdown failed (", type(e).__name__, "), trying a direct download")
        import requests
        r = requests.get("https://drive.usercontent.google.com/download",
                         params={"id": file_id, "export": "download", "confirm": "t"}, timeout=120)
        r.raise_for_status()
        open(dst, "wb").write(r.content)

sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if not (os.path.exists(ZIP) and sha(ZIP) == BUNDLE_SHA256):
    fetch(BUNDLE_FILE_ID, ZIP)
got = sha(ZIP)
assert got == BUNDLE_SHA256, (f"bundle checksum mismatch ({got[:12]} != {BUNDLE_SHA256[:12]}): the Drive file is "
                              "not shared 'anyone with the link', or it was replaced without updating cell 1")
shutil.rmtree(PROJ, ignore_errors=True)
zipfile.ZipFile(ZIP).extractall(PROJ)
assert not glob.glob(f"{PROJ}/data/**/*test*.jsonl", recursive=True), "test data found in the bundle"

OUT_ROOT = f"{WORK}/mrc07_runs"
if ON_COLAB and SAVE_TO_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        OUT_ROOT = "/content/drive/MyDrive/mrc07_runs"
    except Exception as e:
        print("Drive not mounted (", type(e).__name__, "): saving to the session disk; download the zip at the end")
os.makedirs(OUT_ROOT, exist_ok=True)
print("bundle ok", got[:12], "| project:", PROJ, "| runs saved to:", OUT_ROOT)"""),
code("""# 3. Kaggle only: continue a run from an earlier version's output (attach that output as an input first)
if ON_KAGGLE:
    for old in glob.glob("/kaggle/input/**/mrc07_runs/*/*/config.json", recursive=True):
        src = pathlib.Path(old).parent
        dst = pathlib.Path(OUT_ROOT) / src.parent.name / src.name
        if not dst.exists():
            shutil.copytree(src, dst)
            print("resuming from", src)"""),
code("""# 4. Packages. Colab and Kaggle keep their CUDA torch; transformers is pinned to the version of the other runs.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.4.0", "accelerate==1.13.0",
                "bitsandbytes"], check=True)
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| GPU",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"""),
code("""# 5. Hugging Face token, needed only for gated models (Llama, Gemma). Never printed or written to disk.
#    Colab: key icon > add secret HF_TOKEN > switch on "Notebook access". Kaggle: Add-ons > Secrets.
token = os.environ.get("HF_TOKEN")
try:
    if not token and ON_COLAB:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    elif not token and ON_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:  # the message names the problem, never the value
    print(f"secret HF_TOKEN not readable: {type(e).__name__}: {e}")
    if ON_COLAB:
        print("  If the title bar says 'Cannot save changes' you are in someone else's notebook: File > Save a copy"
              " in Drive, open the copy, and grant the secret to it.")
if not token and GATED:
    from getpass import getpass
    token = getpass(f"{MODEL} is gated. Hugging Face token (hidden): ").strip() or None
if token:
    os.environ["HF_TOKEN"] = token
print("HF_TOKEN set" if token else "no HF_TOKEN; fine for ungated models such as Qwen")"""),
code("""# 6. Run. Progress bar + ETA after every batch; every prediction is saved immediately.
#    Rerun this cell (or all cells after a disconnect) to resume.
cmd = [sys.executable, "-u", "experiments/07-llm-offtheshelf-fewshot/run_fewshot.py", "--model", MODEL,
       "--variant", VARIANT, "--split", SPLIT, "--batch-size", str(BATCH_SIZE),
       "--template-kwargs", json.dumps(TEMPLATE_KWARGS), "--out-root", OUT_ROOT]
cmd += ["--load-4bit"] if LOAD_4BIT else []
cmd += ["--limit", str(LIMIT)] if LIMIT else []
cmd += ["--run-name", RUN]
cmd += ["--score-chunk", str(SCORE_CHUNK)] if SCORE_CHUNK else []
env = dict(os.environ, PYTHONIOENCODING="utf-8", TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")  # less fragmentation; an out-of-memory batch is redone 1 by 1
p = subprocess.Popen(cmd, cwd=PROJ, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    if "Warning" not in line and "warn(" not in line:
        print(line, end="", flush=True)
assert p.wait() == 0, "run failed; rerun this cell to resume, or read the log above\""""),
code("""# 7. Error report (general -> specific) for this run, then a timestamped zip of it saved to Google Drive
import datetime as dt
from zoneinfo import ZoneInfo
run_dir = pathlib.Path(OUT_ROOT) / SPLIT / RUN
subprocess.run([sys.executable, "experiments/07-llm-offtheshelf-fewshot/report_fewshot.py", str(run_dir)],
               cwd=PROJ, env=env, check=True)
done = json.loads((run_dir / "progress.json").read_text())["done"]
stamp = dt.datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y%m%d-%H%M%S") + "WIB"
zipped = shutil.make_archive(f"{WORK}/{RUN}_{SPLIT}_{done}q_{stamp}", "zip", root_dir=OUT_ROOT,
                             base_dir=f"{SPLIT}/{RUN}")  # keeps the <split>/<run>/ layout
saved = None
if ON_COLAB and RESULTS_FOLDER_ID:
    try:
        from google.colab import auth
        auth.authenticate_user()
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        f = build("drive", "v3").files().create(
            body={"name": os.path.basename(zipped), "parents": [RESULTS_FOLDER_ID]},
            media_body=MediaFileUpload(zipped, mimetype="application/zip"), fields="id,webViewLink",
            supportsAllDrives=True).execute()
        saved = f["webViewLink"]
    except Exception as e:
        print(f"could not upload to the project results folder ({type(e).__name__}); this account probably has no"
              " edit access to it. Saving to your own Drive instead.")
if not saved and OUT_ROOT.startswith("/content/drive"):
    dst = "/content/drive/MyDrive/mrc07_results"
    os.makedirs(dst, exist_ok=True)
    saved = shutil.copy(zipped, dst)
print("result saved:", saved or zipped)"""),
md("""### Afterwards
The zip is in the project's Drive `results` folder (or your `MyDrive/mrc07_results`). On the project machine unzip it into
`experiments/07-llm-offtheshelf-fewshot/runs/` (it keeps the `<split>/<run>/` layout) and run
`python experiments/07-llm-offtheshelf-fewshot/analyze.py` (and `--split official_val`). Keep the
`.partial.jsonl` file: it is what a later session resumes from."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                 "language_info": {"name": "python"},
                                                 "accelerator": "GPU", "colab": {"gpuType": "T4"}})
nbf.write(nb, out)
print("wrote", out)
