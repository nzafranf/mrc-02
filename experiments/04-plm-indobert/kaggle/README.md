# Kaggle GPU runs for 04-plm-indobert

The T4 runs in `../runs/plm_indobert_full/` were produced this way. Needs the Kaggle CLI 2.x
(`python -m pip install kaggle`, token in `~/.kaggle/access_token` or `kaggle.json`; use `python -m kaggle` on Windows).
The dataset and kernel ids in the two metadata files belong to one team member's account. Change the owner prefix to
your own username if you run this from another account.

1. Stage the dataset (train/dev and code only; the internal test split is never uploaded):
   `python experiments/04-plm-indobert/kaggle/make_dataset.py`
2. Upload it: first time `python -m kaggle datasets create -p experiments/04-plm-indobert/kaggle/build --dir-mode zip`,
   later `python -m kaggle datasets version -p experiments/04-plm-indobert/kaggle/build --dir-mode zip -m "update"`.
3. Push the kernel: `python -m kaggle kernels push -p experiments/04-plm-indobert/kaggle`
4. Poll, then pull the outputs:
   `python -m kaggle kernels status <owner>/pbm-mrc-indobert-full`
   `python -m kaggle kernels output <owner>/pbm-mrc-indobert-full -p experiments/04-plm-indobert/kaggle/out`
   then copy `out/runs/<run-name>/` into `experiments/04-plm-indobert/runs/`.
5. Run `python experiments/04-plm-indobert/analyze.py`.

Other GPU configurations (XLM-R, frozen encoder, LoRA, train fractions) are listed in `../gpu_runs.md`. Copy `run.py`
and change the command to run them the same way.
