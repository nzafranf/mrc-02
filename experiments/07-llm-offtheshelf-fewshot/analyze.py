"""Last stage of this experiment: error analysis of every run in runs/<split>/, written to analysis/ and summary.md.

  python experiments/07-llm-offtheshelf-fewshot/analyze.py                      # dev  -> analysis/
  python experiments/07-llm-offtheshelf-fewshot/analyze.py --split official_val # -> analysis/official_val/

Runs may be partial (they stop at different points of the fixed order); the tables report n, and comparisons use
the questions all compared runs share, which is a prefix of the fixed order.
"""
import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import argparse
from pathlib import Path

from mrc.analysis.experiment import analyze
from mrc.eval_order import SPLIT_FILES

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(SPLIT_FILES), default="dev")
    a = ap.parse_args()
    runs = HERE / "runs" / a.split
    # variant name -> run directory (one run per variant: greedy decoding, no seeds)
    VARIANTS = {p.name: f"runs/{a.split}/{p.name}" for p in sorted(runs.iterdir()) if p.is_dir()} if runs.exists() else {}
    if not VARIANTS:
        raise SystemExit(f"no runs in {runs}")
    analyze(HERE, VARIANTS, split_file=SPLIT_FILES[a.split])
