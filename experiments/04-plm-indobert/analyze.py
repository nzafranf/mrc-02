"""Last stage of this experiment: error analysis of every run in runs/, written to analysis/ and summary.md.

  python experiments/04-plm-indobert/analyze.py
"""
import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
from pathlib import Path

from mrc.analysis.experiment import analyze

HERE = Path(__file__).resolve().parent
# variant name -> run directories (relative to this folder); seeds of one variant are averaged
VARIANTS = {"plm_indobert_full": "runs/plm_indobert_full/seed*"}

if __name__ == "__main__":
    analyze(HERE, VARIANTS, logit_na=["plm_indobert_full"])
