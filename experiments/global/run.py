"""Run a module of the shared `mrc` package from anywhere, e.g.

  python experiments/global/run.py mrc.analysis.compare --n-boot 1000
  python experiments/global/run.py mrc.evaluate --gold data/processed/dev.jsonl --pred <preds.jsonl> --na-threshold-search

Relative paths in the arguments resolve from the project root.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mrc.paths import at_root  # noqa: E402

if len(sys.argv) < 2 or not sys.argv[1].startswith("mrc."):
    sys.exit(__doc__)
module, sys.argv = sys.argv[1], [sys.argv[1]] + sys.argv[2:]
at_root()
runpy.run_module(module, run_name="__main__", alter_sys=True)
