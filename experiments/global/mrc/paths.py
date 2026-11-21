"""Project paths. Everything else imports from here instead of counting parents of __file__.

Layout (project root = three levels above this file):
  data/                      raw, processed, splits
  eda/                       data analysis notebooks and scripts
  experiments/global/        this package (`mrc`), shared configs, cross-model runs of the analysis
  experiments/NN-<name>/     one end-to-end experiment each: code, runs/, analysis/, summary.md
  report/assets/{tables,figures}   cross-model and EDA outputs used by report/latex
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA, RAW, PROC, SPLITS = ROOT / "data", ROOT / "data/raw", ROOT / "data/processed", ROOT / "data/splits"
EXP = ROOT / "experiments"
GLOBAL = EXP / "global"
CONFIGS = GLOBAL / "configs"
ASSETS = ROOT / "report/assets"
TABLES, FIGURES = ASSETS / "tables", ASSETS / "figures"
DOCS = ROOT / "miscellaneous/docs"
EXPERIMENTS = {
    "baselines": EXP / "01-baseline-trivial",
    "classical": EXP / "02-classical-pipeline-lgbm",
    "bidaf": EXP / "03-nn-bidaf",
    "plm": EXP / "04-plm-indobert",
    "llm_local": EXP / "05-llm-local-qwen",
    "llm_api": EXP / "06-llm-api-offtheshelf",
}


def at_root() -> None:
    """Make relative paths (CLI defaults like data/processed/dev.jsonl) resolve from the project root."""
    os.chdir(ROOT)
