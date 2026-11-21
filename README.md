# mrc-02: Indonesian extractive MRC (Proyek Akhir NLP Gasal 2026)

Extractive, SQuAD v2 style reading comprehension on Indonesian Wikipedia passages: answer with a span of the passage,
or abstain (44% of questions are unanswerable). The brief is in
[miscellaneous/docs/Proyek_Akhir_NLP_Gasal_2026__MRC.pdf](miscellaneous/docs/Proyek_Akhir_NLP_Gasal_2026__MRC.pdf).

## Layout
```
data/                 raw/, processed/ (grouped split + slice tags), splits/ (ids + hashes)
eda/                  data EDA and the cross-model per-question error browser (README.md has the scope and findings)
experiments/
  global/             shared code (package `mrc`), configs, cross-model comparison (README.md has the conventions)
  01-baseline-trivial/          always no-answer, sliding window
  02-classical-pipeline-lgbm/   traditional ML pipeline: answerability, sentence retrieval, span ranking
  03-nn-bidaf/                  BiDAF from scratch, with a no-attention ablation
  04-plm-indobert/              IndoBERT fine-tuned (Kaggle T4)
  05-llm-local-qwen/            local instruct LLM through the prompt and grounding harness
  06-llm-api-offtheshelf/       off-the-shelf API LLMs through the same harness
report/
  latex/              main.tex, references.bib
  assets/             cross-model tables and figures used by the report
miscellaneous/
  docs/               status, contract, data card, decisions, error taxonomy, requirements, the brief
  scripts/            render_latex.py
```
Each experiment folder is named `NN-<scope>-<name>` (extended as `NN-<scope>-<subscope>-<name>`). It owns its work end
to end, from its differentiating point to its error analysis. Its `summary.md` is the whole view: what it changes,
the commands, the design, generated results and error analysis, findings and open questions.

## Setup
The environment is pinned with [uv](https://docs.astral.sh/uv/): `pyproject.toml` lists the direct dependencies at the
exact versions every result was produced with, and `uv.lock` pins the full tree with hashes.
```
python -m pip install --user uv      # once (or: winget install astral-sh.uv)
uv sync                              # creates .venv with Python 3.12 and the locked packages (CPU torch)
uv sync --extra gpu                  # adds bitsandbytes for 4-bit LLMs on a CUDA machine
```
Run everything through the environment: `uv run python experiments/03-nn-bidaf/analyze.py`, or activate it
(`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere). In VS Code select `.venv` as the
interpreter and notebook kernel. Without uv: `python -m pip install -r experiments/global/requirements.txt
--extra-index-url https://download.pytorch.org/whl/cpu` (generated from the lock). Kaggle/Colab keep their preinstalled
CUDA torch and pin only `transformers==5.4.0`.

After changing a dependency in `pyproject.toml`: `uv lock`, then regenerate the pip file with
`uv export --no-hashes --no-dev --no-emit-project --format requirements-txt > experiments/global/requirements.txt`
(keep its three header lines) and commit `pyproject.toml`, `uv.lock` and the requirements file together.

Commands work from any folder. Relative paths resolve from the project root. On Windows set `PYTHONIOENCODING=utf-8`
if the console shows encoding errors. LaTeX (MiKTeX or TeX Live) is outside the Python environment.

## Pipeline
| Step | Command | Output |
|---|---|---|
| 1. Split (grouped by passage, seed 13; already done) | `python experiments/global/run.py mrc.prepare_data` | `data/processed/`, `data/splits/hashes.json` |
| 2. Slice tags | `python experiments/global/run.py mrc.analysis.slices` | `data/processed/slices_*.jsonl` |
| 3. EDA | `python eda/scripts/eda.py` (also `leakage_audit.py`, `artifact_probe.py`, `tokenizer_analysis.py`) | `eda/outputs/` |
| 4. Experiments | the commands in each `experiments/NN-*/summary.md`, section 2, ending with `analyze.py` | `runs/`, `analysis/`, `summary.md` |
| 5. Cross-model comparison | `python experiments/global/run.py mrc.analysis.compare` and `... mrc.analysis.per_example` | `report/assets/`, then `eda/notebooks/model_errors.ipynb` |
| 6. Report | `python miscellaneous/scripts/render_latex.py report/latex/main.tex report/main.pdf` | `report/main.pdf` |
| 7. Sealed internal test (once, at the end) | `python experiments/global/run.py mrc.final_test_eval --confirm --preds ...` | `experiments/global/final_test_eval_log.json` |

Score any prediction file:
`python experiments/global/run.py mrc.evaluate --gold data/processed/dev.jsonl --pred <preds.jsonl> --na-threshold-search`.

## Current dev results (no-answer threshold tuned on dev)
| Model | F1 |
|---|---|
| Always no-answer | 46.2 |
| Classical pipeline | 53.4 ± 0.2 |
| BiDAF without attention | 56.5 ± 0.6 |
| BiDAF | 56.6 ± 0.7 |
| IndoBERT | 73.5 ± 0.4 |

Full table with CIs: [report/assets/tables/main_table.md](report/assets/tables/main_table.md). Progress and next steps:
[miscellaneous/docs/STATUS.md](miscellaneous/docs/STATUS.md).

## Rules
[miscellaneous/docs/CONTRACT.md](miscellaneous/docs/CONTRACT.md). The short version: tune on dev only, 3 seeds, the
shared evaluator only, and the internal test split stays sealed until the single final evaluation. Never send test
data to an external API.

## Rendering the report
Needs MiKTeX or TeX Live on PATH. `render_latex.py` uses latexmk when it is installed (otherwise it runs
pdflatex, bibtex, pdflatex, pdflatex), keeps auxiliary files in `report/latex/build/`, and prints the first errors with
line numbers if the build fails. Options: `--engine xelatex|lualatex`, `--clean`.
