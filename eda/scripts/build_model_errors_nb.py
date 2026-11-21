"""Builds eda/notebooks/model_errors.ipynb (run from anywhere)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Model failure browser (dev)

Every dev example with the full passage, gold answers, slice tags and every model's prediction per seed.
Source: `report/assets/tables/per_example_all_models.jsonl`, built by `python experiments/global/run.py mrc.analysis.per_example`
(rerun it after adding a model to `experiments/global/configs/models_registry.yaml`).

Thresholds are tuned per run on dev (same as `report/assets/tables/main_table.md`), so scores are dev-optimistic.
Partial API runs (`llm_*_api`) cover only the first 22 to 25 dev ids and are untuned.
`itest` is not loaded.

Scope: this notebook compares models question by question. Per-experiment analysis (outcome tables, slices, seed stability, calibration, error samples for manual labels) is in `experiments/NN-*/analysis/` and each `summary.md`; aggregated cross-model tables are in `report/assets/tables/`. See `eda/README.md`.""")

code('''import json, re, html, collections
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import HTML, display

pd.set_option("display.max_colwidth", 80)

def find_root(p=Path.cwd()):
    for d in [p, *p.parents]:
        if (d / "report/assets/tables/per_example_all_models.jsonl").exists():
            return d
    raise FileNotFoundError("run: python experiments/global/run.py mrc.analysis.per_example")

ROOT = find_root()
R = [json.loads(l) for l in open(ROOT / "report/assets/tables/per_example_all_models.jsonl", encoding="utf-8")]
BY_ID = {r["id"]: r for r in R}
MODELS = list(R[0]["models"])
TRIVIAL = ["no_answer", "sliding_window", "sliding_window_thr"]
MAIN = [m for m in MODELS if m not in TRIVIAL]
print(len(R), "dev examples; models:", MODELS)''')

code('''# Flat table: one row per example, per-model mean F1 (over seeds) and seed13 category / pred
rows = []
for r in R:
    row = {"id": r["id"], "question": r["question"], "is_impossible": r["is_impossible"],
           "gold": " | ".join(a["text"] for a in r["answers"]) or "<no answer>", **r["slices"]}
    row["ends_space_q"] = r["question"].endswith(" ?")
    for m, v in r["models"].items():
        s = v["seeds"].get("seed13") or next(iter(v["seeds"].values()))
        row[f"{m}.f1"] = v["mean_f1"]
        row[f"{m}.cat"] = s["category"]
        row[f"{m}.pred"] = s["pred"] or "<no answer>"
    for m, v in r["partial_models"].items():
        row[f"{m}.f1"] = v["f1"]; row[f"{m}.cat"] = v["category"]; row[f"{m}.pred"] = v["pred"] or "<no answer>"
    rows.append(row)
df = pd.DataFrame(rows).set_index("id")
df[["question", "gold"] + [f"{m}.f1" for m in MAIN]].head()''')

md("## Overview\nMean F1 per model, error category counts (seed 13), and pairwise agreement.")

code('''display(df[[f"{m}.f1" for m in MODELS]].mean().mul(100).round(2).rename("F1 (mean over seeds)").to_frame())
cats = pd.DataFrame({m: df[f"{m}.cat"].value_counts() for m in MAIN}).fillna(0).astype(int)
cats''')

code('''# Pairwise: how often A is right (mean F1 >= 0.5) where B is wrong, and vice versa
ok = {m: df[f"{m}.f1"] >= 0.5 for m in MAIN}
pair = pd.DataFrame({a: {b: int((ok[a] & ~ok[b]).sum()) for b in MAIN} for a in MAIN}).T
print("cell [A, B] = #examples A gets right and B gets wrong")
pair''')

md("""## Rendering helpers

- **green**: gold answer span(s) in the context
- **yellow**: context word that also appears in the question
- **orange** (in the question): question word that does not occur in the context
- Under the passage, each model's prediction per seed, shown in a window of the passage with the predicted span **boxed**.""")

code('''GREEN, YELLOW, ORANGE = "#8be28b", "#fff176", "#ffb74d"
WORD = re.compile(r"\\w+", re.UNICODE)
CAT_COLOR = {"correct": "#2e7d32", "false_no_answer": "#c62828", "false_answer_on_unanswerable": "#ad1457",
             "boundary_too_long": "#ef6c00", "boundary_too_short": "#f9a825", "partial_overlap": "#8d6e63",
             "wrong_span_same_sentence": "#6a1b9a", "wrong_sentence": "#283593"}

def _runs(text, color):
    out, i = [], 0
    while i < len(text):
        j = i
        while j < len(text) and color[j] == color[i]:
            j += 1
        seg = html.escape(text[i:j])
        out.append(f'<mark style="background:{color[i]};color:#000">{seg}</mark>' if color[i] else seg)
        i = j
    return "".join(out)

def highlight_context(r):
    ctx, q = r["context"], r["question"]
    q_words = {w.lower() for w in WORD.findall(q)}
    color = [None] * len(ctx)
    for m in WORD.finditer(ctx):
        if m.group().lower() in q_words:
            color[m.start():m.end()] = [YELLOW] * (m.end() - m.start())
    for a in r["answers"]:
        st = a["answer_start"]
        for k in range(st, min(st + len(a["text"]), len(ctx))):
            color[k] = GREEN
    return _runs(ctx, color)

def highlight_question(r):
    c_words = {w.lower() for w in WORD.findall(r["context"])}
    q, out, last = r["question"], [], 0
    for m in WORD.finditer(q):
        out.append(html.escape(q[last:m.start()]))
        w = html.escape(m.group())
        out.append(f'<mark style="background:{ORANGE};color:#000">{w}</mark>' if m.group().lower() not in c_words else w)
        last = m.end()
    out.append(html.escape(q[last:]))
    return "".join(out)

def span_window(ctx, pred, start, w=70):
    if not pred:
        return "<i>no answer</i>"
    if start is None or start < 0 or ctx[start:start + len(pred)] != pred:
        start = ctx.find(pred)
    if start < 0:
        return f"<i>not a passage span:</i> {html.escape(pred)}"
    a, b = max(0, start - w), min(len(ctx), start + len(pred) + w)
    return (("..." if a else "") + html.escape(ctx[a:start])
            + f'<span style="outline:2px solid #1565c0;background:#e3f2fd;color:#000">{html.escape(pred)}</span>'
            + html.escape(ctx[start + len(pred):b]) + ("..." if b < len(ctx) else ""))

def _badge(cat):
    return f'<span style="color:#fff;background:{CAT_COLOR.get(cat, "#555")};padding:0 4px;border-radius:3px;font-size:85%">{cat}</span>'

def show(ids, models=None, seeds=("seed13",), partial=True):
    """Render full instances. ids: list of ids or a DataFrame/Index. seeds=None shows all seeds."""
    if isinstance(ids, (pd.DataFrame, pd.Series)):
        ids = ids.index
    models = models or MAIN
    parts = []
    for n, i in enumerate(ids, 1):
        r = BY_ID[i]
        tag = "UNANSWERABLE" if r["is_impossible"] else "answerable"
        gold = " | ".join(a["text"] for a in r["answers"]) or "&lt;no answer&gt;"
        sl = r["slices"]
        meta = ", ".join(f"{k}={sl.get(k)}" for k in ("q_type", "ans_type", "ans_len_bucket", "lex_overlap_bucket", "ctx_len_bucket") if k in sl)
        lines = []
        for m in models:
            for s, v in r["models"][m]["seeds"].items():
                if seeds and s not in seeds:
                    continue
                lines.append(f'<tr><td><b>{m}</b> <small>{s}</small></td><td>{_badge(v["category"])} '
                             f'<small>F1 {v["f1"]:.2f}</small></td><td>{span_window(r["context"], v["pred"], v["start"])}</td></tr>')
        if partial:
            for m, v in r["partial_models"].items():
                raw = html.escape((v.get("raw_output") or "")[:80])
                lines.append(f'<tr><td><b>{m}</b> <small>{v.get("grounding")}</small></td><td>{_badge(v["category"])} '
                             f'<small>F1 {v["f1"]:.2f}</small></td><td>{span_window(r["context"], v["pred"], v["start"])}'
                             f'<br><small>raw: {raw}</small></td></tr>')
        parts.append(
            f'<div style="margin:0 0 16px;padding:8px;border:1px solid #888;border-radius:6px">'
            f'<b>{n}. {highlight_question(r)}</b> <i>({tag})</i> <small><code>{i}</code></small><br>'
            f'<small>gold: {html.escape(gold) if r["answers"] else gold} &nbsp; [{html.escape(meta)}]</small>'
            f'<div style="margin:6px 0">{highlight_context(r)}</div>'
            f'<table style="font-size:90%;text-align:left">{"".join(lines)}</table></div>')
    display(HTML("".join(parts)))''')

md("""## Side-by-side model answers

One row per example, one column per model (seed 13; partial API LLM runs where available). Cell colour = F1 of that answer
(green 1.0, through yellow, to red 0.0); grey italic = the model abstained. The passage is collapsed under each question.

**Ordering:** examples where *every* model gave an answer (no abstention) come first, because they show competing spans
directly. Priority tiers:
1. all main models **and** both API LLMs answered (only possible on the ~22 ids the API runs cover)
2. all main models answered
3. the rest, by number of models that answered""")

code('''API = ["llm_nemotron35_api", "llm_gemini35flash_api"]

def answers_of(r, models=MAIN, partial=API):
    out = {}
    for m in models:
        v = r["models"][m]["seeds"].get("seed13") or next(iter(r["models"][m]["seeds"].values()))
        out[m] = v
    for m in partial:
        if m in r["partial_models"]:
            out[m] = r["partial_models"][m]
    return out

def tier(r):
    a = answers_of(r)
    main_ans = all(a[m]["pred"] for m in MAIN)
    api_ans = all(m in a and a[m]["pred"] for m in API)
    n_ans = sum(bool(v["pred"]) for v in a.values())
    return (1 if main_ans and api_ans else 2 if main_ans else 3), -n_ans

PRIORITY = sorted(BY_ID, key=lambda i: tier(BY_ID[i]))
tiers = pd.Series([tier(BY_ID[i])[0] for i in PRIORITY]).value_counts().sort_index()
print("examples per tier:", tiers.to_dict())

def _f1_color(f):
    # red (0) -> yellow (0.5) -> green (1)
    r_, g_ = (230, int(80 + 340 * f)) if f < .5 else (int(230 - 300 * (f - .5)), 200)
    return f"rgb({min(r_,255)},{min(g_,255)},110)"

def side_by_side(ids, models=None, show_context=True):
    """Table: rows = examples, columns = model answers. ids: list, Index or DataFrame."""
    if isinstance(ids, (pd.DataFrame, pd.Series)):
        ids = ids.index
    cols = models or MAIN + API
    head = "".join(f'<th style="min-width:120px">{m}</th>' for m in cols)
    body = []
    for n, i in enumerate(ids, 1):
        r = BY_ID[i]
        a = answers_of(r)
        gold = " | ".join(x["text"] for x in r["answers"]) or "<no answer>"
        q = highlight_question(r)
        if show_context:
            q += f'<details><summary><small>passage</small></summary><div style="font-size:85%">{highlight_context(r)}</div></details>'
        cells_ = []
        for m in cols:
            if m not in a:
                cells_.append('<td style="color:#999">n/a</td>')
            elif not a[m]["pred"]:
                cells_.append(f'<td style="background:{_f1_color(a[m]["f1"])};color:#333"><i>(no answer)</i></td>')
            else:
                cells_.append(f'<td style="background:{_f1_color(a[m]["f1"])};color:#000" title="{a[m]["category"]}">'
                              f'{html.escape(a[m]["pred"])}<br><small>{a[m]["category"]}</small></td>')
        body.append(f'<tr><td>{n}</td><td style="max-width:360px">{q}<br><small><code>{i}</code> tier {tier(r)[0]}</small></td>'
                    f'<td><b>{html.escape(gold)}</b></td>{"".join(cells_)}</tr>')
    display(HTML('<table style="border-collapse:collapse;font-size:90%;text-align:left" border="1">'
                 f'<tr><th>#</th><th>question</th><th>gold</th>{head}</tr>{"".join(body)}</table>'))''')

code('''# Tier 1 and 2 first: every main model answered (API LLMs too where they ran)
side_by_side(PRIORITY[:25])''')

code('''# Only the disagreements among examples where all main models answered: at least one right and one wrong
all_answered = [i for i in PRIORITY if tier(BY_ID[i])[0] <= 2]
def f1s(i): return [answers_of(BY_ID[i])[m]["f1"] for m in MAIN]
disagree = [i for i in all_answered if max(f1s(i)) >= .5 and min(f1s(i)) < .5]
print(len(all_answered), "all-answered;", len(disagree), "with disagreement")
side_by_side(disagree[:20])''')

md("""## Error-type confusion
Rows and columns are the error categories from `mrc.analysis.compare.classify`
(`correct`, `false_no_answer`, `false_answer_on_unanswerable`, `wrong_sentence`, `wrong_span_same_sentence`,
`boundary_too_long`, `boundary_too_short`, `partial_overlap`). Cell [a, b] counts dev examples where model A made
error type a and model B made error type b on the same question (seed 13 unless stated).

- Large off-diagonal cells in a row show what A's errors turn into under B (for example, BiDAF `false_no_answer` that
  IndoBERT gets `correct` is what pretraining fixes).
- The diagonal minus `correct` counts shared errors, which point at the data (hard or mislabeled questions) rather than the model.
- `cm_ids(a, b, cat_a, cat_b)` returns the ids behind a cell; pass them to `side_by_side` or `show`.""")

code('''CATS = ["correct", "false_no_answer", "false_answer_on_unanswerable", "wrong_sentence",
        "wrong_span_same_sentence", "boundary_too_long", "boundary_too_short", "partial_overlap"]
SHORT = {"correct": "correct", "false_no_answer": "false_noans", "false_answer_on_unanswerable": "false_ans_unans",
         "wrong_sentence": "wrong_sent", "wrong_span_same_sentence": "wrong_span_same_sent",
         "boundary_too_long": "too_long", "boundary_too_short": "too_short", "partial_overlap": "partial"}

def cat_of(r, m, seed="seed13"):
    if m in r["models"]:
        s = r["models"][m]["seeds"]
        return (s.get(seed) or next(iter(s.values())))["category"]
    p = r["partial_models"].get(m)
    return p["category"] if p else None

def confusion(a, b, seed_a="seed13", seed_b="seed13", ids=None):
    """Category crosstab: rows = model a, columns = model b, on the examples both cover."""
    pairs = [(cat_of(r, a, seed_a), cat_of(r, b, seed_b)) for r in R if ids is None or r["id"] in ids]
    pairs = [p for p in pairs if None not in p]
    t = pd.crosstab(pd.Series([p[0] for p in pairs], name=f"{a} ({seed_a})"),
                    pd.Series([p[1] for p in pairs], name=f"{b} ({seed_b})"))
    t = t.reindex(index=CATS, columns=CATS, fill_value=0)
    t.index, t.columns = [SHORT[c] for c in t.index], [SHORT[c] for c in t.columns]
    return t

def show_cm(t, title=""):
    shared = sum(t.iat[k, k] for k in range(1, len(t)))
    sty = (t.style.background_gradient(cmap="Oranges", axis=None)
           .set_caption(f"{title}  (n={t.values.sum()}, shared errors on diagonal={shared})"))
    display(sty)

def cm_ids(a, b, cat_a, cat_b, seed_a="seed13", seed_b="seed13"):
    long = {v: k for k, v in SHORT.items()}
    cat_a, cat_b = long.get(cat_a, cat_a), long.get(cat_b, cat_b)
    return [r["id"] for r in R if cat_of(r, a, seed_a) == cat_a and cat_of(r, b, seed_b) == cat_b]''')

md("""### Outcome table: outcome type (rows) by model (columns)
Rows are grouped by gold: answerable questions end in exactly one of the answerable outcomes, unanswerable ones in
one of the two unanswerable outcomes, so each model's column sums to the group size. Values are the mean count over
seeds (std in the second table); the percent view divides by the group size.""")

code('''OUTCOMES = [("answerable", "correct_ans"), ("answerable", "false_no_answer"), ("answerable", "wrong_sentence"),
            ("answerable", "wrong_span_same_sentence"), ("answerable", "boundary_too_long"),
            ("answerable", "boundary_too_short"), ("answerable", "partial_overlap"),
            ("unanswerable", "correct_no_ans"), ("unanswerable", "false_answer_on_unanswerable")]

def outcome(r, cat):
    if cat == "correct":
        return "correct_no_ans" if r["is_impossible"] else "correct_ans"
    return cat

def outcome_table(models=MAIN, ids=None):
    """Rows (gold, outcome), columns model; mean and std of counts over each model's seeds."""
    rs = [r for r in R if ids is None or r["id"] in ids]
    mean, std = {}, {}
    for m in models:
        per_seed = []
        for seed in rs[0]["models"][m]["seeds"]:
            c = collections.Counter(("unanswerable" if r["is_impossible"] else "answerable",
                                     outcome(r, r["models"][m]["seeds"][seed]["category"])) for r in rs)
            per_seed.append(pd.Series({k: c.get(k, 0) for k in OUTCOMES}))
        t = pd.concat(per_seed, axis=1)
        mean[m], std[m] = t.mean(axis=1), t.std(axis=1, ddof=0)
    idx = pd.MultiIndex.from_tuples(OUTCOMES, names=["gold", "outcome"])
    mean, std = pd.DataFrame(mean).reindex(idx), pd.DataFrame(std).reindex(idx)
    sizes = mean.groupby(level="gold").transform("sum")
    return mean, std, mean / sizes

mean_t, std_t, pct_t = outcome_table(["no_answer"] + MAIN)
assert (mean_t.groupby(level="gold").sum().round(6).nunique(axis=1) == 1).all(), "columns must sum to the group sizes"
display(mean_t.round(1).style.format("{:.1f}").background_gradient(cmap="Oranges", axis=None)
        .set_caption("mean count over seeds"))
display(pct_t.style.format("{:.1%}").background_gradient(cmap="Oranges", axis=None)
        .set_caption("share of the gold group (answerable n=%d, unanswerable n=%d)"
                     % tuple(mean_t.groupby(level="gold").sum().iloc[:, 0])))
display(std_t.style.format("{:.1f}").set_caption("std of counts over seeds (seed noise)"))''')

code('''# Partial API LLM runs: same table on the ids they cover, untuned (n is tiny, read as anecdotes)
for m in API:
    rs = [r for r in R if m in r["partial_models"]]
    if not rs:
        continue
    c = collections.Counter(("unanswerable" if r["is_impossible"] else "answerable",
                             outcome(r, r["partial_models"][m]["category"])) for r in rs)
    base = {mm: collections.Counter(("unanswerable" if r["is_impossible"] else "answerable",
                                     outcome(r, cat_of(r, mm))) for r in rs) for mm in MAIN}
    t = pd.DataFrame({**{mm: [base[mm].get(k, 0) for k in OUTCOMES] for mm in MAIN},
                      m: [c.get(k, 0) for k in OUTCOMES]},
                     index=pd.MultiIndex.from_tuples(OUTCOMES, names=["gold", "outcome"]))
    display(t.style.background_gradient(cmap="Oranges", axis=None)
            .set_caption(f"{m} vs main models (seed 13) on the same {len(rs)} ids"))''')

code('''# The main contrasts: what pretraining changes, what attention changes, classical vs neural
for a, b in [("bidaf", "plm_indobert_full"), ("bidaf_noattn", "bidaf"), ("classical_pipeline", "bidaf"),
             ("classical_pipeline", "plm_indobert_full")]:
    show_cm(confusion(a, b), f"{a} (rows) vs {b} (cols)")''')

code('''# Seed stability: is a model's error type a property of the question or of the seed?
# A strong diagonal means errors are stable; mass off the diagonal means seed noise.
for m in MAIN:
    t = confusion(m, m, "seed13", "seed42")
    stable = np.trace(t.values) / t.values.sum()
    show_cm(t, f"{m}: seed13 (rows) vs seed42 (cols), same category {stable:.1%}")''')

code('''# Shared-error rate for every pair: of the examples both get wrong, how often is it the same error type?
def same_err_rate(a, b):
    both = [(cat_of(r, a), cat_of(r, b)) for r in R]
    both = [p for p in both if p[0] != "correct" and p[1] != "correct"]
    return sum(x == y for x, y in both) / len(both) if both else float("nan"), len(both)

sm = pd.DataFrame({a: {b: same_err_rate(a, b)[0] for b in MAIN} for a in MAIN}).T
nb_ = pd.DataFrame({a: {b: same_err_rate(a, b)[1] for b in MAIN} for a in MAIN}).T
print("same error type | both wrong;  counts of both-wrong examples below")
display(sm.style.format("{:.0%}").background_gradient(cmap="Oranges", axis=None))
display(nb_)''')

code('''# Answerability confusion (gold vs predicted), per model and seed 13: the over-abstention in one view
def answerability(m):
    g = pd.Series(["unanswerable" if r["is_impossible"] else "answerable" for r in R], name="gold")
    p = pd.Series(["no answer" if not (lambda s: (s.get("seed13") or next(iter(s.values())))["pred"])(r["models"][m]["seeds"])
                   else "answer" for r in R], name="predicts")
    return pd.crosstab(g, p).reindex(columns=["answer", "no answer"], fill_value=0)

display(pd.concat({m: answerability(m) for m in MAIN}, axis=1))''')

code('''# Drill into one cell, e.g. BiDAF abstains but IndoBERT is right
ids = cm_ids("bidaf", "plm_indobert_full", "false_noans", "correct")
print(len(ids))
side_by_side(ids[:10])''')

code('''# Optional: the same confusion restricted to a slice, e.g. long answers
long_ids = set(df.index[df.ans_len_bucket == ">6"])
show_cm(confusion("bidaf", "plm_indobert_full", ids=long_ids), "long answers: bidaf vs plm_indobert_full")''')

md("""## Queries
Edit and rerun. `df` columns: `<model>.f1` (mean over seeds), `<model>.cat` and `<model>.pred` (seed 13), plus slice tags.""")

code('''# 1. Hard set: every main model wrong (mean F1 < 0.5)
hard = df[(df[[f"{m}.f1" for m in MAIN]] < 0.5).all(axis=1)]
print(len(hard), "hard examples;", hard.is_impossible.mean().round(3), "unanswerable share")
show(hard.head(5))''')

code('''# 2. Where pretraining helps: IndoBERT right, BiDAF wrong
only_plm = df[(df["plm_indobert_full.f1"] >= 0.5) & (df["bidaf.f1"] < 0.5)]
print(len(only_plm))
show(only_plm.sample(min(5, len(only_plm)), random_state=0))''')

code('''# 3. Where feature engineering wins: classical right, IndoBERT wrong
only_cls = df[(df["classical_pipeline.f1"] >= 0.5) & (df["plm_indobert_full.f1"] < 0.5)]
print(len(only_cls))
show(only_cls.head(5))''')

code('''# 4. One error category for one model, e.g. IndoBERT false no-answers (all seeds shown)
fna = df[df["plm_indobert_full.cat"] == "false_no_answer"]
print(len(fna))
show(fna.head(4), seeds=None)''')

code('''# 5. Slice drill-down, e.g. long answers (>6 words) or questions ending with " ?"
show(df[(df.ans_len_bucket == ">6") & (df["plm_indobert_full.f1"] < 0.5)].head(4))''')

code('''# 6. The partial API LLM runs next to the others
api = [i for i in df.index if BY_ID[i]["partial_models"]]
show(api[:6])''')

code('''# 7. Export any selection for manual labeling
# hard.assign(human_label="", notes="").to_csv(ROOT / "eda/outputs/tables/hard_set_for_labeling.csv", encoding="utf-8")''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
from pathlib import Path
nbf.write(nb, Path(__file__).resolve().parents[2] / "eda/notebooks/model_errors.ipynb")
print("written")
