"""Per-experiment error analysis: the last stage of every experiment, the same for all of them.

Each experiment folder calls `analyze()` from its own `analyze.py` with its variants (a variant is a set of run
directories that differ only by seed, e.g. bidaf and bidaf_noattn in 03-nn-bidaf). Outputs go to
<experiment>/analysis/ and a generated block inside <experiment>/summary.md:

  results.md            EM / F1 / HasAns F1 / NoAns acc per variant (mean ± std over seeds), tuned NA threshold
  outcomes.md, .png     outcome type (rows) x variant (columns): correct_ans, false_no_answer, ... correct_no_ans
  seed_stability.md     per variant: how often two seeds give the same outcome type on a question
  slices.md             F1 per slice value per variant; the weakest slices are listed in summary.md
  significance.md       paired bootstrap of each variant against the reference (first) variant
  error_sample_<v>.md   40 errors stratified by outcome + 10 low-confidence correct answers, for manual labels
  hard_examples.jsonl   questions every run of every variant gets wrong (F1 < 0.5)
  calibration.md, .png  only for variants whose na_prob is a raw logit (PLM)
  per_example.jsonl     id -> variant -> seed -> pred, f1, outcome (feed for notebooks)

Threshold policy is the same as the cross-model table: the no-answer threshold is tuned per run on dev
(dev-optimistic). Runs that cover only part of dev (API LLMs) are scored on the ids they cover; the tables
show n so partial runs are visible. Never pass a test split: this module refuses file names containing "test".

Other evaluation splits (split_file=data/processed/official_val.jsonl): runs are read from preds_<split>.jsonl, the
outputs go to <experiment>/analysis/<split>/ and to a separate AUTO-RESULTS-<split> block in summary.md, and the
threshold is tuned on that split. dev keeps preds_dev.jsonl, analysis/ and the plain AUTO-RESULTS block.
"""
from __future__ import annotations

import glob
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mrc.analysis.compare import classify
from mrc.analysis.slices import sentences
from mrc.evaluate import aggregate, best_na_threshold, load_jsonl, per_example_scores
from mrc.paths import PROC

BEGIN, END = "<!-- AUTO-RESULTS:BEGIN (written by analyze.py, edits here are overwritten) -->", "<!-- AUTO-RESULTS:END -->"
OUTCOMES = [("answerable", "correct_ans"), ("answerable", "false_no_answer"), ("answerable", "wrong_sentence"),
            ("answerable", "wrong_span_same_sentence"), ("answerable", "boundary_too_long"),
            ("answerable", "boundary_too_short"), ("answerable", "partial_overlap"),
            ("unanswerable", "correct_no_ans"), ("unanswerable", "false_answer_on_unanswerable")]
SLICE_KEYS = ["q_type", "ans_type", "ans_len_bucket", "lex_overlap_bucket", "ctx_len_bucket",
              "ans_rel_pos_bucket", "ends_space_qmark"]
HUMAN = ["gold_issue", "multi_hop_miss", "right_type_wrong_entity", "wrong_type", "none", "other"]


def outcome(ex, cat):
    if cat == "correct":
        return "correct_no_ans" if ex["is_impossible"] else "correct_ans"
    return cat


def load_variant(name, run_dirs, gold_by, sents, tune=True, preds_name="preds_dev.jsonl"):
    runs = []
    for d in run_dirs:
        p = Path(d) / preds_name
        if not p.exists():
            continue
        preds = {x["id"]: x for x in load_jsonl(p) if x["id"] in gold_by}
        for x in preds.values():  # some runs store na_prob = null; the evaluator needs it absent instead
            if x.get("na_prob") is None:
                x.pop("na_prob", None)
            if tune and "span_pred" in x:  # 02 --predict-only thresholds pred; tune on the span before that
                x["pred"] = x["span_pred"]
        gold = [gold_by[i] for i in gold_by if i in preds]
        has_na = any("na_prob" in x for x in preds.values())
        thr = best_na_threshold(gold, preds)[0] if has_na and tune else None
        sc = per_example_scores(gold, preds, thr)
        for i, v in sc.items():
            v["na_prob"] = preds[i].get("na_prob")
            v["start"] = preds[i].get("start_char", -1)
            v["outcome"] = outcome(gold_by[i], classify(gold_by[i], v, sents[i]))
        runs.append(dict(seed=Path(d).name, dir=Path(d), thr=thr, scores=sc, agg=aggregate(sc), preds=preds))
    return runs


def md_table(header, rows):
    out = ["| " + " | ".join(map(str, header)) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def ms(xs, fmt="{:.1f}"):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return "n/a"
    return fmt.format(np.mean(xs)) + (f" ± {fmt.format(np.std(xs))}" if len(xs) > 1 else "")


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def analyze(exp_dir, variants, *, reference=None, logit_na=(), untuned=(), n_boot=1000, split_file=None):
    """variants: {name: glob relative to exp_dir}, e.g. {"bidaf": "runs/bidaf/seed*"}.
    reference: variant the others are tested against (default: the first). untuned: variants scored as predicted,
    without the dev-tuned no-answer threshold. logit_na: variants whose na_prob is a raw logit (calibration plot)."""
    exp_dir = Path(exp_dir)
    split_file = Path(split_file or PROC / "dev.jsonl")
    if "test" in split_file.name:
        raise SystemExit("Refusing to analyse a test split.")
    stem = split_file.stem
    preds_name = f"preds_{stem}.jsonl"
    out = exp_dir / "analysis" if stem == "dev" else exp_dir / "analysis" / stem
    rel = out.relative_to(exp_dir).as_posix()  # links in summary.md
    begin, end = (BEGIN, END) if stem == "dev" else (BEGIN.replace("AUTO-RESULTS", f"AUTO-RESULTS-{stem}"),
                                                     END.replace("AUTO-RESULTS", f"AUTO-RESULTS-{stem}"))
    out.mkdir(parents=True, exist_ok=True)
    gold = load_jsonl(split_file)
    gold_by = {g["id"]: g for g in gold}
    sl = {s["id"]: s for s in load_jsonl(PROC / f"slices_{stem}.jsonl")}
    for i, s in sl.items():
        if i in gold_by:
            s["ends_space_qmark"] = str(gold_by[i]["question"].rstrip().endswith(" ?"))
    sents = {i: sentences(g["context"]) for i, g in gold_by.items()}

    V = {}
    for name, pattern in variants.items():
        dirs = sorted(glob.glob(str(exp_dir / pattern)))
        rr = load_variant(name, dirs, gold_by, sents, tune=name not in untuned, preds_name=preds_name)
        if rr:
            V[name] = rr
        else:
            print(f"  skip {name}: no {preds_name} under {pattern}")
    if not V:
        raise SystemExit("no runs found")
    names = list(V)
    ref = reference or names[0]
    ids_of = {m: sorted(set.intersection(*(set(r["scores"]) for r in rr))) for m, rr in V.items()}
    seed_avg = lambda m, ids: np.array([np.mean([r["scores"][i]["f1"] for r in V[m]]) for i in ids])
    sec = {}

    # results
    rows = []
    for m, rr in V.items():
        thr = [r["thr"] for r in rr if r["thr"] is not None]
        rows.append([m, len(rr), len(ids_of[m]), ms([r["agg"]["exact"] for r in rr]), ms([r["agg"]["f1"] for r in rr]),
                     ms([r["agg"]["HasAns_f1"] for r in rr]), ms([r["agg"]["NoAns_acc"] for r in rr]),
                     ms([r["agg"]["pred_noans_rate"] for r in rr]), ms(thr, "{:.3g}") if thr else "untuned"])
    hdr = ["variant", "runs", "n", "EM", "F1", "HasAns F1", "NoAns acc", "predicted no-answer %", "NA threshold"]
    sec["results"] = md_table(hdr, rows)
    (out / "results.md").write_text(f"# {'Dev' if stem == 'dev' else stem} results (NA threshold tuned on {stem}, {stem}-optimistic)\n\n" + sec["results"] + "\n",
                                    encoding="utf-8")

    # outcomes (rows) x variants (columns), percent of the gold group, mean over seeds
    pct, cnt = {}, {}
    for m, rr in V.items():
        per = []
        for r in rr:
            c = Counter((("unanswerable" if gold_by[i]["is_impossible"] else "answerable"), v["outcome"])
                        for i, v in r["scores"].items())
            per.append(c)
        cnt[m] = {k: np.mean([c.get(k, 0) for c in per]) for k in OUTCOMES}
        size = {g: sum(cnt[m][k] for k in OUTCOMES if k[0] == g) for g in ("answerable", "unanswerable")}
        pct[m] = {k: 100 * cnt[m][k] / size[k[0]] if size[k[0]] else float("nan") for k in OUTCOMES}
    rows = [[g, o] + [f"{pct[m][(g, o)]:.1f}% ({cnt[m][(g, o)]:.0f})" for m in names] for g, o in OUTCOMES]
    sec["outcomes"] = md_table(["gold", "outcome"] + names, rows)
    (out / "outcomes.md").write_text(
        "# Outcome type by variant\n\nShare of the gold group (answerable or unanswerable), mean over seeds; "
        "count in brackets. Each column sums to 100% within a group.\n\n" + sec["outcomes"] + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 4, 4.5))
    bottom = np.zeros(len(names))
    for k, col in zip(OUTCOMES, plt.cm.tab10(np.linspace(0, 1, len(OUTCOMES)))):
        v = np.array([cnt[m][k] for m in names])
        ax.bar(names, v, bottom=bottom, color=col, label=f"{k[0][:5]}: {k[1]}")
        bottom += v
    ax.set_ylabel("questions (mean over seeds)")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title(f"Outcomes: {exp_dir.name}")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out / "outcomes.png", dpi=150)
    plt.close(fig)

    # seed stability
    rows = []
    for m, rr in V.items():
        if len(rr) < 2:
            continue
        ids = ids_of[m]
        same = [np.mean([a["scores"][i]["outcome"] == b["scores"][i]["outcome"] for i in ids])
                for k, a in enumerate(rr) for b in rr[k + 1:]]
        flip = np.mean([len({r["scores"][i]["f1"] >= .5 for r in rr}) > 1 for i in ids])
        rows.append([m, len(rr), f"{100 * np.mean(same):.1f}%", f"{100 * flip:.1f}%"])
    sec["seed"] = md_table(["variant", "seeds", "same outcome type (mean over seed pairs)",
                            "questions that flip right/wrong across seeds"], rows) if rows else "single run per variant"
    (out / "seed_stability.md").write_text("# Seed stability\n\n" + sec["seed"] + "\n", encoding="utf-8")

    # slices
    best = max(names, key=lambda m: np.mean([r["agg"]["f1"] for r in V[m]]))
    rows, weak = [], []
    for key in SLICE_KEYS:
        vals = sorted({sl[i][key] for i in gold_by if i in sl and key in sl[i]}, key=str)
        for val in vals:
            row = [key, val]
            for m in names:
                sub = [i for i in ids_of[m] if sl.get(i, {}).get(key) == val]
                f = seed_avg(m, sub).mean() * 100 if sub else float("nan")
                row += [len(sub), f"{f:.1f}"]
                if m == best and len(sub) >= 30:
                    weak.append((f, key, val, len(sub)))
            rows.append(row)
    hdr = ["slice", "value"] + [h for m in names for h in (f"n ({m})", f"F1 {m}")]
    (out / "slices.md").write_text("# F1 by slice (seed-averaged)\n\n" + md_table(hdr, rows) + "\n", encoding="utf-8")
    weak.sort()
    sec["weak"] = md_table(["slice", "value", "n", f"F1 ({best})"], [[k, v, n, f"{f:.1f}"] for f, k, v, n in weak[:6]])

    # significance against the reference variant
    rng = np.random.default_rng(0)
    rows = []
    for m in names:
        if m == ref:
            continue
        ids = sorted(set(ids_of[m]) & set(ids_of[ref]))
        d = seed_avg(m, ids) - seed_avg(ref, ids)
        bs = d[rng.integers(0, len(ids), (n_boot, len(ids)))].mean(1)
        p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / n_boot)
        lo, hi = np.percentile(bs, [2.5, 97.5]) * 100
        rows.append([f"{m} - {ref}", len(ids), f"{d.mean() * 100:+.2f}", f"[{lo:.2f}, {hi:.2f}]", f"{p:.4f}"])
    sec["sig"] = md_table(["comparison", "n", "delta F1", "95% CI", "p (paired bootstrap)"], rows) if rows else ""
    if rows:
        (out / "significance.md").write_text("# Paired bootstrap on seed-averaged per-question F1\n\n" + sec["sig"] + "\n",
                                             encoding="utf-8")

    # hard examples within this experiment
    common = sorted(set.intersection(*(set(ids_of[m]) for m in names)))
    hard = [i for i in common if all(r["scores"][i]["f1"] < .5 for m in names for r in V[m])]
    with open(out / "hard_examples.jsonl", "w", encoding="utf-8") as f:
        for i in hard:
            ex = gold_by[i]
            f.write(json.dumps({"id": i, "question": ex["question"], "gold": [a["text"] for a in ex["answers"]],
                                "is_impossible": ex["is_impossible"],
                                "preds": {m: [r["scores"][i]["pred"] for r in V[m]] for m in names},
                                "slices": {k: sl.get(i, {}).get(k) for k in SLICE_KEYS}}, ensure_ascii=False) + "\n")
    hard_imp = np.mean([gold_by[i]["is_impossible"] for i in hard]) if hard else float("nan")
    sec["hard"] = (f"{len(hard)} of {len(common)} questions are wrong (F1 < 0.5) in every run of every variant; "
                   f"{100 * hard_imp:.0f}% of them are unanswerable ({stem} base rate "
                   f"{100 * np.mean([gold_by[i]['is_impossible'] for i in common]):.0f}%).")

    # error samples for manual labeling
    rnd = random.Random(0)
    for m in names:
        r = next((x for x in V[m] if x["seed"] == "seed13"), V[m][0])
        by = defaultdict(list)
        for i, v in r["scores"].items():
            if v["outcome"] not in ("correct_ans", "correct_no_ans"):
                by[v["outcome"]].append(i)
        for v in by.values():
            rnd.shuffle(v)
        chosen = []
        while len(chosen) < 40 and any(by.values()):
            for k in list(by):
                if by[k] and len(chosen) < 40:
                    chosen.append((k, by[k].pop()))
        corr = [i for i, v in r["scores"].items() if v["outcome"] == "correct_ans" and v["na_prob"] is not None]
        corr.sort(key=lambda i: -r["scores"][i]["na_prob"])
        lines = [f"# Error sample: {m} ({r['seed']})", "",
                 "40 errors stratified by outcome (round robin) + 10 correct answers with the highest na_prob. "
                 f"Fill human_label with one of: {', '.join(HUMAN)}.", "",
                 "| # | id | outcome | question | gold | pred | q_type | human_label | notes |", "|---|---|---|---|---|---|---|---|---|"]
        for n, (k, i) in enumerate(chosen + [("correct_low_conf", i) for i in corr[:10]], 1):
            ex = gold_by[i]
            g = " / ".join(a["text"] for a in ex["answers"]) or "(no answer)"
            lines.append(f"| {n} | {i} | {k} | {esc(ex['question'])} | {esc(g)} | "
                         f"{esc(r['scores'][i]['pred'] or '(no answer)')} | {sl.get(i, {}).get('q_type')} |  |  |")
        p = out / f"error_sample_{m}.md"
        if p.exists() and any(l.count("|") >= 10 and not l.rstrip().endswith("|  |  |") and l.startswith("| ") and
                              l.split("|")[1].strip().isdigit() for l in p.read_text(encoding="utf-8").splitlines()):
            p = out / f"error_sample_{m}.new.md"  # keep a file that already has human labels
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # calibration of raw-logit na_prob
    cal = []
    for m in logit_na:
        if m not in V:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        for r in V[m]:
            ids = [i for i in r["scores"] if r["scores"][i]["na_prob"] is not None]
            p = np.array([1 / (1 + math.exp(-max(-50, min(50, r["scores"][i]["na_prob"])))) for i in ids])
            y = np.array([float(gold_by[i]["is_impossible"]) for i in ids])
            b = np.minimum((p * 10).astype(int), 9)
            xs, ys, ece = [], [], 0.0
            for k in range(10):
                mk = b == k
                if mk.sum():
                    xs.append(p[mk].mean()); ys.append(y[mk].mean())
                    ece += mk.mean() * abs(y[mk].mean() - p[mk].mean())
            ax.plot(xs, ys, "o-", label=f"{r['seed']} (ECE {ece:.3f})")
            cal.append(f"- {m} {r['seed']}: ECE {ece:.4f}, mean sigmoid(na_prob) {p.mean():.3f}, "
                       f"unanswerable rate {y.mean():.3f}, tuned threshold on the raw logit {r['thr']:.3f}")
        ax.set_xlabel("sigmoid(na_prob)"); ax.set_ylabel("fraction unanswerable"); ax.legend(fontsize=8)
        ax.set_title(f"Reliability: {m}")
        fig.tight_layout(); fig.savefig(out / f"reliability_{m}.png", dpi=150); plt.close(fig)
    if cal:
        (out / "calibration.md").write_text(
            "# Calibration of answerability\n\nna_prob is a raw logit difference (null minus best span score); sigmoid is an "
            "uncalibrated monotone map, so ECE mostly shows how far that scale is from a probability. 10 equal-width bins.\n\n"
            + "\n".join(cal) + "\n", encoding="utf-8")

    # per-example feed
    with open(out / "per_example.jsonl", "w", encoding="utf-8") as f:
        for i in gold_by:
            rec = {m: {r["seed"]: {k: r["scores"][i][k] for k in ("pred", "f1", "outcome")}
                       for r in V[m] if i in r["scores"]} for m in names}
            rec = {m: v for m, v in rec.items() if v}
            if rec:
                f.write(json.dumps({"id": i, "variants": rec}, ensure_ascii=False) + "\n")

    # summary block
    block = [begin, "", f"### Results ({stem}, NA threshold tuned on {stem})", "", sec["results"], "",
             "### Outcome types", "", "Share of each gold group; count (mean over seeds) in brackets. "
             f"Figure: [{rel}/outcomes.png]({rel}/outcomes.png).", "", sec["outcomes"], ""]
    if sec["sig"]:
        block += [f"### Variants against `{ref}` (paired bootstrap)", "", sec["sig"], ""]
    block += ["### Seed stability", "", sec["seed"], "", f"### Weakest slices for `{best}`, the best variant (n >= 30). All slices: [{rel}/slices.md]({rel}/slices.md)", "", sec["weak"], "",
              "### Hard questions", "", sec["hard"] + f" List: [{rel}/hard_examples.jsonl]({rel}/hard_examples.jsonl).", ""]
    if cal:
        block += ["### Calibration", ""] + cal + [""]
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    block += ["### Analysis files", "", ", ".join(f"[{n}]({rel}/{n})" for n in files), "", end]
    summ = exp_dir / "summary.md"
    text = summ.read_text(encoding="utf-8") if summ.exists() else f"# {exp_dir.name}\n\n## Results and error analysis\n\n"
    if begin in text and end in text:
        text = text[:text.index(begin)] + "\n".join(block) + text[text.index(end) + len(end):]
    else:
        text = text.rstrip() + "\n\n" + "\n".join(block) + "\n"
    summ.write_text(text, encoding="utf-8")
    print(f"{exp_dir.name}: {', '.join(f'{m} ({len(V[m])} runs, n={len(ids_of[m])})' for m in names)} -> {out}")
    return V
