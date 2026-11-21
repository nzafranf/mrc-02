"""Cross-model comparison on saved dev predictions (no inference): every model in models_registry.yaml side by side.

Outputs report/assets/{tables,figures}: main_table, significance, slice_f1 + heatmaps, error_taxonomy_counts,
hard_set (wrong for every model), taxonomy_per_example. Per-model analysis (error samples for labeling, calibration,
seed stability, outcome tables) lives in each experiment: experiments/NN-*/analyze.py -> analysis/ + summary.md.

python experiments/global/run.py mrc.analysis.compare [--registry experiments/global/configs/models_registry.yaml] [--n-boot 1000]

Threshold policy: for every run with na_prob, the NA threshold is chosen on dev with
mrc.evaluate.best_na_threshold. This is dev-optimistic (threshold tuned and scored on the same
split), so absolute numbers are slightly high; model ranking is unaffected in practice.
"""
from __future__ import annotations

import argparse
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
import yaml

from mrc.analysis.slices import sentences
from mrc.evaluate import (aggregate, best_na_threshold, f1, get_tokens, load_jsonl,
                          normalize_answer, per_example_scores)

from mrc.paths import ROOT
TAB, FIG = ROOT / "report/assets/tables", ROOT / "report/assets/figures"
CATS = ["correct", "false_no_answer", "false_answer_on_unanswerable", "boundary_too_long",
        "boundary_too_short", "partial_overlap", "wrong_span_same_sentence", "wrong_sentence"]
# Taxonomy categories that cannot be decided automatically (miscellaneous/docs/error_taxonomy.md): 1 gold issue,
# 6 multi-hop miss, and the 8 vs 9 split (right type wrong entity / wrong type) inside wrong_span_same_sentence.
HUMAN = ["gold_issue", "multi_hop_miss", "right_type_wrong_entity", "wrong_type"]
SLICE_KEYS = ["q_type", "q_word_raw", "ctx_len_bucket", "ans_len_bucket", "ans_type",
              "lex_overlap_bucket", "ans_rel_pos_bucket", "ends_space_qmark"]
ORDER_SIG = ["no_answer", "classical_pipeline", "bidaf_noattn", "bidaf", "plm_indobert_full"]


def load_runs(reg, gold):
    runs = {}
    for name in reg["order"]:
        for d in sorted(glob.glob(str(ROOT / reg["models"][name]))):
            p = Path(d) / "preds_dev.jsonl"
            if not p.exists():
                continue
            preds = {x["id"]: x for x in load_jsonl(p)}
            has_na = any("na_prob" in x for x in preds.values())
            t = best_na_threshold(gold, preds)[0] if has_na else None
            sc = per_example_scores(gold, preds, t)
            for i, v in sc.items():
                v["na_prob"] = preds.get(i, {}).get("na_prob")
                v["start"] = preds.get(i, {}).get("start_char", -1)
            runs.setdefault(name, []).append(dict(seed=Path(d).name, thr=t, scores=sc, preds=preds,
                                                  agg=aggregate(sc)))
    return runs


def avg_f1(runs_m, ids):
    return np.array([np.mean([r["scores"][i]["f1"] for r in runs_m]) for i in ids])


def write_md(path, header, rows, title=""):
    lines = ([f"# {title}", ""] if title else []) + ["| " + " | ".join(header) + " |",
                                                      "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path, header, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main_table(runs, ids, rng, nb):
    rows = []
    for m, rr in runs.items():
        x = avg_f1(rr, ids)
        bs = x[rng.integers(0, len(x), (nb, len(x)))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5]) * 100
        row = [m, len(rr)]
        for k in ["exact", "f1", "HasAns_f1", "NoAns_acc"]:
            v = [r["agg"][k] for r in rr]
            row.append(f"{np.mean(v):.2f} ± {np.std(v):.2f}")
        row.append(f"[{lo:.2f}, {hi:.2f}]")
        rows.append(row)
    hdr = ["model", "seeds", "EM", "F1", "HasAns_F1", "NoAns_acc", "F1 95% boot CI (seed-avg)"]
    write_md(TAB / "main_table.md", hdr, rows, "Dev results, mean ± std over seeds (NA threshold tuned on dev, dev-optimistic)")
    write_csv(TAB / "main_table.csv", hdr, rows)
    # LaTeX version for report/latex/main.tex (\input{../assets/tables/main_table.tex}); needs booktabs
    tex = ["\\begin{tabular}{lrrrrrl}", "\\toprule",
           "Model & Seeds & EM & F1 & HasAns F1 & NoAns acc & F1 95\\% CI \\\\", "\\midrule"]
    tex += [" & ".join(str(c).replace("_", "\\_").replace("±", "$\\pm$") for c in r) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "main_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return rows


def significance(runs, ids, rng, nb):
    pairs = [(ORDER_SIG[i], ORDER_SIG[i + 1]) for i in range(len(ORDER_SIG) - 1)] + [("bidaf_noattn", "bidaf")]
    pairs = list(dict.fromkeys(pairs))
    rows = []
    idx = rng.integers(0, len(ids), (nb, len(ids)))
    for a, b in pairs:
        if a not in runs or b not in runs:
            continue
        d = avg_f1(runs[b], ids) - avg_f1(runs[a], ids)
        bs = d[idx].mean(1)
        p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / nb)
        lo, hi = np.percentile(bs, [2.5, 97.5]) * 100
        rows.append([f"{b} - {a}", f"{d.mean()*100:+.2f}", f"[{lo:.2f}, {hi:.2f}]", f"{p:.4f}"])
    hdr = ["comparison (B - A)", "delta F1", "95% CI", "p (two-sided paired bootstrap)"]
    write_md(TAB / "significance.md", hdr, rows, "Paired bootstrap on seed-averaged per-example F1")
    return rows


def slice_tables(runs, ids, sl, gold_by):
    sl = {i: dict(s, ends_space_qmark=str(gold_by[i]["question"].rstrip().endswith(" ?"))) for i, s in sl.items()}
    top = [k for k, _ in Counter(s["q_word_raw"] for s in sl.values()).most_common(10)]
    rows, mat, spread = [], {}, {}
    nontriv = [m for m in runs if m not in ("no_answer", "sliding_window", "sliding_window_thr")]
    for key in SLICE_KEYS:
        vals = top if key == "q_word_raw" else sorted({s[key] for s in sl.values()})
        for v in vals:
            sub = [i for i in ids if sl[i][key] == v]
            row, f = [key, v, len(sub)], {}
            for m, rr in runs.items():
                f[m] = avg_f1(rr, sub).mean() * 100
                row.append(f"{f[m]:.1f}")
            rows.append(row)
            mat[(key, v)] = f
            ms = [f[m] for m in nontriv]
            spread[(key, v)] = (max(ms) - min(ms), len(sub))
    hdr = ["slice", "value", "n"] + list(runs)
    write_md(TAB / "slice_f1.md", hdr, rows, "F1 per model per slice value (seed-averaged)")
    write_csv(TAB / "slice_f1.csv", hdr, rows)
    for key in ["q_type", "lex_overlap_bucket"]:
        vs = [v for (k, v) in mat if k == key]
        M = np.array([[mat[(key, v)][m] for v in vs] for m in runs])
        fig, ax = plt.subplots(figsize=(1.2 * len(vs) + 3, 0.5 * len(runs) + 1.5))
        im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(vs)), [f"{v}\n(n={sum(1 for i in ids if sl[i][key] == v)})" for v in vs])
        ax.set_yticks(range(len(runs)), list(runs))
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                ax.text(b, a, f"{M[a, b]:.0f}", ha="center", va="center", color="w" if M[a, b] < 60 else "k", fontsize=8)
        fig.colorbar(im, label="F1")
        ax.set_title(f"F1 by {key}")
        fig.tight_layout()
        fig.savefig(FIG / f"heatmap_{key}.png", dpi=150)
        plt.close(fig)
    return sl, spread


def classify(ex, v, ctx_sents):
    """Taxonomy category for one prediction."""
    if v["em"] == 1:
        return "correct"
    golds = [a for a in ex["answers"] if normalize_answer(a["text"])]
    pred = v["pred"]
    if not golds:
        return "false_answer_on_unanswerable"
    if not normalize_answer(pred):
        return "false_no_answer"
    g = max(golds, key=lambda a: f1(a["text"], pred))
    gt, pt = get_tokens(g["text"]), get_tokens(pred)

    def contains(big, small):
        return any(big[i:i + len(small)] == small for i in range(len(big) - len(small) + 1))
    if contains(pt, gt):
        return "boundary_too_long"
    if contains(gt, pt):
        return "boundary_too_short"
    if f1(g["text"], pred) > 0:
        return "partial_overlap"
    ctx = ex["context"]
    ps = v["start"] if v["start"] is not None and v["start"] >= 0 else ctx.find(pred)
    sid = lambda c: next((i for i, (s, e) in enumerate(ctx_sents) if s <= c < e + 1), len(ctx_sents) - 1)
    return "wrong_span_same_sentence" if sid(ps) == sid(g["answer_start"]) else "wrong_sentence"


def taxonomy(runs, gold_by, ids):
    tax = {m: {r["seed"]: {} for r in rr} for m, rr in runs.items()}
    sents = {i: sentences(gold_by[i]["context"]) for i in ids}
    for m, rr in runs.items():
        for r in rr:
            for i in ids:
                tax[m][r["seed"]][i] = classify(gold_by[i], r["scores"][i], sents[i])
    dist = {}
    for m, rr in runs.items():
        s = "seed13" if "seed13" in tax[m] else rr[0]["seed"]
        c = Counter(tax[m][s].values())
        dist[m] = {k: c.get(k, 0) for k in CATS}
    rows = [[m] + [dist[m][k] for k in CATS] for m in runs]
    write_md(TAB / "error_taxonomy_counts.md", ["model (seed13)"] + CATS, rows,
             "Automatic error taxonomy counts, seed 13 (n=803). Needs human judgement: " + ", ".join(HUMAN))
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(runs))
    for k, c in zip(CATS[1:], plt.cm.tab10(np.linspace(0, 1, 10))):
        v = np.array([dist[m][k] for m in runs], float)
        ax.bar(list(runs), v, bottom=bottom, label=k, color=c)
        bottom += v
    ax.set_ylabel("errors (seed 13)")
    ax.set_title("Error category distribution per model (seed 13; dev, n=803)")
    ax.legend(fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "error_taxonomy_stacked.png", dpi=150)
    plt.close(fig)
    return tax, dist


def hard_set(runs, gold_by, ids, sl, reg):
    ms = [m for m in runs if m not in reg["trivial"]]
    hard = [i for i in ids if all(r["scores"][i]["f1"] < 0.5 for m in ms for r in runs[m])]
    with open(TAB / "hard_set.jsonl", "w", encoding="utf-8") as f:
        for i in hard:
            ex = gold_by[i]
            g = next((a for a in ex["answers"] if a["text"].strip()), None)
            if g:
                s = g["answer_start"]
                snip = ex["context"][max(0, s - 150): s + len(g["text"]) + 150]
            else:
                snip = ex["context"][:300]
            o = {"id": i, "question": ex["question"], "context_snippet": snip,
                 "gold": [a["text"] for a in ex["answers"]], "is_impossible": ex["is_impossible"],
                 "preds": {m: [r["scores"][i]["pred"] for r in runs[m]] for m in ms},
                 "slices": {k: sl[i][k] for k in ["q_type", "ans_type", "lex_overlap_bucket", "ctx_len_bucket", "ans_len_bucket"]}}
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    rows = []
    for key in ["q_type", "ans_type", "lex_overlap_bucket", "ctx_len_bucket", "ans_len_bucket", "is_impossible"]:
        get = (lambda i: gold_by[i]["is_impossible"]) if key == "is_impossible" else (lambda i, k=key: sl[i][k])
        ch, ca = Counter(get(i) for i in hard), Counter(get(i) for i in ids)
        for v in sorted(ca, key=str):
            rows.append([key, v, ch.get(v, 0), f"{100 * ch.get(v, 0) / max(1, len(hard)):.1f}%",
                         f"{100 * ca[v] / len(ids):.1f}%", f"{100 * ch.get(v, 0) / ca[v]:.1f}%"])
    write_md(TAB / "hard_set_composition.md",
             ["slice", "value", "n_hard", "share_of_hard", "share_of_dev", "hard_rate_in_slice"], rows,
             f"Hard set composition: {len(hard)} of {len(ids)} dev examples, F1<0.5 for every seed of {', '.join(ms)}")
    return hard, rows


def calibration(runs, reg, gold_by, ids):
    lines = ["# Calibration of answerability (PLM)", "",
             "p_na = sigmoid(na_prob). na_prob is a RAW logit difference (null score minus best span score), "
             "not a trained probability; sigmoid is an uncalibrated monotone map, so ECE mainly shows how far "
             "the raw logit scale is from a probability. 10 equal-width bins, ECE = sum_b (n_b/N)|acc_b - conf_b|.", ""]
    for m in reg.get("logit_na", []):
        if m not in runs:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        for r in runs[m]:
            p = np.array([1 / (1 + math.exp(-max(-50, min(50, r["preds"][i]["na_prob"])))) for i in ids])
            y = np.array([float(gold_by[i]["is_impossible"]) for i in ids])
            bins = np.minimum((p * 10).astype(int), 9)
            xs, ys, ece = [], [], 0.0
            for b in range(10):
                mk = bins == b
                if mk.sum():
                    xs.append(p[mk].mean())
                    ys.append(y[mk].mean())
                    ece += mk.mean() * abs(y[mk].mean() - p[mk].mean())
            ax.plot(xs, ys, "o-", label=f"{r['seed']} (ECE {ece:.3f})")
            lines.append(f"- {m} {r['seed']}: ECE = {ece:.4f}; mean p_na = {p.mean():.3f}; unanswerable rate = "
                         f"{y.mean():.3f}; dev-tuned threshold on raw logit = {r['thr']:.3f}")
        ax.set_xlabel("sigmoid(na_prob)")
        ax.set_ylabel("fraction unanswerable")
        ax.set_title(f"Reliability diagram: {m}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f"reliability_{m}.png", dpi=150)
        plt.close(fig)
    (TAB / "calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def error_samples(runs, tax, gold_by, ids, sl, reg):
    rnd = random.Random(0)
    esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
    for m, rr in runs.items():
        if m in reg["trivial"]:
            continue
        r = next((x for x in rr if x["seed"] == "seed13"), rr[0])
        t = tax[m][r["seed"]]
        by = defaultdict(list)
        for i in ids:
            if t[i] != "correct":
                by[t[i]].append(i)
        for v in by.values():
            rnd.shuffle(v)
        chosen = []
        while len(chosen) < 40 and any(by.values()):
            for k in list(by):
                if by[k] and len(chosen) < 40:
                    chosen.append((k, by[k].pop()))
        corr = [i for i in ids if t[i] == "correct" and r["scores"][i]["pred"] and r["scores"][i]["na_prob"] is not None]
        corr.sort(key=lambda i: -r["scores"][i]["na_prob"])
        low = [("correct_low_conf", i) for i in corr[:10]]
        lines = [f"# Error sample: {m} ({r['seed']})", "",
                 "40 errors stratified by automatic category (round robin) + 10 correct answers with highest na_prob. "
                 f"Human labels: {', '.join(HUMAN)}, none, other.", "",
                 "| # | id | auto_category | question | gold | pred | q_type | human_label | notes |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for n, (k, i) in enumerate(chosen + low, 1):
            ex = gold_by[i]
            g = " / ".join(a["text"] for a in ex["answers"]) or "(no answer)"
            lines.append(f"| {n} | {i} | {k} | {esc(ex['question'])} | {esc(g)} | "
                         f"{esc(r['scores'][i]['pred'] or '(no answer)')} | {sl[i]['q_type']} |  |  |")
        (TAB / f"error_sample_{m}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(ROOT / "experiments/global/configs/models_registry.yaml"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--split-file", default=str(ROOT / "data/processed/dev.jsonl"))
    ap.add_argument("--slices-file", default=str(ROOT / "data/processed/slices_dev.jsonl"))
    a = ap.parse_args()
    if "test" in Path(a.split_file).name:
        raise SystemExit("Refusing to analyse a test split.")
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    reg = yaml.safe_load(open(a.registry, encoding="utf-8"))
    gold = load_jsonl(a.split_file)
    gold_by = {g["id"]: g for g in gold}
    ids = [g["id"] for g in gold]
    sl = {s["id"]: s for s in load_jsonl(a.slices_file)}
    runs = load_runs(reg, gold)
    rng = np.random.default_rng(0)
    mt = main_table(runs, ids, rng, a.n_boot)
    sg = significance(runs, ids, rng, a.n_boot)
    sl2, spread = slice_tables(runs, ids, sl, gold_by)
    tax, dist = taxonomy(runs, gold_by, ids)
    hard, comp = hard_set(runs, gold_by, ids, sl2, reg)
    json.dump({m: d for m, d in tax.items()}, open(TAB / "taxonomy_per_example.json", "w"))
    print("\n".join(" | ".join(map(str, r)) for r in mt), "\n")
    print("\n".join(" | ".join(r) for r in sg), "\nhard", len(hard))
    print("\n".join(" | ".join(map(str, r)) for r in comp))
    top = sorted(((round(v[0], 1), k, v[1]) for k, v in spread.items() if v[1] >= 30), reverse=True)[:8]
    print("top spreads", top)


if __name__ == "__main__":
    main()
