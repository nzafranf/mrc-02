"""EDA on train+dev only. Figures -> eda/outputs/figures/eda_*.png, tables -> eda/outputs/tables/eda_*.md"""
import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "experiments" / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

P, F, T = ROOT / "data/processed", ROOT / "eda/outputs/figures", ROOT / "eda/outputs/tables"
F.mkdir(parents=True, exist_ok=True); T.mkdir(parents=True, exist_ok=True)


def load(s):
    sl = pd.read_json(P / f"slices_{s}.jsonl", lines=True)
    raw = pd.read_json(P / f"{s}.jsonl", lines=True)[["id", "context"]]
    sl = sl.merge(raw, on="id"); sl["split"] = s
    return sl


df = pd.concat([load("train"), load("dev")], ignore_index=True)
df["imp"] = df.is_impossible.astype(int)


def md(t, name, idx=True):
    (T / f"eda_{name}.md").write_text(t.to_markdown(index=idx, floatfmt=".3f"), encoding="utf-8")


# sizes + length stats
size = df.groupby("split").agg(n=("id", "size"), imp_rate=("imp", "mean"), contexts=("context", "nunique"),
                               multi_gold=("n_gold", lambda s: (s > 1).mean()), citation=("has_citation", "mean"))
md(size, "sizes")
ans = df[df.imp == 0]
ls = pd.DataFrame({"context_words": df.ctx_len_words.describe(), "question_words": df.q_len_words.describe(),
                   "answer_words": ans.ans_len_words.describe()}).T
md(ls, "lengths")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
for a, (s, t) in zip(ax, [(df.ctx_len_words, "context words"), (df.q_len_words, "question words"), (ans.ans_len_words, "answer words")]):
    a.hist(s.clip(upper=s.quantile(.99)), bins=30, color="#4C78A8"); a.set_title(t)
plt.tight_layout(); plt.savefig(F / "eda_lengths.png", dpi=130); plt.close()

# question words
qw = df.groupby("q_word").agg(n=("id", "size"), imp_rate=("imp", "mean")).sort_values("n", ascending=False)
md(qw.head(20), "qword")
qr = df.groupby("q_word_raw").agg(n=("id", "size"), imp_rate=("imp", "mean")).query("n>=20").sort_values("n", ascending=False)
md(qr, "qword_raw")
qt = df.groupby("q_type").agg(n=("id", "size"), imp_rate=("imp", "mean"))
md(qt, "qtype")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
top = qw.head(10); ax[0].bar(top.index, top.imp_rate, color="#E45756"); ax[0].axhline(df.imp.mean(), ls="--", c="k")
ax[0].set_title("unanswerable rate per q_word"); ax[0].tick_params(axis="x", rotation=45)
ax[1].bar(qt.index, qt.imp_rate, color="#E45756"); ax[1].axhline(df.imp.mean(), ls="--", c="k")
ax[1].set_title("unanswerable rate per q_type"); ax[1].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig(F / "eda_unanswerable_rate.png", dpi=130); plt.close()

# answer type / position / len
at = ans.ans_type.value_counts(normalize=True).to_frame("share"); md(at, "anstype")
pos = ans.ans_rel_pos_bucket.value_counts(normalize=True).sort_index().to_frame("share"); md(pos, "ans_position")
sidx = ans.ans_sent_idx.clip(upper=10).value_counts(normalize=True).sort_index().to_frame("share"); md(sidx, "ans_sent_idx")
abk = ans.ans_len_bucket.value_counts(normalize=True).to_frame("share"); md(abk, "anslen_bucket")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].bar(at.index, at.share, color="#4C78A8"); ax[0].tick_params(axis="x", rotation=30); ax[0].set_title("answer type")
ax[1].bar(sidx.index.astype(str), sidx.share, color="#4C78A8"); ax[1].set_title("answer sentence index (>=10 clipped)")
plt.tight_layout(); plt.savefig(F / "eda_answer.png", dpi=130); plt.close()

# overlap
ov = df.groupby("imp").lex_overlap.describe(); ov.index = ["answerable", "unanswerable"]; md(ov, "overlap")
ob = pd.crosstab(df.lex_overlap_bucket, df.imp, normalize="columns"); ob.columns = ["answerable", "unanswerable"]; md(ob, "overlap_bucket")
ob2 = df.groupby("lex_overlap_bucket").agg(n=("id", "size"), imp_rate=("imp", "mean")); md(ob2, "overlap_imp_rate")
plt.figure(figsize=(5, 3.2))
for k, l in [(0, "answerable"), (1, "unanswerable")]:
    plt.hist(df[df.imp == k].lex_overlap, bins=15, alpha=.6, label=l, density=True)
plt.legend(); plt.xlabel("question content-word overlap with best/answer sentence"); plt.tight_layout()
plt.savefig(F / "eda_overlap.png", dpi=130); plt.close()

# slices vs imp
for c in ["ctx_len_bucket", "has_citation", "n_gold"]:
    md(df.groupby(c).agg(n=("id", "size"), imp_rate=("imp", "mean")), f"slice_{c}")

# questions per context
qpc = df.groupby(["split", "context"]).size()
md(qpc.groupby("split").describe(), "q_per_context")
md(qpc.value_counts().sort_index().to_frame("n_contexts"), "q_per_context_hist")
plt.figure(figsize=(5, 3.2)); qpc.clip(upper=15).hist(bins=15); plt.xlabel("questions per context (clipped 15)")
plt.tight_layout(); plt.savefig(F / "eda_q_per_context.png", dpi=130); plt.close()
# unanswerable share within contexts
g = df.groupby("context").imp.agg(["size", "mean"]); g = g[g["size"] >= 2]
md(pd.Series({"contexts>=2q": len(g), "all-answerable": (g["mean"] == 0).mean(), "all-unanswerable": (g["mean"] == 1).mean(),
              "mixed": ((g["mean"] > 0) & (g["mean"] < 1)).mean()}).to_frame("share/count"), "context_mix")
print("done"); print(size); print(qw.head(8)); print(qt); print(ov); print(g["mean"].describe())
