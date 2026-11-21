"""Answerability artifact probe: can unanswerability be predicted without reading context+question jointly?
Train on train, eval on dev (train+dev only; itest never read). Plus 5-fold GroupKFold CV on train."""
import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "experiments" / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

P = ROOT / "data/processed"
def load(s):
    r = pd.read_json(P / f"{s}.jsonl", lines=True)
    sl = pd.read_json(P / f"slices_{s}.jsonl", lines=True).drop(columns=["is_impossible"])
    return r.merge(sl, on="id")
tr, dv = load("train"), load("dev")
y_tr, y_dv = tr.is_impossible.astype(int).values, dv.is_impossible.astype(int).values
norm = lambda x: re.sub(r"\s+", " ", re.sub(r"\[\d+\]", "", x.lower())).strip()[:120]
groups = tr.context.map(norm).values

def fmt(df):
    q = df.question
    f = pd.DataFrame({
        "space_before_q": q.str.contains(r"\s\?\s*$"), "ends_q": q.str.strip().str.endswith("?"),
        "dbl_space": q.str.contains(r"\s{2}"), "starts_lower": q.str[0].str.islower(),
        "n_caps_ratio": q.map(lambda s: sum(c.isupper() for c in s) / max(1, len(s))), "qlen": q.str.len() / 100})
    return f.astype(float).values

def ovf(df):
    return np.c_[df.lex_overlap, df.lex_overlap_bucket.eq("low"), df.lex_overlap_bucket.eq("high"),
                 df.ctx_len_words / 300, df.q_len_words / 20, df.has_citation]

def make(variant):
    """returns fit_transform/transform closures over a dataframe"""
    vs = []
    if variant in ("question", "question+overlap"):
        vs += [("question", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
               ("question", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, lowercase=False))]
    if variant == "context":
        vs += [("context", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=100000))]
    if variant == "question_wordsonly_lower":
        vs += [("question", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True))]
    def fit(df):
        m = [v.fit_transform(df[c]) for c, v in vs]
        if variant == "question+overlap": m.append(csr_matrix(ovf(df)))
        if variant == "format_only": m = [csr_matrix(fmt(df))]
        if variant == "question+format": m.append(csr_matrix(fmt(df)))
        return hstack(m).tocsr()
    def tf(df):
        m = [v.transform(df[c]) for c, v in vs]
        if variant == "question+overlap": m.append(csr_matrix(ovf(df)))
        if variant == "format_only": m = [csr_matrix(fmt(df))]
        if variant == "question+format": m.append(csr_matrix(fmt(df)))
        return hstack(m).tocsr()
    return fit, tf

def run(variant, trd, ytr, ted, yte):
    fit, tf = make(variant)
    X = fit(trd); clf = LogisticRegression(C=2.0, max_iter=2000).fit(X, ytr)
    p = clf.predict_proba(tf(ted))[:, 1]
    return accuracy_score(yte, p > .5), roc_auc_score(yte, p)

variants = ["question", "question+format", "format_only", "context", "question+overlap"]
out = {"majority_dev_acc": max(y_dv.mean(), 1 - y_dv.mean()), "majority_train_acc": max(y_tr.mean(), 1 - y_tr.mean()), "variants": {}}
# fix: question+format needs word tfidf too
_orig = make
def make(variant):
    if variant == "question+format":
        f0, t0 = _orig("question")
        return (lambda d: hstack([f0(d), csr_matrix(fmt(d))]).tocsr()), (lambda d: hstack([t0(d), csr_matrix(fmt(d))]).tocsr())
    return _orig(variant)
for v in variants:
    acc, auc = run(v, tr, y_tr, dv, y_dv)
    cv = []
    for a, b in GroupKFold(5).split(tr, y_tr, groups):
        cv.append(run(v, tr.iloc[a], y_tr[a], tr.iloc[b], y_tr[b]))
    cv = np.array(cv)
    out["variants"][v] = dict(dev_acc=round(acc, 4), dev_auc=round(auc, 4), cv_acc=round(cv[:, 0].mean(), 4), cv_acc_sd=round(cv[:, 0].std(), 4),
                              cv_auc=round(cv[:, 1].mean(), 4), cv_auc_sd=round(cv[:, 1].std(), 4))
    print(v, out["variants"][v], flush=True)

# format cue rates on train+dev
al = pd.concat([tr, dv]); f = pd.DataFrame(fmt(al), columns=["space_before_q", "ends_q", "dbl_space", "starts_lower", "caps", "qlen"])
f["imp"] = al.is_impossible.values
cues = {c: {"rate": round(float(f[c].mean()), 3), "imp_rate_if_1": round(float(f[f[c] > 0].imp.mean()), 3), "imp_rate_if_0": round(float(f[f[c] == 0].imp.mean()), 3)}
        for c in ["space_before_q", "ends_q", "dbl_space", "starts_lower"]}
out["format_cues"] = cues
# within-context pairs: contexts with exactly 2 questions
g = al.groupby(al.context.map(norm))
pairs = [d for _, d in g if len(d) == 2]
mixed = sum(d.is_impossible.sum() == 1 for d in pairs)
out["pairs"] = {"n_2q_contexts": len(pairs), "exactly_one_unanswerable": int(mixed)}
# top LR coefficients (question word model)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True); X = vec.fit_transform(al.question)
clf = LogisticRegression(C=2, max_iter=2000).fit(X, al.is_impossible.astype(int)); nm = np.array(vec.get_feature_names_out())
o = np.argsort(clf.coef_[0]); out["top_unans_tokens"] = nm[o[-15:]][::-1].tolist(); out["top_ans_tokens"] = nm[o[:15]].tolist()
json.dump(out, open(ROOT / "eda/outputs/tables/artifact_probe.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ["majority_dev_acc", "format_cues", "pairs", "top_unans_tokens", "top_ans_tokens"]}, indent=1, ensure_ascii=False))
