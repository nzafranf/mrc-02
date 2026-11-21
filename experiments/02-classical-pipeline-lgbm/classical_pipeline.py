"""Traditional-ML pipeline for Indonesian extractive MRC.

  question + context
     |
     v
  [Stage B] sentence retriever (BM25/TF-IDF/overlap features, LR ranker)  -> top-k sentences
     |                                                                          |
     v                                                                          v
  [Stage C] span ranker (candidate n-grams + hand features, LightGBM)  -> best span + score
  [Stage A] answerability classifier (LightGBM on retrieval/overlap/type features) -> na_prob
     |
     v
  final: "" if na_prob > threshold (tuned on dev) else best span

Each stage is a class that can be fit and evaluated alone. Features live in named groups so
`--ablation` can drop one group at a time.

  python experiments/02-classical-pipeline-lgbm/classical_pipeline.py --seed 13 --split-file data/processed/dev.jsonl
  python experiments/02-classical-pipeline-lgbm/classical_pipeline.py --predict-only --split-file X.jsonl --out Y.jsonl
  python experiments/02-classical-pipeline-lgbm/classical_pipeline.py --ablation --seed 42
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import argparse
import datetime
import json
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from mrc.evaluate import (aggregate, best_na_threshold, evaluate, f1, gold_texts, load_jsonl,
                          normalize_answer, per_example_scores)
from mrc.qa_text import (MONTHS, QTYPES, UNITS, IdfStats, Tok, content_stems, is_year, qtype,
                                tokenize, tokenize_context)

TOP_K = 2          # sentences passed to stage C
MAX_WORDS = 8      # max words per candidate span
SYMS = ["di", "pada", "tahun", "oleh", "sejak", "dari", "yaitu", "adalah", "ke", "dengan", "bernama", "sekitar",
        "sebesar", "dalam", "hingga", "sampai", "antara", "kepada", "untuk", "dan", "yang", "ini", "itu",
        "(", ")", ",", ".", "-", "\"", "'", ":", "persen", "juta", "ribu", "orang", "km"]
SYM_ID = {s: i + 1 for i, s in enumerate(SYMS)}
BOUND = len(SYMS) + 1  # sentence boundary marker id


# --------------------------------------------------------------------------------------- preprocessing
@dataclass
class Prep:
    ex: dict
    toks: list[Tok]
    spans: list[tuple[int, int]]
    ranges: list[tuple[int, int]]
    qt: list[Tok]
    qtp: int
    qs: list[str]           # question content stems
    gold: list[str]         # normalized gold strings ([] if unanswerable / no gold)
    gold_sents: set[int]    # sentence ids containing a gold answer start


def prep(ex: dict) -> Prep:
    toks, spans, ranges = tokenize_context(ex["context"])
    qt = tokenize(ex["question"])
    gold, gs = [], set()
    if not ex.get("is_impossible", True):
        gold = [normalize_answer(t) for t in gold_texts(ex) if normalize_answer(t)]
        for a in ex.get("answers", []):
            for si, (s, e) in enumerate(spans):
                if s <= a["answer_start"] < e + 1:
                    gs.add(si)
    return Prep(ex, toks, spans, ranges, qt, qtype(qt), content_stems(qt), gold, gs)


# --------------------------------------------------------------------------------------- stage B
class SentenceRetriever:
    """Rank sentences of one context for one question. Features -> standardized LR trained on 'sentence holds the answer'."""
    FEATS = ["tfidf", "bm25", "w_overlap", "overlap_ratio", "n_overlap", "bigram_overlap", "pos", "rel_pos", "is_first",
             "n_sent", "len", "type_hits", "tfidf_rel", "bm25_rel", "overlap_rel", "qtype"]

    def __init__(self, idf: IdfStats, seed: int):
        self.idf, self.seed = idf, seed
        self.model = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
        self.scaler = StandardScaler()

    def features(self, p: Prep) -> np.ndarray:
        qs = p.qs
        qset = set(qs)
        w = {s: self.idf.idf(s) for s in qset}
        wtot = sum(w.values()) or 1.0
        qbig = set(zip(qs, qs[1:]))
        rows = []
        for si, (a, b) in enumerate(p.ranges):
            st = [t.stem for t in p.toks[a:b] if not t.punct]
            sset = set(st)
            ov = qset & sset
            wov = sum(w[s] for s in ov)
            big = len(qbig & set(zip(st, st[1:])))
            hits = self._type_hits(p, a, b)
            rows.append([self.idf.tfidf_cos(qs, st), self.idf.bm25(qs, st), wov, wov / wtot, len(ov), big, si,
                         si / max(len(p.ranges) - 1, 1), float(si == 0), len(p.ranges), b - a, hits, 0, 0, 0, p.qtp])
        X = np.array(rows, dtype=float)
        for col, rel in ((0, 12), (1, 13), (2, 14)):
            X[:, rel] = X[:, col] / (X[:, col].max() + 1e-9)
        return X

    @staticmethod
    def _type_hits(p: Prep, a: int, b: int) -> int:
        ts = p.toks[a:b]
        if p.qtp in (QTYPES["when"], QTYPES["num"]):
            return sum(t.digit or t.low in MONTHS for t in ts)
        if p.qtp in (QTYPES["who"], QTYPES["where"]):
            return sum(t.cap for t in ts[1:])
        return 0

    def fit(self, preps: list[Prep]):
        X, y = [], []
        for p in preps:
            if not p.gold_sents:
                continue
            f = self.features(p)
            X.append(f)
            y += [int(i in p.gold_sents) for i in range(len(f))]
        X = np.vstack(X)
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def score(self, p: Prep, feats: np.ndarray | None = None) -> np.ndarray:
        f = self.features(p) if feats is None else feats
        return self.model.predict_proba(self.scaler.transform(f))[:, 1]

    def evaluate(self, preps: list[Prep], scored: dict[str, np.ndarray], feats: dict[str, np.ndarray]) -> dict:
        r1 = r3 = n = 0
        rk = {"tfidf": 0, "bm25": 0}
        for p in preps:
            if not p.gold_sents:
                continue
            n += 1
            order = list(np.argsort(-scored[p.ex["id"]]))
            r1 += order[0] in p.gold_sents
            r3 += bool(set(order[:3]) & p.gold_sents)
            f = feats[p.ex["id"]]
            rk["tfidf"] += int(np.argmax(f[:, 0])) in p.gold_sents
            rk["bm25"] += int(np.argmax(f[:, 1])) in p.gold_sents
        return {"n_answerable": n, "recall@1": r1 / n, "recall@3": r3 / n,
                "recall@1_tfidf_only": rk["tfidf"] / n, "recall@1_bm25_only": rk["bm25"] / n}


# --------------------------------------------------------------------------------------- stage A
A_GROUPS = {
    "question": ["qtype", "q_len", "q_has_tahun", "q_n_content", "q_n_caps"],
    "retrieval": ["max_tfidf", "max_bm25", "second_tfidf", "margin_tfidf", "ctx_tfidf", "mean_tfidf"],
    "overlap": ["best_overlap_ratio", "max_overlap_ratio", "missing_frac", "missing_w", "missing_n", "max_bigram",
                "ctx_coverage"],
    "entity_type": ["cap_cover", "type_hits_best", "type_hits_ctx", "num_cover"],
    "window": ["win10", "win5", "win_gap"],
    "context": ["ctx_len", "n_sent"],
}


class AnswerabilityClassifier:
    """LightGBM on question/retrieval/overlap/type features; outputs na_prob = P(unanswerable)."""

    def __init__(self, idf: IdfStats, seed: int, params: dict | None = None):
        self.idf, self.seed = idf, seed
        self.names = [n for g in A_GROUPS.values() for n in g]
        self.params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20,
                           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                           seed=seed, bagging_seed=seed, feature_fraction_seed=seed, deterministic=True,
                           force_row_wise=True, num_threads=4)
        self.params.update(params or {})
        self.rounds = 250
        self.booster: lgb.Booster | None = None

    def features(self, p: Prep) -> list[float]:
        qs, qset = p.qs, set(p.qs)
        w = {s: self.idf.idf(s) for s in qset}
        wtot = sum(w.values()) or 1.0
        sents = [[t.stem for t in p.toks[a:b] if not t.punct] for a, b in p.ranges]
        cos = [self.idf.tfidf_cos(qs, s) for s in sents]
        tf = sorted(cos, reverse=True) or [0.0]
        bm = [self.idf.bm25(qs, s) for s in sents]
        ratios = [sum(w[x] for x in qset & set(s)) / wtot for s in sents]
        best = int(np.argmax([c + 1e-6 * r for c, r in zip(cos, ratios)]))
        allst = [t.stem for t in p.toks if not t.punct]
        cset = set(allst)
        missing = [s for s in qset if s not in cset]
        bigs = [len(set(zip(qs, qs[1:])) & set(zip(s, s[1:]))) for s in sents]
        caps = [t for t in p.qt[1:] if t.cap and not t.punct]
        nums = [t for t in p.qt if t.digit]
        ctx_low = {t.low for t in p.toks}
        a, b = p.ranges[best]
        th_best = SentenceRetriever._type_hits(p, a, b)
        th_ctx = SentenceRetriever._type_hits(p, 0, len(p.toks))
        match_w = [w.get(t.stem, 0.0) if not t.punct else 0.0 for t in p.toks]
        wins = {}
        for L in (5, 10):
            vals = []
            for i in range(0, max(len(p.toks) - L + 1, 1)):
                seen = {p.toks[k].stem for k in range(i, min(i + L, len(p.toks))) if match_w[k] > 0}
                vals.append(sum(w[s] for s in seen) / wtot)
            wins[L] = max(vals) if vals else 0.0
        second = tf[1] if len(tf) > 1 else 0.0
        return [p.qtp, len(p.qt), float("tahun" in {t.low for t in p.qt}), len(qs), len(caps),
                tf[0], max(bm) if bm else 0.0, second, tf[0] - second,
                self.idf.tfidf_cos(qs, allst), float(np.mean(tf)),
                ratios[best], max(ratios), len(missing) / max(len(qset), 1), sum(w[s] for s in missing) / wtot, len(missing),
                max(bigs) if bigs else 0, 1 - sum(w[s] for s in missing) / wtot,
                (sum(t.low in ctx_low or t.stem in cset for t in caps) / len(caps)) if caps else 1.0,
                th_best, th_ctx,
                (sum(t.low in ctx_low for t in nums) / len(nums)) if nums else 1.0,
                wins[10], wins[5], wins[10] - wins[5], len(p.toks), len(p.ranges)]

    def fit(self, X: np.ndarray, y: np.ndarray, cols: list[int] | None = None):
        cols = cols if cols is not None else list(range(X.shape[1]))
        self.cols = cols
        ds = lgb.Dataset(X[:, cols], y, feature_name=[self.names[c] for c in cols])
        self.booster = lgb.train(self.params, ds, self.rounds)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X[:, self.cols])

    def importance(self) -> dict[str, float]:
        g = self.booster.feature_importance("gain")
        return dict(zip(self.booster.feature_name(), g / g.sum()))


# --------------------------------------------------------------------------------------- stage C
C_GROUPS = {
    "type_match": ["q_type", "q_tahun", "c_is_year", "c_has_year", "c_is_num", "c_has_digit", "c_has_month", "c_cap_frac",
                   "c_first_cap", "c_all_cap", "c_has_unit", "type_match", "c_in_paren"],
    "overlap": ["ov_frac", "ov_cnt", "ov_first", "ov_all"],
    "proximity": ["d_left", "d_right", "d_min", "n_match_w3", "n_match_w6", "w_match_w5", "mean_dist", "frac_left",
                  "n_match_sent"],
    "context_words": ["left1_id", "right1_id", "left2_id", "left1_in_q", "right1_in_q", "left2_in_q", "right2_in_q"],
    "shape_boundary": ["c_len_tok", "c_len_chr", "rel_pos_sent", "is_sent_start", "is_sent_end", "rel_pos_ctx",
                       "left_is_cap", "right_is_cap", "left_is_digit", "right_is_digit", "maximal_run"],
    "pos_heuristics": ["stop_frac", "verb_like_frac", "left_verb_like", "right_verb_like", "internal_punct", "n_comma_in"],
    "sentence": ["s_rank", "s_score", "s_score_rel", "s_tfidf", "s_bm25", "s_overlap"],
}
C_NAMES = [n for g in C_GROUPS.values() for n in g]
KEEP_INNER_PUNCT = set(",-()\"'/:&.%")


def _verb_like(t: Tok) -> bool:
    return (not t.punct and not t.cap and len(t.low) > 5 and t.low.startswith(("me", "ber", "ter", "di")))


def gen_candidates(p: Prep, si: int) -> list[tuple[int, int]]:
    """All word n-grams (<= MAX_WORDS words) inside sentence si, filtered by boundary rules."""
    a, b = p.ranges[si]
    toks = p.toks
    out = []
    for i in range(a, b):
        t = toks[i]
        if t.punct or (t.stop and not t.cap and not t.digit):
            continue
        words = 0
        depth = 0
        for j in range(i, b):
            u = toks[j]
            if u.punct:
                if u.text not in KEEP_INNER_PUNCT:
                    break
                depth += (u.text == "(") - (u.text == ")")
                if depth < 0:
                    break
                continue
            words += 1
            if words > MAX_WORDS:
                break
            if u.stop and not u.cap and not u.digit:
                continue
            if depth == 0:
                out.append((i, j))
    return out


class SpanRanker:
    """Candidate generation + feature extraction + LightGBM scoring of spans inside the top-k sentences."""

    def __init__(self, idf: IdfStats, seed: int, objective: str = "binary", params: dict | None = None):
        self.idf, self.seed, self.objective = idf, seed, objective
        self.params = dict(objective=objective, learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
                           feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                           seed=seed, bagging_seed=seed, feature_fraction_seed=seed, deterministic=True,
                           force_row_wise=True, num_threads=4)
        if objective == "lambdarank":
            self.params.update(metric="ndcg", eval_at=[1], lambdarank_truncation_level=10)
        self.params.update(params or {})
        self.rounds = 300
        self.booster: lgb.Booster | None = None

    def build(self, p: Prep, sent_scores: np.ndarray, sent_feats: np.ndarray):
        """Return (rows, cands, labels) for the top-k sentences of one example."""
        order = list(np.argsort(-sent_scores))[:TOP_K]
        rows, cands, labels = [], [], []
        qset = set(p.qs)
        qlow = {t.low for t in p.qt}
        w = {s: self.idf.idf(s) for s in qset}
        ctx = p.ex["context"]
        for rank, si in enumerate(order):
            a, b = p.ranges[si]
            matched = [i for i in range(a, b) if p.toks[i].stem in qset]
            sfeat = [rank, sent_scores[si], sent_scores[si] - sent_scores[order[0]], sent_feats[si, 0], sent_feats[si, 1],
                     sent_feats[si, 3]]
            for (i, j) in gen_candidates(p, si):
                rows.append(self._cand_features(p, a, b, i, j, matched, qset, qlow, w, sfeat))
                cs, ce = p.toks[i].s, p.toks[j].e
                cands.append((cs, ce, si))
                labels.append(int(normalize_answer(ctx[cs:ce]) in p.gold) if p.gold else 0)
        return rows, cands, labels

    def _cand_features(self, p, a, b, i, j, matched, qset, qlow, w, sfeat):
        T = p.toks
        span = T[i:j + 1]
        words = [t for t in span if not t.punct]
        nw = len(words)
        c_is_year = float(nw == 1 and is_year(words[0].text))
        c_has_year = float(any(is_year(t.text) for t in words))
        c_is_num = float(all(t.digit for t in words))
        c_has_digit = float(any(t.digit for t in words))
        c_has_month = float(any(t.low in MONTHS for t in words))
        cap_frac = sum(t.cap for t in words) / nw
        first_cap = float(words[0].cap)
        all_cap = float(all(t.cap or t.digit for t in words))
        has_unit = float(any(t.low in UNITS for t in words)) + float(i > a and T[i - 1].low in UNITS) * 0.5
        qtp = p.qtp
        if qtp == QTYPES["when"]:
            tm = max(c_is_year, c_has_year, c_has_month, c_has_digit * 0.5)
        elif qtp == QTYPES["num"]:
            tm = max(c_is_num, c_has_digit * 0.7)
        elif qtp in (QTYPES["who"], QTYPES["where"]):
            tm = all_cap * (1 - c_has_digit)
        else:
            tm = 0.0
        depth = 0
        for k in range(a, i):
            depth += (T[k].text == "(") - (T[k].text == ")")
        ov = [t.stem in qset or t.low in qlow for t in words]
        outside = [m for m in matched if m < i or m > j]
        dl = min((i - m for m in outside if m < i), default=30)
        dr = min((m - j for m in outside if m > j), default=30)
        dists = [min(abs(i - m), abs(j - m)) for m in outside]
        w5 = sum(w.get(T[m].stem, 0.0) for m, d in zip(outside, dists) if d <= 5)
        mean_d = float(np.mean(dists)) if dists else 30.0
        frac_left = (sum(m < i for m in outside) / len(outside)) if outside else 0.5

        def tk(k):
            return T[k] if a <= k < b else None
        l1, r1, l2, r2 = tk(i - 1), tk(j + 1), tk(i - 2), tk(j + 2)

        def sid(t):
            return BOUND if t is None else SYM_ID.get(t.low, 0)

        def inq(t):
            return float(t is not None and (t.stem in qset or t.low in qlow) and not t.punct)

        def capw(t):
            return float(t is not None and t.cap and not t.punct)
        sent_len = b - a
        maximal = float(first_cap and not capw(l1) and not capw(r1))
        return [qtp, float("tahun" in {t.low for t in p.qt}), c_is_year, c_has_year, c_is_num, c_has_digit, c_has_month,
                cap_frac, first_cap, all_cap, has_unit, tm, float(depth > 0),
                sum(ov) / nw, float(sum(ov)), float(ov[0]), float(all(ov)),
                dl, dr, min(dl, dr), sum(d <= 3 for d in dists), sum(d <= 6 for d in dists), w5, mean_d, frac_left, len(outside),
                sid(l1), sid(r1), sid(l2), inq(l1), inq(r1), inq(l2), inq(r2),
                nw, T[j].e - T[i].s, (i - a) / sent_len, float(i == a), float(j == b - 1 or (j == b - 2 and T[b - 1].punct)),
                i / max(len(T), 1), capw(l1), capw(r1),
                float(l1 is not None and l1.digit), float(r1 is not None and r1.digit), maximal,
                sum(t.stop for t in words) / nw, sum(_verb_like(t) for t in words) / nw,
                float(l1 is not None and _verb_like(l1)), float(r1 is not None and _verb_like(r1)),
                float(any(t.punct for t in span)), float(sum(t.text == "," for t in span)),
                *sfeat]

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, cols: list[int] | None = None):
        cols = cols if cols is not None else list(range(X.shape[1]))
        self.cols = cols
        names = [C_NAMES[c] for c in cols]
        cat = [n for n in ("q_type", "left1_id", "right1_id", "left2_id") if n in names]
        ds = lgb.Dataset(X[:, cols], y, feature_name=names, categorical_feature=cat or "auto",
                         group=groups[groups > 0] if self.objective == "lambdarank" else None)
        if self.objective == "lambdarank":
            keep = np.repeat(groups > 0, groups)
            ds = lgb.Dataset(X[keep][:, cols], y[keep], feature_name=names, categorical_feature=cat or "auto",
                             group=groups[groups > 0])
        self.booster = lgb.train(self.params, ds, self.rounds)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X[:, self.cols])

    def importance(self) -> dict[str, float]:
        g = self.booster.feature_importance("gain")
        return dict(zip(self.booster.feature_name(), g / max(g.sum(), 1e-9)))


# --------------------------------------------------------------------------------------- pipeline
class Featurized:
    """Everything computed once per split so that ablations only retrain models."""

    def __init__(self, preps, retr: SentenceRetriever, A: AnswerabilityClassifier, C: SpanRanker):
        self.preps = preps
        self.sfeats = {p.ex["id"]: retr.features(p) for p in preps}
        self.sscores = {p.ex["id"]: retr.score(p, self.sfeats[p.ex["id"]]) for p in preps}
        self.XA = np.array([A.features(p) for p in preps], dtype=float)
        rows, labels, self.cands, grp, owner = [], [], [], [], []
        for n, p in enumerate(preps):
            r, c, l = C.build(p, self.sscores[p.ex["id"]], self.sfeats[p.ex["id"]])
            rows += r
            labels += l
            self.cands.append(c)
            grp.append(len(r))
        self.XC = np.array(rows, dtype=float).reshape(-1, len(C_NAMES))
        self.yC = np.array(labels)
        self.grp = np.array(grp)
        self.offs = np.concatenate([[0], np.cumsum(self.grp)])
        self.yA = np.array([int(p.ex.get("is_impossible", False)) for p in preps])


def top_spans(fz: Featurized, scores: np.ndarray) -> list[tuple[str, int, float, int]]:
    """Per example (text, start_char, score, candidate index) of the top candidate."""
    out = []
    for n, p in enumerate(fz.preps):
        k = fz.grp[n]
        if k == 0:
            out.append(("", -1, 0.0, -1))
            continue
        sc = scores[fz.offs[n]: fz.offs[n] + k]
        m = int(np.argmax(sc))
        s, e, _ = fz.cands[n][m]
        out.append((p.ex["context"][s:e], s, float(sc[m]), m))
    return out


def make_preds(fz: Featurized, spans, na) -> dict[str, dict]:
    return {p.ex["id"]: {"id": p.ex["id"], "pred": sp[0], "na_prob": float(n), "score": sp[2], "start_char": sp[1]}
            for p, sp, n in zip(fz.preps, spans, na)}


def final_scores(gold: list[dict], preds: dict) -> dict:
    t, _ = best_na_threshold(gold, preds)
    m = aggregate(per_example_scores(gold, preds, t))
    m["threshold"] = t
    return m


class ClassicalPipeline:
    def __init__(self, seed: int, objective: str = "lambdarank"):
        self.seed, self.objective = seed, objective

    def fit(self, train: list[dict], log=print):
        t0 = time.time()
        self.idf = IdfStats(list({e["context"] for e in train}))
        tp = [prep(e) for e in train]
        self.B = SentenceRetriever(self.idf, self.seed).fit(tp)
        log(f"stage B fitted {time.time()-t0:.1f}s")
        self.A = AnswerabilityClassifier(self.idf, self.seed)
        self.C = SpanRanker(self.idf, self.seed, self.objective)
        self.fz_train = Featurized(tp, self.B, self.A, self.C)
        log(f"train featurized: A {self.fz_train.XA.shape}, C {self.fz_train.XC.shape}, pos {self.fz_train.yC.sum()} {time.time()-t0:.1f}s")
        self.A.fit(self.fz_train.XA, self.fz_train.yA)
        self.C.fit(self.fz_train.XC, self.fz_train.yC, self.fz_train.grp)
        log(f"stages A, C fitted {time.time()-t0:.1f}s")
        return self

    def featurize(self, examples: list[dict]) -> Featurized:
        return Featurized([prep(e) for e in examples], self.B, self.A, self.C)

    def predict(self, fz: Featurized) -> tuple[dict, list, np.ndarray]:
        spans = top_spans(fz, self.C.predict(fz.XC))
        na = self.A.predict(fz.XA)
        return make_preds(fz, spans, na), spans, na


# --------------------------------------------------------------------------------------- evaluation of stages
def stage_metrics(pipe: ClassicalPipeline, fz: Featurized, spans, na, thr: float) -> dict:
    ans = [i for i, p in enumerate(fz.preps) if p.gold]
    y = fz.yA
    m: dict = {"A": {"auc": float(roc_auc_score(y, na)), "acc_at_thr": float(np.mean((na > thr).astype(int) == y)),
                     "acc_at_0.5": float(np.mean((na > 0.5).astype(int) == y)), "threshold": thr}}
    m["B"] = pipe.B.evaluate(fz.preps, fz.sscores, fz.sfeats)
    m["B"][f"recall@{TOP_K}"] = float(np.mean([bool(set(np.argsort(-fz.sscores[fz.preps[i].ex["id"]])[:TOP_K])
                                                    & fz.preps[i].gold_sents) for i in ans]))
    has_gold_cand, em_given, f1_given, em_all, em_given1 = [], [], [], [], []
    for i in ans:
        p = fz.preps[i]
        off, k = int(fz.offs[i]), int(fz.grp[i])
        has_gold_cand.append(bool(fz.yC[off: off + k].sum() > 0))
        pred = spans[i][0]
        em = float(normalize_answer(pred) in p.gold)
        em_all.append(em)
        top = list(np.argsort(-fz.sscores[p.ex["id"]]))
        if set(top[:TOP_K]) & p.gold_sents:
            em_given.append(em)
            f1_given.append(max(f1(g, pred) for g in gold_texts(p.ex)))
        if top[0] in p.gold_sents:
            em_given1.append(em)
    m["C"] = {"candidate_recall": float(np.mean(has_gold_cand)),
              "avg_candidates_per_question": float(fz.grp.mean()),
              "span_em_given_gold_sentence_in_topk": float(np.mean(em_given)),
              "span_f1_given_gold_sentence_in_topk": float(np.mean(f1_given)),
              "span_em_given_gold_sentence_rank1": float(np.mean(em_given1)),
              "span_em_on_answerable_unconditional": float(np.mean(em_all))}
    return m


def error_attribution(fz: Featurized, preds: dict, gold: list[dict], thr: float) -> tuple[dict, list[dict]]:
    """Attribute each dev error (EM = 0) to the earliest failing stage: A (answerability), B (gold sentence not in
    top-k), C_gen (gold not among candidates), C_rank (gold candidate not ranked first)."""
    sc = per_example_scores(gold, preds, thr)
    counts = {"A_false_answer": 0, "A_false_no_answer": 0, "B": 0, "C_gen": 0, "C_rank": 0}
    errs = []
    for i, p in enumerate(fz.preps):
        r = sc[p.ex["id"]]
        if r["em"] == 1:
            continue
        if not p.gold:
            stage = "A_false_answer"
        elif r["pred"] == "":
            stage = "A_false_no_answer"
        else:
            top = set(np.argsort(-fz.sscores[p.ex["id"]])[:TOP_K])
            off, k = int(fz.offs[i]), int(fz.grp[i])
            if not (top & p.gold_sents):
                stage = "B"
            elif fz.yC[off: off + k].sum() == 0:
                stage = "C_gen"
            else:
                stage = "C_rank"
        counts[stage] += 1
        errs.append({"id": p.ex["id"], "stage": stage, "question": p.ex["question"], "gold": gold_texts(p.ex)[0],
                     "pred": r["pred"], "f1": r["f1"], "na_prob": preds[p.ex["id"]]["na_prob"]})
    counts["total_errors"] = sum(counts.values())
    counts["n"] = len(fz.preps)
    return counts, errs


# --------------------------------------------------------------------------------------- ablation
def ablation(pipe: ClassicalPipeline, fzd: Featurized, dev: list[dict]):
    spans0, na0 = top_spans(fzd, pipe.C.predict(fzd.XC)), pipe.A.predict(fzd.XA)
    base = final_scores(dev, make_preds(fzd, spans0, na0))
    ans = [i for i, p in enumerate(fzd.preps) if p.gold]

    def has_f1(spans):
        return 100 * float(np.mean([max(f1(g, spans[i][0]) for g in gold_texts(fzd.preps[i].ex)) for i in ans]))
    base_hasf1 = has_f1(spans0)
    ft = pipe.fz_train
    a_rows, c_rows = [], []
    for g, feats in A_GROUPS.items():
        cols = [i for i, n in enumerate(pipe.A.names) if n not in feats]
        A = AnswerabilityClassifier(pipe.idf, pipe.seed).fit(ft.XA, ft.yA, cols)
        na = A.predict(fzd.XA)
        r = final_scores(dev, make_preds(fzd, spans0, na))
        a_rows.append({"group": g, "n_feat": len(feats), "f1": r["f1"], "d_f1": r["f1"] - base["f1"], "em": r["exact"],
                       "auc": float(roc_auc_score(fzd.yA, na)), "noans_acc": r["NoAns_acc"]})
    for g, feats in C_GROUPS.items():
        cols = [i for i, n in enumerate(C_NAMES) if n not in feats]
        C = SpanRanker(pipe.idf, pipe.seed, pipe.objective).fit(ft.XC, ft.yC, ft.grp, cols)
        spans = top_spans(fzd, C.predict(fzd.XC))
        r = final_scores(dev, make_preds(fzd, spans, na0))
        h = has_f1(spans)
        c_rows.append({"group": g, "n_feat": len(feats), "f1": r["f1"], "d_f1": r["f1"] - base["f1"], "em": r["exact"],
                       "hasans_f1": h, "d_hasans_f1": h - base_hasf1})
    return a_rows, c_rows, {"base": base, "base_hasans_f1": base_hasf1}


def ablation_markdown(a_rows, c_rows, info, seed) -> str:
    b = info["base"]
    s = [f"# Feature-group ablation (dev, seed {seed})", "",
         "Each row retrains one stage without one feature group, keeping the other stage at its full-feature model. "
         "F1 is overall dev F1 at the dev-tuned no-answer threshold (re-tuned per row). Negative delta means the group helps.", "",
         f"Full model: F1 {b['f1']:.2f}, EM {b['exact']:.2f}, span-only HasAns F1 (ignores stage A, answerable questions) {info['base_hasans_f1']:.2f}.", "",
         "## Stage C (span ranker)", "", "| dropped group | #feat | F1 | dF1 | EM | span F1 (answerable, no thr) | d span F1 |", "|---|---|---|---|---|---|---|"]
    for r in sorted(c_rows, key=lambda r: r["d_f1"]):
        s.append(f"| {r['group']} | {r['n_feat']} | {r['f1']:.2f} | {r['d_f1']:+.2f} | {r['em']:.2f} | {r['hasans_f1']:.2f} | {r['d_hasans_f1']:+.2f} |")
    s += ["", "## Stage A (answerability classifier)", "", "| dropped group | #feat | F1 | dF1 | EM | AUC | NoAns acc |", "|---|---|---|---|---|---|---|"]
    for r in sorted(a_rows, key=lambda r: r["d_f1"]):
        s.append(f"| {r['group']} | {r['n_feat']} | {r['f1']:.2f} | {r['d_f1']:+.2f} | {r['em']:.2f} | {r['auc']:.3f} | {r['noans_acc']:.2f} |")
    s += ["", "Note: single seed, 803 dev questions; differences below about 0.5 F1 are within noise."]
    return "\n".join(s) + "\n"


def importance_markdown(pipe: ClassicalPipeline, seed: int) -> str:
    def table(title, imp, groups):
        inv = {n: g for g, ns in groups.items() for n in ns}
        out = [f"## {title}", "", "| feature | group | gain share |", "|---|---|---|"]
        out += [f"| {n} | {inv[n]} | {v:.3f} |" for n, v in sorted(imp.items(), key=lambda kv: -kv[1])[:15]]
        gs: dict[str, float] = {}
        for n, v in imp.items():
            gs[inv[n]] = gs.get(inv[n], 0) + v
        out += ["", "Gain share by group: " + ", ".join(f"{g} {v:.3f}" for g, v in sorted(gs.items(), key=lambda kv: -kv[1])), ""]
        return "\n".join(out)
    coefs = dict(zip(SentenceRetriever.FEATS, pipe.B.model.coef_[0]))
    b = ["## Stage B (standardized LR coefficients)", "", "| feature | coef |", "|---|---|"]
    b += [f"| {n} | {v:+.3f} |" for n, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1]))]
    return (f"# Feature importance (seed {seed}, LightGBM gain share, top 15)\n\n" + table("Stage A", pipe.A.importance(), A_GROUPS)
            + "\n" + table("Stage C", pipe.C.importance(), C_GROUPS) + "\n" + "\n".join(b) + "\n")


# --------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--split-file", default="data/processed/dev.jsonl")
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--objective", default="lambdarank", choices=["binary", "lambdarank"])
    ap.add_argument("--ablation", action="store_true")
    a = ap.parse_args()
    exp = ROOT / "experiments/02-classical-pipeline-lgbm"
    out = exp / f"runs/classical_pipeline/seed{a.seed}"
    logf: list[str] = []

    def log(msg):
        print(msg, flush=True)
        logf.append(msg)

    train = load_jsonl(ROOT / "data/processed/train.jsonl")
    dev = load_jsonl(ROOT / "data/processed/dev.jsonl")
    pipe = ClassicalPipeline(a.seed, a.objective).fit(train, log)
    fzd = pipe.featurize(dev)
    preds, spans, na = pipe.predict(fzd)
    thr, _ = best_na_threshold(dev, preds)
    log(f"dev threshold {thr:.4f}")

    if a.predict_only:
        split = load_jsonl(a.split_file)
        pr, _, _ = pipe.predict(pipe.featurize(split))
        dest = Path(a.out or "preds.jsonl")
        if dest.suffix != ".jsonl":
            dest.mkdir(parents=True, exist_ok=True)
            dest = dest / "preds.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            for e in split:
                r = pr[e["id"]]
                f.write(json.dumps({"id": r["id"], "pred": "" if r["na_prob"] > thr else r["pred"], "na_prob": r["na_prob"],
                                    "score": r["score"], "start_char": r["start_char"], "span_pred": r["pred"],
                                    "na_threshold": thr}, ensure_ascii=False) + "\n")
        log(f"wrote {dest}")
        return

    out.mkdir(parents=True, exist_ok=True)
    split_name = Path(a.split_file).stem
    with open(out / f"preds_{split_name}.jsonl", "w", encoding="utf-8") as f:
        for e in dev:
            r = preds[e["id"]]
            f.write(json.dumps({"id": r["id"], "pred": r["pred"], "na_prob": r["na_prob"], "score": r["score"],
                                "start_char": r["start_char"]}, ensure_ascii=False) + "\n")
    m = evaluate(ROOT / "data/processed/dev.jsonl", out / f"preds_{split_name}.jsonl", search=True)
    (out / "metrics_dev.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    sm = stage_metrics(pipe, fzd, spans, na, thr)
    counts, errs = error_attribution(fzd, preds, dev, thr)
    final = m["at_best_threshold"]
    (out / "stage_metrics.json").write_text(json.dumps({"stages": sm, "errors": counts, "final_at_dev_thr": final}, indent=2), encoding="utf-8")
    (out / "errors_dev.json").write_text(json.dumps(errs, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "importance.json").write_text(json.dumps({"A": pipe.A.importance(), "C": pipe.C.importance()}, indent=2), encoding="utf-8")
    cfg = {"model": "classical_pipeline", "seed": a.seed, "objective": a.objective, "top_k_sentences": TOP_K, "max_words": MAX_WORDS,
           "A_params": pipe.A.params, "A_rounds": pipe.A.rounds, "C_params": pipe.C.params, "C_rounds": pipe.C.rounds,
           "na_threshold": thr, "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    log(f"DEV EM {final['exact']:.2f} F1 {final['f1']:.2f} HasAns_F1 {final['HasAns_f1']:.2f} NoAns_acc {final['NoAns_acc']:.2f}")
    log(json.dumps(sm, indent=1))
    log(json.dumps(counts))
    if a.ablation:
        a_rows, c_rows, info = ablation(pipe, fzd, dev)
        (exp / "analysis/ablation.md").write_text(ablation_markdown(a_rows, c_rows, info, a.seed), encoding="utf-8")
        (exp / "analysis/importance.md").write_text(importance_markdown(pipe, a.seed), encoding="utf-8")
        log("wrote analysis/ablation.md, analysis/importance.md")
    (out / "log.txt").write_text("\n".join(logf), encoding="utf-8")


if __name__ == "__main__":
    main()
