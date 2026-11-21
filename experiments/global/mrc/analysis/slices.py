"""Deterministic per-example slice tags (input-only features, no model).

CLI: python experiments/global/run.py mrc.analysis.slices [--split-file path ...]  -> data/processed/slices_{split}.jsonl
"""
import argparse
import json
import re
from pathlib import Path

from mrc.paths import ROOT
PROC = ROOT / "data/processed"

STOP = set("""yang di ke dari dan atau pada untuk dengan adalah itu ini oleh dalam sebagai akan telah sudah juga
tidak ada apa siapa siapakah apakah kapan kapankah dimana dimanakah mana manakah bagaimana bagaimanakah mengapa kenapa
berapa berapakah saat sebuah suatu para si sang tersebut yaitu yakni agar karena jika bila saja lebih sangat
mereka dia ia nya kah lah pun serta antara hingga sampai secara dapat bisa seorang oleh atas bawah""".split())

QW_NORM = {"kapankah": "kapan", "siapakah": "siapa", "apakah": "apa", "dimanakah": "dimana", "dimana": "dimana",
           "manakah": "mana", "bagaimanakah": "bagaimana", "berapakah": "berapa", "mengapakah": "mengapa",
           "kenapa": "mengapa", "adakah": "ada", "bilakah": "bila"}
QWORDS = {"apa", "siapa", "kapan", "dimana", "di", "mana", "bagaimana", "berapa", "mengapa", "kenapa", "bila",
          "yang", "dari", "ke", "pada", "untuk", "dengan", "adakah", "apakah"}

TOK = re.compile(r"\w+", re.U)
MONTHS = "januari februari maret april mei juni juli agustus september oktober november desember".split()


def words(s):
    return TOK.findall(s)


def q_word(q):
    ws = words(q.lower())
    raw = ws[0] if ws else ""
    # first interrogative token anywhere in the first 3 words, else first word
    for w in ws[:3]:
        n = QW_NORM.get(w, w)
        if n in {"kapan", "siapa", "apa", "dimana", "mana", "bagaimana", "berapa", "mengapa", "bila", "ada"}:
            return raw, n
    return raw, QW_NORM.get(raw, raw)


def q_type(q, qw):
    ql = q.lower()
    if qw in ("kapan", "bila") or re.search(r"\b(tahun berapa|tanggal berapa|bulan apa|pada tahun)\b", ql):
        return "time"
    if qw == "siapa" or re.search(r"\bsiapa", ql):
        return "person"
    if qw == "dimana" or qw == "mana" and re.search(r"\b(negara|kota|daerah|wilayah|provinsi)\b", ql) \
            or re.search(r"\b(di mana|dimana|ke mana|dari mana|di)\b", " ".join(ql.split()[:2])):
        return "location"
    if qw == "berapa" or re.search(r"\bberapa\b|\bjumlah\b", ql):
        return "number"
    if qw in ("apa", "mana") or re.search(r"\bapa\b", ql):
        return "entity_or_def"
    if qw in ("ada",) or re.match(r"^(apakah|adakah|benarkah)\b", ql):
        return "yesno_like"
    return "other"


def bucket(v, edges, labels):
    for e, l in zip(edges, labels):
        if v < e:
            return l
    return labels[-1]


def sentences(ctx):
    """List of (start, end) char spans."""
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.!?])(?:\[\d+\])*\s+(?=[A-Z0-9\"'(])", ctx):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(ctx)))
    return spans


def content_words(s):
    return {w for w in words(s.lower()) if w not in STOP and len(w) > 1}


def overlap(qcw, sent):
    if not qcw:
        return 0.0
    sw = set(words(sent.lower()))
    return len(qcw & sw) / len(qcw)


def ans_type(a):
    t = a.strip()
    if not t:
        return "none"
    tl = t.lower()
    if re.fullmatch(r"(1[0-9]|20)\d\d(-an)?", tl) or any(m in tl.split() for m in MONTHS) and re.search(r"\d", tl):
        return "date_year"
    if re.search(r"\d", tl) and len(re.sub(r"[\d\s.,%/-]", "", tl)) <= max(3, len(tl) // 3):
        return "number"
    ws = t.split()
    if t[0].isupper() or t[0].isdigit():
        caps = sum(w[0].isupper() for w in ws if w[0].isalpha())
        alpha = sum(w[0].isalpha() for w in ws)
        if 1 <= len(ws) <= 4 and alpha and caps == alpha:
            return "person_like_cap" if len(ws) >= 2 and not re.search(r"\b(of|the|republik|kota|negara)\b", tl) \
                else "other_cap"
        return "other_cap"
    return "lowercase_phrase"


def tag_example(ex):
    ctx, q = ex["context"], ex["question"]
    raw, qw = q_word(q)
    golds = [a for a in ex.get("answers", []) if a.get("text", "").strip()]
    sents = sentences(ctx)
    qcw = content_words(q)
    out = {"id": ex["id"], "q_word": qw, "q_word_raw": raw, "q_type": q_type(q, qw),
           "is_impossible": bool(ex.get("is_impossible", not golds)), "n_gold": len(golds),
           "has_citation": bool(re.search(r"\[\d+\]", ctx)), "n_ctx_sents": len(sents)}
    nw = len(words(ctx))
    out["ctx_len_words"], out["q_len_words"] = nw, len(words(q))
    out["ctx_len_bucket"] = bucket(nw, [60, 120, 251], ["<60", "60-120", "120-250", ">250"])
    ovs = [overlap(qcw, ctx[s:e]) for s, e in sents]
    if golds:
        a = golds[0]
        at, st = a["text"], a["answer_start"]
        na = len(words(at))
        out["ans_len_words"] = na
        out["ans_len_bucket"] = bucket(na, [2, 4, 7], ["1", "2-3", "4-6", ">6"])
        out["ans_type"] = ans_type(at)
        out["ans_rel_pos_bucket"] = f"Q{min(3, int(4 * st / max(1, len(ctx)))) + 1}"
        si = next((i for i, (s, e) in enumerate(sents) if s <= st < e + 1), len(sents) - 1)
        out["ans_sent_idx"] = si
        ov = ovs[si]
    else:
        out.update(ans_len_words=0, ans_len_bucket="none", ans_type="none", ans_rel_pos_bucket="none", ans_sent_idx=-1)
        ov = max(ovs) if ovs else 0.0
    out["lex_overlap"] = round(ov, 4)
    out["lex_overlap_bucket"] = "low" if ov < 0.34 else ("high" if ov >= 0.67 else "mid")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", nargs="*", default=[str(PROC / f"{s}.jsonl") for s in ("train", "dev", "itest")])
    a = ap.parse_args()
    for p in a.split_file:
        p = Path(p)
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        out = PROC / f"slices_{p.stem}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(tag_example(r), ensure_ascii=False) + "\n")
        print(p.stem, len(rows), "->", out)


if __name__ == "__main__":
    main()
