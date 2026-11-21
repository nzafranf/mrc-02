"""Shared text utilities for the non-neural models: tokenizer with offsets, sentence splitter,
Indonesian stopwords, light stemmer, IDF statistics, question-type detection, token typing."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|\w+(?:-\w+)*|[^\w\s]")

STOPWORDS = set("""
yang di dan itu dengan untuk tidak ini dari dalam akan pada juga saya ke karena tersebut bisa ada mereka lebih kamu
kami kita atau oleh saat harus sebagai sudah telah dapat adalah itulah merupakan ialah yaitu yakni serta sebuah
seorang suatu para sebuah antara setelah sebelum sejak hingga sampai ketika sehingga namun tetapi tapi hanya
sangat lalu kemudian masih pun lagi bagi dalam atas bawah selama sedang agar supaya jika bila apabila seperti
sebagai terhadap tentang oleh secara dia ia nya lain semua setiap beberapa banyak kembali setelahnya
apa apakah siapa siapakah kapan kapankah dimana dimanakah mana manakah berapa berapakah mengapa kenapa bagaimana
bagaimanakah di mana mananya manakah bernama disebut disebutkan dikenal
""".split())
QUESTION_WORDS = set("apa apakah siapa siapakah kapan kapankah dimana dimanakah mana manakah berapa berapakah mengapa kenapa bagaimana bagaimanakah".split())
MONTHS = set("januari februari maret april mei juni juli agustus september oktober november desember "
             "jan feb mar apr jun jul agu agt sep okt nov des".split())
UNITS = set("persen juta ribu miliar milyar triliun km kg meter orang jiwa tahun abad hari bulan jam menit kilometer "
            "ton hektare hektar rupiah dolar km2 m2 mil derajat unit buah kali".split())
ABBREV = {"dr", "prof", "ir", "no", "st", "jl", "hj", "h", "mr", "tn", "bpk", "drs", "dll", "dsb", "yth", "sdr", "ny", "vs", "cth"}

QTYPES = {"other": 0, "when": 1, "who": 2, "num": 3, "where": 4, "what": 5, "why": 6, "how": 7}
_PREFIXES = ("meng", "meny", "men", "mem", "peng", "peny", "pen", "pem", "ber", "ter", "me", "pe", "di", "ke", "se", "be")
_SUFFIXES = ("kah", "lah", "pun", "nya", "kan", "an", "i")


def is_punct(t: str) -> bool:
    return not (t[0].isalnum() or t[0] == "_")


def stem(w: str) -> str:
    w = w.lower()
    if w.isdigit() or len(w) < 5:
        return w
    for s in _SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 4:
            w = w[: -len(s)]
            break
    for p in _PREFIXES:
        if w.startswith(p) and len(w) - len(p) >= 4:
            w = w[len(p):]
            break
    return w


@dataclass
class Tok:
    text: str
    s: int
    e: int
    low: str
    stem: str
    punct: bool
    stop: bool
    digit: bool
    cap: bool
    sent: int = 0


def tokenize(text: str) -> list[Tok]:
    out = []
    for m in TOKEN_RE.finditer(text):
        t = m.group()
        p = is_punct(t)
        low = t.lower()
        out.append(Tok(t, m.start(), m.end(), low, low if p else stem(t), p, low in STOPWORDS,
                       t[0].isdigit(), t[0].isupper()))
    return out


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Char spans of sentences. Boundary = .?! (+ optional [n] citations) + space + uppercase/digit/quote."""
    spans, start = [], 0
    for m in re.finditer(r"[.!?](?:\[\d+\])*(?:\s+)", text):
        nxt = text[m.end(): m.end() + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in "\"“(‘'"):
            continue
        prev = re.search(r"(\w+)$", text[start: m.start()])
        if text[m.start()] == "." and prev and (len(prev.group(1)) == 1 or prev.group(1).lower() in ABBREV):
            continue
        spans.append((start, m.start() + 1 + len(re.match(r"(?:\[\d+\])*", text[m.start() + 1:]).group())))
        start = m.end()
    if start < len(text.rstrip()):
        spans.append((start, len(text.rstrip())))
    return spans or [(0, len(text))]


def tokenize_context(ctx: str) -> tuple[list[Tok], list[tuple[int, int]], list[tuple[int, int]]]:
    """Returns tokens (with .sent set), sentence char spans, sentence token ranges [a,b)."""
    toks = tokenize(ctx)
    spans = split_sentences(ctx)
    ranges, k = [], 0
    for si, (a, b) in enumerate(spans):
        first = k
        while k < len(toks) and toks[k].s < b:
            toks[k].sent = si
            k += 1
        ranges.append((first, k))
    return toks, spans, ranges


def qtype(qtoks: list[Tok]) -> int:
    words = [t.low for t in qtoks]
    ws = set(words)
    if ws & {"kapan", "kapankah", "bila"}:
        return QTYPES["when"]
    if ws & {"siapa", "siapakah", "siapakan"}:
        return QTYPES["who"]
    if ws & {"berapa", "berapakah"}:
        return QTYPES["num"]
    if ws & {"mana", "manakah", "dimana", "dimanakah", "kemana"}:
        return QTYPES["where"]
    if ws & {"mengapa", "kenapa"}:
        return QTYPES["why"]
    if ws & {"bagaimana", "bagaimanakah"}:
        return QTYPES["how"]
    if ws & {"apa", "apakah"}:
        return QTYPES["what"]
    if "tahun" in ws and words and words[0] in ("pada", "di", "tahun"):
        return QTYPES["when"]
    return QTYPES["other"]


def is_year(t: str) -> bool:
    return bool(re.fullmatch(r"(1[0-9]|20)\d\d", t))


class IdfStats:
    """Sentence-level IDF + BM25 statistics fitted on training contexts."""

    def __init__(self, contexts: list[str]):
        df: Counter = Counter()
        n, total = 0, 0
        for c in contexts:
            toks, _, ranges = tokenize_context(c)
            for a, b in ranges:
                ss = {t.stem for t in toks[a:b] if not t.punct}
                df.update(ss)
                n += 1
                total += b - a
        self.n, self.df, self.avgdl = n, df, total / max(n, 1)

    def idf(self, w: str) -> float:
        return math.log(1 + (self.n - self.df.get(w, 0) + 0.5) / (self.df.get(w, 0) + 0.5))

    def bm25(self, qstems: list[str], sent_stems: list[str], k1=1.5, b=0.75) -> float:
        tf = Counter(sent_stems)
        dl = len(sent_stems)
        s = 0.0
        for w in set(qstems):
            f = tf.get(w, 0)
            if f:
                s += self.idf(w) * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self.avgdl))
        return s

    def tfidf_cos(self, qstems: list[str], sent_stems: list[str]) -> float:
        qt, st = Counter(qstems), Counter(sent_stems)
        num = sum(qt[w] * st[w] * self.idf(w) ** 2 for w in qt if w in st)
        dq = math.sqrt(sum((c * self.idf(w)) ** 2 for w, c in qt.items()))
        ds = math.sqrt(sum((c * self.idf(w)) ** 2 for w, c in st.items()))
        return num / (dq * ds) if dq and ds else 0.0


def content_stems(qtoks: list[Tok]) -> list[str]:
    return [t.stem for t in qtoks if not t.punct and not t.stop and t.low not in QUESTION_WORDS]
