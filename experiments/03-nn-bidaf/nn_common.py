"""Tokenization, vocab, featurization and batching shared by the from-scratch neural models."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import torch

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PAD, UNK, NULL = 0, 1, 2
MAX_WORD_CHARS = 16


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """Regex word tokenizer keeping char offsets: [(token, start, end)]."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


class Vocab:
    def __init__(self, words: list[str], chars: list[str]):
        self.itos = ["<pad>", "<unk>", "<null>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.citos = ["<pad>", "<unk>"] + chars
        self.cstoi = {c: i for i, c in enumerate(self.citos)}

    @classmethod
    def build(cls, examples: list[dict], min_freq: int = 2, min_char_freq: int = 5) -> "Vocab":
        wc, cc = Counter(), Counter()
        for ex in examples:
            for text in (ex["context"], ex["question"]):
                for tok, _, _ in tokenize(text):
                    wc[tok.lower()] += 1
                    cc.update(tok)
        words = sorted(w for w, c in wc.items() if c >= min_freq)
        chars = sorted(c for c, n in cc.items() if n >= min_char_freq)
        return cls(words, chars)

    def wid(self, tok: str) -> int:
        return self.stoi.get(tok.lower(), UNK)

    def cids(self, tok: str) -> list[int]:
        return [self.cstoi.get(c, UNK) for c in tok[:MAX_WORD_CHARS]]

    def state(self) -> dict:
        return {"words": self.itos[3:], "chars": self.citos[2:]}

    @classmethod
    def from_state(cls, s: dict) -> "Vocab":
        return cls(s["words"], s["chars"])


@dataclass
class Feat:
    id: str
    ctx_tok: list[tuple[str, int, int]]  # truncated, WITHOUT null
    q_tok: list[tuple[str, int, int]]
    c_w: list[int]  # with NULL prepended
    q_w: list[int]
    c_c: list[list[int]]
    q_c: list[list[int]]
    start: int  # token index in c_w space (0 = no answer); -1 if lost
    end: int
    lost: bool  # answerable but answer fell outside truncated context
    n_ctx_words: int  # whitespace words of original context (for buckets)


def featurize(ex: dict, vocab: Vocab, max_ctx: int = 400, max_q: int = 40) -> Feat:
    ctx = tokenize(ex["context"])[:max_ctx]
    q = tokenize(ex["question"])[:max_q] or [("?", 0, 1)]
    start = end = 0
    lost = False
    ans = ex.get("answers") or []
    if not ex.get("is_impossible", False) and ans and ans[0]["text"].strip():
        a0 = ans[0]["answer_start"]
        a1 = a0 + len(ans[0]["text"])
        idx = [i for i, (_, s, e) in enumerate(ctx) if s < a1 and e > a0]
        if idx:
            start, end = idx[0] + 1, idx[-1] + 1
        else:
            lost, start, end = True, -1, -1
    c_w = [NULL] + [vocab.wid(t) for t, _, _ in ctx]
    c_c = [[PAD]] + [vocab.cids(t) for t, _, _ in ctx]
    return Feat(ex["id"], ctx, q, c_w, [vocab.wid(t) for t, _, _ in q], c_c,
                [vocab.cids(t) for t, _, _ in q], start, end, lost, len(ex["context"].split()))


def _pad2(seqs: list[list[int]]) -> torch.Tensor:
    m = max(len(s) for s in seqs)
    return torch.tensor([s + [0] * (m - len(s)) for s in seqs], dtype=torch.long)


def _padc(seqs: list[list[list[int]]]) -> torch.Tensor:
    m = max(len(s) for s in seqs)
    out = torch.zeros(len(seqs), m, MAX_WORD_CHARS, dtype=torch.long)
    for i, s in enumerate(seqs):
        for j, w in enumerate(s):
            if w:
                out[i, j, : len(w)] = torch.tensor(w[:MAX_WORD_CHARS])
    return out


def collate(feats: list[Feat]) -> dict:
    return {
        "c_w": _pad2([f.c_w for f in feats]), "q_w": _pad2([f.q_w for f in feats]),
        "c_c": _padc([f.c_c for f in feats]), "q_c": _padc([f.q_c for f in feats]),
        "start": torch.tensor([f.start for f in feats]), "end": torch.tensor([f.end for f in feats]),
    }
