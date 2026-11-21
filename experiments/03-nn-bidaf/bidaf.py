"""BiDAF-lite for Indonesian extractive MRC with a learnable NULL position for no-answer.

Train:   python experiments/03-nn-bidaf/bidaf.py --out-dir experiments/03-nn-bidaf/runs/bidaf/seed13 --seed 13
Ablate:  add --no-attention / --rnn gru / --no-char
Predict: python experiments/03-nn-bidaf/bidaf.py --predict-only --checkpoint X/model.pt --split-file Y --out Z
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mrc.evaluate import evaluate, load_jsonl
from nn_common import PAD, Feat, Vocab, collate, featurize

NEG = -1e30
MAX_ANS_LEN = 30


class RNN(nn.Module):
    """Single-layer bidirectional RNN over padded batches.

    Built from two unidirectional nets (the backward one runs on length-reversed sequences) instead of
    packed sequences: PyTorch's CPU packed RNN backward was ~8x slower, and this keeps padding from leaking
    into the backward direction. Outputs at padded positions are zeroed."""

    def __init__(self, kind: str, inp: int, hid: int):
        super().__init__()
        cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[kind]
        self.fw = cls(inp, hid, batch_first=True)
        self.bw = cls(inp, hid, batch_first=True)

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        t = torch.arange(L, device=x.device).unsqueeze(0)
        valid = t < lens.unsqueeze(1)
        rev = torch.where(valid, lens.unsqueeze(1) - 1 - t, t)  # reverse inside each sequence
        xr = torch.gather(x, 1, rev.unsqueeze(-1).expand_as(x))
        of, _ = self.fw(x)
        ob, _ = self.bw(xr)
        ob = torch.gather(ob, 1, rev.unsqueeze(-1).expand_as(ob))
        return torch.cat([of, ob], -1) * valid.unsqueeze(-1)


class Highway(nn.Module):
    def __init__(self, d: int, n: int = 2):
        super().__init__()
        self.l = nn.ModuleList(nn.Linear(d, 2 * d) for _ in range(n))

    def forward(self, x):
        for l in self.l:
            h, g = l(x).chunk(2, -1)
            g = torch.sigmoid(g)
            x = g * F.relu(h) + (1 - g) * x
        return x


class BiDAF(nn.Module):
    def __init__(self, n_words: int, n_chars: int, emb: int = 100, hid: int = 80, char_emb: int = 16,
                 char_filters: int = 50, dropout: float = 0.2, rnn: str = "lstm",
                 use_char: bool = True, attention: bool = True):
        super().__init__()
        self.use_char, self.attention, self.hid = use_char, attention, hid
        self.word_emb = nn.Embedding(n_words, emb, padding_idx=PAD)
        nn.init.normal_(self.word_emb.weight, std=0.1)
        d = emb
        if use_char:
            self.char_emb = nn.Embedding(n_chars, char_emb, padding_idx=PAD)
            self.char_conv = nn.Conv1d(char_emb, char_filters, 5, padding=2)
            d += char_filters
        self.highway = Highway(d)
        self.enc = RNN(rnn, d, hid)
        D = 2 * hid
        if attention:
            self.att_w = nn.Linear(3 * D, 1)
            gdim = 4 * D
        else:
            gdim = 3 * D
        self.proj = nn.Linear(gdim, D)  # cheap bottleneck before the modeling RNN (CPU budget)
        self.mod = RNN(rnn, D, hid)
        self.end_rnn = RNN(rnn, D, hid)
        self.out_start = nn.Linear(gdim + D, 1)
        self.out_end = nn.Linear(gdim + D, 1)
        self.drop = nn.Dropout(dropout)

    def embed(self, w, c):
        x = self.word_emb(w)
        if self.use_char:
            B, L, W = c.shape
            ce = self.char_emb(c.view(B * L, W)).transpose(1, 2)
            ce = F.relu(self.char_conv(self.drop(ce))).max(-1)[0].view(B, L, -1)
            x = torch.cat([x, ce], -1)
        return self.highway(self.drop(x))

    def forward(self, b: dict, return_attn: bool = False):
        cw, qw = b["c_w"], b["q_w"]
        cm, qm = cw != PAD, qw != PAD
        cl, ql = cm.sum(1), qm.sum(1)
        C = self.drop(self.enc(self.embed(cw, b["c_c"]), cl))  # B,Lc,D
        Q = self.drop(self.enc(self.embed(qw, b["q_c"]), ql))  # B,Lq,D
        attn = None
        if self.attention:
            Lc = C.size(1)
            # S_ij = w^T [c_i ; q_j ; c_i * q_j], computed without materializing B,Lc,Lq,3D
            w = self.att_w.weight[0]
            D = C.size(-1)
            s = (C @ w[:D].unsqueeze(-1)) + (Q @ w[D:2 * D].unsqueeze(-1)).transpose(1, 2) \
                + torch.bmm(C * w[2 * D:], Q.transpose(1, 2)) + self.att_w.bias
            s = s.masked_fill(~qm.unsqueeze(1), NEG)
            a = F.softmax(s, dim=2)  # context-to-query
            U = a @ Q
            sc = s.masked_fill(~cm.unsqueeze(2), NEG)
            bq = F.softmax(sc.max(2)[0], dim=1)  # query-to-context
            h = (bq.unsqueeze(1) @ C).expand(-1, Lc, -1)
            G = torch.cat([C, U, C * U, C * h], -1)
            attn = a
        else:
            qbar = (Q * qm.unsqueeze(-1)).sum(1, keepdim=True) / ql.view(-1, 1, 1)
            qbar = qbar.expand(-1, C.size(1), -1)
            G = torch.cat([C, qbar, C * qbar], -1)
        M = self.drop(self.mod(F.relu(self.proj(self.drop(G))), cl))
        M2 = self.drop(self.end_rnn(M, cl))
        G = self.drop(G)
        ls = self.out_start(torch.cat([G, M], -1)).squeeze(-1).masked_fill(~cm, NEG)
        le = self.out_end(torch.cat([G, M2], -1)).squeeze(-1).masked_fill(~cm, NEG)
        return (ls, le, attn) if return_attn else (ls, le)


GROUPS = {"embed": ("word_emb", "char_emb", "char_conv"), "encoder": ("highway", "enc."),
          "attention": ("att_w",), "modeling": ("proj", "mod.", "end_rnn"), "output": ("out_",)}


def group_of(name: str) -> str:
    for g, keys in GROUPS.items():
        if any(name.startswith(k) for k in keys):
            return g
    return "other"


def grad_norms(model: nn.Module) -> dict:
    sq: dict[str, float] = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            sq[group_of(n)] = sq.get(group_of(n), 0.0) + float(p.grad.pow(2).sum())
    return {g: v ** 0.5 for g, v in sq.items()}


def decode(ls: torch.Tensor, le: torch.Tensor):
    """Return (na_prob, span_score, start, end) per row. Index 0 is NULL; spans need end>=start, len<=30."""
    ps, pe = F.softmax(ls, 1), F.softmax(le, 1)
    p0 = ps[:, 0] * pe[:, 0]
    L = ps.size(1)
    joint = ps.unsqueeze(2) * pe.unsqueeze(1)  # B,L,L
    i = torch.arange(L)
    band = (i.view(1, -1) >= i.view(-1, 1)) & (i.view(1, -1) - i.view(-1, 1) < MAX_ANS_LEN)
    band[0, :] = False
    band[:, 0] = False
    joint = joint.masked_fill(~band, 0.0).view(len(ps), -1)
    best, idx = joint.max(1)
    na = p0 / (p0 + best + 1e-12)
    return na, best, idx // L, idx % L


@torch.no_grad()
def predict(model: BiDAF, feats: list[Feat], contexts: dict[str, str], bs: int = 64) -> dict:
    model.eval()
    order = sorted(range(len(feats)), key=lambda i: len(feats[i].c_w))
    out = {}
    for k in range(0, len(order), bs):
        fb = [feats[i] for i in order[k:k + bs]]
        ls, le = model(collate(fb))
        na, sc, s, e = decode(ls, le)
        for f, n, c, si, ei in zip(fb, na.tolist(), sc.tolist(), s.tolist(), e.tolist()):
            if si == 0 or ei == 0:  # no valid span (empty context)
                text, sch = "", -1
            else:
                sch, ech = f.ctx_tok[si - 1][1], f.ctx_tok[ei - 1][2]
                text = contexts[f.id][sch:ech]
            pred = "" if n > 0.5 else text
            out[f.id] = {"id": f.id, "pred": pred, "na_prob": n, "score": c,
                         "start_char": -1 if pred == "" else sch}
    return out


def write_preds(model, feats, exs, path: Path) -> dict:
    preds = predict(model, feats, {e["id"]: e["context"] for e in exs})
    with open(path, "w", encoding="utf-8") as f:
        for e in exs:
            f.write(json.dumps(preds[e["id"]], ensure_ascii=False) + "\n")
    return preds


def batches(feats: list[Feat], bs: int, rng: random.Random):
    """Length-bucketed shuffled batches (speeds up CPU training)."""
    idx = list(range(len(feats)))
    rng.shuffle(idx)
    chunks = [sorted(idx[i:i + bs * 20], key=lambda j: len(feats[j].c_w)) for i in range(0, len(idx), bs * 20)]
    bl = [c[i:i + bs] for c in chunks for i in range(0, len(c), bs)]
    rng.shuffle(bl)
    return [[feats[j] for j in b] for b in bl]


def loss_fn(ls, le, b):
    return F.cross_entropy(ls, b["start"]) + F.cross_entropy(le, b["end"])


def build_model(a: dict, vocab: Vocab) -> BiDAF:
    return BiDAF(len(vocab.itos), len(vocab.citos), a["emb"], a["hidden"], dropout=a["dropout"], rnn=a["rnn"],
                 use_char=not a["no_char"], attention=not a["no_attention"])


def save_attention(model, feats_by_id, ids, path):
    model.eval()
    d = {"ids": np.array(ids)}
    with torch.no_grad():
        for i in ids:
            f = feats_by_id[i]
            _, _, a = model(collate([f]), return_attn=True)
            d[f"{i}__attn"] = a[0].numpy()
            d[f"{i}__ctx"] = np.array(["<NULL>"] + [t for t, _, _ in f.ctx_tok])
            d[f"{i}__q"] = np.array([t for t, _, _ in f.q_tok])
    np.savez(path, **d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", default="data/processed/train.jsonl")
    ap.add_argument("--dev-file", default="data/processed/dev.jsonl")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--emb", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--max-ctx", type=int, default=400)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--rnn", choices=["lstm", "gru", "rnn"], default="lstm")
    ap.add_argument("--no-char", action="store_true")
    ap.add_argument("--no-attention", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--save-attn", default=None)
    ap.add_argument("--skip-overfit", action="store_true")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    if args.predict_only:
        ck = torch.load(args.checkpoint, weights_only=False)
        cfg = ck["config"]
        vocab = Vocab.from_state(ck["vocab"])
        model = build_model(cfg, vocab)
        model.load_state_dict(ck["model"])
        exs = load_jsonl(args.split_file)
        feats = [featurize(e, vocab, cfg["max_ctx"]) for e in exs]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_preds(model, feats, exs, Path(args.out))
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "w", encoding="utf-8")

    def log(s: str):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tr, dv = load_jsonl(args.train_file), load_jsonl(args.dev_file)
    if args.max_train:
        tr = tr[:args.max_train]
    vocab = Vocab.build(tr, args.min_freq)
    ftr = [featurize(e, vocab, args.max_ctx) for e in tr]
    fdv = [featurize(e, vocab, args.max_ctx) for e in dv]
    n_ans_tr = sum(1 for e in tr if not e["is_impossible"])
    n_ans_dv = sum(1 for e in dv if not e["is_impossible"])
    lost_tr, lost_dv = sum(f.lost for f in ftr), sum(f.lost for f in fdv)
    n_trunc = sum(len(f.ctx_tok) >= args.max_ctx for f in ftr)
    log(f"vocab words={len(vocab.itos)} chars={len(vocab.citos)}")
    log(f"truncation at {args.max_ctx} tokens: gold answers lost train {lost_tr}/{n_ans_tr} "
        f"({100*lost_tr/n_ans_tr:.2f}%), dev {lost_dv}/{n_ans_dv} ({100*lost_dv/n_ans_dv:.2f}%); "
        f"train examples with truncated context {n_trunc}")
    ftr = [f for f in ftr if not f.lost]  # cannot supervise a span outside the window
    cfg = dict(vars(args))
    cfg["timestamp"] = datetime.now().isoformat(timespec="seconds")
    cfg["train_gold_lost"] = lost_tr
    cfg["dev_gold_lost"] = lost_dv
    model = build_model(cfg, vocab)
    cfg["n_params"] = sum(p.numel() for p in model.parameters())
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    log(f"params={cfg['n_params']} emb={args.emb} hidden={args.hidden} dropout={args.dropout} rnn={args.rnn} "
        f"char={not args.no_char} attention={not args.no_attention} lr={args.lr} bs={args.batch_size}")
    rng = random.Random(args.seed)

    if not args.skip_overfit:  # sanity check: memorize one batch with dropout off
        ob = collate(ftr[:args.batch_size])
        m2 = build_model({**cfg, "dropout": 0.0}, vocab)
        o2 = torch.optim.Adam(m2.parameters(), lr=3e-3)
        m2.train()
        for step in range(1, 401):
            o2.zero_grad()
            ls, le = m2(ob)
            l = loss_fn(ls, le, ob)
            l.backward()
            o2.step()
            if step == 1 or step % 50 == 0:
                log(f"[overfit-1-batch] step {step} loss {l.item():.4f}")
            if l.item() < 0.02:
                log(f"[overfit-1-batch] step {step} loss {l.item():.4f} (reached <0.02)")
                break
        torch.manual_seed(args.seed)
        model = build_model(cfg, vocab)  # fresh model for real training
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1, best_ep, bad = -1.0, 0, 0
    hist = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n, nb, gn = 0.0, 0, 0, {}
        for fb in batches(ftr, args.batch_size, rng):
            b = collate(fb)
            opt.zero_grad()
            ls, le = model(b)
            l = loss_fn(ls, le, b)
            l.backward()
            for g, v in grad_norms(model).items():
                gn[g] = gn.get(g, 0.0) + v
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot += l.item() * len(fb)
            n += len(fb)
            nb += 1
        gn = {g: v / nb for g, v in gn.items()}
        write_preds(model, fdv, dv, out / "_tmp_dev.jsonl")
        m = evaluate(args.dev_file, out / "_tmp_dev.jsonl")
        dt = time.time() - t0
        hist.append({"epoch": ep, "train_loss": tot / n, "dev_em": m["exact"], "dev_f1": m["f1"],
                     "has_f1": m["HasAns_f1"], "na_acc": m["NoAns_acc"], "pred_na_rate": m["pred_noans_rate"],
                     "grad_norm": gn, "time": dt})
        log(f"epoch {ep} loss {tot/n:.4f} devEM {m['exact']:.2f} devF1 {m['f1']:.2f} "
            f"HasF1 {m['HasAns_f1']:.2f} NoAcc {m['NoAns_acc']:.2f} predNA {m['pred_noans_rate']:.1f}% "
            f"gradnorm {json.dumps({g: round(v, 3) for g, v in gn.items()})} time {dt:.0f}s")
        if m["f1"] > best_f1:
            best_f1, best_ep, bad = m["f1"], ep, 0
            torch.save({"model": model.state_dict(), "config": cfg, "vocab": vocab.state()}, out / "model.pt")
        else:
            bad += 1
            if bad >= args.patience:
                log(f"early stop at epoch {ep}, best epoch {best_ep} devF1 {best_f1:.2f}")
                break
    (out / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    ck = torch.load(out / "model.pt", weights_only=False)
    model.load_state_dict(ck["model"])
    write_preds(model, fdv, dv, out / "preds_dev.jsonl")
    (out / "_tmp_dev.jsonl").unlink(missing_ok=True)
    m = evaluate(args.dev_file, out / "preds_dev.jsonl", search=True)
    (out / "metrics_dev.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    log(f"best epoch {best_ep}: EM {m['exact']:.2f} F1 {m['f1']:.2f} HasAns_f1 {m['HasAns_f1']:.2f} "
        f"NoAns_acc {m['NoAns_acc']:.2f}; at best threshold F1 {m['at_best_threshold']['f1']:.2f}")
    if args.save_attn:
        ids = sorted(e["id"] for e in dv if not e["is_impossible"])[:20]
        save_attention(model, {f.id: f for f in fdv}, ids, args.save_attn)
        log(f"saved attention for {len(ids)} dev ids to {args.save_attn}")


if __name__ == "__main__":
    main()
