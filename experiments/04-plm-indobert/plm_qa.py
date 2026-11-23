"""Pretrained-encoder extractive QA (SQuAD v2 style) with a plain PyTorch loop.

Train:   python experiments/04-plm-indobert/plm_qa.py --model-name indobenchmark/indobert-base-p1 --seed 13
Predict: python experiments/04-plm-indobert/plm_qa.py --predict-only --checkpoint X.pt --split-file data/processed/dev.jsonl --out preds.jsonl
Offsets: python experiments/04-plm-indobert/plm_qa.py --check-offsets --model-name FacebookAI/xlm-roberta-base

Prediction: n-best (20) start/end logits per window, max span 30 tokens, spans mapped back to the ORIGINAL
context through tokenizer offsets (never detokenized text). na_prob = null_score - best_span_score (raw
logit difference, higher = more likely unanswerable), where null_score = min over windows of
(start_logit[CLS] + end_logit[CLS]).
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import argparse
import sys
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from mrc.evaluate import best_na_threshold, evaluate, load_jsonl, normalize_answer



# ----------------------------------------------------------------------------- data
def pick_examples(rows: list[dict], frac: float, max_n: int | None, seed: int) -> list[dict]:
    """Random subset of context groups (frac), then optionally a random subset of examples (max_n)."""
    rng = random.Random(seed)
    if frac < 1.0:
        ctxs = sorted({r["context"] for r in rows})
        rng.shuffle(ctxs)
        keep = set(ctxs[: max(1, round(len(ctxs) * frac))])
        rows = [r for r in rows if r["context"] in keep]
    if max_n and len(rows) > max_n:
        rows = rng.sample(rows, max_n)
    return rows


def gold_span(ex: dict) -> tuple[int, int] | None:
    """(start_char, end_char) of the first gold answer, or None if unanswerable."""
    if ex.get("is_impossible") or not ex.get("answers"):
        return None
    a = ex["answers"][0]
    if not a["text"]:
        return None
    return a["answer_start"], a["answer_start"] + len(a["text"])


def tokenize(tok, rows: list[dict], max_len: int, stride: int):
    return tok([r["question"].strip() for r in rows], [r["context"] for r in rows], truncation="only_second",
               max_length=max_len, stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True)


def token_span(offsets, seq_ids, s_char: int, e_char: int) -> tuple[int, int] | None:
    """Token span covering [s_char, e_char) inside the context part of a window, or None if outside."""
    ctx = [i for i, s in enumerate(seq_ids) if s == 1]
    if not ctx or offsets[ctx[0]][0] > s_char or offsets[ctx[-1]][1] < e_char:
        return None
    i = ctx[0]
    while i <= ctx[-1] and offsets[i][0] <= s_char:
        i += 1
    j = ctx[-1]
    while j >= ctx[0] and offsets[j][1] >= e_char:
        j -= 1
    return i - 1, j + 1


def build_train_features(tok, rows, max_len, stride):
    enc = tokenize(tok, rows, max_len, stride)
    feats = []
    for k in range(len(enc["input_ids"])):
        ex = rows[enc["overflow_to_sample_mapping"][k]]
        seq = enc.sequence_ids(k)
        sp, ep = 0, 0  # CLS index
        g = gold_span(ex)
        if g is not None:
            t = token_span(enc["offset_mapping"][k], seq, *g)
            if t:
                sp, ep = t
        f = {"input_ids": enc["input_ids"][k], "attention_mask": enc["attention_mask"][k],
             "start_positions": sp, "end_positions": ep}
        if "token_type_ids" in enc:
            f["token_type_ids"] = enc["token_type_ids"][k]
        feats.append(f)
    return feats


def build_eval_features(tok, rows, max_len, stride):
    enc = tokenize(tok, rows, max_len, stride)
    feats = []
    for k in range(len(enc["input_ids"])):
        seq = enc.sequence_ids(k)
        f = {"input_ids": enc["input_ids"][k], "attention_mask": enc["attention_mask"][k],
             "ex": enc["overflow_to_sample_mapping"][k],
             "offsets": [o if seq[i] == 1 else None for i, o in enumerate(enc["offset_mapping"][k])]}
        if "token_type_ids" in enc:
            f["token_type_ids"] = enc["token_type_ids"][k]
        feats.append(f)
    return feats


def collate(batch: list[dict], pad_id: int, keys=("input_ids", "attention_mask", "token_type_ids",
                                                  "start_positions", "end_positions")):
    L = max(len(b["input_ids"]) for b in batch)
    out = {}
    for k in keys:
        if k not in batch[0]:
            continue
        if k in ("start_positions", "end_positions"):
            out[k] = torch.tensor([b[k] for b in batch])
        else:
            pad = pad_id if k == "input_ids" else 0
            out[k] = torch.tensor([b[k] + [pad] * (L - len(b[k])) for b in batch])
    return out


# ----------------------------------------------------------------------------- offset check
def check_offsets(tok, rows, max_len, stride) -> dict:
    """Does context[offsets of the gold token span] reproduce the gold answer text exactly?"""
    n = exact = norm_ok = no_window = gold_inconsistent = 0
    examples = []
    for ex in rows:
        g = gold_span(ex)
        if g is None:
            continue
        n += 1
        if ex["context"][g[0]:g[1]] != ex["answers"][0]["text"]:
            gold_inconsistent += 1
        enc = tokenize(tok, [ex], max_len, stride)
        found = None
        for k in range(len(enc["input_ids"])):
            t = token_span(enc["offset_mapping"][k], enc.sequence_ids(k), *g)
            if t:
                off = enc["offset_mapping"][k]
                found = ex["context"][off[t[0]][0]:off[t[1]][1]]
                break
        if found is None:
            no_window += 1
        elif found == ex["answers"][0]["text"]:
            exact += 1
        else:
            if normalize_answer(found) == normalize_answer(ex["answers"][0]["text"]):
                norm_ok += 1
            if len(examples) < 5:
                examples.append({"gold": ex["answers"][0]["text"], "recon": found})
    mism = n - exact - no_window
    return {"answerable": n, "exact": exact, "mismatch_exact": mism, "mismatch_rate": mism / max(n, 1),
            "mismatch_but_normalized_equal": norm_ok, "no_window_contains_answer": no_window,
            "gold_text_vs_context_inconsistent": gold_inconsistent, "examples": examples}


# ----------------------------------------------------------------------------- postprocess
def postprocess(rows, feats, start_logits, end_logits, n_best=20, max_ans_len=30) -> dict[str, dict]:
    by_ex: dict[int, list[int]] = {}
    for fi, f in enumerate(feats):
        by_ex.setdefault(f["ex"], []).append(fi)
    out = {}
    for ei, ex in enumerate(rows):
        null = math.inf
        best = (-math.inf, "", -1)
        for fi in by_ex.get(ei, []):
            sl, el, off = start_logits[fi], end_logits[fi], feats[fi]["offsets"]
            null = min(null, float(sl[0] + el[0]))
            for s in np.argsort(sl)[::-1][:n_best]:
                for e in np.argsort(el)[::-1][:n_best]:
                    if s >= len(off) or e >= len(off) or off[s] is None or off[e] is None:
                        continue
                    if e < s or e - s + 1 > max_ans_len:
                        continue
                    sc = float(sl[s] + el[e])
                    if sc > best[0]:
                        best = (sc, ex["context"][off[s][0]:off[e][1]], off[s][0])
        if best[2] < 0:
            best = (-1e4, "", -1)
        if null is math.inf:
            null = 0.0
        out[ex["id"]] = {"id": ex["id"], "pred": best[1], "na_prob": null - best[0], "score": best[0],
                         "start_char": best[2]}
    return out


@torch.no_grad()
def predict(model, tok, rows, args, device) -> dict[str, dict]:
    model.eval()
    feats = build_eval_features(tok, rows, args.max_len, args.doc_stride)
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]))
    sl: list = [None] * len(feats)
    el: list = [None] * len(feats)
    bs = args.eval_batch_size
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        batch = collate([feats[i] for i in idx], tok.pad_token_id)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device.type, enabled=args.fp16 and device.type == "cuda"):
            o = model(**batch)
        s, e = o.start_logits.float().cpu().numpy(), o.end_logits.float().cpu().numpy()
        for j, i in enumerate(idx):
            n = len(feats[i]["input_ids"])
            sl[i], el[i] = s[j, :n], e[j, :n]
    return postprocess(rows, feats, sl, el, args.n_best, args.max_answer_len)


def write_preds(preds: dict, rows, path: Path):
    """pred is always the best span so na_threshold can be applied later; pred_argmax = null beats span."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in rows:
            p = dict(preds[ex["id"]])
            p["pred_argmax"] = "" if p["na_prob"] > 0 else p["pred"]
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- model
def build_model(args):
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name, num_labels=2)
    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.QUESTION_ANS, r=args.lora_r, lora_alpha=2 * args.lora_r,
                         lora_dropout=0.1, target_modules=["query", "value"])
        model = get_peft_model(model, cfg)  # qa_outputs head stays trainable (modules_to_save)
    elif args.freeze_encoder:
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith("qa_outputs")
    return model


def trainable_state(model) -> dict:
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="indobenchmark/indobert-base-p1")
    ap.add_argument("--run-name", default="plm_qa")
    ap.add_argument("--out-dir", default=None, help="default experiments/<run-name>/runs/seed<seed>")
    ap.add_argument("--train-file", default="data/processed/train.jsonl")
    ap.add_argument("--dev-file", default="data/processed/dev.jsonl")
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--doc-stride", type=int, default=128)
    ap.add_argument("--n-best", type=int, default=20)
    ap.add_argument("--max-answer-len", type=int, default=30)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-eval-samples", type=int, default=None)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--save-checkpoint", action="store_true")
    ap.add_argument("--also-predict", nargs="*", default=[],
                    help="after training, predict these split files with the best epoch and write "
                         "preds_<stem>.jsonl next to preds_dev.jsonl (no checkpoint needed)")
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--out", default=None, help="predict-only: output preds jsonl")
    ap.add_argument("--check-offsets", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps requested but MPS is unavailable")

    if args.check_offsets:
        tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        rows = load_jsonl(ROOT / args.train_file)
        r = check_offsets(tok, rows, args.max_len, args.doc_stride)
        r["model"] = args.model_name
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return

    if args.predict_only:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        for k, v in ck["args"].items():
            if k in ("model_name", "lora", "lora_r", "freeze_encoder", "max_len", "doc_stride"):
                setattr(args, k, v)
        tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        model = build_model(args)
        model.load_state_dict(ck["state"], strict=False)
        model.to(device)
        rows = load_jsonl(args.split_file)
        preds = predict(model, tok, rows, args, device)
        write_preds(preds, rows, Path(args.out))
        print(f"wrote {args.out}; dev-tuned na threshold stored in checkpoint: {ck.get('na_threshold')}")
        return

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "experiments/04-plm-indobert/runs" / args.run_name / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "log.txt", "w", encoding="utf-8")

    def log(msg: str):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_rows = pick_examples(load_jsonl(ROOT / args.train_file), args.train_frac, args.max_train_samples, args.seed)
    dev_all = load_jsonl(ROOT / args.dev_file)
    dev_rows = dev_all[: args.max_eval_samples] if args.max_eval_samples else dev_all
    gold_dev_path = out_dir / "_dev_subset.jsonl"
    with open(gold_dev_path, "w", encoding="utf-8") as f:
        for r in dev_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    feats = build_train_features(tok, train_rows, args.max_len, args.doc_stride)
    log(f"train examples {len(train_rows)} -> windows {len(feats)}; dev examples {len(dev_rows)}")

    model = build_model(args).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr, n_all = sum(p.numel() for p in params), sum(p.numel() for p in model.parameters())
    log(f"params trainable {n_tr:,} / total {n_all:,}")
    cfg = {**vars(args), "n_train_examples": len(train_rows), "n_train_windows": len(feats),
           "trainable_params": n_tr, "total_params": n_all, "timestamp": datetime.now().isoformat(timespec="seconds"),
           "na_prob": "raw logit diff: null_score - best_span_score"}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    nd = ("bias", "LayerNorm.weight", "layer_norm.weight")
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": [p for n, p in named if not any(x in n for x in nd)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in named if any(x in n for x in nd)], "weight_decay": 0.0}], lr=args.lr)
    steps_per_epoch = math.ceil(len(feats) / (args.batch_size * args.grad_accum))
    total = steps_per_epoch * args.epochs
    warm = int(total * args.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    scaler = torch.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    gen = random.Random(args.seed)
    best = {"f1": -1.0}
    best_state, best_preds = None, None
    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = list(range(len(feats)))
        gen.shuffle(idx)
        run_loss, nb = 0.0, 0
        for b in range(0, len(idx), args.batch_size):
            batch = collate([feats[i] for i in idx[b:b + args.batch_size]], tok.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, enabled=args.fp16 and device.type == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss / args.grad_accum).backward()
            run_loss += loss.item()
            nb += 1
            if nb % args.grad_accum == 0 or b + args.batch_size >= len(idx):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 5 == 0:
                    log(f"epoch {epoch} step {step}/{total} loss {run_loss / nb:.4f} lr {sched.get_last_lr()[0]:.2e} "
                        f"elapsed {time.time() - t0:.0f}s ({(time.time() - t0) / step:.2f}s/step)")
        train_loss = run_loss / max(nb, 1)
        preds = predict(model, tok, dev_rows, args, device)
        thr, f1 = best_na_threshold(dev_rows, preds)
        log(f"== epoch {epoch} train_loss {train_loss:.4f} dev F1@best_thr {f1:.2f} (thr {thr:.3f}) "
            f"elapsed {time.time() - t0:.0f}s")
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch, "na_threshold": thr}
            best_state = trainable_state(model)
            best_preds = preds
    write_preds(best_preds, dev_rows, out_dir / "preds_dev.jsonl")
    metrics = evaluate(gold_dev_path, out_dir / "preds_dev.jsonl", search=True)
    metrics["best_epoch"] = best["epoch"]
    metrics["train_seconds"] = time.time() - t0
    (out_dir / "metrics_dev.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    gold_dev_path.unlink()
    log(f"best epoch {best['epoch']}  dev at best threshold: {json.dumps(metrics['at_best_threshold'])}")
    if args.save_checkpoint:
        torch.save({"args": vars(args), "state": best_state, "na_threshold": best["na_threshold"]},
                   out_dir / "checkpoint.pt")
        log("saved checkpoint.pt (trainable params only)")
    if args.also_predict:
        model.load_state_dict(best_state, strict=False)
        for sf in args.also_predict:
            sf = Path(sf) if Path(sf).is_absolute() else ROOT / sf
            if "test" in sf.name and "official" not in sf.name:
                raise SystemExit(f"refusing to predict the sealed internal test split: {sf}")
            rows = load_jsonl(sf)
            write_preds(predict(model, tok, rows, args, device), rows, out_dir / f"preds_{sf.stem}.jsonl")
            log(f"wrote preds_{sf.stem}.jsonl ({len(rows)} questions, best epoch {best['epoch']})")


if __name__ == "__main__":
    main()
