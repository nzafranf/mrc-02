"""Subword fertility / % words split / context length overflow on unique train+dev contexts (tokenizer only)."""
import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "experiments" / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import json, re
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ctxs = sorted({json.loads(l)["context"] for s in ("train", "dev") for l in open(ROOT / f"data/processed/{s}.jsonl", encoding="utf-8")})
words_per = [re.findall(r"\S+", c) for c in ctxs]
sample_words = [w for ws in words_per[:600] for w in ws]
out = {}
for name in ["indobenchmark/indobert-base-p1", "FacebookAI/xlm-roberta-base", "Qwen/Qwen2.5-1.5B-Instruct"]:
    try:
        tok = AutoTokenizer.from_pretrained(name)
    except Exception as e:
        out[name] = {"error": str(e)[:200]}; print(name, "FAILED", e); continue
    lens = np.array([len(tok(c, add_special_tokens=False)["input_ids"]) for c in ctxs])
    # word-level fertility on whitespace words (sample); count wordpieces without specials
    enc = tok(sample_words, add_special_tokens=False)["input_ids"]
    f = np.array([max(1, len(e)) for e in enc])
    out[name] = {"n_contexts": len(ctxs), "tokens_per_word": round(float(f.mean()), 3), "pct_words_split": round(float((f > 1).mean() * 100), 1),
                 "ctx_tokens_mean": round(float(lens.mean()), 1), "ctx_tokens_p95": int(np.percentile(lens, 95)), "ctx_tokens_max": int(lens.max()),
                 "pct_ctx_gt_384": round(float((lens > 384).mean() * 100), 1), "pct_ctx_gt_512": round(float((lens > 512).mean() * 100), 1)}
    print(name, out[name], flush=True)
json.dump(out, open(ROOT / "eda/outputs/tables/tokenizer_analysis.json", "w"), indent=1)
