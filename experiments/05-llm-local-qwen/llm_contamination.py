"""Contamination probe: does the LM continue a dev context verbatim?

For N dev contexts (distinct, >= --min-chars), show the first 50% of the words and ask the model to
continue the paragraph exactly. Score = LCS(tokens(continuation), tokens(true second half)) / len(reference),
i.e. ROUGE-L recall over normalized tokens, with the reference cut to ~0.6 * --max-new-tokens words.
Control: the same contexts with sentences shuffled (so the "true" second half cannot be memorized);
the reference there is the second half of the shuffled text. High original score with low control score
suggests memorization; similar scores suggest generic copy/fluency only.

  python experiments/05-llm-local-qwen/llm_contamination.py --n 50 --model-name Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import argparse
import json
import random
import re
import time

from mrc.evaluate import load_jsonl, normalize_answer
from mrc.llm_qa import ROOT, load_model

INSTR = ("Berikut adalah awal sebuah paragraf. Lanjutkan paragraf tersebut persis seperti teks aslinya, "
         "kata demi kata, tanpa komentar tambahan.\n\n")
SENT = re.compile(r"(?<=[.!?])\s+")


def lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def split_half(text: str) -> tuple[str, str]:
    words = text.split()
    h = len(words) // 2
    return " ".join(words[:h]), " ".join(words[h:])


def shuffle_sentences(text: str, rng: random.Random) -> str:
    sents = [s for s in SENT.split(text) if s]
    rng.shuffle(sents)
    return " ".join(sents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--split-file", default="data/processed/dev.jsonl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="experiments/05-llm-local-qwen/contamination")
    a = ap.parse_args()
    import torch
    rng = random.Random(a.seed)
    seen, ctxs = set(), []
    for ex in load_jsonl(ROOT / a.split_file):
        c = ex["context"]
        if c not in seen and len(c) >= a.min_chars and len(SENT.split(c)) >= 3:
            seen.add(c)
            ctxs.append(c)
    ctxs = rng.sample(ctxs, min(a.n, len(ctxs)))
    tok, model = load_model(a.model_name, a.device, a.load_4bit)
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    def run(texts: list[str]):
        rows = []
        for t in texts:
            first, second = split_half(t)
            ref_words = second.split()[: int(a.max_new_tokens * 0.6)]
            prompt = tok.apply_chat_template([{"role": "user", "content": INSTR + first}], tokenize=False,
                                             add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, temperature=None,
                                   top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
            cont = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            r = normalize_answer(" ".join(ref_words)).split()
            c = normalize_answer(cont).split()[: len(r) + 10]
            rows.append({"rouge_l_recall": lcs_len(c, r) / max(1, len(r)), "continuation": cont,
                         "reference": " ".join(ref_words)})
        return rows

    t0 = time.time()
    orig = run(ctxs)
    ctrl = run([shuffle_sentences(c, rng) for c in ctxs])

    def mean(rows):
        return sum(r["rouge_l_recall"] for r in rows) / len(rows)

    def frac(rows):
        return sum(r["rouge_l_recall"] >= 0.5 for r in rows) / len(rows)

    summ = {"model": a.model_name, "n": len(ctxs), "original_mean_rougeL_recall": mean(orig),
            "control_shuffled_mean_rougeL_recall": mean(ctrl), "original_frac_ge_0.5": frac(orig),
            "control_frac_ge_0.5": frac(ctrl), "seconds": time.time() - t0}
    tag = a.model_name.split("/")[-1]
    (out_dir / f"{tag}.json").write_text(
        json.dumps({"summary": summ, "original": orig, "control": ctrl}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
