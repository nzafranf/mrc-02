"""Prompted extractive QA with a decoder LM (chat template), SQuAD v2 style.

Pipeline per example: build chat prompt (0-shot / fixed k-shot / retrieved k-shot) ->
greedy decode -> parse output -> ground to a context span -> na_prob from first-token probs.

Output parsing (defined up front):
  1. take the first non-empty line, strip whitespace
  2. drop prefixes such as "Jawaban:", "Answer:", "Jawab -"
  3. strip wrapping quotes / backticks / asterisks; strip trailing "."
  4. no-answer phrases (TIDAK ADA JAWABAN, tidak ada jawaban, no answer, tidak disebutkan, ...) -> ""
Grounding (field `grounding`):
  noans      parsed output is empty / no-answer phrase -> pred ""
  exact      output is a case-insensitive substring of context -> original-cased substring
  snapped    otherwise the context n-gram (word level) with best token F1 >= SNAP_MIN_F1 is used
  ungrounded best F1 < SNAP_MIN_F1: pred keeps the cleaned model text (counted as hallucination)

na_prob, method --na-method (default seqlogprob):
  seqlogprob  teacher-forced scoring on the same chat prompt. lp_noans = total log-prob of the continuation
              "TIDAK ADA JAWABAN" + end-of-turn token; lp_ans = total log-prob of the answer string + end-of-turn
              token. The answer string is the final grounded prediction (`pred`). If the greedy output is itself
              no-answer there is no pred, so lp_ans = max over up to 3 candidate context spans picked by
              question overlap (see candidate_spans). na_prob = sigmoid(lp_noans - lp_ans). Length-normalized
              values lpn_noans / lpn_ans are stored too but not used.
  first_token legacy: p_no / (p_no + p_other) at the first decoding step, for the first token of the
              no-answer string (ambiguous with answers starting with the same token, e.g. "T" in Qwen).
  `na_prob_first` (first-token value) is always stored.
score: mean log-prob of the generated tokens (excluding EOS).

Usage:
  python experiments/global/run.py mrc.llm_qa --split-file data/processed/dev.jsonl --limit 100 --shots 0 --prompt-id id1
  python experiments/global/run.py mrc.llm_qa --model-name Qwen/Qwen2.5-7B-Instruct --device cuda --load-4bit --shots 3
  python experiments/global/run.py mrc.llm_qa --backend openrouter --api-model google/gemma-4-31b-it:free --shots 0  (see experiments/06-llm-api-offtheshelf/README.md)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import re
import time
from pathlib import Path

import yaml

from mrc.evaluate import evaluate, f1, load_jsonl, normalize_answer

from mrc.paths import ROOT
SNAP_MIN_F1 = 0.5
_PREFIX = re.compile(r"^\s*(jawaban|jawab|answer|respons|response)\s*(akhir)?\s*[:\-–]\s*", re.I)
_NOANS = re.compile(
    r"^(tidak\s+ada\s+jawaban|tidak\s+ada\s+(informasi|jawabannya)|tidak\s+(disebutkan|tercantum|dapat\s+dijawab|"
    r"diketahui|terdapat\s+jawaban)|no\s+answer|not\s+(mentioned|found)|unanswerable|n/?a)\b", re.I)


# ---------------------------------------------------------------- prompts
def load_prompts(path: Path | None = None) -> dict:
    return yaml.safe_load((path or ROOT / "experiments/global/configs/llm_prompts.yaml").read_text(encoding="utf-8"))


def user_turn(cfg: dict, prompt_id: str, context: str, question: str) -> str:
    return (f"{cfg['prompts'][prompt_id]}\n\n{cfg['context_label']}: {context}\n\n"
            f"{cfg['question_label']}: {question}")


def gold_answer(ex: dict, no_ans: str) -> str:
    return ex["answers"][0]["text"] if ex["answers"] else no_ans


def fixed_demos(train: list[dict], k: int, seed: int = 1234, max_ctx_chars: int = 700) -> list[dict]:
    """Balanced answerable/unanswerable, fixed by seed, restricted to short contexts."""
    rng = random.Random(seed)
    pool = [e for e in train if len(e["context"]) <= max_ctx_chars]
    ans = rng.sample([e for e in pool if not e["is_impossible"]], k)
    noa = rng.sample([e for e in pool if e["is_impossible"]], k)
    out = []
    for i in range(k):  # alternate, starting with answerable
        out.append(ans[i // 2] if i % 2 == 0 else noa[i // 2])
    return out[:k]


class Retriever:
    """TF-IDF over train questions; returns nearest answerable/unanswerable alternately."""

    def __init__(self, train: list[dict], max_ctx_chars: int = 900):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.pool = [e for e in train if len(e["context"]) <= max_ctx_chars]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.mat = self.vec.fit_transform([e["question"] for e in self.pool])
        self.imp = [e["is_impossible"] for e in self.pool]

    def get(self, question: str, k: int) -> list[dict]:
        sims = (self.mat @ self.vec.transform([question]).T).toarray().ravel()
        order = sims.argsort()[::-1]
        ans = [i for i in order[:400] if not self.imp[i]]
        noa = [i for i in order[:400] if self.imp[i]]
        picks = []
        for j in range(k):
            src = ans if j % 2 == 0 else noa
            picks.append(src[j // 2])
        # most similar demo placed last (closest to the query)
        picks.sort(key=lambda i: sims[i])
        return [self.pool[i] for i in picks]


def build_messages(cfg, prompt_id, ex, demos, order_seed, system: str | None = None) -> list[dict]:
    demos = list(demos)
    if order_seed is not None and demos:
        random.Random(order_seed * 100003 + 7).shuffle(demos)
    no_ans = cfg["no_answer_string"]
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    for d in demos:
        msgs.append({"role": "user", "content": user_turn(cfg, prompt_id, d["context"], d["question"])})
        msgs.append({"role": "assistant", "content": gold_answer(d, no_ans)})
    msgs.append({"role": "user", "content": user_turn(cfg, prompt_id, ex["context"], ex["question"])})
    return msgs


# ---------------------------------------------------------------- parsing / grounding
def parse_output(raw: str) -> str:
    """Return cleaned answer text, or "" for no-answer."""
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    s = lines[0]
    for _ in range(2):
        s = _PREFIX.sub("", s)
        s = s.strip().strip("\"'`*“”‘’«»").strip()
    if _NOANS.match(s):
        return ""
    s = s.rstrip(".").strip() if not re.search(r"\b\w\.$", s) else s  # keep abbreviations like "A.S."
    return s


def _word_spans(text: str):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def ground(answer: str, context: str) -> tuple[str, str, int]:
    """Return (pred, grounding, start_char)."""
    if not answer:
        return "", "noans", -1
    i = context.lower().find(answer.lower())
    if i >= 0:
        return context[i:i + len(answer)], "exact", i
    spans = _word_spans(context)
    toks = [normalize_answer(context[a:b]) for a, b in spans]
    m = max(1, len(answer.split()))
    best, best_f = None, 0.0
    for n in range(max(1, m - 2), m + 3):
        for s in range(0, len(spans) - n + 1):
            cand = " ".join(t for t in toks[s:s + n] if t)
            if not cand:
                continue
            sc = f1(answer, cand)
            if sc > best_f:
                best_f, best = sc, (s, s + n)
    if best is not None and best_f >= SNAP_MIN_F1:
        a, b = spans[best[0]][0], spans[best[1] - 1][1]
        text = context[a:b].rstrip(".,;:!?")
        return text, "snapped", a
    return answer, "ungrounded", -1


# ---------------------------------------------------------------- model
def load_model(name: str, device: str, load_4bit: bool, dtype: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = {"": 0}
    else:
        dt_ = {"auto": torch.float16 if device == "cuda" else torch.float32,
               "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        kw["dtype"] = dt_
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if not load_4bit:
        model.to(device)
    model.eval()
    return tok, model


def generate_batch(tok, model, msgs_batch, max_new_tokens, no_tok_id):
    """Greedy decode a batch. Returns list of (raw_text, na_prob, mean_logprob)."""
    import torch
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_batch]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                             top_p=None, top_k=None, return_dict_in_generate=True, output_scores=True,
                             pad_token_id=tok.pad_token_id)
    seqs = out.sequences[:, enc["input_ids"].shape[1]:]
    first = torch.log_softmax(out.scores[0].float(), dim=-1)
    res = []
    for b in range(seqs.shape[0]):
        ids = seqs[b].tolist()
        n_keep = len(ids)
        eos_ids = {tok.eos_token_id, tok.pad_token_id}
        gen_eos = getattr(model.generation_config, "eos_token_id", None)
        eos_ids |= set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
        for j, t in enumerate(ids):
            if t in eos_ids:
                n_keep = j
                break
        raw = tok.decode(ids[:n_keep], skip_special_tokens=True)
        lp = first[b]
        p_no = lp[no_tok_id].exp().item()
        other = lp.clone()
        other[no_tok_id] = -1e9
        p_other = other.max().exp().item()
        na_prob = p_no / (p_no + p_other + 1e-12)
        tl = [torch.log_softmax(out.scores[j][b].float(), -1)[ids[j]].item() for j in range(n_keep)]
        res.append((raw, na_prob, sum(tl) / len(tl) if tl else 0.0))
    return res, texts


# ---------------------------------------------------------------- sequence log-prob scoring
def end_token_id(tok, model) -> int:
    g = getattr(model.generation_config, "eos_token_id", None)
    g = g[0] if isinstance(g, list) else g
    return g if g is not None else tok.eos_token_id


def score_continuations(tok, model, prompt_text: str, conts: list[str], return_ids: bool = False):
    """Teacher-forced log-prob of each continuation string (+ end-of-turn token) given prompt_text.

    Prompt and continuation are tokenized separately and concatenated (the prompt ends in
    "<|im_start|>assistant\\n", so nothing can merge across the boundary). Right padding, one forward pass.
    Returns list of (total_logprob, n_tokens[, scored_ids])."""
    import torch
    p_ids = tok(prompt_text, add_special_tokens=False).input_ids
    P = len(p_ids)
    end_id = end_token_id(tok, model)
    c_list = [tok.encode(c, add_special_tokens=False) + [end_id] for c in conts]
    L = max(len(c) for c in c_list)
    ids = torch.full((len(conts), P + L), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for b, c in enumerate(c_list):
        ids[b, :P + len(c)] = torch.tensor(p_ids + c)
        mask[b, :P + len(c)] = 1
    ids, mask = ids.to(model.device), mask.to(model.device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
    # token at index t is predicted by logits at index t-1: continuation tokens P..P+L-1 use logits P-1..P+L-2
    lp = torch.log_softmax(logits[:, P - 1:P + L - 1].float(), dim=-1)
    out = []
    for b, c in enumerate(c_list):
        tgt = ids[b, P:P + len(c)]
        tl = lp[b, torch.arange(len(c)), tgt]
        item = (tl.sum().item(), len(c))
        out.append(item + (tgt.tolist(),) if return_ids else item)
    return out


_STOP = {"yang", "dan", "di", "ke", "dari", "pada", "untuk", "dengan", "adalah", "itu", "ini", "apa", "siapa",
         "kapan", "dimana", "mana", "berapa", "bagaimana", "mengapa", "apakah", "dalam", "oleh", "atau", "the"}
_PUNCT = ".,;:!?()\"'"


def candidate_spans(question: str, context: str, k: int = 3) -> list[str]:
    """Up to k answer-like spans from the context, used when the model's greedy output is no-answer.
    Sentences are ranked by content-word overlap with the question (ties: earlier first). Within the best
    sentences, candidates are runs of capitalized words / numbers (entity-like, not sentence-initial) whose
    words are not all in the question."""
    q = {t for t in normalize_answer(question).split() if t not in _STOP}
    sents = [x for x in re.split(r"(?<=[.!?])\s+", context) if x.strip()]
    ranked = sorted(range(len(sents)), key=lambda i: (-len(q & set(normalize_answer(sents[i]).split())), i))

    def cap(w: str) -> bool:
        w = w.strip(_PUNCT)
        return bool(w) and (w[0].isupper() or w[0].isdigit())

    out, seen = [], set()
    for si in ranked:
        toks = sents[si].split()
        j = 1
        while j < len(toks):
            if cap(toks[j]):
                e = j
                while e + 1 < len(toks) and cap(toks[e + 1]):
                    e += 1
                span = " ".join(toks[j:e + 1]).strip(_PUNCT)
                key = normalize_answer(span)
                if key and key not in seen and not set(key.split()) <= q:
                    seen.add(key)
                    out.append(span)
                    if len(out) == k:
                        return out
                j = e + 1
            else:
                j += 1
    return out


def seq_na_scores(tok, model, prompt_text: str, no_ans: str, answer: str, question: str, context: str) -> dict:
    """lp_noans / lp_ans (totals), length-normalized variants and na_prob_seq = sigmoid(lp_noans - lp_ans).
    `answer` is the final grounded pred; if empty (greedy was no-answer) the best of candidate_spans is used."""
    cands = [answer] if answer else candidate_spans(question, context)
    res = score_continuations(tok, model, prompt_text, [no_ans] + cands)
    (lp_no, n_no), rest = res[0], res[1:]
    if rest:
        j = max(range(len(rest)), key=lambda i: rest[i][0])
        lp_a, n_a = rest[j]
        ans_scored = cands[j]
    else:  # no candidate span found: lean towards abstaining
        lp_a, n_a, ans_scored = lp_no - 20.0, 1, ""
    d = lp_no - lp_a
    na = 1.0 / (1.0 + math.exp(-d)) if d > -700 else 0.0
    return {"lp_noans": lp_no, "lp_ans": lp_a, "lpn_noans": lp_no / n_no, "lpn_ans": lp_a / n_a,
            "na_prob_seq": na, "ans_scored": ans_scored}


def self_test(tok, model, cfg, no_ans: str) -> None:
    """Assert scored positions are exactly the continuation tokens and match an unpadded reference forward."""
    import torch
    ex = {"context": "Jakarta adalah ibu kota Indonesia.", "question": "Apa ibu kota Indonesia?"}
    text = tok.apply_chat_template(build_messages(cfg, "id1", ex, [], None), tokenize=False,
                                   add_generation_prompt=True)
    conts = [no_ans, "Jakarta", "Tahun 1945 di Jakarta"]
    res = score_continuations(tok, model, text, conts, return_ids=True)
    end_id = end_token_id(tok, model)
    p_ids = tok(text, add_special_tokens=False).input_ids
    for c, (tot, n, ids) in zip(conts, res):
        dec = tok.decode(ids[:-1])
        assert dec == c, (dec, c)
        assert ids[-1] == end_id and n == len(ids)
        full = torch.tensor([p_ids + ids]).to(model.device)
        with torch.no_grad():
            lg = torch.log_softmax(model(input_ids=full).logits[0].float(), -1)
        ref = sum(lg[len(p_ids) + t - 1, ids[t]].item() for t in range(n))
        assert abs(ref - tot) < 1e-2 * max(1.0, abs(ref)), (c, ref, tot)
        print(f"ok {c!r}: scored slice decodes to {dec!r} + {tok.decode([ids[-1]])!r}; total lp {tot:.3f} "
              f"(unpadded reference {ref:.3f}); n_tokens {n}")
    print("prompt tail:", repr(tok.decode(p_ids[-4:])))
    print("candidate_spans demo:", candidate_spans(
        "Apa ibu kota Indonesia?", "Jakarta adalah ibu kota Indonesia. Presiden pertama adalah Soekarno sejak 1945."))
    print("SELF-TEST PASSED")


def run_api(a, cfg, data, demos_fixed, retr, client, name, out_dir) -> None:
    """API backend: same messages, parser, grounding and output files as the local path."""
    from mrc import llm_api
    log = open(out_dir / "log.txt", "w", encoding="utf-8")

    def P(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    P(f"run {name}; backend {a.backend}; model {client.model}; n={len(data)}; rpm {a.rpm}; mock {a.mock}; "
      f"cache {client.cache_path.name} ({len(client.cache)} entries)")
    preds, t_start = [], time.time()
    for i, ex in enumerate(data):
        msgs = build_messages(cfg, a.prompt_id, ex, retr.get(ex["question"], a.shots) if retr else demos_fixed,
                              a.demo_order_seed, a.system)
        rec = client.chat(msgs)
        raw, lp1 = llm_api.extract(rec)
        parsed = parse_output(raw)
        pred, g, start = ground(parsed, ex["context"])
        p = {"id": ex["id"], "pred": pred, "score": lp1 if lp1 is not None else 0.0, "start_char": start,
             "raw_output": raw, "grounding": g}
        if lp1 is not None:
            p["na_prob"] = llm_api.crude_na_prob(lp1, parsed == "")
        preds.append(p)
        if i % 10 == 0:
            P(f"{i + 1}/{len(data)} calls={client.n_calls} cached={client.n_cached}")
    elapsed = time.time() - t_start
    has_na = all("na_prob" in p for p in preds)
    with open(out_dir / "preds_dev.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    gold_path = out_dir / "_gold_subset.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    metrics = evaluate(gold_path, out_dir / "preds_dev.jsonl", search=has_na)
    gold_path.unlink()
    n = len(preds)
    gr = {k: sum(p["grounding"] == k for p in preds) / n for k in ["exact", "snapped", "ungrounded", "noans"]}
    metrics["grounding_rates"] = gr
    metrics["grounding_counts"] = {k: sum(p["grounding"] == k for p in preds) for k in gr}
    metrics["hallucination_rate_ungrounded"] = gr["ungrounded"]
    metrics["format_failure_rate"] = sum(
        1 for p in preds if p["grounding"] != "noans" and len(p["raw_output"].split()) > 25) / n
    metrics["na_prob_available"] = has_na
    metrics["api_calls_this_run"], metrics["api_cache_hits"] = client.n_calls, client.n_cached
    (out_dir / "metrics_dev.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cfgd = vars(a) | {"run_name": name, "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                      "run_date": dt.date.today().isoformat(), "model_requested": client.model,
                      "provider_returned_models": sorted(client.returned_models),
                      "api_params": client.params, "base_url": client.base_url,
                      "fixed_demo_ids": [d["id"] for d in demos_fixed], "n_examples": n,
                      "snap_min_f1": SNAP_MIN_F1, "elapsed_sec": elapsed}
    (out_dir / "config.json").write_text(json.dumps(cfgd, indent=2), encoding="utf-8")
    P(json.dumps({k: metrics[k] for k in ["exact", "f1", "HasAns_f1", "NoAns_acc", "pred_noans_rate",
                                          "grounding_counts", "api_calls_this_run", "api_cache_hits"]
                  if k in metrics}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "openrouter", "gemini"], default="hf",
                    help="hf = local transformers (default); others call an OpenAI-compatible API")
    ap.add_argument("--api-model", default=None, help="API model id (default per backend, see llm_api.BACKENDS)")
    ap.add_argument("--base-url", default=None, help="override API base URL")
    ap.add_argument("--rpm", type=float, default=15, help="API client-side requests per minute")
    ap.add_argument("--logprobs", action="store_true", help="API: request logprobs, derive crude na_prob")
    ap.add_argument("--reasoning-effort", default=None, help="API: pass reasoning_effort (e.g. none) if supported")
    ap.add_argument("--mock", action="store_true", help="API: fake responder, no network, no key")
    ap.add_argument("--allow-test", action="store_true", help="API: allow split files with 'test' in the name")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--split-file", default="data/processed/dev.jsonl")
    ap.add_argument("--train-file", default="data/processed/train.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="first N examples")
    ap.add_argument("--prompt-id", default="id1")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--demo-mode", choices=["fixed", "retrieved"], default="fixed")
    ap.add_argument("--demo-order-seed", type=int, default=None)
    ap.add_argument("--demo-seed", type=int, default=1234, help="seed choosing the fixed demos")
    ap.add_argument("--system", default=None, help="optional system prompt")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--adapter-dir", default=None, help="LoRA adapter from experiments/05-llm-local-qwen/llm_lora.py")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out-root", default=None,
                    help="default experiments/05-llm-local-qwen/runs (hf) or experiments/06-llm-api-offtheshelf/runs (API)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--prompts-file", default=None)
    ap.add_argument("--na-method", choices=["first_token", "seqlogprob"], default="seqlogprob")
    ap.add_argument("--self-test", action="store_true", help="check sequence-scoring token positions, then exit")
    a = ap.parse_args()

    random.seed(a.seed)
    cfg = load_prompts(Path(a.prompts_file) if a.prompts_file else None)
    no_ans = cfg["no_answer_string"]
    api = a.backend != "hf"
    if api:
        from mrc import llm_api
        if "test" in Path(a.split_file).name.lower():
            if not a.allow_test:
                raise SystemExit("Refusing to send a split with 'test' in its name to a third-party API. "
                                 "Pass --allow-test only once the course confirms it is acceptable.")
            print("WARNING: sending test data to a third-party API provider.", flush=True)
        try:
            client = llm_api.ApiClient(a.backend, a.api_model, a.base_url, a.rpm, a.mock, logprobs=a.logprobs,
                                       max_tokens=a.max_new_tokens, reasoning_effort=a.reasoning_effort)
        except llm_api.ApiKeyMissing as e:
            raise SystemExit(f"error: {e}")
        a.model_name = client.model
    short = a.model_name.split("/")[-1].replace(":", "_")
    a.out_root = a.out_root or ("experiments/06-llm-api-offtheshelf/runs" if api else "experiments/05-llm-local-qwen/runs")
    name = a.run_name or (f"{short}_{a.prompt_id}_{a.shots}shot"
                          + (f"_{a.demo_mode}" if a.shots else "")
                          + (f"_ord{a.demo_order_seed}" if a.demo_order_seed is not None and a.shots else "")
                          + (f"_n{a.limit}" if a.limit else ""))
    out_dir = ROOT / a.out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_jsonl(ROOT / a.split_file if not Path(a.split_file).is_absolute() else a.split_file)
    if a.limit:
        data = data[:a.limit]
    demos_fixed, retr = [], None
    if a.shots:
        train = load_jsonl(ROOT / a.train_file)
        if a.demo_mode == "fixed":
            demos_fixed = fixed_demos(train, a.shots, a.demo_seed)
        else:
            retr = Retriever(train)

    if api:
        run_api(a, cfg, data, demos_fixed, retr, client, name, out_dir)
        return
    tok, model = load_model(a.model_name, a.device, a.load_4bit, a.dtype)
    if a.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter_dir).eval()
    no_tok_id = tok.encode(no_ans, add_special_tokens=False)[0]
    if a.self_test:
        self_test(tok, model, cfg, no_ans)
        return
    log = open(out_dir / "log.txt", "w", encoding="utf-8")

    def P(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    P(f"run {name}; model {a.model_name}; n={len(data)}; no-answer first token id {no_tok_id} "
      f"= {tok.decode([no_tok_id])!r}; fixed demo ids {[d['id'] for d in demos_fixed]}")
    preds, t_start = [], time.time()
    for i in range(0, len(data), a.batch_size):
        batch = data[i:i + a.batch_size]
        msgs = [build_messages(cfg, a.prompt_id, ex,
                               retr.get(ex["question"], a.shots) if retr else demos_fixed,
                               a.demo_order_seed, a.system) for ex in batch]
        gen, texts = generate_batch(tok, model, msgs, a.max_new_tokens, no_tok_id)
        for ex, text, (raw, nap, sc) in zip(batch, texts, gen):
            pred, g, start = ground(parse_output(raw), ex["context"])
            rec = {"id": ex["id"], "pred": pred, "na_prob": nap, "na_prob_first": nap, "score": sc,
                   "start_char": start, "raw_output": raw, "grounding": g}
            if a.na_method == "seqlogprob":
                # scored string is exactly the final pred (what gets thresholded) when greedy gave an answer
                sq = seq_na_scores(tok, model, text, no_ans, pred, ex["question"], ex["context"])
                rec["na_prob"] = sq.pop("na_prob_seq")
                rec.update(sq)
            preds.append(rec)
        if (i // a.batch_size) % 10 == 0:
            P(f"{len(preds)}/{len(data)} {(time.time() - t_start) / len(preds):.2f}s/ex")
    elapsed = time.time() - t_start

    with open(out_dir / "preds_dev.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    gold_path = out_dir / "_gold_subset.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    metrics = evaluate(gold_path, out_dir / "preds_dev.jsonl", search=True)
    gold_path.unlink()
    n = len(preds)
    gr = {k: sum(p["grounding"] == k for p in preds) / n for k in ["exact", "snapped", "ungrounded", "noans"]}
    metrics["grounding_rates"] = gr
    metrics["hallucination_rate_ungrounded"] = gr["ungrounded"]
    metrics["format_failure_rate"] = sum(
        1 for p in preds if p["grounding"] != "noans" and len(p["raw_output"].split()) > 25) / n
    metrics["sec_per_example"] = elapsed / n
    metrics["na_method"] = a.na_method
    try:
        from sklearn.metrics import roc_auc_score
        y = [int(ex["is_impossible"]) for ex in data]
        if 0 < sum(y) < len(y):
            metrics["auc_na_prob"] = roc_auc_score(y, [p["na_prob"] for p in preds])
            metrics["auc_na_prob_first_token"] = roc_auc_score(y, [p["na_prob_first"] for p in preds])
    except ImportError:
        pass
    (out_dir / "metrics_dev.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cfgd = vars(a) | {"run_name": name, "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                      "fixed_demo_ids": [d["id"] for d in demos_fixed], "n_examples": n,
                      "snap_min_f1": SNAP_MIN_F1}
    (out_dir / "config.json").write_text(json.dumps(cfgd, indent=2), encoding="utf-8")
    P(json.dumps({k: metrics[k] for k in ["exact", "f1", "HasAns_f1", "NoAns_acc", "pred_noans_rate",
                                          "grounding_rates", "sec_per_example", "best_na_threshold", "auc_na_prob",
                                          "auc_na_prob_first_token"] if k in metrics}, indent=1))
    P("at best threshold:", json.dumps(metrics["at_best_threshold"]))


if __name__ == "__main__":
    main()
