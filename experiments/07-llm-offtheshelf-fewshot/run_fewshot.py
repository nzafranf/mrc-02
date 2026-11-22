"""Few-shot off-the-shelf LLM on dev or the official validation set, in the fixed order, resumable.

  python experiments/07-llm-offtheshelf-fewshot/run_fewshot.py --model Qwen/Qwen2.5-7B-Instruct --load-4bit \
      --split dev --variant fs_id [--limit 200] [--batch-size 4]
  python experiments/07-llm-offtheshelf-fewshot/run_fewshot.py --backend openrouter --model <api id> --split dev

What makes runs comparable and safe to interrupt:
  * Questions are processed in the fixed order data/splits/order_<split>.txt (mrc.eval_order, global seed 13).
    `--limit N` means "the first N questions of that order", so two runs that stopped at different points share
    their common prefix, and a prefix is a stratified sample (same unanswerable share as the split).
  * Progress: one line per batch (bar, done/stop, s/question, ETA and finish time) on screen and in log.txt, and
    progress.json in the run folder (rewritten atomically after every batch; read it from another cell or tab).
  * Every prediction is appended to preds_<split>.partial.jsonl and fsynced as soon as it is made. Rerunning the same
    command continues after the last finished question; a half-written last line is dropped and redone.
  * config.json holds a fingerprint (model, backend, prompt hash, demo hash, order hash, decoding settings). A rerun
    with a different fingerprint in the same run folder is refused, so a resumed run never mixes settings.
  * The demos are the frozen demos.json (see select_demos.py), checked against the train split on every start.
After each session (also after --limit stops) the script writes preds_<split>.jsonl (finished questions, in order)
and metrics_<split>.json (scored on those questions; no-answer threshold tuned when na_prob exists). For dev the
files are named preds_dev.jsonl / metrics_dev.json, the run format every analyze.py reads.

na_prob (local models): mrc.llm_qa.seq_na_scores, i.e. sigmoid(log p("unanswerable") - log p(best answer span)) by
teacher forcing on the same prompt. API backends return no usable log-probs, so their runs are untuned.
Split files with "test" in the name are refused; only dev and official_val have a fixed order.
"""
import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import argparse
import datetime as dt
import json
import os
import platform
import time
from pathlib import Path

from fewshot import HERE, build_messages, load_demos, load_prompts, no_answer, prompt_hash
from mrc.eval_order import GLOBAL_SEED, SPLIT_FILES, load_order, order_path, sha256
from mrc.evaluate import evaluate, load_jsonl
from mrc.llm_qa import SNAP_MIN_F1, ground, parse_output

FINGERPRINT_KEYS = ["backend", "model", "variant", "split", "prompt_sha256", "demos_sha256", "order_sha256",
                    "max_new_tokens", "load_4bit", "dtype", "template_kwargs", "na_method", "system_role",
                    "reasoning_effort"]


def read_partial(path: Path, order: list[str]) -> list[dict]:
    """Finished predictions; drops a half-written last line; checks they are exactly a prefix of the order."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    done = []
    for k, line in enumerate(lines):
        try:
            done.append(json.loads(line))
        except json.JSONDecodeError:
            if k != len(lines) - 1:
                raise SystemExit(f"{path} is corrupt at line {k + 1} (not the last line); inspect it by hand")
            print(f"dropping a half-written last line in {path.name}")
    ids = [d["id"] for d in done]
    if ids != order[:len(ids)]:
        raise SystemExit(f"{path} does not follow the fixed order of the split; refusing to resume")
    with open(path, "w", encoding="utf-8", newline="\n") as f:  # rewrite without the dropped line
        f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in done)
    return done


def finalize(out_dir: Path, split: str, done: list[dict], gold_by: dict, n_total: int, meta: dict) -> dict:
    name = "dev" if split == "dev" else split
    preds_path = out_dir / f"preds_{name}.jsonl"
    with open(preds_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in done)
    gold_path = out_dir / "_gold_subset.jsonl"
    with open(gold_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(gold_by[p["id"]], ensure_ascii=False) + "\n" for p in done)
    has_na = all("na_prob" in p for p in done)
    m = evaluate(gold_path, preds_path, search=has_na)
    gold_path.unlink()
    n = len(done)
    g = {k: sum(p["grounding"] == k for p in done) for k in ["exact", "snapped", "ungrounded", "noans"]}
    m.update({"n_done": n, "n_split": n_total, "complete": n == n_total, "covered": f"first {n} of the fixed order",
              "grounding_counts": g, "hallucination_rate_ungrounded": g["ungrounded"] / n,
              "format_failure_rate": sum(p["grounding"] != "noans" and len(p["raw_output"].split()) > 25
                                         for p in done) / n,
              "empty_output_rate": sum(not p["raw_output"].strip() for p in done) / n,
              "na_prob_available": has_na, **meta})
    if has_na:
        try:
            from sklearn.metrics import roc_auc_score
            y = [int(gold_by[p["id"]]["is_impossible"]) for p in done]
            if 0 < sum(y) < n:
                m["auc_na_prob"] = roc_auc_score(y, [p["na_prob"] for p in done])
        except ImportError:
            pass
    (out_dir / f"metrics_{name}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m


def system_role_supported(tok) -> bool:
    probe = [{"role": "system", "content": "SYS-PROBE"}, {"role": "user", "content": "hi"}]
    try:
        return "SYS-PROBE" in tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
    except Exception:  # e.g. Gemma templates raise on a system turn
        return False


def model_revision(name: str) -> str | None:
    try:
        from huggingface_hub import model_info
        return model_info(name).sha
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="HF model id (local) or API model id")
    ap.add_argument("--backend", choices=["hf", "openrouter", "gemini"], default="hf")
    ap.add_argument("--split", choices=sorted(SPLIT_FILES), default="dev")
    ap.add_argument("--variant", default="fs_id", help="prompt variant in prompts.yaml")
    ap.add_argument("--limit", type=int, default=None, help="stop after the first N questions of the fixed order")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--score-chunk", type=int, default=None,
                    help="na_prob scoring: continuations per forward pass (default all 4 at once; 1 = least memory)")
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--load-4bit", action="store_true", help="bitsandbytes NF4 (needs a CUDA GPU)")
    ap.add_argument("--template-kwargs", default="{}", help='JSON for apply_chat_template, e.g. \'{"enable_thinking": false}\'')
    ap.add_argument("--checkpoint-every", type=int, default=100,
                    help="also rewrite preds_<split>.jsonl + metrics every N questions (0 = only at the end)")
    ap.add_argument("--zero-shot", action="store_true", help="no demos (ablation: instruction only)")
    ap.add_argument("--no-na-score", action="store_true", help="skip the na_prob scoring pass (faster, untuned)")
    ap.add_argument("--rpm", type=float, default=15, help="API requests per minute")
    ap.add_argument("--reasoning-effort", default=None, help="API: reasoning_effort, e.g. none")
    ap.add_argument("--run-name", default=None, help="default: <model>_<variant>")
    ap.add_argument("--out-root", default=None, help="default: experiments/07-llm-offtheshelf-fewshot/runs "
                                                     "(on Colab point it at Google Drive so a disconnect loses nothing)")
    a = ap.parse_args()

    if "test" in SPLIT_FILES[a.split].name:
        raise SystemExit("refusing a test split")
    cfg = load_prompts()
    if a.variant not in cfg["variants"]:
        raise SystemExit(f"unknown variant {a.variant}; choose from {list(cfg['variants'])}")
    no_ans = no_answer(cfg)
    order = load_order(a.split)
    gold_by = {e["id"]: e for e in load_jsonl(SPLIT_FILES[a.split])}
    demos, demos_sha = load_demos()
    if a.zero_shot:
        demos, demos_sha = [], "zero-shot"
    demo_ctx = {d["context"] for d in demos}
    assert not any(gold_by[i]["context"] in demo_ctx for i in order), "a demo passage occurs in the evaluation split"
    tkw = json.loads(a.template_kwargs)
    api = a.backend != "hf"

    run_name = a.run_name or f"{a.model.split('/')[-1].replace(':', '_')}_{a.variant}" + ("_0shot" if a.zero_shot else "")
    out_dir = Path(a.out_root or HERE / "runs") / a.split / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "log.txt", "a", encoding="utf-8")

    def P(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    # ---- model / client
    if api:
        from mrc import llm_api
        try:
            client = llm_api.ApiClient(a.backend, a.model, None, a.rpm, False, max_tokens=a.max_new_tokens,
                                       reasoning_effort=a.reasoning_effort)
        except llm_api.ApiKeyMissing as e:
            raise SystemExit(f"error: {e}")
        sys_role, na_method = True, "none"
    else:
        import torch
        from mrc.llm_qa import generate_batch, load_model, seq_na_scores
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok, model = load_model(a.model, device, a.load_4bit, a.dtype)
        sys_role = system_role_supported(tok)
        no_tok_id = tok.encode(no_ans, add_special_tokens=False)[0]
        na_method = "none" if a.no_na_score else "seqlogprob"

    fp = {"backend": a.backend, "model": a.model, "variant": a.variant, "split": a.split,
          "prompt_sha256": prompt_hash(cfg, a.variant), "demos_sha256": demos_sha,
          "order_sha256": sha256(order_path(a.split)), "max_new_tokens": a.max_new_tokens,
          "load_4bit": a.load_4bit, "dtype": a.dtype, "template_kwargs": tkw, "na_method": na_method,
          "system_role": sys_role, "reasoning_effort": a.reasoning_effort}
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        diff = {k: (old.get("fingerprint", {}).get(k), fp[k]) for k in FINGERPRINT_KEYS
                if old.get("fingerprint", {}).get(k) != fp[k]}
        if diff:
            raise SystemExit(f"{out_dir} holds a run with other settings {diff}; use another --run-name")
        conf = old
    else:
        conf = {"run_name": run_name, "fingerprint": fp, "global_seed": GLOBAL_SEED,
                "demo_ids": [d["id"] for d in demos], "no_answer_string": no_ans, "snap_min_f1": SNAP_MIN_F1,
                "decoding": "greedy", "model_revision": None if api else model_revision(a.model),
                "created": dt.datetime.now().isoformat(timespec="seconds"), "sessions": []}
    session = {"start": dt.datetime.now().isoformat(timespec="seconds"), "host": platform.node(),
               "python": platform.python_version(), "batch_size": a.batch_size, "limit": a.limit}
    if not api:
        import transformers
        session |= {"torch": torch.__version__, "transformers": transformers.__version__,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if a.load_4bit:
            import bitsandbytes
            session["bitsandbytes"] = bitsandbytes.__version__
    conf["sessions"].append(session)
    cfg_path.write_text(json.dumps(conf, indent=2), encoding="utf-8")

    partial = out_dir / f"preds_{a.split}.partial.jsonl"
    done = read_partial(partial, order)
    stop = min(a.limit or len(order), len(order))
    todo = order[len(done):stop]
    P(f"== {run_name} on {a.split}: {len(done)} done, {len(todo)} to do (stop at {stop}/{len(order)}); "
      f"model {a.model}; system role {sys_role}; demos {demos_sha[:12]}; na {na_method}")

    t0, n_start, n_oom, n_new = time.time(), len(done), 0, 0
    with open(partial, "a", encoding="utf-8", newline="\n") as f:

        def process(exs, score_chunk):
            """Predict exs and save each record as soon as it is complete."""
            msgs = [build_messages(cfg, a.variant, demos, ex, sys_role) for ex in exs]
            if api:
                from mrc import llm_api
                raw, _ = llm_api.extract(client.chat(msgs[0]))
                outs, texts = [(raw, None, 0.0)], [None]
            else:
                outs, texts = generate_batch(tok, model, msgs, a.max_new_tokens, no_tok_id, tkw)
            for ex, text, (raw, na_first, sc) in zip(exs, texts, outs):
                pred, g, start = ground(parse_output(raw), ex["context"])
                rec = {"id": ex["id"], "pred": pred, "score": sc, "start_char": start, "raw_output": raw,
                       "grounding": g}
                if not api:
                    rec["na_prob_first"] = na_first
                    if na_method == "seqlogprob":
                        sq = seq_na_scores(tok, model, text, no_ans, pred, ex["question"], ex["context"],
                                           chunk=score_chunk)
                        rec["na_prob"] = sq.pop("na_prob_seq")
                        rec.update(sq)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")  # saved per question, before the next one
                f.flush()
                os.fsync(f.fileno())
                done.append(rec)

        bs = 1 if api else a.batch_size
        for b in range(0, len(todo), bs):
            batch = [gold_by[i] for i in todo[b:b + bs]]
            before = len(done)
            try:
                process(batch, a.score_chunk)
            except Exception as e:
                if api or "OutOfMemoryError" not in type(e).__name__:
                    raise
                # out of GPU memory: free it and redo the unsaved questions of this batch one at a time,
                # scoring one continuation per pass (same predictions, less memory, slower)
                n_oom += 1
                rest = batch[len(done) - before:]
                del e
                torch.cuda.empty_cache()
                P(f"  out of GPU memory on a batch of {len(batch)}; redoing {len(rest)} question(s) one at a time")
                for ex in rest:
                    process([ex], 1)
                torch.cuda.empty_cache()
            n_new = len(done) - n_start
            el = time.time() - t0
            spq = el / max(n_new, 1)
            left = len(todo) - n_new
            eta = dt.datetime.now() + dt.timedelta(seconds=spq * left)
            counts = {k: sum(p["grounding"] == k for p in done) for k in ("exact", "snapped", "ungrounded", "noans")}
            prog = {"run": run_name, "split": a.split, "done": len(done), "stop": stop, "split_size": len(order),
                    "percent": round(100 * len(done) / stop, 1), "sec_per_question": round(spq, 2),
                    "elapsed_min_session": round(el / 60, 1), "eta_min": round(spq * left / 60, 1),
                    "eta_clock": eta.isoformat(timespec="minutes"), "grounding_so_far": counts,
                    "updated": dt.datetime.now().isoformat(timespec="seconds")}
            tmp = out_dir / "progress.json.tmp"
            tmp.write_text(json.dumps(prog, indent=1), encoding="utf-8")
            os.replace(tmp, out_dir / "progress.json")  # atomic: never half-written
            bar = "#" * int(30 * len(done) / stop)
            P(f"  [{bar:<30}] {len(done)}/{stop} {prog['percent']:5.1f}% | {spq:.2f}s/q | elapsed "
              f"{prog['elapsed_min_session']:.1f} min | ETA {prog['eta_min']:.1f} min ({eta:%H:%M}) | "
              f"noans {counts['noans']} ungrounded {counts['ungrounded']}")
            if a.checkpoint_every and len(done) % a.checkpoint_every < bs and left:
                finalize(out_dir, a.split, done, gold_by, len(order), {"checkpoint": True})
                P(f"  checkpoint: preds/metrics written for the first {len(done)}")

    if not done:
        conf["sessions"][-1] |= {"end": dt.datetime.now().isoformat(timespec="seconds"), "n_new": 0}
        cfg_path.write_text(json.dumps(conf, indent=2), encoding="utf-8")
        P("nothing to score yet")
        return
    m = finalize(out_dir, a.split, done, gold_by, len(order),
                 {"sec_per_question_last_session": (time.time() - t0) / max(n_new, 1)})
    conf["sessions"][-1] |= {"end": dt.datetime.now().isoformat(timespec="seconds"), "n_new": n_new, "oom_retries": n_oom,
                             "n_done": len(done)}
    cfg_path.write_text(json.dumps(conf, indent=2), encoding="utf-8")
    keys = ["n_done", "exact", "f1", "HasAns_f1", "NoAns_acc", "pred_noans_rate", "best_na_threshold", "auc_na_prob",
            "grounding_counts", "empty_output_rate"]
    P(json.dumps({k: m[k] for k in keys if k in m}, indent=1))
    if "at_best_threshold" in m:
        P("at best threshold:", json.dumps(m["at_best_threshold"]))


if __name__ == "__main__":
    main()
