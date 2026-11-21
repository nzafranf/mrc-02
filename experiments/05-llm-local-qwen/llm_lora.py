"""QLoRA / LoRA SFT stub for Indonesian extractive QA (chat format, loss on assistant tokens only).

Target = exact gold span, or "TIDAK ADA JAWABAN" for unanswerable. Train split only.
The saved adapter is evaluated with `python experiments/global/run.py mrc.llm_qa --adapter-dir <dir>/adapter`.

Local smoke (CPU, 5 steps):
  python experiments/05-llm-local-qwen/llm_lora.py --max-steps 5 --batch-size 2 --out-dir experiments/05-llm-local-qwen/lora_smoke
Colab T4 (QLoRA):
  python experiments/05-llm-local-qwen/llm_lora.py --model-name Qwen/Qwen2.5-7B-Instruct --device cuda --load-4bit \
      --epochs 1 --batch-size 4 --grad-accum 4 --lr 2e-4 --out-dir experiments/05-llm-local-qwen/lora_qwen7b
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

from mrc.evaluate import load_jsonl
from mrc.llm_qa import ROOT, gold_answer, load_model, load_prompts, user_turn


def encode(tok, cfg, prompt_id, ex, max_len):
    msgs = [{"role": "user", "content": user_turn(cfg, prompt_id, ex["context"], ex["question"])}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    end = tok.convert_ids_to_tokens(tok.eos_token_id)  # <|im_end|> for Qwen instruct
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok(gold_answer(ex, cfg["no_answer_string"]) + end, add_special_tokens=False)["input_ids"]
    ids = (p + r)[-max_len:]
    labels = ([-100] * len(p) + r)[-max_len:]
    return ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train-file", default="data/processed/train.jsonl")
    ap.add_argument("--prompt-id", default="id1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", default="experiments/05-llm-local-qwen/lora_smoke")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    cfg = load_prompts()
    tok, model = load_model(a.model_name, a.device, a.load_4bit, "fp32" if a.device == "cpu" else "auto")
    if a.load_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.train()

    train = load_jsonl(ROOT / a.train_file)
    random.shuffle(train)
    steps_total = a.max_steps if a.max_steps > 0 else int(a.epochs * len(train) / (a.batch_size * a.grad_accum))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    warm = max(1, int(0.05 * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 1 - s / max(1, steps_total)))
    out_dir = ROOT / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "log.txt", "w", encoding="utf-8")
    t0, pos = time.time(), 0
    pad = tok.pad_token_id
    for step in range(steps_total):
        tot = 0.0
        for _ in range(a.grad_accum):
            batch = [encode(tok, cfg, a.prompt_id, train[(pos + j) % len(train)], a.max_len)
                     for j in range(a.batch_size)]
            pos += a.batch_size
            n = max(len(i) for i, _ in batch)
            dev = model.device
            ids = torch.tensor([i + [pad] * (n - len(i)) for i, _ in batch]).to(dev)
            lab = torch.tensor([l + [-100] * (n - len(l)) for _, l in batch]).to(dev)
            att = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i, _ in batch]).to(dev)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / a.grad_accum
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        msg = f"step {step + 1}/{steps_total} loss {tot:.4f} elapsed {time.time() - t0:.1f}s"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
    model.save_pretrained(out_dir / "adapter")
    (out_dir / "config.json").write_text(json.dumps(vars(a), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
