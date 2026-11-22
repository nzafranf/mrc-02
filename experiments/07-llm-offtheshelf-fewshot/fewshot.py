"""Prompt and demo handling for 07-llm-offtheshelf-fewshot (imported by select_demos.py and run_fewshot.py).

The demos live in demos.json, written once by select_demos.py. Every run loads them through load_demos(), which
re-reads each demo from the train split by id and refuses to run if the text differs from the frozen copy or the
file's hash changed, so all runs see exactly the same examples in the same order.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from mrc.evaluate import load_jsonl
from mrc.paths import PROC

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts.yaml"
DEMOS = HERE / "demos.json"
DEMO_FIELDS = ("id", "context", "question", "is_impossible", "answer")


def load_prompts() -> dict:
    return yaml.safe_load(PROMPTS.read_text(encoding="utf-8"))


def demos_hash(demos: list[dict]) -> str:
    blob = json.dumps([{k: d[k] for k in DEMO_FIELDS} for d in demos], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def demo_record(ex: dict, slot: str) -> dict:
    ans = "" if ex["is_impossible"] else ex["answers"][0]["text"]
    return {"id": ex["id"], "slot": slot, "is_impossible": ex["is_impossible"], "question": ex["question"],
            "answer": ans, "answer_start": -1 if ex["is_impossible"] else ex["answers"][0]["answer_start"],
            "context": ex["context"]}


def load_demos(train_file: Path = PROC / "train.jsonl") -> tuple[list[dict], str]:
    """Frozen demos in prompt order, after checking them against the train split. Returns (demos, sha256)."""
    meta = json.loads(DEMOS.read_text(encoding="utf-8"))
    demos = meta["demos"]
    h = demos_hash(demos)
    if h != meta["sha256"]:
        raise SystemExit(f"{DEMOS.name} was edited by hand (hash mismatch). Rebuild it with select_demos.py "
                         f"(use --pin to choose specific ids).")
    train = {e["id"]: e for e in load_jsonl(train_file)}
    for d in demos:
        e = train.get(d["id"])
        if e is None:
            raise SystemExit(f"demo {d['id']} is not in {train_file}; demos must come from the train split")
        if demo_record(e, d["slot"]) | {"slot": d["slot"]} != d:
            raise SystemExit(f"demo {d['id']} differs from its train record")
    return demos, h


def no_answer(cfg: dict) -> str:
    return cfg["no_answer_string"]


def user_turn(v: dict, context: str, question: str) -> str:
    return f"{v['context_label']}: {context}\n\n{v['question_label']}: {question}"


def build_messages(cfg: dict, variant: str, demos: list[dict], ex: dict, system_role: bool = True) -> list[dict]:
    """system instruction, demos as user/assistant pairs, then the query. Without a system role the instruction
    is put at the top of the first user turn (same text, same order)."""
    v = cfg["variants"][variant]
    msgs = [{"role": "system", "content": v["system"]}] if system_role else []
    for d in demos:
        msgs.append({"role": "user", "content": user_turn(v, d["context"], d["question"])})
        msgs.append({"role": "assistant", "content": d["answer"] or no_answer(cfg)})
    msgs.append({"role": "user", "content": user_turn(v, ex["context"], ex["question"])})
    if not system_role:
        msgs[0] = {"role": "user", "content": v["system"] + "\n\n" + msgs[0]["content"]}
    return msgs


def prompt_hash(cfg: dict, variant: str) -> str:
    v = cfg["variants"][variant]
    blob = json.dumps({"variant": v, "no_answer": no_answer(cfg)}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
