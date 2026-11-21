"""Join every model's dev prediction into one record per dev example, with the full instance.

Output: report/assets/tables/per_example_all_models.jsonl, one line per dev example:
  id, question, context, answers [{text, answer_start}], is_impossible, slices {...},
  models: {model: {"seeds": {seed: {pred, start, em, f1, na_prob, category}}, "mean_f1", "n_seeds_correct"}}
  partial_models: same shape, only for API/LLM runs that covered this id (no threshold tuning).

Thresholds and categories come from mrc.analysis.compare (same as main_table.md): the no-answer
threshold is tuned per run on dev, so these are dev-optimistic.

Usage: python experiments/global/run.py mrc.analysis.per_example
"""
import json
from pathlib import Path

import yaml

from mrc.analysis.compare import ROOT, classify, load_runs, sentences
from mrc.evaluate import load_jsonl, per_example_scores

DEV = ROOT / "data/processed/dev.jsonl"
SLICES = ROOT / "data/processed/slices_dev.jsonl"
OUT = ROOT / "report/assets/tables/per_example_all_models.jsonl"
# Partial runs: scored as-is on the ids they cover. Add more here.
PARTIAL = {
    "llm_nemotron35_api": ROOT / "experiments/06-llm-api-offtheshelf/runs/nemotron35_super_id1_0shot",
    "llm_gemini35flash_api": ROOT / "experiments/06-llm-api-offtheshelf/runs/gemini_35flash_id1_0shot",
}


def main():
    gold = load_jsonl(DEV)
    gold_by = {e["id"]: e for e in gold}
    sl = {s["id"]: s for s in load_jsonl(SLICES)}
    reg = yaml.safe_load(open(ROOT / "experiments/global/configs/models_registry.yaml", encoding="utf-8"))
    runs = load_runs(reg, gold)
    sents = {i: sentences(gold_by[i]["context"]) for i in gold_by}

    partial = {}
    for name, d in PARTIAL.items():
        p = d / "preds_dev.jsonl"
        if not p.exists():
            continue
        preds = {x["id"]: x for x in load_jsonl(p)}
        sub = [gold_by[i] for i in preds if i in gold_by]
        sc = per_example_scores(sub, preds, None)
        for i, v in sc.items():
            v["start"] = preds[i].get("start_char", -1)
            v["raw_output"] = preds[i].get("raw_output")
            v["grounding"] = preds[i].get("grounding")
        partial[name] = sc

    with open(OUT, "w", encoding="utf-8") as f:
        for e in gold:
            i = e["id"]
            rec = {k: e[k] for k in ("id", "question", "context", "answers", "is_impossible")}
            rec["slices"] = {k: v for k, v in sl.get(i, {}).items() if k not in ("id", "is_impossible")}
            rec["models"] = {}
            for m, rr in runs.items():
                seeds = {}
                for r in rr:
                    v = r["scores"][i]
                    seeds[r["seed"]] = {"pred": v["pred"], "start": v["start"], "em": v["em"],
                                        "f1": round(v["f1"], 4), "na_prob": v["na_prob"],
                                        "category": classify(e, v, sents[i])}
                rec["models"][m] = {"seeds": seeds,
                                    "mean_f1": round(sum(s["f1"] for s in seeds.values()) / len(seeds), 4),
                                    "n_seeds_correct": sum(s["em"] for s in seeds.values())}
            rec["partial_models"] = {}
            for m, sc in partial.items():
                if i in sc:
                    v = sc[i]
                    rec["partial_models"][m] = {"pred": v["pred"], "raw_output": v["raw_output"],
                                                "grounding": v["grounding"], "start": v["start"], "em": v["em"],
                                                "f1": round(v["f1"], 4), "category": classify(e, v, sents[i])}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(gold)} examples; models: {list(runs)}; partial: "
          f"{ {m: len(sc) for m, sc in partial.items()} })")


if __name__ == "__main__":
    main()
