"""Run an API LLM on dev in growing batches, stopping early if a batch looks broken.

Each stage reruns mrc.llm_qa (experiments/global) with a larger --limit and the same run name. Responses are cached,
so a stage only pays for its new examples. After each stage a gate checks the cumulative preds.

Usage:
  python experiments/06-llm-api-offtheshelf/run_llm_api_staged.py --backend openrouter --api-model google/gemma-4-31b-it:free \
      --prompt-id id1 --shots 0 --run-name gemma4_31b_id1_0shot [--stages 5 25 100 803] [--yes]
Without --yes the script pauses after each passing stage and asks to continue.
"""
import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

GATES = {  # threshold, description
    "empty_rate_max": (0.02, "empty raw outputs (provider returned nothing, or reasoning used the token budget)"),
    "ungrounded_rate_max": (0.35, "outputs that cannot be mapped to a passage span"),
    "noans_rate_min": (0.03, "no-answer predictions (lower means the instruction is ignored)"),
    "noans_rate_max": (0.95, "no-answer predictions (higher means collapse to abstaining)"),
}


def check(preds: list[dict], stage_n: int) -> list[str]:
    n = len(preds)
    fails = []
    if n < stage_n:
        fails.append(f"only {n} preds written, expected {stage_n}")
    if n == 0:
        return fails
    empty = sum(not (p.get("raw_output") or "").strip() for p in preds) / n
    g = collections.Counter(p.get("grounding") for p in preds)
    ungrounded, noans = g["ungrounded"] / n, g["noans"] / n
    print(f"  n={n} grounding={dict(g)} empty={empty:.2f}")
    if empty > GATES["empty_rate_max"][0]:
        fails.append(f"empty rate {empty:.2f}: {GATES['empty_rate_max'][1]}")
    if n >= 20:  # rate gates are meaningless on tiny samples
        if ungrounded > GATES["ungrounded_rate_max"][0]:
            fails.append(f"ungrounded rate {ungrounded:.2f}: {GATES['ungrounded_rate_max'][1]}")
        if noans < GATES["noans_rate_min"][0]:
            fails.append(f"no-answer rate {noans:.2f}: {GATES['noans_rate_min'][1]}")
        if noans > GATES["noans_rate_max"][0]:
            fails.append(f"no-answer rate {noans:.2f}: {GATES['noans_rate_max'][1]}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["openrouter", "gemini"])
    ap.add_argument("--api-model", required=True)
    ap.add_argument("--prompt-id", default="id1")
    ap.add_argument("--shots", default="0")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--stages", type=int, nargs="+", default=[5, 25, 100, 803])
    ap.add_argument("--rpm", default="15")
    ap.add_argument("--yes", action="store_true", help="do not pause between passing stages")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="extra args passed to llm_qa after --")
    a = ap.parse_args()
    run_dir = ROOT / "experiments/06-llm-api-offtheshelf/runs" / a.run_name
    extra = [x for x in a.extra if x != "--"]
    for i, n in enumerate(a.stages):
        print(f"\n== stage {i + 1}/{len(a.stages)}: first {n} dev examples")
        cmd = [sys.executable, str(ROOT / "experiments/global/run.py"), "mrc.llm_qa", "--backend", a.backend, "--api-model", a.api_model,
               "--prompt-id", a.prompt_id, "--shots", a.shots, "--run-name", a.run_name, "--rpm", a.rpm,
               "--limit", str(n), *extra]
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"STOP: llm_qa exited with code {r.returncode} at stage {n}. Check {run_dir / 'log.txt'}.")
        preds = [json.loads(l) for l in open(run_dir / "preds_dev.jsonl", encoding="utf-8")]
        if i == 0:
            print("  sample raw outputs:")
            for p in preds[:5]:
                print(f"    {p['id']}: {p.get('raw_output', '')[:100]!r} -> {p.get('grounding')}")
        fails = check(preds, n)
        m = json.loads((run_dir / "metrics_dev.json").read_text(encoding="utf-8"))
        print(f"  F1={m['f1']:.1f} EM={m['exact']:.1f} HasAns_f1={m['HasAns_f1']:.1f} NoAns_acc={m['NoAns_acc']:.1f}")
        if fails:
            sys.exit("STOP at stage %d:\n  - " % n + "\n  - ".join(fails))
        if not a.yes and i < len(a.stages) - 1 and input("  gate passed, continue? [y/N] ").strip().lower() != "y":
            sys.exit("stopped by user")
    print("\nall stages passed")


if __name__ == "__main__":
    main()
