"""Heatmaps of the context-to-query attention saved by bidaf (--save-attn).

python experiments/03-nn-bidaf/plot_attention.py --npz experiments/03-nn-bidaf/analysis/attn_seed13.npz --out-dir experiments/03-nn-bidaf/analysis/attn_plots
Rows are context tokens (cropped to the window around the most attended tokens), columns are question tokens.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # shared code: experiments/global/mrc
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "global"))
from mrc.paths import ROOT, at_root  # noqa: E402
at_root()  # relative paths resolve from the project root
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="experiments/03-nn-bidaf/analysis/attn_seed13.npz")
    ap.add_argument("--out-dir", default="experiments/03-nn-bidaf/analysis/attn_plots")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--window", type=int, default=40)
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=True)
    ids = [str(i) for i in d["ids"]][: a.n]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in ids:
        att, ctx, q = d[f"{i}__attn"], d[f"{i}__ctx"], d[f"{i}__q"]
        att = att[:, : len(q)]
        peak = int(att.max(1).argmax())  # most sharply aligned context row
        lo = max(0, min(peak - a.window // 2, len(ctx) - a.window))
        hi = min(len(ctx), lo + a.window)
        fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(q)), 0.28 * (hi - lo) + 1.5))
        im = ax.imshow(att[lo:hi], aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(q)), q, rotation=70, ha="right")
        ax.set_yticks(range(hi - lo), ctx[lo:hi])
        ax.set_xlabel("question token")
        ax.set_ylabel("context token")
        ax.set_title(f"context-to-query attention {i}", fontsize=9)
        fig.colorbar(im, ax=ax, label="attention weight")
        fig.tight_layout()
        fig.savefig(out / f"attn_{i}.png", dpi=130)
        plt.close(fig)
        print("wrote", out / f"attn_{i}.png")


if __name__ == "__main__":
    main()
