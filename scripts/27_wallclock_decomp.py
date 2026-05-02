"""
Wall-clock decomposition for Tier 3 (on-policy training).

Each iter's wall-clock is decomposed into:
  - rollout_secs: time spent in VLLM (env + LLM)
  - train_secs: time in HF teacher-forcing forward + backward + optim
  - save_secs: writing LoRA checkpoint to disk

Plot: stacked bar chart per iter; aggregated total.

Outputs:
  results/figures/wallclock_decomp.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
import sys; sys.path.insert(0, str(ROOT))
from src.figutil import safe_savefig

OUT = ROOT / "results"
FIG = OUT / "figures"


def load(p):
    rows = []
    with open(p) as f:
        for line in f: rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    bl = load(ROOT / "runs" / "onpolicy_baseline" / "train.jsonl")
    gt = load(ROOT / "runs" / "onpolicy_gated" / "train.jsonl")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Stacked bar: rollout / train / save per iter
    for ax, df, name, color_set in [
        (axes[0], bl, "baseline", ["#1f77b4", "#aec7e8", "#c6dbef"]),
        (axes[1], gt, "gated", ["#ff7f0e", "#ffbb78", "#fdd0a2"]),
    ]:
        ax.bar(df.iter, df.rollout_secs, color=color_set[0], label="rollout")
        ax.bar(df.iter, df.train_secs, bottom=df.rollout_secs,
                color=color_set[1], label="train")
        ax.bar(df.iter, df.save_secs,
                bottom=df.rollout_secs + df.train_secs,
                color=color_set[2], label="save")
        ax.set_xlabel("iteration")
        ax.set_ylabel("wall-clock (s)")
        total = df.rollout_secs.sum() + df.train_secs.sum() + df.save_secs.sum()
        roll_pct = df.rollout_secs.sum() / total * 100
        train_pct = df.train_secs.sum() / total * 100
        ax.set_title(f"{name}: total {total:.0f}s "
                     f"(rollout {roll_pct:.1f}%, train {train_pct:.1f}%)")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(0, 600)

    plt.tight_layout()
    safe_savefig(fig, FIG / "wallclock_decomp.png", dpi=140)
    plt.close(fig)
    print(f"wrote {FIG / 'wallclock_decomp.png'}")

    # Print summary
    for df, name in [(bl, "baseline"), (gt, "gated")]:
        roll = df.rollout_secs.sum()
        tr = df.train_secs.sum()
        sv = df.save_secs.sum()
        tot = roll + tr + sv
        print(f"\n{name}:")
        print(f"  rollout: {roll:7.0f}s ({roll/tot*100:.1f}%)")
        print(f"  train:   {tr:7.0f}s ({tr/tot*100:.1f}%)")
        print(f"  save:    {sv:7.0f}s ({sv/tot*100:.1f}%)")
        print(f"  total:   {tot:7.0f}s")

    # Where does the saving come from?
    bl_roll = bl.rollout_secs.sum()
    gt_roll = gt.rollout_secs.sum()
    print(f"\nRollout saving: {bl_roll:.0f}s -> {gt_roll:.0f}s "
          f"({(bl_roll - gt_roll)/bl_roll*100:.2f}%)")
    bl_tot = bl.cumulative_wall_clock_sec.iloc[-1]
    gt_tot = gt.cumulative_wall_clock_sec.iloc[-1]
    print(f"Total saving:   {bl_tot:.0f}s -> {gt_tot:.0f}s "
          f"({(bl_tot - gt_tot)/bl_tot*100:.2f}%)")


if __name__ == "__main__":
    main()
