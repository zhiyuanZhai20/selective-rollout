"""Compare baseline vs gated training runs.

Reads runs/baseline/train.jsonl and runs/gated/train.jsonl, makes a
side-by-side comparison plot and a CSV summary.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.figutil import safe_savefig

OUT = ROOT / "results"
FIG = OUT / "figures"


def load(p):
    rows = []
    if not p.exists():
        return pd.DataFrame()
    with open(p) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarise(df, name):
    if df.empty:
        print(f"\n[{name}] no data")
        return None
    print(f"\n[{name}] {len(df)} steps")
    print(f"  total wall-clock: {df.step_secs.sum():.1f}s")
    print(f"  rollout time:     {df.rollout_secs.sum():.1f}s "
          f"({df.rollout_secs.sum() / df.step_secs.sum() * 100:.1f}% of total)")
    print(f"  groups cut:       {df.n_groups_cut.sum()} / {df.n_groups.sum()}")
    print(f"  zero-var groups:  {df.n_zero_variance.sum()} / {df.n_groups.sum()}")
    print(f"  rewards mean (final 5 steps):  {df.rewards_mean.tail(5).mean():.3f}")
    print(f"  rewards mean (first 5 steps):  {df.rewards_mean.head(5).mean():.3f}")
    return {
        "name": name,
        "n_steps": len(df),
        "total_wall_secs": float(df.step_secs.sum()),
        "rollout_secs": float(df.rollout_secs.sum()),
        "n_groups": int(df.n_groups.sum()),
        "n_groups_cut": int(df.n_groups_cut.sum()),
        "n_zero_var": int(df.n_zero_variance.sum()),
        "rewards_first5": float(df.rewards_mean.head(5).mean()),
        "rewards_last5": float(df.rewards_mean.tail(5).mean()),
        "loss_first5": float(df.loss.head(5).mean()),
        "loss_last5": float(df.loss.tail(5).mean()),
        "grad_norm_mean": float(df.grad_norm.mean()),
    }


def main():
    bl = load(ROOT / "runs" / "baseline" / "train.jsonl")
    gt = load(ROOT / "runs" / "gated" / "train.jsonl")
    bl_s = summarise(bl, "baseline")
    gt_s = summarise(gt, "gated")
    rows = [s for s in (bl_s, gt_s) if s is not None]
    if not rows:
        print("\nno data to plot")
        return
    pd.DataFrame(rows).to_csv(OUT / "training_summary.csv", index=False)
    print(f"\nwrote {OUT / 'training_summary.csv'}")

    if bl.empty or gt.empty:
        print("\n(only one run available — skipping comparison plot)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    # 1. cumulative wall-clock
    axes[0, 0].plot(bl.step, np.cumsum(bl.step_secs), color="#1f77b4",
                     marker="o", label="baseline")
    axes[0, 0].plot(gt.step, np.cumsum(gt.step_secs), color="#ff7f0e",
                     marker="o", label="gated")
    axes[0, 0].set_xlabel("training step")
    axes[0, 0].set_ylabel("cumulative wall-clock (s)")
    axes[0, 0].set_title("Wall-clock (gate saves time)")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # 2. rewards mean
    axes[0, 1].plot(bl.step, bl.rewards_mean, color="#1f77b4",
                     marker="o", label="baseline")
    axes[0, 1].plot(gt.step, gt.rewards_mean, color="#ff7f0e",
                     marker="o", label="gated")
    axes[0, 1].set_xlabel("training step")
    axes[0, 1].set_ylabel("rewards mean (per step)")
    axes[0, 1].set_title("Reward (gate doesn't hurt)")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # 3. groups cut by step (gated only)
    axes[1, 0].bar(gt.step, gt.n_groups_cut, color="#ff7f0e", alpha=0.7,
                    label="gated cuts")
    axes[1, 0].bar(bl.step, bl.n_zero_variance, color="#1f77b4", alpha=0.4,
                    label="baseline zero-var (cuts that COULD have been made)")
    axes[1, 0].set_xlabel("training step")
    axes[1, 0].set_ylabel("# groups")
    axes[1, 0].set_title("Gate firing rate (vs zero-var ground truth)")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3, axis="y")

    # 4. grad norm
    axes[1, 1].plot(bl.step, bl.grad_norm, color="#1f77b4",
                     marker="o", label="baseline")
    axes[1, 1].plot(gt.step, gt.grad_norm, color="#ff7f0e",
                     marker="o", label="gated")
    axes[1, 1].set_xlabel("training step")
    axes[1, 1].set_ylabel("gradient norm")
    axes[1, 1].set_title("Gradient norm (gate preserves signal)")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    safe_savefig(fig, FIG / "training_compare.png", dpi=140)
    plt.close(fig)
    print(f"wrote {FIG / 'training_compare.png'}")


if __name__ == "__main__":
    main()
