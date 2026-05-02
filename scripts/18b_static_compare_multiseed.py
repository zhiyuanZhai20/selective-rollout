"""4-seed compare figure for Tier 2 (off-policy GRPO training A/B).

Loads runs/static_{arm}_s{seed}/train.jsonl for arm in {baseline,gated}
and seed in {7, 13, 23, 42}, plots seed-mean curves with ±1 std bands.
Mirrors the structure of scripts/30_compare_multiseed.py.
"""
from __future__ import annotations
import json
import sys
import statistics
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

SEEDS = [7, 13, 23, 42]


def load(arm, seed):
    p = ROOT / "runs" / f"static_{arm}_s{seed}" / "train.jsonl"
    return pd.DataFrame([json.loads(l) for l in open(p)])


def main():
    bl = [load("baseline", s) for s in SEEDS]
    gt = [load("gated", s) for s in SEEDS]
    n_steps = min(len(d) for d in bl + gt)
    print(f"Common step count: {n_steps}")

    # Per-seed totals
    bl_walls = [d.step_secs[:n_steps].sum() for d in bl]
    gt_walls = [d.step_secs[:n_steps].sum() for d in gt]
    saves = [(b - g) / b * 100 for b, g in zip(bl_walls, gt_walls)]
    save_mean, save_std = statistics.mean(saves), statistics.stdev(saves)
    print(f"per-seed wall-clock saving: {[f'{x:.1f}%' for x in saves]}")
    print(f"mean ± std: {save_mean:.1f} ± {save_std:.1f}%")

    bl_grads_run = [d.grad_norm[:n_steps].mean() for d in bl]
    gt_grads_run = [d.grad_norm[:n_steps].mean() for d in gt]
    print(f"baseline grad_norm: {statistics.mean(bl_grads_run):.4f} ± "
          f"{statistics.stdev(bl_grads_run):.4f}")
    print(f"gated    grad_norm: {statistics.mean(gt_grads_run):.4f} ± "
          f"{statistics.stdev(gt_grads_run):.4f}")
    amp = [g/b for b, g in zip(bl_grads_run, gt_grads_run)]
    print(f"grad amplification (gated/baseline): {statistics.mean(amp):.2f} ± "
          f"{statistics.stdev(amp):.2f}")

    bl_items_run = [d.n_train_items[:n_steps].mean() for d in bl]
    gt_items_run = [d.n_train_items[:n_steps].mean() for d in gt]
    print(f"baseline items/step: {statistics.mean(bl_items_run):.0f} ± "
          f"{statistics.stdev(bl_items_run):.0f}")
    print(f"gated    items/step: {statistics.mean(gt_items_run):.0f} ± "
          f"{statistics.stdev(gt_items_run):.0f}")

    bl_cuts = [d.n_groups_cut[:n_steps].sum() for d in bl]
    gt_cuts = [d.n_groups_cut[:n_steps].sum() for d in gt]
    print(f"gated groups cut per run (out of {n_steps*4}): "
          f"{statistics.mean(gt_cuts):.0f} ± {statistics.stdev(gt_cuts):.0f}")

    # Build per-step arrays (rows: seed, cols: step)
    bl_secs = np.stack([d.step_secs.values[:n_steps] for d in bl])
    gt_secs = np.stack([d.step_secs.values[:n_steps] for d in gt])
    bl_grad = np.stack([d.grad_norm.values[:n_steps] for d in bl])
    gt_grad = np.stack([d.grad_norm.values[:n_steps] for d in gt])
    bl_loss = np.stack([d.loss.values[:n_steps] for d in bl])
    gt_loss = np.stack([d.loss.values[:n_steps] for d in gt])
    bl_items = np.stack([d.n_train_items.values[:n_steps] for d in bl])
    gt_items = np.stack([d.n_train_items.values[:n_steps] for d in gt])

    steps = np.arange(n_steps)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    BLUE = "#1f77b4"
    ORANGE = "#ff7f0e"

    # 1. cumulative wall-clock (mean ± std across seeds)
    bl_cum = np.cumsum(bl_secs, axis=1)
    gt_cum = np.cumsum(gt_secs, axis=1)
    ax = axes[0, 0]
    ax.plot(steps, bl_cum.mean(0), marker="o", color=BLUE, label="baseline")
    ax.fill_between(steps, bl_cum.mean(0) - bl_cum.std(0),
                    bl_cum.mean(0) + bl_cum.std(0), color=BLUE, alpha=0.2)
    ax.plot(steps, gt_cum.mean(0), marker="o", color=ORANGE, label="gated")
    ax.fill_between(steps, gt_cum.mean(0) - gt_cum.std(0),
                    gt_cum.mean(0) + gt_cum.std(0), color=ORANGE, alpha=0.2)
    ax.set_xlabel("training step")
    ax.set_ylabel("cumulative wall-clock (s)")
    ax.set_title(f"Wall-clock (gated saves {save_mean:.1f}$\\pm${save_std:.1f}\\%)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    # 2. grad norm (5-step rolling mean for smoothness)
    win = 3
    def smooth(y):
        return pd.Series(y).rolling(win, center=True, min_periods=1).mean().values
    bl_g_smooth = np.stack([smooth(r) for r in bl_grad])
    gt_g_smooth = np.stack([smooth(r) for r in gt_grad])
    ax = axes[0, 1]
    ax.plot(steps, bl_g_smooth.mean(0), marker="o", color=BLUE, label="baseline")
    ax.fill_between(steps, bl_g_smooth.mean(0) - bl_g_smooth.std(0),
                    bl_g_smooth.mean(0) + bl_g_smooth.std(0),
                    color=BLUE, alpha=0.2)
    ax.plot(steps, gt_g_smooth.mean(0), marker="o", color=ORANGE, label="gated")
    ax.fill_between(steps, gt_g_smooth.mean(0) - gt_g_smooth.std(0),
                    gt_g_smooth.mean(0) + gt_g_smooth.std(0),
                    color=ORANGE, alpha=0.2)
    ax.set_xlabel("training step")
    ax.set_ylabel("gradient $L^2$ norm")
    grad_amp = statistics.mean(gt_grads_run) / statistics.mean(bl_grads_run)
    ax.set_title(f"Gradient norm (gated $\\approx${grad_amp*100:.0f}\\% of baseline)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)

    # 3. items per step
    ax = axes[1, 0]
    ax.bar(steps - 0.18, bl_items.mean(0), width=0.36, color=BLUE, alpha=0.85,
           label="baseline")
    ax.errorbar(steps - 0.18, bl_items.mean(0), yerr=bl_items.std(0),
                fmt="none", ecolor="black", alpha=0.4, capsize=2)
    ax.bar(steps + 0.18, gt_items.mean(0), width=0.36, color=ORANGE, alpha=0.85,
           label="gated")
    ax.errorbar(steps + 0.18, gt_items.mean(0), yerr=gt_items.std(0),
                fmt="none", ecolor="black", alpha=0.4, capsize=2)
    ax.set_xlabel("training step")
    ax.set_ylabel("train items per step (= traj $\\times$ action positions)")
    ax.set_title("Compute per step (gate drops cut groups)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3, axis="y")

    # 4. loss (smoothed)
    bl_l_smooth = np.stack([smooth(r) for r in bl_loss])
    gt_l_smooth = np.stack([smooth(r) for r in gt_loss])
    ax = axes[1, 1]
    ax.plot(steps, bl_l_smooth.mean(0), marker="o", color=BLUE, label="baseline")
    ax.fill_between(steps, bl_l_smooth.mean(0) - bl_l_smooth.std(0),
                    bl_l_smooth.mean(0) + bl_l_smooth.std(0),
                    color=BLUE, alpha=0.2)
    ax.plot(steps, gt_l_smooth.mean(0), marker="o", color=ORANGE, label="gated")
    ax.fill_between(steps, gt_l_smooth.mean(0) - gt_l_smooth.std(0),
                    gt_l_smooth.mean(0) + gt_l_smooth.std(0),
                    color=ORANGE, alpha=0.2)
    ax.axhline(0, ls=":", c="gray", alpha=0.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("policy-gradient loss")
    ax.set_title("Training loss (smoothed)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = FIG / "training_static_multiseed.png"
    safe_savefig(fig, out_path, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out_path}")

    # Also write a summary CSV for the paper
    summary = pd.DataFrame([
        {"arm": "baseline", "wall_mean": statistics.mean(bl_walls),
         "wall_std": statistics.stdev(bl_walls),
         "grad_mean": statistics.mean(bl_grads_run),
         "grad_std": statistics.stdev(bl_grads_run),
         "items_mean": statistics.mean(bl_items_run),
         "cuts_mean": 0},
        {"arm": "gated", "wall_mean": statistics.mean(gt_walls),
         "wall_std": statistics.stdev(gt_walls),
         "grad_mean": statistics.mean(gt_grads_run),
         "grad_std": statistics.stdev(gt_grads_run),
         "items_mean": statistics.mean(gt_items_run),
         "cuts_mean": statistics.mean(gt_cuts)},
    ])
    summary.to_csv(OUT / "training_static_multiseed_summary.csv", index=False)
    print(f"wrote {OUT / 'training_static_multiseed_summary.csv'}")


if __name__ == "__main__":
    main()
