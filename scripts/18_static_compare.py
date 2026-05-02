"""Compare runs/static_baseline vs runs/static_gated.

Produces:
  results/training_static_summary.csv
  results/figures/training_static.png
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


def main():
    bl = load(ROOT / "runs" / "static_baseline" / "train.jsonl")
    gt = load(ROOT / "runs" / "static_gated" / "train.jsonl")
    assert len(bl) > 0 and len(gt) > 0, "missing run logs"

    print(f"baseline: {len(bl)} steps, total {bl.step_secs.sum():.1f}s")
    print(f"  cuts: {bl.n_groups_cut.sum()} (should be 0; gate off)")
    print(f"  zero-var groups: {bl.n_zero_variance.sum()}")
    print(f"  items per step: {bl.n_train_items.mean():.1f} ± {bl.n_train_items.std():.1f}")
    print(f"  step_secs: {bl.step_secs.mean():.1f} ± {bl.step_secs.std():.1f}")
    print(f"  loss first 5 / last 5: {bl.loss.head(5).mean():+.4f} / {bl.loss.tail(5).mean():+.4f}")
    print(f"  grad_norm mean: {bl.grad_norm.mean():.3f}")

    print(f"\ngated: {len(gt)} steps, total {gt.step_secs.sum():.1f}s")
    print(f"  cuts: {gt.n_groups_cut.sum()} ({gt.n_groups_cut.sum()/(len(gt)*4)*100:.1f}% of groups cut)")
    print(f"  zero-var groups: {gt.n_zero_variance.sum()}")
    print(f"  items per step: {gt.n_train_items.mean():.1f} ± {gt.n_train_items.std():.1f}")
    print(f"  step_secs: {gt.step_secs.mean():.1f} ± {gt.step_secs.std():.1f}")
    print(f"  loss first 5 / last 5: {gt.loss.head(5).mean():+.4f} / {gt.loss.tail(5).mean():+.4f}")
    print(f"  grad_norm mean: {gt.grad_norm.mean():.3f}")

    delta_total = (bl.step_secs.sum() - gt.step_secs.sum()) / bl.step_secs.sum()
    delta_per_step = (bl.step_secs.mean() - gt.step_secs.mean()) / bl.step_secs.mean()
    delta_items = (bl.n_train_items.sum() - gt.n_train_items.sum()) / bl.n_train_items.sum()
    print(f"\n=== headline numbers ===")
    print(f"  Δ total wall-clock: {delta_total*100:+.2f}% (gated vs baseline)")
    print(f"  Δ mean step time:   {delta_per_step*100:+.2f}%")
    print(f"  Δ total train items: {delta_items*100:+.2f}%")
    print(f"  gated cut rate: {gt.n_groups_cut.sum()/(len(gt)*4)*100:.1f}%  "
          f"(vs offline-predicted 23/100 = 23.0%)")
    print(f"  baseline grad_norm mean: {bl.grad_norm.mean():.3f}")
    print(f"  gated    grad_norm mean: {gt.grad_norm.mean():.3f}  "
          f"({gt.grad_norm.mean() / bl.grad_norm.mean() * 100:.1f}% of baseline)")

    rows = [
        {"run": "baseline", "n_steps": len(bl),
         "total_step_secs": float(bl.step_secs.sum()),
         "mean_step_secs": float(bl.step_secs.mean()),
         "total_items": int(bl.n_train_items.sum()),
         "mean_items_per_step": float(bl.n_train_items.mean()),
         "n_groups_cut": int(bl.n_groups_cut.sum()),
         "loss_first5": float(bl.loss.head(5).mean()),
         "loss_last5":  float(bl.loss.tail(5).mean()),
         "grad_norm_mean": float(bl.grad_norm.mean())},
        {"run": "gated", "n_steps": len(gt),
         "total_step_secs": float(gt.step_secs.sum()),
         "mean_step_secs": float(gt.step_secs.mean()),
         "total_items": int(gt.n_train_items.sum()),
         "mean_items_per_step": float(gt.n_train_items.mean()),
         "n_groups_cut": int(gt.n_groups_cut.sum()),
         "loss_first5": float(gt.loss.head(5).mean()),
         "loss_last5":  float(gt.loss.tail(5).mean()),
         "grad_norm_mean": float(gt.grad_norm.mean())},
    ]
    pd.DataFrame(rows).to_csv(OUT / "training_static_summary.csv", index=False)
    print(f"\nwrote {OUT / 'training_static_summary.csv'}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    # 1. cumulative wall-clock
    axes[0, 0].plot(bl.step, np.cumsum(bl.step_secs),
                     marker="o", color="#1f77b4", label="baseline")
    axes[0, 0].plot(gt.step, np.cumsum(gt.step_secs),
                     marker="o", color="#ff7f0e", label="gated")
    axes[0, 0].set_xlabel("training step")
    axes[0, 0].set_ylabel("cumulative wall-clock (s)")
    axes[0, 0].set_title(f"Wall-clock (gated saves {delta_total*100:.1f}%)")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # 2. grad norm
    axes[0, 1].plot(bl.step, bl.grad_norm, marker="o", color="#1f77b4", label="baseline")
    axes[0, 1].plot(gt.step, gt.grad_norm, marker="o", color="#ff7f0e", label="gated")
    axes[0, 1].set_xlabel("training step")
    axes[0, 1].set_ylabel("gradient L2 norm")
    axes[0, 1].set_title("Gradient norm (gate preserves signal)")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # 3. items per step (compute proxy)
    axes[1, 0].bar(bl.step - 0.18, bl.n_train_items, width=0.36,
                    color="#1f77b4", label="baseline")
    axes[1, 0].bar(gt.step + 0.18, gt.n_train_items, width=0.36,
                    color="#ff7f0e", label="gated")
    axes[1, 0].set_xlabel("training step")
    axes[1, 0].set_ylabel("train items (= traj×kept-step)")
    axes[1, 0].set_title("Compute per step (gate cuts items)")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3, axis="y")

    # 4. loss
    axes[1, 1].plot(bl.step, bl.loss, marker="o", color="#1f77b4", label="baseline")
    axes[1, 1].plot(gt.step, gt.loss, marker="o", color="#ff7f0e", label="gated")
    axes[1, 1].axhline(0, ls=":", c="gray", alpha=0.5)
    axes[1, 1].set_xlabel("training step")
    axes[1, 1].set_ylabel("loss (PG, lower = higher reward)")
    axes[1, 1].set_title("Training loss")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    safe_savefig(fig, FIG / "training_static.png", dpi=140)
    plt.close(fig)
    print(f"wrote {FIG / 'training_static.png'}")


if __name__ == "__main__":
    main()
