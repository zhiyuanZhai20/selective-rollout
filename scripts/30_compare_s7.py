"""Generate the headline 4-panel compare figure from seed=7 R1 data."""
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
    with open(p) as f:
        for line in f: rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    bl = load(ROOT / "runs" / "onpolicy_baseline_s7" / "train.jsonl")
    gt = load(ROOT / "runs" / "onpolicy_gated_s7" / "train.jsonl")
    bl_e = load(ROOT / "runs" / "onpolicy_baseline_s7" / "eval.jsonl")
    gt_e = load(ROOT / "runs" / "onpolicy_gated_s7" / "eval.jsonl")

    delta_wall = (bl.cumulative_wall_clock_sec.iloc[-1] -
                  gt.cumulative_wall_clock_sec.iloc[-1]) / bl.cumulative_wall_clock_sec.iloc[-1]
    print(f"wall saved: {delta_wall*100:+.2f}%")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # eval
    axes[0, 0].plot(bl_e.iter, bl_e.success_rate * 100, marker="o",
                     color="#1f77b4", label="baseline (R1)", linewidth=2, markersize=8)
    axes[0, 0].plot(gt_e.iter, gt_e.success_rate * 100, marker="o",
                     color="#ff7f0e", label="gated R1 ($d_K\!<\!0.12$)",
                     linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("training iteration")
    axes[0, 0].set_ylabel("held-out success rate (%)")
    axes[0, 0].set_title("Held-out eval (50 valid_seen tasks, seed=7)")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # cumulative wall-clock
    axes[0, 1].plot(bl.iter, bl.cumulative_wall_clock_sec / 60, marker=".",
                     color="#1f77b4", label="baseline", linewidth=2)
    axes[0, 1].plot(gt.iter, gt.cumulative_wall_clock_sec / 60, marker=".",
                     color="#ff7f0e", label="gated", linewidth=2)
    axes[0, 1].set_xlabel("training iteration")
    axes[0, 1].set_ylabel("cumulative wall-clock (min)")
    axes[0, 1].set_title(f"Wall-clock (gated saves {delta_wall*100:.1f}% end-to-end)")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # training reward (rolling 5)
    win = 5
    axes[1, 0].plot(bl.iter, bl.rewards_mean.rolling(win, min_periods=1).mean(),
                     color="#1f77b4", label=f"baseline ({win}-iter rolling)",
                     linewidth=2)
    axes[1, 0].plot(gt.iter, gt.rewards_mean.rolling(win, min_periods=1).mean(),
                     color="#ff7f0e", label=f"gated ({win}-iter rolling)",
                     linewidth=2)
    axes[1, 0].fill_between(bl.iter, bl.rewards_mean, color="#1f77b4", alpha=0.15)
    axes[1, 0].fill_between(gt.iter, gt.rewards_mean, color="#ff7f0e", alpha=0.15)
    axes[1, 0].set_xlabel("training iteration")
    axes[1, 0].set_ylabel("mean reward")
    axes[1, 0].set_title("On-policy training reward")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # gate firings vs zv
    axes[1, 1].bar(gt.iter, gt.n_groups_cut, color="#ff7f0e", alpha=0.8,
                    label="gated cuts per iter")
    axes[1, 1].bar(bl.iter, bl.n_zero_variance, color="#1f77b4", alpha=0.4,
                    label="baseline zero-var (ground truth)")
    axes[1, 1].set_xlabel("training iteration")
    axes[1, 1].set_ylabel("# groups (out of 10)")
    axes[1, 1].set_title("Gate firings vs zero-variance ground truth")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    safe_savefig(fig, FIG / "onpolicy_compare_s7.png", dpi=140)
    plt.close(fig)
    print(f"wrote {FIG / 'onpolicy_compare_s7.png'}")


if __name__ == "__main__":
    main()
