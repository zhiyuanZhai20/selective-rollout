"""4-panel compare figure averaged across 4 R1 seeds (7, 13, 23, 42).

Replaces the noisy single-seed (seed=7) plot. Curves show mean across
seeds; shaded band = mean +/- 1 std across the 4 seeds.
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

SEEDS = [7, 13, 23, 42]
BL_DIRS = {7: "onpolicy_baseline_s7", 13: "onpolicy_baseline_s13",
           23: "onpolicy_baseline_s23", 42: "onpolicy_baseline"}
GT_DIRS = {7: "onpolicy_gated_s7", 13: "onpolicy_gated_s13",
           23: "onpolicy_gated_s23", 42: "onpolicy_gated_s42_R1"}


def load_train(d):
    rows = [json.loads(l) for l in open(ROOT / "runs" / d / "train.jsonl")]
    return pd.DataFrame(rows)


def load_eval(d):
    rows = [json.loads(l) for l in open(ROOT / "runs" / d / "eval.jsonl")]
    return pd.DataFrame(rows)


def stack(dfs, col):
    """Align on iter and return mean, std arrays of shape (n_iters,)."""
    arrs = [df[col].values for df in dfs]
    n = min(len(a) for a in arrs)
    arrs = np.stack([a[:n] for a in arrs])
    return arrs.mean(axis=0), arrs.std(axis=0), arrs[0].shape[0]


def main():
    bl_train = [load_train(BL_DIRS[s]) for s in SEEDS]
    gt_train = [load_train(GT_DIRS[s]) for s in SEEDS]
    bl_eval = [load_eval(BL_DIRS[s]) for s in SEEDS]
    gt_eval = [load_eval(GT_DIRS[s]) for s in SEEDS]

    saved_pcts = []
    for b, g in zip(bl_train, gt_train):
        saved_pcts.append(
            (b.cumulative_wall_clock_sec.iloc[-1] -
             g.cumulative_wall_clock_sec.iloc[-1]) /
            b.cumulative_wall_clock_sec.iloc[-1] * 100)
    saved_mean, saved_std = np.mean(saved_pcts), np.std(saved_pcts, ddof=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. held-out eval (smoothed with a 3-iter centered rolling mean to
    # reduce jitter from the small N=50 eval set; iter-0 and iter-60
    # endpoints are anchored to the raw value so the start and end
    # points of the curve match the table).
    def smooth(y):
        s = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values.copy()
        s[0] = y[0]
        s[-1] = y[-1]
        return s
    bl_iter = bl_eval[0].iter.values
    bl_succ = np.stack([df.success_rate.values * 100 for df in bl_eval])
    gt_succ = np.stack([df.success_rate.values * 100 for df in gt_eval])
    bl_mean_s = smooth(bl_succ.mean(0))
    gt_mean_s = smooth(gt_succ.mean(0))
    bl_std_s = smooth(bl_succ.std(0))
    gt_std_s = smooth(gt_succ.std(0))
    ax = axes[0, 0]
    ax.plot(bl_iter, bl_mean_s, marker="o", color="#1f77b4",
            label="baseline", linewidth=2, markersize=7)
    ax.fill_between(bl_iter, bl_mean_s - bl_std_s,
                    bl_mean_s + bl_std_s, color="#1f77b4", alpha=0.20)
    ax.plot(bl_iter, gt_mean_s, marker="o", color="#ff7f0e",
            label="gated", linewidth=2, markersize=7)
    ax.fill_between(bl_iter, gt_mean_s - gt_std_s,
                    gt_mean_s + gt_std_s, color="#ff7f0e", alpha=0.20)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("held-out success rate (%)")
    ax.set_title(f"Held-out eval ({len(SEEDS)}-seed mean $\\pm$ std, $n\\!=\\!{len(SEEDS)}$)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    # 2. cumulative wall-clock
    iters = bl_train[0].iter.values
    bl_wall = np.stack([df.cumulative_wall_clock_sec.values / 60 for df in bl_train])
    gt_wall = np.stack([df.cumulative_wall_clock_sec.values / 60 for df in gt_train])
    ax = axes[0, 1]
    ax.plot(iters, bl_wall.mean(0), color="#1f77b4", label="baseline", linewidth=2)
    ax.fill_between(iters, bl_wall.mean(0) - bl_wall.std(0),
                    bl_wall.mean(0) + bl_wall.std(0), color="#1f77b4", alpha=0.20)
    ax.plot(iters, gt_wall.mean(0), color="#ff7f0e", label="gated", linewidth=2)
    ax.fill_between(iters, gt_wall.mean(0) - gt_wall.std(0),
                    gt_wall.mean(0) + gt_wall.std(0), color="#ff7f0e", alpha=0.20)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("cumulative wall-clock (min)")
    ax.set_title(f"Wall-clock (gated saves {saved_mean:.1f}$\\pm${saved_std:.1f}\\% end-to-end)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    # 3. training reward over uncut groups only (gated cut groups removed
    # from the mean because the gate truncated them at step K=10 before most
    # rewards could arrive, biasing the recorded reward downward).
    win = 5
    bl_rew = np.stack([df.rewards_mean.values for df in bl_train])
    # gated: rewards_mean is over all 10 groups; assume the n_cut cut groups
    # contribute reward ~0 (truncated at step 10 before completion in ALFWorld),
    # back out the mean over the (10 - n_cut) uncut groups.
    gt_rew_uncut = []
    for df in gt_train:
        rm = df.rewards_mean.values
        nc = df.n_groups_cut.values
        ng = df.n_groups.values
        n_uncut = ng - nc
        # avoid div-by-zero (no iter has all 10 cut, but be safe):
        gt_rew_uncut.append(np.where(n_uncut > 0, rm * ng / np.maximum(n_uncut, 1), 0.0))
    gt_rew_u = np.stack(gt_rew_uncut)

    bl_smooth = pd.Series(bl_rew.mean(0)).rolling(win, min_periods=1).mean()
    gt_smooth = pd.Series(gt_rew_u.mean(0)).rolling(win, min_periods=1).mean()
    bl_band = pd.Series(bl_rew.std(0)).rolling(win, min_periods=1).mean()
    gt_band = pd.Series(gt_rew_u.std(0)).rolling(win, min_periods=1).mean()
    ax = axes[1, 0]
    ax.plot(iters, bl_smooth, color="#1f77b4", label="baseline", linewidth=2)
    ax.fill_between(iters, bl_smooth - bl_band, bl_smooth + bl_band,
                    color="#1f77b4", alpha=0.20)
    ax.plot(iters, gt_smooth, color="#ff7f0e",
            label="gated (uncut groups only)", linewidth=2)
    ax.fill_between(iters, gt_smooth - gt_band, gt_smooth + gt_band,
                    color="#ff7f0e", alpha=0.20)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("mean training reward")
    ax.set_title("On-policy training reward, 5-iter rolling")
    ax.set_ylim(0, 0.8)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    # 4. gate firings vs baseline zero-variance (mean across seeds)
    bl_zv = np.stack([df.n_zero_variance.values for df in bl_train])
    gt_cut = np.stack([df.n_groups_cut.values for df in gt_train])
    ax = axes[1, 1]
    ax.bar(iters, bl_zv.mean(0), color="#1f77b4", alpha=0.45,
           label="baseline zero-var (ground truth)")
    ax.bar(iters, gt_cut.mean(0), color="#ff7f0e", alpha=0.85,
           label="gated cuts per iter")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("# groups (out of 10), seed-mean")
    ax.set_title("Gate firings vs zero-variance baseline groups")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = FIG / "onpolicy_compare_multiseed.png"
    safe_savefig(fig, out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"saving: mean={saved_mean:.2f}%, std={saved_std:.2f}%, per-seed={saved_pcts}")


if __name__ == "__main__":
    main()
