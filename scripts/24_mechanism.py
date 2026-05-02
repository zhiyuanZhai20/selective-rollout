"""
Mechanism deep-dive on existing on-policy training logs.

Aggregates over the 4 seeds {7, 13, 23, 42} (the unsuffixed
runs/onpolicy_{baseline,gated}/ dirs are seed=42).

Computes:
  1. zv-rate evolution over 60 iters (5-iter rolling, mean ± std across seeds)
  2. gradient-norm trajectory + mean during early/late training
  3. Cut rate vs zv-rate per iter (gated only) — does the gate
     keep up as the policy converges?
  4. Per-iter dilution rate breakdown

Outputs:
  results/mechanism_baseline_per_iter.csv  (per-seed per-iter)
  results/mechanism_gated_per_iter.csv     (per-seed per-iter)
  results/figures/mechanism_evolution.png
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

SEEDS = ["42", "7", "13", "23"]
G = 8
ITEMS_PER_ITER = 80  # 10 prompts * G


def seed_dir(arm: str, seed: str) -> Path:
    """For the main R1 single-axis result, seed 42 lives in
    runs/onpolicy_gated_s42_R1/ (the unsuffixed runs/onpolicy_gated/
    is a legacy R3 OR-rule run). Other seeds use the standard
    runs/onpolicy_{arm}_s{seed}/ layout. Baselines are gate-agnostic."""
    if arm == "gated" and seed == "42":
        return ROOT / "runs" / "onpolicy_gated_s42_R1"
    suffix = "" if seed == "42" else f"_s{seed}"
    return ROOT / "runs" / f"onpolicy_{arm}{suffix}"


def load(p):
    rows = []
    with open(p) as f:
        for line in f: rows.append(json.loads(line))
    return pd.DataFrame(rows)


def compute_per_iter(arm: str, seed: str) -> pd.DataFrame:
    df = load(seed_dir(arm, seed) / "train.jsonl")
    df["seed"] = seed
    df["arm"] = arm
    if arm == "baseline":
        df["zv_rate"] = df.n_zero_variance / df.n_groups
        df["cut_rate"] = 0.0
        df["recall_per_iter"] = np.nan
        df["dilution_pct"] = df.n_zero_variance * G / ITEMS_PER_ITER * 100
    else:
        # gated: cut groups stop at K=10 with reward 0, so they look
        # zv at the train-log level; we report the zv rate among the
        # groups that the gate let through (uncut).
        df["zv_rate"] = (
            (df.n_zero_variance - df.n_groups_cut).clip(lower=0)
            / (df.n_groups - df.n_groups_cut).clip(lower=1)
        )
        df["cut_rate"] = df.n_groups_cut / df.n_groups
        df["recall_per_iter"] = df.apply(
            lambda r: r.n_groups_cut / max(1, r.n_zero_variance), axis=1
        )
        # dilution after cuts: surviving zv groups in the train batch
        df["dilution_pct"] = (
            (df.n_zero_variance - df.n_groups_cut).clip(lower=0) * G
            / df.n_train_items * 100
        )
    return df


def aggregate(arm: str) -> pd.DataFrame:
    """Returns iter-indexed df with mean/std across seeds for each metric."""
    seed_dfs = [compute_per_iter(arm, s) for s in SEEDS]
    all_df = pd.concat(seed_dfs, ignore_index=True)
    cols = ["zv_rate", "cut_rate", "recall_per_iter", "dilution_pct", "grad_norm"]
    grouped = all_df.groupby("iter")[cols].agg(["mean", "std"]).reset_index()
    grouped.columns = ["iter"] + [f"{c}_{stat}" for c in cols for stat in ("mean", "std")]
    # 5-iter rolling on the mean
    win = 5
    for c in cols:
        grouped[f"{c}_mean_roll5"] = grouped[f"{c}_mean"].rolling(win, min_periods=1).mean()
    return grouped, all_df


def main():
    bl, bl_all = aggregate("baseline")
    gt, gt_all = aggregate("gated")

    # Save per-seed per-iter for downstream uses
    bl_all.to_csv(OUT / "mechanism_baseline_per_iter.csv", index=False)
    gt_all.to_csv(OUT / "mechanism_gated_per_iter.csv", index=False)

    # Plots --- 4 panels, mean line + ±1 std band across seeds
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def band(ax, x, mean, std, color, label):
        ax.plot(x, mean, color=color, linewidth=2, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)

    # 1. ZV rate evolution
    ax = axes[0, 0]
    band(ax, bl.iter, bl.zv_rate_mean_roll5 * 100, bl.zv_rate_std * 100,
         "#1f77b4", f"baseline (mean={bl_all.zv_rate.mean()*100:.0f}%)")
    band(ax, gt.iter, gt.zv_rate_mean_roll5 * 100, gt.zv_rate_std * 100,
         "#ff7f0e", f"gated, uncut only (mean={gt_all.zv_rate.mean()*100:.0f}%)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("zero-variance rate (%)")
    ax.set_title("Zero-variance rate over training (n=4 seeds)")
    ax.legend(); ax.grid(alpha=0.3)

    # 2. Gate cut rate vs zv rate (gated only)
    ax = axes[0, 1]
    band(ax, gt.iter, gt.zv_rate_mean_roll5 * 100, gt.zv_rate_std * 100,
         "#1f77b4", "zv rate (ground truth)")
    band(ax, gt.iter, gt.cut_rate_mean_roll5 * 100, gt.cut_rate_std * 100,
         "#ff7f0e", "cut rate (gate fired)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("rate (%)")
    ax.set_title("Gate keeps up with zv ground truth")
    ax.legend(); ax.grid(alpha=0.3)

    # 3. Gradient L2 norm trajectory
    ax = axes[1, 0]
    band(ax, bl.iter, bl.grad_norm_mean_roll5, bl.grad_norm_std,
         "#1f77b4", f"baseline (mean={bl_all.grad_norm.mean():.3f})")
    band(ax, gt.iter, gt.grad_norm_mean_roll5, gt.grad_norm_std,
         "#ff7f0e", f"gated (mean={gt_all.grad_norm.mean():.3f})")
    ratio = gt_all.grad_norm.mean() / bl_all.grad_norm.mean()
    ax.set_xlabel("iteration")
    ax.set_ylabel("gradient L2-norm")
    ax.set_title(f"Gradient amplification: gated/baseline = {ratio:.2f}×")
    ax.legend(); ax.grid(alpha=0.3)

    # 4. Dilution rate
    ax = axes[1, 1]
    band(ax, bl.iter, bl.dilution_pct_mean_roll5, bl.dilution_pct_std,
         "#1f77b4", f"baseline (mean={bl_all.dilution_pct.mean():.1f}%)")
    band(ax, gt.iter, gt.dilution_pct_mean_roll5, gt.dilution_pct_std,
         "#ff7f0e", f"gated (mean={gt_all.dilution_pct.mean():.1f}%)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("zero-advantage items in batch (%)")
    delta_pp = bl_all.dilution_pct.mean() - gt_all.dilution_pct.mean()
    ax.set_title(f"Dilution: gate strips {delta_pp:.1f} pp on average")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    safe_savefig(fig, FIG / "mechanism_evolution.png", dpi=140)
    plt.close(fig)
    print(f"wrote {FIG / 'mechanism_evolution.png'}")

    # Headline numbers (4-seed averages)
    print("\n=== Mechanism summary (4-seed mean, ± std across seeds) ===")
    bl_zv = bl_all.zv_rate * 100
    gt_zv = gt_all.zv_rate * 100
    bl_dil = bl_all.dilution_pct
    gt_dil = gt_all.dilution_pct
    print(f"Baseline zv rate: {bl_zv.mean():.1f}% (per-seed mean ± std: "
          f"{bl_all.groupby('seed').zv_rate.mean().mean()*100:.1f} ± "
          f"{bl_all.groupby('seed').zv_rate.mean().std()*100:.1f})")
    print(f"Gated zv (uncut): {gt_zv.mean():.1f}% (per-seed mean ± std: "
          f"{gt_all.groupby('seed').zv_rate.mean().mean()*100:.1f} ± "
          f"{gt_all.groupby('seed').zv_rate.mean().std()*100:.1f})")
    print(f"Gated cut rate:   {(gt_all.cut_rate*100).mean():.1f}%")
    print(f"Per-iter recall:  {(gt_all.recall_per_iter*100).mean():.1f}%")
    print(f"Baseline dilution: {bl_dil.mean():.1f}%")
    print(f"Gated dilution:    {gt_dil.mean():.1f}%")
    print(f"Reduction:         {bl_dil.mean() - gt_dil.mean():.1f} pp")
    pred = (1 - gt_dil.mean()/100) / (1 - bl_dil.mean()/100)
    print(f"Predicted grad amplification: {pred:.3f}×")
    measured = gt_all.grad_norm.mean() / bl_all.grad_norm.mean()
    print(f"Measured grad amplification:  {measured:.3f}×")
    # Per-seed grad ratio (matches the body's std)
    per_seed_ratio = (gt_all.groupby("seed").grad_norm.mean()
                      / bl_all.groupby("seed").grad_norm.mean())
    print(f"Per-seed measured: {per_seed_ratio.to_dict()}")
    print(f"   mean ± std across seeds: {per_seed_ratio.mean():.3f} ± "
          f"{per_seed_ratio.std():.3f}")


if __name__ == "__main__":
    main()
