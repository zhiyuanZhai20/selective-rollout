"""Compare runs/onpolicy_baseline vs runs/onpolicy_gated.

Reads:
    runs/<name>/train.jsonl  — per-iter training log
    runs/<name>/eval.jsonl   — periodic held-out eval
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
    if not p.exists(): return pd.DataFrame()
    rows = []
    with open(p) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarise(name: str, train: pd.DataFrame, ev: pd.DataFrame):
    if train.empty:
        print(f"\n[{name}] no training log")
        return None
    last = train.iloc[-1]
    print(f"\n[{name}] {len(train)} iters in {last.cumulative_wall_clock_sec:.0f}s")
    print(f"  rollout: {train.rollout_secs.sum():.0f}s "
          f"({train.rollout_secs.sum() / last.cumulative_wall_clock_sec * 100:.1f}% of total)")
    print(f"  train:   {train.train_secs.sum():.0f}s")
    print(f"  cuts:    {train.n_groups_cut.sum()}/{train.n_groups.sum()} "
          f"({train.n_groups_cut.sum() / max(train.n_groups.sum(), 1) * 100:.1f}%)")
    print(f"  zero-var: {train.n_zero_variance.sum()}/{train.n_groups.sum()}")
    print(f"  reward mean (first 5 / last 5): "
          f"{train.rewards_mean.head(5).mean():.3f} / {train.rewards_mean.tail(5).mean():.3f}")
    print(f"  grad_norm mean: {train.grad_norm.mean():.3f}")
    if not ev.empty:
        print(f"  eval success rate: {ev.success_rate.iloc[0]*100:.1f}% (iter 0) "
              f"-> {ev.success_rate.iloc[-1]*100:.1f}% (iter {ev.iter.iloc[-1]})")
        print(f"  eval trajectory: {[f'{r*100:.0f}%' for r in ev.success_rate]}")
    return {
        "name": name,
        "n_iters": len(train),
        "total_wall_sec": float(last.cumulative_wall_clock_sec),
        "rollout_sec": float(train.rollout_secs.sum()),
        "n_groups": int(train.n_groups.sum()),
        "n_groups_cut": int(train.n_groups_cut.sum()),
        "n_zero_var": int(train.n_zero_variance.sum()),
        "reward_first5": float(train.rewards_mean.head(5).mean()),
        "reward_last5": float(train.rewards_mean.tail(5).mean()),
        "grad_norm_mean": float(train.grad_norm.mean()),
        "eval_iter0":  float(ev.success_rate.iloc[0])  if not ev.empty else None,
        "eval_final":  float(ev.success_rate.iloc[-1]) if not ev.empty else None,
    }


def main():
    bl = load(ROOT / "runs" / "onpolicy_baseline" / "train.jsonl")
    bl_e = load(ROOT / "runs" / "onpolicy_baseline" / "eval.jsonl")
    gt = load(ROOT / "runs" / "onpolicy_gated" / "train.jsonl")
    gt_e = load(ROOT / "runs" / "onpolicy_gated" / "eval.jsonl")

    bl_s = summarise("baseline", bl, bl_e)
    gt_s = summarise("gated",    gt, gt_e)

    rows = [r for r in (bl_s, gt_s) if r is not None]
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "onpolicy_summary.csv", index=False)
        print(f"\nwrote {OUT / 'onpolicy_summary.csv'}")

    if bl.empty or gt.empty:
        return

    # Headline numbers
    delta_wall = (bl.cumulative_wall_clock_sec.iloc[-1] - gt.cumulative_wall_clock_sec.iloc[-1]) / bl.cumulative_wall_clock_sec.iloc[-1]
    print(f"\n=== HEADLINE ===")
    print(f"  Wall-clock saved by gated: {delta_wall*100:+.2f}%  "
          f"({bl.cumulative_wall_clock_sec.iloc[-1]:.0f}s -> {gt.cumulative_wall_clock_sec.iloc[-1]:.0f}s)")
    if not bl_e.empty and not gt_e.empty:
        print(f"  Final eval: baseline {bl_e.success_rate.iloc[-1]*100:.1f}% vs "
              f"gated {gt_e.success_rate.iloc[-1]*100:.1f}%")
        print(f"  Pre-eval:   baseline {bl_e.success_rate.iloc[0]*100:.1f}% vs "
              f"gated {gt_e.success_rate.iloc[0]*100:.1f}%")

    # 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. eval success rate over iterations
    if not bl_e.empty:
        axes[0, 0].plot(bl_e.iter, bl_e.success_rate * 100, marker="o",
                         color="#1f77b4", label="baseline", linewidth=2)
    if not gt_e.empty:
        axes[0, 0].plot(gt_e.iter, gt_e.success_rate * 100, marker="o",
                         color="#ff7f0e", label="gated", linewidth=2)
    axes[0, 0].set_xlabel("training iteration")
    axes[0, 0].set_ylabel("held-out success rate (%)")
    axes[0, 0].set_title("Held-out eval success rate (50 valid_seen tasks)")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # 2. cumulative wall-clock
    axes[0, 1].plot(bl.iter, bl.cumulative_wall_clock_sec / 60, marker=".",
                     color="#1f77b4", label="baseline", linewidth=2)
    axes[0, 1].plot(gt.iter, gt.cumulative_wall_clock_sec / 60, marker=".",
                     color="#ff7f0e", label="gated", linewidth=2)
    axes[0, 1].set_xlabel("training iteration")
    axes[0, 1].set_ylabel("cumulative wall-clock (min)")
    axes[0, 1].set_title(f"Wall-clock (gated saves {delta_wall*100:.1f}% end-to-end)")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # 3. training reward (per-iter mean)
    win = 5
    axes[1, 0].plot(bl.iter, bl.rewards_mean.rolling(win, min_periods=1).mean(),
                     color="#1f77b4", label=f"baseline ({win}-iter rolling mean)",
                     linewidth=2)
    axes[1, 0].plot(gt.iter, gt.rewards_mean.rolling(win, min_periods=1).mean(),
                     color="#ff7f0e", label=f"gated ({win}-iter rolling mean)",
                     linewidth=2)
    axes[1, 0].fill_between(bl.iter, bl.rewards_mean, color="#1f77b4", alpha=0.15)
    axes[1, 0].fill_between(gt.iter, gt.rewards_mean, color="#ff7f0e", alpha=0.15)
    axes[1, 0].set_xlabel("training iteration")
    axes[1, 0].set_ylabel("mean reward (per training-iter rollouts)")
    axes[1, 0].set_title("On-policy training reward")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # 4. gate cuts per iter (gated only)
    axes[1, 1].bar(gt.iter, gt.n_groups_cut, color="#ff7f0e", alpha=0.8,
                    label="gated cuts per iter")
    axes[1, 1].bar(bl.iter, bl.n_zero_variance, color="#1f77b4", alpha=0.4,
                    label="baseline zero-var groups (could-cut)")
    axes[1, 1].set_xlabel("training iteration")
    axes[1, 1].set_ylabel("# groups (out of {})".format(int(bl.n_groups.iloc[0]) if not bl.empty else 10))
    axes[1, 1].set_title("Gate firings vs zero-variance ground truth")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    safe_savefig(fig, FIG / "onpolicy_compare.png", dpi=140)
    plt.close(fig)
    print(f"wrote {FIG / 'onpolicy_compare.png'}")


if __name__ == "__main__":
    main()
