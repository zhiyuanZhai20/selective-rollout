"""
Dense K sweep: K ∈ {5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,25}.

The original metrics.parquet only has metrics @ K∈{5,10,15,20}. We need
to recompute (d_K, τ_K) at the missing K values from the raw rollouts.

For each K, sweep d_L ∈ [0.02, 0.30], τ_H ∈ [0.50, 1.00] under R3 OR-rule
and report:
  - best (precision ≥ 0.80) operating point
  - savings = TP * (T_max - K) / (N * T_max)

Outputs:
  results/K_sweep_dense.csv
  results/figures/K_sweep_dense.png
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

ROLLOUTS = ROOT / "data" / "rollouts.jsonl"
OUT = ROOT / "results"
FIG = OUT / "figures"
T_MAX = 30


def levenshtein_normed(a, b):
    """Length-normalised Levenshtein distance over action sequences."""
    if not a and not b: return 0.0
    if not a: return 1.0
    if not b: return 1.0
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost,
            )
    return dp[n][m] / max(n, m)


def compute_metrics(group, K):
    """Returns (d_K, tau_K) for this group at step K."""
    trajs = group["trajectories"]
    G = len(trajs)
    # Get first-K actions for each
    prefixes = []
    n_term = 0
    for t in trajs:
        steps = t["steps"]
        # truncate at terminated_at
        end = min(K, t.get("terminated_at", len(steps)) + 1, len(steps))
        prefix = [s["action"] for s in steps[:end]]
        prefixes.append(prefix)
        # is this trajectory terminated by step K?
        ta = t.get("terminated_at")
        if ta is not None and ta < K:
            n_term += 1
    # mean pairwise edit distance
    pairs = []
    for i in range(G):
        for j in range(i+1, G):
            pairs.append(levenshtein_normed(prefixes[i], prefixes[j]))
    d_K = float(np.mean(pairs))
    tau_K = n_term / G
    return d_K, tau_K


def main():
    groups = []
    with open(ROLLOUTS) as f:
        for line in f: groups.append(json.loads(line))
    n = len(groups)
    print(f"Loaded {n} groups")

    # Compute zv label
    zv_per_group = []
    for g in groups:
        rs = [t["total_reward"] for t in g["trajectories"]]
        zv_per_group.append(int(np.var(rs) == 0))
    zv = np.array(zv_per_group, bool)
    print(f"Zero-variance: {zv.sum()}/{n}")

    Ks = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25]
    rows = []
    for K in Ks:
        # compute (d_K, t_K) for all groups
        ds = np.zeros(n); ts = np.zeros(n)
        for i, g in enumerate(groups):
            d, t = compute_metrics(g, K)
            ds[i] = d; ts[i] = t

        # sweep R3
        best = None
        for d_L in np.linspace(0.02, 0.30, 29):
            for t_H in np.linspace(0.50, 1.00, 11):
                cut = (ds < d_L) | (ts >= t_H)
                tp = int((cut & zv).sum())
                fp = int((cut & ~zv).sum())
                fn = int((~cut & zv).sum())
                prec = tp / max(1, tp + fp)
                if prec < 0.80:
                    continue
                rec = tp / max(1, tp + fn)
                savings = tp * (T_MAX - K) / (n * T_MAX)
                if best is None or savings > best["savings"]:
                    best = {
                        "K": K, "d_L": d_L, "t_H": t_H,
                        "TP": tp, "FP": fp, "prec": prec,
                        "rec": rec, "savings": savings,
                    }
        if best is None:
            print(f"K={K}: no operating point clears precision floor 0.80")
            rows.append({"K": K, "savings": 0, "prec": 0, "rec": 0})
        else:
            print(f"K={K:2d}: TP={best['TP']:2d} FP={best['FP']:2d} "
                  f"prec={best['prec']:.3f} rec={best['rec']:.3f} "
                  f"savings={best['savings']*100:.2f}% "
                  f"d_L={best['d_L']:.2f} t_H={best['t_H']:.2f}")
            rows.append(best)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "K_sweep_dense.csv", index=False)

    # Plot: savings vs K
    fig, ax = plt.subplots(figsize=(7, 4.5))
    valid = df[df.savings > 0]
    ax.plot(valid.K, valid.savings * 100, marker="o", linewidth=2,
             color="#ff7f0e", markersize=8)
    # mark our chosen K=10
    ours = df[df.K == 10].iloc[0] if len(df[df.K == 10]) else None
    if ours is not None and ours.savings > 0:
        ax.scatter([10], [ours.savings * 100], color="red", s=200, zorder=5,
                    label="OURS (K=10)")
    ax.set_xlabel("K (gate evaluation step)")
    ax.set_ylabel("compute saved (%)")
    ax.set_title(f"Savings vs K under precision ≥ 0.80 floor (T_max={T_MAX})")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    safe_savefig(fig, FIG / "K_sweep_dense.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote {FIG / 'K_sweep_dense.png'}")


if __name__ == "__main__":
    main()
