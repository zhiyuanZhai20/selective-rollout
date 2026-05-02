"""
R1-only K sweep: single-axis gate d_K < d_L sweep across K and d_L.

For each K ∈ {5, 6, ..., 25}, find the best d_L that gives precision ≥ 0.80
on the offline N=100 buffer, and report savings/recall.

Outputs:
  results/R1_K_sweep.csv
  results/figures/R1_K_sweep.png
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
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[n][m] / max(n, m)


def compute_d(group, K):
    trajs = group["trajectories"]
    G = len(trajs)
    prefixes = []
    for t in trajs:
        steps = t["steps"]
        end = min(K, t.get("terminated_at", len(steps)) + 1, len(steps))
        prefix = [s["action"] for s in steps[:end]]
        prefixes.append(prefix)
    pairs = []
    for i in range(G):
        for j in range(i+1, G):
            pairs.append(levenshtein_normed(prefixes[i], prefixes[j]))
    return float(np.mean(pairs))


def main():
    groups = []
    with open(ROLLOUTS) as f:
        for line in f: groups.append(json.loads(line))
    n = len(groups)
    zv = np.array([int(np.var([t["total_reward"] for t in g["trajectories"]]) == 0) for g in groups], bool)

    Ks = list(range(5, 26))
    rows = []
    for K in Ks:
        ds = np.array([compute_d(g, K) for g in groups])
        # Sweep d_L
        best = None
        for d_L in np.linspace(0.02, 0.40, 39):
            cut = ds < d_L
            tp = int((cut & zv).sum())
            fp = int((cut & ~zv).sum())
            fn = int((~cut & zv).sum())
            prec = tp / max(1, tp + fp)
            if prec < 0.80:
                continue
            rec = tp / max(1, tp + fn)
            savings = tp * (T_MAX - K) / (n * T_MAX)
            if best is None or savings > best["savings"]:
                best = {"K": K, "d_L": d_L, "TP": tp, "FP": fp,
                        "prec": prec, "rec": rec, "savings": savings}
        if best is None:
            print(f"K={K:2d}: no operating point >= 0.80 precision")
            rows.append({"K": K, "savings": 0, "prec": 0, "rec": 0,
                         "d_L": np.nan, "TP": 0, "FP": 0})
        else:
            print(f"K={K:2d}: TP={best['TP']:2d} FP={best['FP']} "
                  f"prec={best['prec']:.3f} rec={best['rec']:.3f} "
                  f"savings={best['savings']*100:5.2f}% d_L={best['d_L']:.2f}")
            rows.append(best)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "R1_K_sweep.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    valid = df[df.savings > 0]
    ax.plot(valid.K, valid.savings * 100, marker="o", linewidth=2,
             color="#1f77b4", markersize=8, label="R1 (single axis)")
    if 10 in df.K.values:
        sel = df[df.K == 10].iloc[0]
        if sel.savings > 0:
            ax.scatter([10], [sel.savings * 100], color="red", s=200, zorder=5,
                        label=f"chosen K=10")
    ax.set_xlabel("K (gate evaluation step)")
    ax.set_ylabel("compute saved (%)")
    ax.set_title(f"R1 single-axis gate: savings vs K under prec ≥ 0.80 (T_max={T_MAX})")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    safe_savefig(fig, FIG / "R1_K_sweep.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote {FIG / 'R1_K_sweep.png'}")


if __name__ == "__main__":
    main()
