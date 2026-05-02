"""
Final visualization: pick the WINNING gate and show it clearly.

Compare three candidate rules at the operating point that gives best savings
under prec ≥ 0.80, side-by-side at K=10 (early termination = max savings):

  W1: R1 — single threshold on div  (1 param, baseline)
  W2: R3 — div OR term-high         (2 params, our hybrid)
  W3: R2 — term U-shape only        (most-novel-feature, no div)

Plus: head-to-head numbers and a "cut diagram" panel.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.figutil import safe_savefig
ROLLOUTS = ROOT / "data" / "rollouts.jsonl"
METRICS = ROOT / "data" / "metrics.parquet"
FIG_DIR = ROOT / "results" / "figures"
MAX_STEPS = 30
N = 100


def load(K):
    groups = []
    with open(ROLLOUTS) as f:
        for line in f:
            groups.append(json.loads(line))

    def lab(g):
        rs = [t["total_reward"] for t in g["trajectories"]]
        n = len(rs); won = sum(1 for r in rs if r > 0)
        if won == 0: return "all_fail"
        if won == n: return "all_succeed"
        return "mixed"

    df = pd.DataFrame({
        "task_id": [g["task_id"] for g in groups],
        "label": [lab(g) for g in groups],
        "reward_var": [float(np.var([t["total_reward"] for t in g["trajectories"]]))
                       for g in groups],
    })
    df["zero_var"] = (df.reward_var == 0).astype(int)
    metrics = pd.read_parquet(METRICS)
    df = df.merge(metrics[["task_id", f"prefix_edit_distance_mean@{K}",
                           f"termination_fraction@{K}"]], on="task_id")
    df = df.rename(columns={f"prefix_edit_distance_mean@{K}": "div",
                            f"termination_fraction@{K}": "term"})
    return df


def metrics_of(cut, df, K):
    z = df.zero_var.values.astype(bool)
    cut = np.asarray(cut, bool)
    TP = int((cut & z).sum()); FP = int((cut & ~z).sum())
    FN = int((~cut & z).sum()); TN = int((~cut & ~z).sum())
    prec = TP / max(1, TP + FP); rec = TP / max(1, TP + FN)
    savings = TP * (MAX_STEPS - K) / (MAX_STEPS * N)
    return {"K": K, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "prec": prec, "rec": rec, "savings": savings, "cut_count": int(cut.sum())}


def main():
    K = 10
    df = load(K)
    div = df["div"].values; term = df["term"].values

    # Three rules, each tuned on the data (one shot — small dataset, descriptive)
    # W1: best R1 at prec >= 0.80 (from sweep): div < 0.12
    cut_W1 = div < 0.12
    # W2: best R3 at prec >= 0.80 (winner): div < 0.12 OR term >= 0.90
    cut_W2 = (div < 0.12) | (term >= 0.90)
    # W3: pure term U-shape: term <= 0 OR term >= 0.875 (handpicked simple)
    # tune at K=10
    cut_W3 = (term <= 0.0) | (term >= 0.875)

    rows = [
        ("W1: div<0.12 (1 param)", cut_W1),
        ("W2: div<0.12 OR term≥0.90 (2 params)", cut_W2),
        ("W3: term≤0 OR term≥0.875 (1 param U)", cut_W3),
    ]
    print(f"=== K={K} ===")
    print(f"{'rule':50s}  {'TP':>3s}/{'FP':>3s}  {'prec':>5s}  {'rec':>5s}  {'save%':>5s}")
    for name, c in rows:
        m = metrics_of(c, df, K)
        print(f"{name:50s}  {m['TP']:>3d}/{m['FP']:>3d}  {m['prec']:>5.2f}  "
              f"{m['rec']:>5.2f}  {m['savings']*100:>5.1f}%")

    # Also tune R2 at K=15, K=20 for comparison
    print()
    for K2 in (15, 20):
        d2 = load(K2)
        for thr_h in (0.875, 0.95):
            cut = (d2["term"].values <= 0) | (d2["term"].values >= thr_h)
            m = metrics_of(cut, d2, K2)
            print(f"W3 @K={K2}, term-extreme≥{thr_h}: TP={m['TP']}/{m['FP']}, "
                  f"prec={m['prec']:.2f}, rec={m['rec']:.2f}, save={m['savings']*100:.1f}%")

    # ===== Visualization =====
    plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)
    colors = {"all_succeed": "#2ca02c", "all_fail": "#1f77b4", "mixed": "#ff7f0e"}
    markers = {"all_succeed": "o", "all_fail": "o", "mixed": "D"}
    rng = np.random.default_rng(42)

    for ax, (name, cut) in zip(axes, rows):
        # shaded cut zones
        if "div<0.12" in name:
            ax.add_patch(Rectangle((-0.05, -0.05), 0.17, 1.15, color="red", alpha=0.10))
        if "term≥0.90" in name:
            ax.add_patch(Rectangle((-0.05, 0.90), 1.10, 0.20, color="red", alpha=0.10))
        if "term≤0 OR term≥0.875" in name:
            ax.add_patch(Rectangle((-0.05, -0.05), 1.10, 0.05, color="red", alpha=0.10))
            ax.add_patch(Rectangle((-0.05, 0.875), 1.10, 0.20, color="red", alpha=0.10))

        for L in ("all_succeed", "all_fail", "mixed"):
            sub = df[df.label == L].reset_index(drop=True)
            jx = rng.normal(0, 0.005, size=len(sub))
            jy = rng.normal(0, 0.012, size=len(sub))
            ax.scatter(sub["div"].values + jx, sub["term"].values + jy,
                       c=colors[L], marker=markers[L],
                       alpha=0.78, s=55, edgecolor="k", linewidth=0.4,
                       label=f"{L} (n={len(sub)})")

        m = metrics_of(cut, df, K)
        title_main = name.split(":", 1)[1].strip() if ":" in name else name
        ax.set_title(f"{name.split(':')[0]}: {title_main}\n"
                     f"TP={m['TP']}/39  FP={m['FP']}/61  prec={m['prec']:.2f}  "
                     f"rec={m['rec']:.2f}  save={m['savings']*100:.1f}%",
                     fontsize=10)
        ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel(f"prefix_edit_distance_mean@K={K}")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f"termination_fraction@K={K}")
    axes[0].legend(loc="center right", fontsize=8)
    plt.suptitle(f"Three gate variants at K={K} (act at step 10/30)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    safe_savefig(fig, FIG_DIR / "winner_gates_K10.png", dpi=140, bbox_inches="tight")
    print(f"\nwrote {FIG_DIR / 'winner_gates_K10.png'}")

    # Final summary plot: savings vs K for the three rules
    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    K_vals = (5, 10, 15, 20)
    rule_results = {"R1 (div only)": [], "R3 (div + term-high)": [],
                    "R2 (term U-shape)": []}
    for Kv in K_vals:
        d2 = load(Kv)
        # R1 best at prec>=0.80
        best = None
        for dl in np.linspace(0.02, 0.4, 39):
            c = d2["div"].values < dl
            m = metrics_of(c, d2, Kv)
            if m["prec"] >= 0.80 and (best is None or m["savings"] > best["savings"]):
                best = m
        rule_results["R1 (div only)"].append((best["savings"]*100, best["FP"]) if best else (0, 0))

        # R3 best
        best = None
        for dl in np.linspace(0.02, 0.3, 15):
            for th in np.linspace(0.5, 1.0, 11):
                c = (d2["div"].values < dl) | (d2["term"].values >= th)
                m = metrics_of(c, d2, Kv)
                if m["prec"] >= 0.80 and (best is None or m["savings"] > best["savings"]):
                    best = m
        rule_results["R3 (div + term-high)"].append((best["savings"]*100, best["FP"]) if best else (0, 0))

        # R2 best
        best = None
        for tl in np.linspace(0.0, 0.15, 4):
            for th in np.linspace(0.5, 1.0, 11):
                c = (d2["term"].values <= tl) | (d2["term"].values >= th)
                m = metrics_of(c, d2, Kv)
                if m["prec"] >= 0.80 and (best is None or m["savings"] > best["savings"]):
                    best = m
        rule_results["R2 (term U-shape)"].append((best["savings"]*100, best["FP"]) if best else (0, 0))

    for name, vals in rule_results.items():
        ys = [v[0] for v in vals]
        ax2.plot(K_vals, ys, marker="o", label=name, linewidth=2)
        for K_, (y, fp) in zip(K_vals, vals):
            ax2.annotate(f"FP={fp}", (K_, y), xytext=(5, 5),
                         textcoords="offset points", fontsize=7)
    ax2.set_xlabel("Gate timestep K")
    ax2.set_ylabel("Compute saved (%)")
    ax2.set_title("Best savings under prec ≥ 0.80, by rule × K")
    ax2.set_xticks(K_vals)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")
    plt.tight_layout()
    safe_savefig(fig2, FIG_DIR / "gate_savings_by_K.png", dpi=140)
    print(f"wrote {FIG_DIR / 'gate_savings_by_K.png'}")

    # Print the savings table
    print("\nSavings (%) under prec≥0.80:")
    print(f"{'rule':30s} " + " ".join(f"K={k:2d}" for k in K_vals))
    for name, vals in rule_results.items():
        print(f"{name:30s} " + " ".join(f"{v[0]:5.1f}" for v in vals))


if __name__ == "__main__":
    main()
