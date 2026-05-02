"""Make a single-axis Figure 2: d_K=10 distribution by group label
with the chosen d_L=0.12 threshold marked, consistent with the
single-axis R1 gate in §3.2.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.figutil import safe_savefig

ROLLOUTS = ROOT / "data" / "rollouts.jsonl"
METRICS = ROOT / "data" / "metrics.parquet"
OUT = ROOT / "results" / "figures"
K = 10
DL = 0.12


def main():
    groups = []
    with open(ROLLOUTS) as f:
        for line in f: groups.append(json.loads(line))

    metrics = pd.read_parquet(METRICS)
    rows = []
    for g in groups:
        rs = [t["total_reward"] for t in g["trajectories"]]
        won = sum(1 for r in rs if r > 0)
        lost = sum(1 for r in rs if r == 0)
        if won == 0: label = "all_fail"
        elif lost == 0: label = "all_succeed"
        else: label = "mixed"
        rows.append({"task_id": g["task_id"], "label": label})
    df = pd.DataFrame(rows)
    df = df.merge(metrics[["task_id", f"prefix_edit_distance_mean@{K}"]], on="task_id")
    df = df.rename(columns={f"prefix_edit_distance_mean@{K}": "d_K"})

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 26)
    colors = {"all_succeed": "#2ca02c", "all_fail": "#d62728", "mixed": "#1f77b4"}
    labels = {"all_succeed": f"all-succeed (n={(df.label=='all_succeed').sum()})",
              "all_fail":    f"all-fail (n={(df.label=='all_fail').sum()})",
              "mixed":       f"mixed (n={(df.label=='mixed').sum()})"}

    for lbl in ["all_succeed", "all_fail", "mixed"]:
        sub = df[df.label == lbl]
        ax.hist(sub.d_K, bins=bins, alpha=0.65, color=colors[lbl],
                label=labels[lbl], edgecolor="black", linewidth=0.4)

    ax.axvline(DL, color="red", linestyle="--", linewidth=2,
               label=f"$d_L\\!=\\!{DL}$ (gate threshold)")
    ax.set_xlabel(f"$d_{{K={K}}}$  (prefix edit distance, in-group mean)")
    ax.set_ylabel("# groups")
    ax.set_title(f"Single-axis gate: $d_{{K={K}}}$ distribution stratified by group label\n"
                 f"groups left of red line are cut by $\\mathrm{{Cut}}_{{K, d_L}}$")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    # text annotations: TP / FP counts within cut region
    cut_mask = df.d_K < DL
    n_zv = ((df.label == "all_succeed") | (df.label == "all_fail")) & cut_mask
    n_mixed_in_cut = (df.label == "mixed") & cut_mask
    ax.text(0.02, ax.get_ylim()[1] * 0.85,
            f"In cut region ($d_K\\!<\\!{DL}$):\n"
            f"  TP (zero-var) = {int(n_zv.sum())}\n"
            f"  FP (mixed)    = {int(n_mixed_in_cut.sum())}\n"
            f"  precision     = {n_zv.sum()/cut_mask.sum():.2f}",
            fontsize=9, verticalalignment="top",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.95))

    plt.tight_layout()
    safe_savefig(fig, OUT / "single_axis_K10.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT / 'single_axis_K10.png'}")


if __name__ == "__main__":
    main()
