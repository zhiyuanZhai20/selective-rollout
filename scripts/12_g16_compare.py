"""
Phase B follow-up — compare G=8 (data/rollouts.jsonl) vs G=16
(data/rollouts_g16.jsonl).

Reports:
  * Group composition: all_fail / mixed / all_succeed share at each G
  * Spearman ρ and AUROC at the winner cell (prefix_edit_distance_mean @ K=15)
  * Operating-point recall and savings of the published gate (K=10, d=0.12, t=0.90)
    re-evaluated on G=16

Outputs:
  results/g16_compare.csv
  results/figures/g16_vs_g8_panel.png
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.figutil import safe_savefig

OUT_DIR = ROOT / "results"
FIG_DIR = OUT_DIR / "figures"
MAX_STEPS = 30


def load_groups(path: Path):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def label_for(g):
    rs = [t["total_reward"] for t in g["trajectories"]]
    n = len(rs); won = sum(1 for r in rs if r > 0)
    if won == 0: return "all_fail"
    if won == n: return "all_succeed"
    return "mixed"


def auroc(scores, labels):
    """labels: 1 for positive class, 0 for negative."""
    s = np.asarray(scores, float); l = np.asarray(labels, int)
    pos = s[l == 1]; neg = s[l == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n = 0; tot = 0
    for p in pos:
        n += int((neg < p).sum()) + 0.5 * int((neg == p).sum())
        tot += len(neg)
    return n / tot


def eval_gate(div, term, zv, K, dl, th, n):
    cut = (div < dl) | (term >= th)
    TP = int((cut & zv).sum()); FP = int((cut & ~zv).sum())
    FN = int((~cut & zv).sum()); TN = int((~cut & ~zv).sum())
    prec = TP / max(1, TP + FP); rec = TP / max(1, TP + FN)
    savings = TP * (MAX_STEPS - K) / (MAX_STEPS * n)
    return TP, FP, prec, rec, savings


def summarise(groups, metrics, label_str):
    df = pd.DataFrame({
        "task_id": [g["task_id"] for g in groups],
        "label": [label_for(g) for g in groups],
        "reward_var": [float(np.var([t["total_reward"] for t in g["trajectories"]]))
                       for g in groups],
    })
    df["zero_var"] = (df.reward_var == 0).astype(int)
    df = df.merge(metrics, on="task_id")
    n = len(df)
    composition = df.label.value_counts().to_dict()
    print(f"\n=== {label_str} (N={n}, G inferred from data) ===")
    print(f"  composition: all_fail={composition.get('all_fail',0)}  "
          f"mixed={composition.get('mixed',0)}  "
          f"all_succeed={composition.get('all_succeed',0)}  "
          f"zero_var={int(df.zero_var.sum())} ({df.zero_var.mean()*100:.0f}%)")

    rows = []
    for K in (5, 10, 15, 20):
        d_col = f"prefix_edit_distance_mean@{K}"
        t_col = f"termination_fraction@{K}"
        d = df[d_col].values; t = df[t_col].values
        zv = df.zero_var.values.astype(bool)
        rho, p = spearmanr(d, df.reward_var.values)
        au = auroc(d, 1 - zv)  # AUROC for non-zero-variance class
        rows.append({"K": K, "rho": rho, "p": p, "auroc": au})
        print(f"  K={K:2d}: ρ(div, reward_var)={rho:+.3f} (p={p:.1e})  "
              f"AUROC(div→non-zv)={au:.3f}")

    # Apply published gate (K=10, d=0.12, t=0.90)
    K, dl, th = 10, 0.12, 0.90
    d = df[f"prefix_edit_distance_mean@{K}"].values
    t = df[f"termination_fraction@{K}"].values
    zv = df.zero_var.values.astype(bool)
    TP, FP, prec, rec, savings = eval_gate(d, t, zv, K, dl, th, n)
    print(f"\n  Published gate (K={K}, d<{dl}, term≥{th}):")
    print(f"    TP={TP}  FP={FP}  prec={prec:.3f}  rec={rec:.3f}  savings={savings*100:.2f}%")
    return df, pd.DataFrame(rows), {"TP": TP, "FP": FP, "prec": prec,
                                    "rec": rec, "savings": savings}


def main():
    out = []

    # G=8
    if (ROOT / "data" / "rollouts.jsonl").exists():
        g8 = load_groups(ROOT / "data" / "rollouts.jsonl")
        m8 = pd.read_parquet(ROOT / "data" / "metrics.parquet")
        df8, corr8, gate8 = summarise(g8, m8, "G=8 (data/rollouts.jsonl)")
        for r in corr8.to_dict("records"):
            out.append({"G": 8, **r})
        out.append({"G": 8, "K": 10, "metric": "gate", **gate8})

    # G=16
    g16_path = ROOT / "data" / "rollouts_g16.jsonl"
    m16_path = ROOT / "data" / "metrics_g16.parquet"
    if g16_path.exists() and m16_path.exists():
        g16 = load_groups(g16_path)
        m16 = pd.read_parquet(m16_path)
        df16, corr16, gate16 = summarise(g16, m16, "G=16 (data/rollouts_g16.jsonl)")
        for r in corr16.to_dict("records"):
            out.append({"G": 16, **r})
        out.append({"G": 16, "K": 10, "metric": "gate", **gate16})
    else:
        print("\n(no G=16 data yet — skipping)")

    df_out = pd.DataFrame(out)
    df_out.to_csv(OUT_DIR / "g16_compare.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'g16_compare.csv'}")

    # Side-by-side panel: ρ at each K, G=8 vs G=16
    if "G" in df_out.columns and (df_out.G == 16).any():
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for G in (8, 16):
            sub = df_out[(df_out.G == G) & df_out.K.isin([5, 10, 15, 20])
                          & df_out["rho"].notna()]
            ax.plot(sub.K, sub["rho"], marker="o", label=f"G={G}")
        ax.set_xlabel("K"); ax.set_ylabel("Spearman ρ(div, reward_var)")
        ax.set_title("Divergence-variance correlation by group size")
        ax.set_xticks([5, 10, 15, 20])
        ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout()
        safe_savefig(fig, FIG_DIR / "g16_vs_g8_panel.png", dpi=140)
        print(f"wrote {FIG_DIR / 'g16_vs_g8_panel.png'}")


if __name__ == "__main__":
    main()
