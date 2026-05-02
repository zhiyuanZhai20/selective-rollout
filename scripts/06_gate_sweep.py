"""
Comprehensive gate sweep across K values and rule forms.

Objective: find the SIMPLEST gate that gives the highest compute savings.

Compute-savings metric:
    savings_frac = TP * (max_steps - K) / (max_steps * N_groups)
                   ───────────────────────────────────────────
    fraction of total rollout work avoided by cutting TP groups at step K.

We also report the "net" compute saved including a soft penalty for each FP
(losing a useful gradient group = we still pay max_steps for the rollout AND
the gradient signal is lost):
    net_saved = savings - FP_penalty * FP / N_groups

We don't pretend to know FP_penalty exactly. We report savings at multiple
operating points (different precision floors) and let the user pick.

Rule families tested (in order of simplicity):
  R1: cut iff div < d_low                      (1 parameter)
  R2: cut iff term <= t_low OR term >= t_high  (1-2 parameter on TERM only)
  R3: cut iff div < d_low OR term >= t_high    (asymmetric two-axis, 2 param)
  R4: cut iff div < d_low OR (div > d_high AND (term <= t_low OR term >= t_high))
                                                (the full 4-param version)

Evaluation: max_steps=30, N=100. We report each rule at three precision floors
(prec ≥ 0.85, ≥ 0.80, ≥ 0.75) and the F1-optimal point.
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.figutil import safe_savefig
ROLLOUTS = ROOT / "data" / "rollouts.jsonl"
METRICS = ROOT / "data" / "metrics.parquet"
OUT_DIR = ROOT / "results"
FIG_DIR = OUT_DIR / "figures"

MAX_STEPS = 30
N = 100  # groups


def load_data():
    groups = []
    with open(ROLLOUTS) as f:
        for line in f:
            groups.append(json.loads(line))

    metrics = pd.read_parquet(METRICS)

    def label(g):
        rs = [t["total_reward"] for t in g["trajectories"]]
        n = len(rs); won = sum(1 for r in rs if r > 0)
        if won == 0: return "all_fail"
        if won == n: return "all_succeed"
        return "mixed"

    df = pd.DataFrame({
        "task_id": [g["task_id"] for g in groups],
        "label": [label(g) for g in groups],
        "reward_var": [float(np.var([t["total_reward"] for t in g["trajectories"]]))
                       for g in groups],
    })
    df["zero_var"] = (df.reward_var == 0).astype(int)
    return df, metrics


def eval_cut(cut, zero_var):
    cut = np.asarray(cut, bool); zv = np.asarray(zero_var, bool)
    TP = int((cut & zv).sum()); FP = int((cut & ~zv).sum())
    FN = int((~cut & zv).sum()); TN = int((~cut & ~zv).sum())
    prec = TP / max(1, TP + FP)
    rec = TP / max(1, TP + FN)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return TP, FP, FN, TN, prec, rec, f1


def compute_savings(TP, FP, K):
    """Fraction of total rollout-step compute saved by the gate."""
    return TP * (MAX_STEPS - K) / (MAX_STEPS * N)


def sweep_R1(df, K, div):
    """cut iff div < d"""
    out = []
    for d in np.linspace(0.02, 0.4, 39):
        cut = div < d
        TP, FP, FN, TN, prec, rec, f1 = eval_cut(cut, df.zero_var)
        out.append({"rule": "R1", "K": K, "d_low": d,
                    "TP": TP, "FP": FP, "prec": prec, "rec": rec, "f1": f1,
                    "savings": compute_savings(TP, FP, K),
                    "n_params": 1})
    return out


def sweep_R2(df, K, term):
    """cut iff term <= tl OR term >= th"""
    out = []
    for tl in np.linspace(0.0, 0.2, 5):
        for th in np.linspace(0.5, 1.0, 11):
            cut = (term <= tl) | (term >= th)
            TP, FP, FN, TN, prec, rec, f1 = eval_cut(cut, df.zero_var)
            out.append({"rule": "R2", "K": K,
                        "t_low": tl, "t_high": th,
                        "TP": TP, "FP": FP, "prec": prec, "rec": rec, "f1": f1,
                        "savings": compute_savings(TP, FP, K),
                        "n_params": 2})
    return out


def sweep_R3(df, K, div, term):
    """cut iff div < d_low OR term >= t_high"""
    out = []
    for dl in np.linspace(0.02, 0.3, 15):
        for th in np.linspace(0.5, 1.0, 11):
            cut = (div < dl) | (term >= th)
            TP, FP, FN, TN, prec, rec, f1 = eval_cut(cut, df.zero_var)
            out.append({"rule": "R3", "K": K, "d_low": dl, "t_high": th,
                        "TP": TP, "FP": FP, "prec": prec, "rec": rec, "f1": f1,
                        "savings": compute_savings(TP, FP, K),
                        "n_params": 2})
    return out


def sweep_R3b(df, K, div, term):
    """cut iff div < d_low OR (term <= t_low AND div > d_low)
    i.e. term=0 catches all_fail (lock-step or chaotic), low div catches all_succeed.
    Single div threshold + single term threshold = 2 params, simpler than R4.
    """
    out = []
    for dl in np.linspace(0.02, 0.3, 15):
        for tl in [0.0, 0.05, 0.1, 0.15]:
            cut = (div < dl) | (term <= tl)
            TP, FP, FN, TN, prec, rec, f1 = eval_cut(cut, df.zero_var)
            out.append({"rule": "R3b", "K": K, "d_low": dl, "t_low": tl,
                        "TP": TP, "FP": FP, "prec": prec, "rec": rec, "f1": f1,
                        "savings": compute_savings(TP, FP, K),
                        "n_params": 2})
    return out


def sweep_R4(df, K, div, term):
    """cut iff div < d_low OR (div > d_high AND (term <= t_low OR term >= t_high))"""
    out = []
    for dl in np.linspace(0.05, 0.25, 9):
        for dh in np.linspace(0.4, 0.7, 7):
            for tl in [0.0, 0.05, 0.10]:
                for th in np.linspace(0.5, 1.0, 6):
                    if th <= tl: continue
                    cut = (div < dl) | ((div > dh) & ((term <= tl) | (term >= th)))
                    TP, FP, FN, TN, prec, rec, f1 = eval_cut(cut, df.zero_var)
                    out.append({"rule": "R4", "K": K, "d_low": dl, "d_high": dh,
                                "t_low": tl, "t_high": th,
                                "TP": TP, "FP": FP, "prec": prec, "rec": rec, "f1": f1,
                                "savings": compute_savings(TP, FP, K),
                                "n_params": 4})
    return out


def main():
    df, metrics = load_data()
    print(f"Groups: {len(df)} | Zero-var: {int(df.zero_var.sum())} "
          f"(all_succeed={int((df.label=='all_succeed').sum())}, "
          f"all_fail={int((df.label=='all_fail').sum())}) | "
          f"Non-zero: {int((1-df.zero_var).sum())} (mixed)")

    all_rows = []
    for K in (5, 10, 15, 20):
        d_col = f"prefix_edit_distance_mean@{K}"
        t_col = f"termination_fraction@{K}"
        sub = df.merge(metrics[["task_id", d_col, t_col]], on="task_id")
        div = sub[d_col].values
        term = sub[t_col].values
        all_rows.extend(sweep_R1(sub, K, div))
        all_rows.extend(sweep_R2(sub, K, term))
        all_rows.extend(sweep_R3(sub, K, div, term))
        all_rows.extend(sweep_R3b(sub, K, div, term))
        all_rows.extend(sweep_R4(sub, K, div, term))

    res = pd.DataFrame(all_rows)
    res.to_csv(OUT_DIR / "gate_sweep_full.csv", index=False)

    # ===== Summary tables =====
    print("\n========== Best rule by (rule, K, prec_floor) ==========")
    for prec_floor in (0.85, 0.80, 0.75):
        print(f"\n--- precision floor ≥ {prec_floor} (sorted by savings desc) ---")
        rows = []
        for rule in ("R1", "R2", "R3", "R3b", "R4"):
            for K in (5, 10, 15, 20):
                cands = res[(res["rule"] == rule) & (res["K"] == K)
                            & (res["prec"] >= prec_floor) & (res["TP"] > 0)]
                if len(cands) == 0: continue
                best = cands.sort_values("savings", ascending=False).iloc[0]
                rows.append(best)
        if not rows:
            print("  (no rule meets floor)"); continue
        out = pd.DataFrame(rows)[["rule", "K", "TP", "FP", "prec", "rec",
                                   "savings", "f1", "n_params",
                                   "d_low", "d_high", "t_low", "t_high"]]
        out = out.sort_values("savings", ascending=False)
        with pd.option_context("display.max_rows", None,
                                "display.width", 160,
                                "display.float_format", lambda x: f"{x:.3f}"):
            print(out.to_string(index=False))

    # ===== Pick winner: best savings at prec >= 0.80 =====
    print("\n\n========== WINNER (savings at prec ≥ 0.80) ==========")
    cands = res[(res["prec"] >= 0.80) & (res["TP"] > 0)]
    winner = cands.sort_values("savings", ascending=False).iloc[0]
    print(winner.to_dict())

    # ===== Also report best F1 ignoring precision floor =====
    print("\n========== Best F1 across all ==========")
    winF = res.sort_values("f1", ascending=False).iloc[0]
    print(winF.to_dict())

    # ===== Per-rule savings at all precision levels =====
    print("\n========== Pareto front: savings vs FP, by rule × K (prec floor 0.80) ==========")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    colors = {5: "#aaa", 10: "#1f77b4", 15: "#ff7f0e", 20: "#2ca02c"}
    markers = {"R1": "o", "R2": "s", "R3": "^", "R3b": "v", "R4": "D"}
    rule_descr = {
        "R1": "div < dL  [1 param]",
        "R2": "term ≤ tL OR term ≥ tH  [2]",
        "R3": "div < dL OR term ≥ tH  [2]",
        "R3b": "div < dL OR term ≤ tL  [2]",
        "R4": "low_div OR (high_div AND term extreme)  [4]",
    }
    # Pareto envelope: for each (rule, K), plot savings vs FP frontier
    for rule in ("R1", "R2", "R3", "R3b", "R4"):
        for K in (10, 15, 20):
            sub = res[(res["rule"] == rule) & (res["K"] == K)
                       & (res["prec"] >= 0.80)]
            if len(sub) == 0: continue
            sub = sub.sort_values("FP")
            # take pareto: max savings at each FP
            front = sub.groupby("FP")["savings"].max().reset_index()
            axes[0].plot(front["FP"], front["savings"] * 100,
                         marker=markers[rule], color=colors[K],
                         linewidth=1.2, alpha=0.85,
                         label=f"{rule}@K={K}")
    axes[0].set_xlabel("False positives (out of 61 mixed)")
    axes[0].set_ylabel("Compute saved (%)")
    axes[0].set_title("Pareto front: prec ≥ 0.80")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=7, ncol=2)

    # Recall vs savings, color = K
    for rule in ("R1", "R3", "R3b", "R4"):
        for K in (10, 15, 20):
            sub = res[(res["rule"] == rule) & (res["K"] == K)
                       & (res["prec"] >= 0.80)]
            if len(sub) == 0: continue
            sub = sub.sort_values("rec")
            front = sub.groupby("rec")["savings"].max().reset_index()
            axes[1].plot(front["rec"], front["savings"] * 100,
                         marker=markers[rule], color=colors[K],
                         linewidth=1.2, alpha=0.8,
                         label=f"{rule}@K={K}")
    axes[1].set_xlabel("Recall on zero-var (39 total)")
    axes[1].set_ylabel("Compute saved (%)")
    axes[1].set_title("Savings vs recall: prec ≥ 0.80")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=7, ncol=2)
    plt.tight_layout()
    safe_savefig(fig, FIG_DIR / "gate_sweep_pareto.png", dpi=140)
    print(f"\nwrote {FIG_DIR / 'gate_sweep_pareto.png'}")


if __name__ == "__main__":
    main()
