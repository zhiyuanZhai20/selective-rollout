"""
Two-axis hybrid gate.

Insight from 04: termination_fraction has a U-shape vs zero-variance:
  - all_succeed groups: HIGH term frac (everyone finishes early)
  - all_fail groups:    LOW term frac (no one finishes)
  - mixed groups:       MID term frac
A monotonic gate (or single Spearman correlation) misses this entirely.

Proposed gate logic at step K:
  CUT iff
    div LOW                       (covers coordinated zero-var: most all-succeed
                                   and lock-step all-fail)
    OR (div HIGH AND term_frac LOW)
                                  (covers chaotic all-fail: 8 trajs flailing,
                                   none has finished)
    OR (div HIGH AND term_frac HIGH)
                                  (covers any all-succeed leaking to high div —
                                   rare but cheap to add)

The "div MID" band stays as KEEP (most uncertain region).

We tune (div_low, div_high, term_low, term_high) on K=15 using the 100-group
data. Held-out evaluation isn't possible at this scale, so this is descriptive
of what's achievable; the gate would need re-tuning per env in the real project.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.figutil import safe_savefig

ROOT = Path(__file__).resolve().parent.parent
ROLLOUTS = ROOT / "data" / "rollouts.jsonl"
METRICS = ROOT / "data" / "metrics.parquet"
OUT_FIG = ROOT / "results" / "figures" / "two_axis_gate_K15.png"
OUT_CSV = ROOT / "results" / "two_axis_gate_eval.csv"


def label_group(g):
    rs = [t["total_reward"] for t in g["trajectories"]]
    n = len(rs); won = sum(1 for r in rs if r > 0)
    if won == 0: return "all_fail"
    if won == n: return "all_succeed"
    return "mixed"


def main():
    groups = []
    with open(ROLLOUTS) as f:
        for line in f:
            groups.append(json.loads(line))

    metrics = pd.read_parquet(METRICS)
    K = 15
    div_col = f"prefix_edit_distance_mean@{K}"
    term_col = f"termination_fraction@{K}"

    df = pd.DataFrame({
        "task_id": [g["task_id"] for g in groups],
        "label": [label_group(g) for g in groups],
        "reward_var": [float(np.var([t["total_reward"] for t in g["trajectories"]]))
                       for g in groups],
    })
    df = df.merge(metrics[["task_id", div_col, term_col]], on="task_id")
    df["zero_var"] = (df.reward_var == 0).astype(int)
    df = df.rename(columns={div_col: "div", term_col: "term"})

    # group breakdown
    print("Label counts:", df.label.value_counts().to_dict())
    print("\nMean div / term by label:")
    print(df.groupby("label")[["div", "term"]].agg(["mean", "median"]).round(3))
    print()

    # Sweep candidate thresholds
    div_lows = np.linspace(0.05, 0.25, 9)
    div_highs = np.linspace(0.4, 0.7, 7)
    term_lows = np.linspace(0.0, 0.2, 5)
    term_highs = np.linspace(0.5, 0.95, 7)

    rows = []

    def eval_gate(pred_cut: np.ndarray):
        TP = (pred_cut & (df.zero_var == 1)).sum()
        FP = (pred_cut & (df.zero_var == 0)).sum()
        FN = (~pred_cut & (df.zero_var == 1)).sum()
        TN = (~pred_cut & (df.zero_var == 0)).sum()
        return int(TP), int(FP), int(FN), int(TN)

    # Baseline: just div < div_low
    print("=== Baseline (single low-div cut) ===")
    print(f"{'div<':>6s}  {'cut':>4s}  {'TP':>3s}/{'FP':>3s}  {'prec':>5s}  {'recall':>6s}")
    for dl in div_lows:
        cut = (df["div"].values < dl)
        TP, FP, FN, TN = eval_gate(cut)
        prec = TP / max(1, TP + FP)
        rec = TP / max(1, TP + FN)
        print(f"{dl:>6.2f}  {cut.sum():>4d}  {TP:>3d}/{FP:>3d}  {prec:>5.2f}  {rec:>6.2f}")
        rows.append({"gate": "low_div_only", "div_low": dl,
                     "div_high": np.nan, "term_low": np.nan, "term_high": np.nan,
                     "cut_count": int(cut.sum()),
                     "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                     "precision": prec, "recall": rec,
                     "f1": 2 * prec * rec / max(1e-9, prec + rec)})

    # Two-axis: div_low OR (div_high AND term_low) OR (div_high AND term_high)
    print("\n=== Two-axis gate sweep ===")
    print("rule: cut iff (div<dL) or (div>dH AND term<tL) or (div>dH AND term>=tH)")
    best = None
    for dl in div_lows:
        for dh in div_highs:
            for tl in term_lows:
                for th in term_highs:
                    if th <= tl: continue
                    cut = (
                        (df["div"].values < dl)
                        | ((df["div"].values > dh) & (df["term"].values < tl))
                        | ((df["div"].values > dh) & (df["term"].values >= th))
                    )
                    TP, FP, FN, TN = eval_gate(cut)
                    prec = TP / max(1, TP + FP)
                    rec = TP / max(1, TP + FN)
                    f1 = 2 * prec * rec / max(1e-9, prec + rec)
                    rows.append({"gate": "two_axis", "div_low": dl,
                                 "div_high": dh, "term_low": tl, "term_high": th,
                                 "cut_count": int(cut.sum()),
                                 "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                                 "precision": prec, "recall": rec, "f1": f1})
                    if best is None or f1 > best["f1"]:
                        best = rows[-1]
    print(f"\nBest two-axis (max F1):  {best}")

    # Compare a few "high-precision" operating points
    print("\n=== Pick the best gate at FP <= K (sweep K) ===")
    print(f"{'gate':14s}  {'fp<=':>4s}  {'TP':>3s}  {'recall':>6s}  params")
    for max_fp in (0, 1, 2, 3, 5):
        # baseline best
        cands = [r for r in rows if r["gate"] == "low_div_only" and r["FP"] <= max_fp]
        if cands:
            b = max(cands, key=lambda r: r["TP"])
            print(f"low_div_only    {max_fp:>4d}  {b['TP']:>3d}  {b['recall']:>6.2f}  div<{b['div_low']:.2f}")
        cands = [r for r in rows if r["gate"] == "two_axis" and r["FP"] <= max_fp]
        if cands:
            b = max(cands, key=lambda r: r["TP"])
            print(f"two_axis        {max_fp:>4d}  {b['TP']:>3d}  {b['recall']:>6.2f}  "
                  f"dL={b['div_low']:.2f} dH={b['div_high']:.2f} "
                  f"tL={b['term_low']:.2f} tH={b['term_high']:.2f}")

    # Save table
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    # Visualization: 2D scatter of (div, term) colored by label, with the
    # best two-axis gate region shaded.
    best_two = max([r for r in rows if r["gate"] == "two_axis"], key=lambda r: r["f1"])
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
    colors = {"all_succeed": "#2ca02c", "all_fail": "#1f77b4", "mixed": "#ff7f0e"}
    markers = {"all_succeed": "o", "all_fail": "o", "mixed": "D"}
    rng = np.random.default_rng(42)
    for lab in ["all_succeed", "all_fail", "mixed"]:
        sub = df[df.label == lab].reset_index(drop=True)
        # tiny jitter on term axis since values are discrete (multiples of 1/8)
        jx = rng.normal(0, 0.005, size=len(sub))
        jy = rng.normal(0, 0.012, size=len(sub))
        ax.scatter(sub["div"].values + jx, sub["term"].values + jy,
                   c=colors[lab], marker=markers[lab],
                   alpha=0.75, s=55, edgecolor="k", linewidth=0.4,
                   label=f"{lab} (n={len(sub)})")

    # Shade the cut region
    dL = best_two["div_low"]; dH = best_two["div_high"]
    tL = best_two["term_low"]; tH = best_two["term_high"]
    ax.axvspan(0, dL, color="red", alpha=0.08, label=f"CUT: low div (<{dL:.2f})")
    ax.axhspan(0, tL, xmin=(dH - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
               color="red", alpha=0.08)
    # Just draw rectangles directly
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((dH, 0), 1.0 - dH, tL, color="red", alpha=0.10))
    ax.add_patch(Rectangle((dH, tH), 1.0 - dH, 1.0 - tH, color="red", alpha=0.10))
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(f"prefix_edit_distance_mean@K={K}")
    ax.set_ylabel(f"termination_fraction@K={K}")
    ax.set_title(f"Two-axis gate @ K={K}\n"
                 f"cut: div<{dL:.2f} OR (div>{dH:.2f} AND (term<{tL:.2f} OR term≥{tH:.2f}))\n"
                 f"TP={best_two['TP']}/39  FP={best_two['FP']}/61  "
                 f"prec={best_two['precision']:.2f}  recall={best_two['recall']:.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    safe_savefig(fig, OUT_FIG, dpi=140)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
