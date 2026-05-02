"""
Phase A — paper-grade ablation table + 5-fold CV stability.

Reads data/metrics.parquet + data/rollouts.jsonl.
Writes:
    results/ablation_table.csv
    results/cv_stability.csv
    results/figures/ablation_bar.png
    results/figures/cv_thresholds.png

Ablation: at fixed K=10 (winner), report best operating point per rule
under precision >= 0.80.

5-fold CV: split N=100 groups into 5 folds.  On each train fold, pick
the (d_low, t_high) for R3 that maximizes savings subject to prec >= 0.80
on that fold; evaluate on the held-out test fold.  Report savings + prec
mean / std across folds.
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

OUT_DIR = ROOT / "results"
FIG_DIR = OUT_DIR / "figures"
MAX_STEPS = 30
N = 100


def load() -> pd.DataFrame:
    """Return df with task_id, label, zero_var, div@K, term@K for K in {5,10,15,20}."""
    groups = []
    with open(ROOT / "data" / "rollouts.jsonl") as f:
        for line in f:
            groups.append(json.loads(line))
    metrics = pd.read_parquet(ROOT / "data" / "metrics.parquet")

    def label(g):
        rs = [t["total_reward"] for t in g["trajectories"]]
        n = len(rs); won = sum(1 for r in rs if r > 0)
        if won == 0: return "all_fail"
        if won == n: return "all_succeed"
        return "mixed"

    base = pd.DataFrame({
        "task_id": [g["task_id"] for g in groups],
        "label": [label(g) for g in groups],
        "reward_var": [float(np.var([t["total_reward"] for t in g["trajectories"]]))
                       for g in groups],
    })
    base["zero_var"] = (base.reward_var == 0).astype(int)
    keep = ["task_id"]
    for K in (5, 10, 15, 20):
        keep += [f"prefix_edit_distance_mean@{K}", f"termination_fraction@{K}"]
    return base.merge(metrics[keep], on="task_id")


def eval_cut(cut, zv):
    cut = np.asarray(cut, bool); zv = np.asarray(zv, bool)
    TP = int((cut & zv).sum()); FP = int((cut & ~zv).sum())
    FN = int((~cut & zv).sum()); TN = int((~cut & ~zv).sum())
    prec = TP / max(1, TP + FP)
    rec = TP / max(1, TP + FN)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return TP, FP, FN, TN, prec, rec, f1


def savings(TP, K, n):
    return TP * (MAX_STEPS - K) / (MAX_STEPS * n)


# ---------- ablation ----------
def best_R1(df, K, prec_floor=0.80):
    div = df[f"prefix_edit_distance_mean@{K}"].values
    zv = df.zero_var.values
    rows = []
    for d in np.linspace(0.02, 0.4, 39):
        cut = div < d
        TP, FP, FN, TN, prec, rec, f1 = eval_cut(cut, zv)
        if prec >= prec_floor and TP > 0:
            rows.append({"d_low": d, "TP": TP, "FP": FP, "prec": prec,
                         "rec": rec, "f1": f1,
                         "savings": savings(TP, K, len(df))})
    return max(rows, key=lambda r: r["savings"]) if rows else None


def best_R2_term(df, K, prec_floor=0.80):
    term = df[f"termination_fraction@{K}"].values
    zv = df.zero_var.values
    rows = []
    for tl in np.linspace(0.0, 0.2, 9):
        for th in np.linspace(0.5, 1.0, 11):
            cut = (term <= tl) | (term >= th)
            TP, FP, FN, TN, prec, rec, f1 = eval_cut(cut, zv)
            if prec >= prec_floor and TP > 0:
                rows.append({"t_low": tl, "t_high": th, "TP": TP, "FP": FP,
                             "prec": prec, "rec": rec, "f1": f1,
                             "savings": savings(TP, K, len(df))})
    return max(rows, key=lambda r: r["savings"]) if rows else None


def best_R3_OR(df, K, prec_floor=0.80):
    div = df[f"prefix_edit_distance_mean@{K}"].values
    term = df[f"termination_fraction@{K}"].values
    zv = df.zero_var.values
    rows = []
    for dl in np.linspace(0.02, 0.30, 15):
        for th in np.linspace(0.5, 1.0, 11):
            cut = (div < dl) | (term >= th)
            TP, FP, FN, TN, prec, rec, f1 = eval_cut(cut, zv)
            if prec >= prec_floor and TP > 0:
                rows.append({"d_low": dl, "t_high": th, "TP": TP, "FP": FP,
                             "prec": prec, "rec": rec, "f1": f1,
                             "savings": savings(TP, K, len(df))})
    return max(rows, key=lambda r: r["savings"]) if rows else None


def best_R3_AND(df, K, prec_floor=0.80):
    """cut iff div < d_low AND term >= t_high — much stricter."""
    div = df[f"prefix_edit_distance_mean@{K}"].values
    term = df[f"termination_fraction@{K}"].values
    zv = df.zero_var.values
    rows = []
    for dl in np.linspace(0.05, 0.5, 19):
        for th in np.linspace(0.0, 1.0, 21):
            cut = (div < dl) & (term >= th)
            TP, FP, FN, TN, prec, rec, f1 = eval_cut(cut, zv)
            if prec >= prec_floor and TP > 0:
                rows.append({"d_low": dl, "t_high": th, "TP": TP, "FP": FP,
                             "prec": prec, "rec": rec, "f1": f1,
                             "savings": savings(TP, K, len(df))})
    return max(rows, key=lambda r: r["savings"]) if rows else None


def best_R4(df, K, prec_floor=0.80):
    div = df[f"prefix_edit_distance_mean@{K}"].values
    term = df[f"termination_fraction@{K}"].values
    zv = df.zero_var.values
    rows = []
    for dl in np.linspace(0.05, 0.25, 9):
        for dh in np.linspace(0.4, 0.7, 7):
            for tl in [0.0, 0.05, 0.10]:
                for th in np.linspace(0.5, 1.0, 6):
                    if th <= tl: continue
                    cut = (div < dl) | ((div > dh) & ((term <= tl) | (term >= th)))
                    TP, FP, FN, TN, prec, rec, f1 = eval_cut(cut, zv)
                    if prec >= prec_floor and TP > 0:
                        rows.append({"d_low": dl, "d_high": dh,
                                     "t_low": tl, "t_high": th,
                                     "TP": TP, "FP": FP, "prec": prec,
                                     "rec": rec, "f1": f1,
                                     "savings": savings(TP, K, len(df))})
    return max(rows, key=lambda r: r["savings"]) if rows else None


def run_ablation(df, K=10):
    rows = []
    for name, fn, n_params in [
        ("R0 (no gate)", None, 0),
        ("R1: div < dL", best_R1, 1),
        ("R2: term ≤ tL OR term ≥ tH", best_R2_term, 2),
        ("R3 (OR): div < dL OR term ≥ tH", best_R3_OR, 2),
        ("R3' (AND): div < dL AND term ≥ tH", best_R3_AND, 2),
        ("R4: div<dL OR (div>dH AND term extreme)", best_R4, 4),
    ]:
        if fn is None:
            rows.append({"rule": name, "n_params": 0, "TP": 0, "FP": 0,
                         "prec": np.nan, "rec": 0.0, "f1": 0.0, "savings": 0.0,
                         "params": ""})
            continue
        best = fn(df, K)
        if best is None:
            rows.append({"rule": name, "n_params": n_params,
                         "TP": 0, "FP": 0, "prec": np.nan,
                         "rec": 0.0, "f1": 0.0, "savings": 0.0,
                         "params": "(no operating point ≥0.80 prec)"})
            continue
        params_str = ", ".join(f"{k}={v:.2f}" for k, v in best.items()
                               if k in ("d_low", "d_high", "t_low", "t_high"))
        rows.append({"rule": name, "n_params": n_params,
                     "TP": best["TP"], "FP": best["FP"],
                     "prec": best["prec"], "rec": best["rec"],
                     "f1": best["f1"], "savings": best["savings"],
                     "params": params_str})
    return pd.DataFrame(rows)


# ---------- 5-fold CV ----------
def run_cv_fixed(df, K=10, dl=0.12, th=0.90, seed=42, n_folds=5):
    """Apply the published (fixed) gate to each held-out fold.

    This measures external validity of the *paper-recommended* thresholds:
    given (d_low, t_high) chosen on the full data, how stable are
    precision/recall/savings on fresh subsamples?
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    rows = []
    for f in range(n_folds):
        test = df.iloc[folds[f]].reset_index(drop=True)
        div_te = test[f"prefix_edit_distance_mean@{K}"].values
        term_te = test[f"termination_fraction@{K}"].values
        zv_te = test.zero_var.values
        cut = (div_te < dl) | (term_te >= th)
        TP, FP, FN, TN, prec, rec, f1 = eval_cut(cut, zv_te)
        rows.append({"fold": f, "n": len(test), "n_zero_var": int(zv_te.sum()),
                     "TP": TP, "FP": FP,
                     "prec": prec, "rec": rec, "f1": f1,
                     "savings": savings(TP, K, len(test))})
    return pd.DataFrame(rows)


def run_cv_relearn(df, K=10, prec_floor=0.85, seed=42, n_folds=5):
    """Learn thresholds on each train fold; evaluate on test fold.

    Uses prec_floor=0.85 on train to leave buffer for test-set noise.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    rows = []
    for f in range(n_folds):
        test_idx = folds[f]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != f])
        train = df.iloc[train_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        best = best_R3_OR(train, K, prec_floor=prec_floor)
        if best is None:
            rows.append({"fold": f, "d_low": np.nan, "t_high": np.nan,
                         "train_savings": np.nan, "test_savings": np.nan,
                         "test_prec": np.nan, "test_rec": np.nan,
                         "test_TP": 0, "test_FP": 0})
            continue
        dl, th = best["d_low"], best["t_high"]
        div_te = test[f"prefix_edit_distance_mean@{K}"].values
        term_te = test[f"termination_fraction@{K}"].values
        zv_te = test.zero_var.values
        cut = (div_te < dl) | (term_te >= th)
        TP, FP, FN, TN, prec, rec, f1 = eval_cut(cut, zv_te)
        rows.append({"fold": f, "d_low": dl, "t_high": th,
                     "train_savings": best["savings"],
                     "test_savings": savings(TP, K, len(test)),
                     "test_prec": prec, "test_rec": rec,
                     "test_TP": TP, "test_FP": FP})
    return pd.DataFrame(rows)


# ---------- main ----------
def main():
    df = load()
    print(f"Loaded {len(df)} groups | zero_var={int(df.zero_var.sum())}")

    # Ablation at K=10 (winner) and K=15 (compare)
    print("\n=== Ablation @ K=10 (precision floor 0.80) ===")
    abl_10 = run_ablation(df, K=10)
    abl_10["K"] = 10
    print(abl_10[["rule", "n_params", "TP", "FP", "prec", "rec", "f1", "savings", "params"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Ablation @ K=15 (precision floor 0.80) ===")
    abl_15 = run_ablation(df, K=15)
    abl_15["K"] = 15
    print(abl_15[["rule", "n_params", "TP", "FP", "prec", "rec", "f1", "savings", "params"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    abl = pd.concat([abl_10, abl_15], ignore_index=True)
    abl.to_csv(OUT_DIR / "ablation_table.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'ablation_table.csv'}")

    # 5-fold CV — published-thresholds external validity
    print("\n=== 5-fold CV: published gate (d=0.12, t=0.90) on held-out folds ===")
    cv_fixed = run_cv_fixed(df, K=10, dl=0.12, th=0.90)
    print(cv_fixed.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nsavings:  {cv_fixed['savings'].mean():.3f} ± {cv_fixed['savings'].std():.3f}")
    print(f"prec:     {cv_fixed['prec'].mean():.3f} ± {cv_fixed['prec'].std():.3f}")
    print(f"rec:      {cv_fixed['rec'].mean():.3f} ± {cv_fixed['rec'].std():.3f}")

    # 5-fold CV — relearn thresholds with buffer
    print("\n=== 5-fold CV: relearn thresholds at train prec≥0.85 (buffer for test noise) ===")
    cv_re = run_cv_relearn(df, K=10, prec_floor=0.85)
    print(cv_re.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\ntest_savings: {cv_re['test_savings'].mean():.3f} ± {cv_re['test_savings'].std():.3f}")
    print(f"test_prec:    {cv_re['test_prec'].mean():.3f} ± {cv_re['test_prec'].std():.3f}")
    print(f"test_rec:     {cv_re['test_rec'].mean():.3f} ± {cv_re['test_rec'].std():.3f}")

    cv_combined = pd.concat([cv_fixed.assign(mode="fixed"),
                             cv_re.assign(mode="relearn")], ignore_index=True)
    cv_combined.to_csv(OUT_DIR / "cv_stability.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'cv_stability.csv'}")
    cv = cv_re  # for the plot below

    # Plot: ablation bar chart
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rules_short = ["R0", "R1\n(div)", "R2\n(term)", "R3 OR\n(div, term)",
                   "R3' AND\n(div, term)", "R4\n(4 param)"]
    sav_10 = abl_10["savings"].values * 100
    sav_15 = abl_15["savings"].values * 100
    x = np.arange(len(rules_short))
    w = 0.38
    ax.bar(x - w/2, sav_10, width=w, color="#1f77b4", label="K=10")
    ax.bar(x + w/2, sav_15, width=w, color="#ff7f0e", label="K=15")
    for xi, v in zip(x - w/2, sav_10):
        ax.text(xi, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w/2, sav_15):
        ax.text(xi, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(rules_short, fontsize=9)
    ax.set_ylabel("Compute saved (%)")
    ax.set_title("Gate ablation @ precision ≥ 0.80")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    safe_savefig(fig, FIG_DIR / "ablation_bar.png", dpi=140)
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'ablation_bar.png'}")

    # Plot: CV thresholds
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].scatter(range(5), cv["d_low"], c="#1f77b4", s=60)
    axes[0].axhline(0.12, ls="--", c="red", alpha=0.5, label="full-data fit (0.12)")
    axes[0].set_xticks(range(5)); axes[0].set_xlabel("fold")
    axes[0].set_ylabel("d_low (best on train)")
    axes[0].set_title("Threshold $d_L$ across folds"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].scatter(range(5), cv["t_high"], c="#ff7f0e", s=60)
    axes[1].axhline(0.90, ls="--", c="red", alpha=0.5, label="full-data fit (0.90)")
    axes[1].set_xticks(range(5)); axes[1].set_xlabel("fold")
    axes[1].set_ylabel("t_high (best on train)")
    axes[1].set_title("Threshold $t_H$ across folds"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    safe_savefig(fig, FIG_DIR / "cv_thresholds.png", dpi=140)
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'cv_thresholds.png'}")


if __name__ == "__main__":
    main()
