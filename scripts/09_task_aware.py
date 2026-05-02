"""
Phase A — task-aware gate analysis.

Approach:
  1. Per task type, find the best (metric, K, threshold) by F1 (or savings at prec≥0.80).
  2. Train a leave-one-out task-aware gate: for each held-out group, pick its
     task-type's threshold (fitted on the other groups in that task type).
  3. Compare to the global K=10 gate (d=0.12, t=0.90) on the same data.

Outputs:
  results/task_aware_per_type.csv
  results/task_aware_summary.csv
  results/figures/task_aware_compare.png
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


def load() -> pd.DataFrame:
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

    def task_type(task_id: str) -> str:
        # task_id like "pick_two_obj_and_place-SoapBar-None-GarbageCan-418/trial_..."
        return task_id.split("-")[0]

    base = pd.DataFrame({
        "task_id": [g["task_id"] for g in groups],
        "task_type": [task_type(g["task_id"]) for g in groups],
        "label": [label(g) for g in groups],
        "reward_var": [float(np.var([t["total_reward"] for t in g["trajectories"]]))
                       for g in groups],
    })
    base["zero_var"] = (base.reward_var == 0).astype(int)
    return base.merge(metrics, on="task_id", suffixes=("", "_drop"))


def eval_cut(cut, zv):
    cut = np.asarray(cut, bool); zv = np.asarray(zv, bool)
    TP = int((cut & zv).sum()); FP = int((cut & ~zv).sum())
    FN = int((~cut & zv).sum()); TN = int((~cut & ~zv).sum())
    prec = TP / max(1, TP + FP)
    rec = TP / max(1, TP + FN)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return TP, FP, prec, rec, f1


def savings(TP, K, n):
    return TP * (MAX_STEPS - K) / (MAX_STEPS * n)


def best_R3_OR_for_subset(sub, K, prec_floor=0.80):
    """Find best (d_low, t_high) for OR rule on this subset; return None if no
    valid operating point exists."""
    if len(sub) < 4:
        return None
    div = sub[f"prefix_edit_distance_mean@{K}"].values
    term = sub[f"termination_fraction@{K}"].values
    zv = sub.zero_var.values
    if zv.sum() == 0 or zv.sum() == len(zv):
        return None
    rows = []
    for dl in np.linspace(0.0, 0.4, 21):
        for th in np.linspace(0.5, 1.0, 11):
            cut = (div < dl) | (term >= th)
            TP, FP, prec, rec, f1 = eval_cut(cut, zv)
            if prec >= prec_floor and TP > 0:
                rows.append({"d_low": dl, "t_high": th, "TP": TP, "FP": FP,
                             "prec": prec, "rec": rec, "f1": f1,
                             "savings": savings(TP, K, len(sub))})
    if not rows:
        return None
    return max(rows, key=lambda r: r["savings"])


def main():
    df = load()
    K = 10
    prec_floor = 0.80
    print(f"Loaded {len(df)} groups | task types: {df.task_type.nunique()}")

    # Per-type best operating point @ K=10 (fitted on full per-type data)
    rows = []
    for tt, sub in df.groupby("task_type"):
        composition = sub.label.value_counts().to_dict()
        best = best_R3_OR_for_subset(sub, K=K, prec_floor=prec_floor)
        if best is None:
            rows.append({"task_type": tt, "n": len(sub),
                         "n_zero_var": int(sub.zero_var.sum()),
                         "n_all_fail": composition.get("all_fail", 0),
                         "n_all_succeed": composition.get("all_succeed", 0),
                         "n_mixed": composition.get("mixed", 0),
                         "d_low": np.nan, "t_high": np.nan,
                         "TP": 0, "FP": 0,
                         "prec": np.nan, "rec": 0.0, "f1": 0.0,
                         "savings": 0.0,
                         "note": "no valid operating point"})
            continue
        rows.append({"task_type": tt, "n": len(sub),
                     "n_zero_var": int(sub.zero_var.sum()),
                     "n_all_fail": composition.get("all_fail", 0),
                     "n_all_succeed": composition.get("all_succeed", 0),
                     "n_mixed": composition.get("mixed", 0),
                     **{k: best[k] for k in ("d_low", "t_high", "TP", "FP",
                                              "prec", "rec", "f1", "savings")},
                     "note": ""})
    per_type = pd.DataFrame(rows).sort_values("n", ascending=False)
    print("\n=== Per-task-type best R3 OR @ K=10, prec≥0.80 ===")
    print(per_type.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    per_type.to_csv(OUT_DIR / "task_aware_per_type.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'task_aware_per_type.csv'}")

    # Leave-one-out task-aware gate vs global gate
    # For each group, fit per-type thresholds on (its task type, excluding self) then test on it.
    div_all = df[f"prefix_edit_distance_mean@{K}"].values
    term_all = df[f"termination_fraction@{K}"].values
    zv_all = df.zero_var.values
    n = len(df)

    cut_global = (div_all < 0.12) | (term_all >= 0.90)
    TP_g, FP_g, prec_g, rec_g, f1_g = eval_cut(cut_global, zv_all)
    sav_g = savings(TP_g, K, n)
    print(f"\nGlobal gate K=10 (d=0.12, t=0.90): "
          f"TP={TP_g}, FP={FP_g}, prec={prec_g:.3f}, rec={rec_g:.3f}, savings={sav_g:.3f}")

    # Per-type fitted thresholds (whole subset; LOO-CV would shift only one
    # group, gives essentially the same thresholds for 24-group classes — we
    # report whole-subset fits but exclude the type-specific fitted point only
    # if the type didn't yield one).
    cut_ta = np.zeros(n, bool)
    for i, row in df.reset_index(drop=True).iterrows():
        tt = row.task_type
        sub_others = df[(df.task_type == tt) & (df.task_id != row.task_id)]
        best = best_R3_OR_for_subset(sub_others, K=K, prec_floor=prec_floor)
        if best is None:
            # fall back to global thresholds for this type
            dl, th = 0.12, 0.90
        else:
            dl, th = best["d_low"], best["t_high"]
        cut_ta[i] = (div_all[i] < dl) or (term_all[i] >= th)
    TP_t, FP_t, prec_t, rec_t, f1_t = eval_cut(cut_ta, zv_all)
    sav_t = savings(TP_t, K, n)
    print(f"Task-aware gate K=10:               "
          f"TP={TP_t}, FP={FP_t}, prec={prec_t:.3f}, rec={rec_t:.3f}, savings={sav_t:.3f}")

    summary = pd.DataFrame([
        {"gate": "global (d=0.12, t=0.90)", "K": K, "TP": TP_g, "FP": FP_g,
         "prec": prec_g, "rec": rec_g, "f1": f1_g, "savings": sav_g},
        {"gate": "task-aware (LOO per-type fit)", "K": K, "TP": TP_t, "FP": FP_t,
         "prec": prec_t, "rec": rec_t, "f1": f1_t, "savings": sav_t},
    ])
    print("\n=== Global vs task-aware ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.to_csv(OUT_DIR / "task_aware_summary.csv", index=False)
    print(f"wrote {OUT_DIR / 'task_aware_summary.csv'}")

    # Figure: per-type savings bar chart
    fig, ax = plt.subplots(figsize=(10, 4.2))
    pt = per_type.copy()
    short = {
        "look_at_obj_in_light": "look@light",
        "pick_and_place_simple": "pick&place",
        "pick_clean_then_place_in_recep": "pick·clean",
        "pick_cool_then_place_in_recep": "pick·cool",
        "pick_heat_then_place_in_recep": "pick·heat",
        "pick_two_obj_and_place": "pick·two",
    }
    pt["short"] = pt["task_type"].map(short)
    x = np.arange(len(pt))
    sav = pt["savings"].values * 100
    bars = ax.bar(x, sav, color="#1f77b4")
    for xi, v in zip(x, sav):
        ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(pt["short"], fontsize=9)
    ax.set_ylabel("Within-type compute saved (%)")
    ax.set_title("Task-aware R3 OR @ K=10, precision ≥ 0.80")
    ax.axhline(sav_g * 100, ls="--", c="red", alpha=0.5,
               label=f"global gate avg {sav_g*100:.1f}%")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    safe_savefig(fig, FIG_DIR / "task_aware_compare.png", dpi=140)
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'task_aware_compare.png'}")


if __name__ == "__main__":
    main()
