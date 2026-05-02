"""Refit the gate thresholds on G=16 data and compare to G=8 thresholds."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results"
MAX_STEPS = 30


def load_groups(path):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def eval_or(div, term, zv, dl, th):
    cut = (div < dl) | (term >= th)
    TP = int((cut & zv).sum()); FP = int((cut & ~zv).sum())
    FN = int((~cut & zv).sum())
    prec = TP / max(1, TP + FP); rec = TP / max(1, TP + FN)
    return TP, FP, prec, rec


def main():
    g16 = load_groups(ROOT / "data" / "rollouts_g16.jsonl")
    m16 = pd.read_parquet(ROOT / "data" / "metrics_g16.parquet")

    rows = []
    for g in g16:
        rs = [t["total_reward"] for t in g["trajectories"]]
        rows.append({"task_id": g["task_id"], "zv": int(np.var(rs) == 0)})
    df = pd.DataFrame(rows).merge(m16, on="task_id")
    n = len(df)
    zv = df.zv.values.astype(bool)

    print(f"G=16  N={n}  zero_var={int(zv.sum())}")

    print("\n--- Refitted gate at K=10 (precision >= 0.80) ---")
    best = None
    for K in (5, 10, 15, 20):
        d = df[f"prefix_edit_distance_mean@{K}"].values
        t = df[f"termination_fraction@{K}"].values
        for dl in np.linspace(0.0, 0.4, 41):
            for th in np.linspace(0.5, 1.0, 11):
                TP, FP, prec, rec = eval_or(d, t, zv, dl, th)
                if prec >= 0.80 and TP > 0:
                    sav = TP * (MAX_STEPS - K) / (MAX_STEPS * n)
                    if best is None or sav > best[0]:
                        best = (sav, K, dl, th, TP, FP, prec, rec)

    if best is None:
        print("  no operating point at prec >= 0.80")
    else:
        sav, K, dl, th, TP, FP, prec, rec = best
        print(f"  best:  K={K}  d_low={dl:.3f}  t_high={th:.2f}")
        print(f"         TP={TP}  FP={FP}  prec={prec:.3f}  rec={rec:.3f}  savings={sav*100:.2f}%")

    # Same form as G=8 winner: K=10, d=0.12, t=0.90
    print("\n--- Apply G=8 published thresholds (K=10, d=0.12, t=0.90) on G=16 ---")
    for K in (10, 15):
        d = df[f"prefix_edit_distance_mean@{K}"].values
        t = df[f"termination_fraction@{K}"].values
        TP, FP, prec, rec = eval_or(d, t, zv, 0.12, 0.90)
        sav = TP * (MAX_STEPS - K) / (MAX_STEPS * n)
        print(f"  K={K} d=0.12 t=0.90:  TP={TP}  FP={FP}  prec={prec:.3f}  rec={rec:.3f}  savings={sav*100:.2f}%")

    # Per-K best operating point at G=16, prec>=0.80, OR rule
    print("\n--- Per-K best (G=16, OR rule, prec>=0.80) ---")
    table = []
    for K in (5, 10, 15, 20):
        d = df[f"prefix_edit_distance_mean@{K}"].values
        t = df[f"termination_fraction@{K}"].values
        bestK = None
        for dl in np.linspace(0.0, 0.4, 41):
            for th in np.linspace(0.5, 1.0, 11):
                TP, FP, prec, rec = eval_or(d, t, zv, dl, th)
                if prec >= 0.80 and TP > 0:
                    sav = TP * (MAX_STEPS - K) / (MAX_STEPS * n)
                    if bestK is None or sav > bestK[0]:
                        bestK = (sav, dl, th, TP, FP, prec, rec)
        if bestK:
            sav, dl, th, TP, FP, prec, rec = bestK
            table.append({"K": K, "d_low": dl, "t_high": th,
                          "TP": TP, "FP": FP, "prec": prec, "rec": rec,
                          "savings": sav})
            print(f"  K={K:2d}: d={dl:.3f}, t={th:.2f}  "
                  f"TP={TP}  FP={FP}  prec={prec:.3f}  rec={rec:.3f}  savings={sav*100:.2f}%")
    pd.DataFrame(table).to_csv(OUT_DIR / "g16_refit.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'g16_refit.csv'}")


if __name__ == "__main__":
    main()
