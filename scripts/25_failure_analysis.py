"""
Failure-case analysis: inspect the 4 false positive groups at the chosen
gate (K=10, d_L=0.12, t_H=0.90) on the 100-task offline buffer.

Reports for each FP:
  - task_id and task type
  - reward distribution (G=8 trajectories)
  - d_K, t_K values (why they triggered)
  - mean step length
  - sample first-K actions to see why they "looked like" zero-var

Outputs:
  results/false_positives.csv
  results/false_positives_detail.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ROLLOUTS = ROOT / "data" / "rollouts.jsonl"
METRICS = ROOT / "data" / "metrics.parquet"
OUT = ROOT / "results"
K = 10


def main():
    groups = []
    with open(ROLLOUTS) as f:
        for line in f: groups.append(json.loads(line))
    by_id = {g["task_id"]: g for g in groups}

    metrics = pd.read_parquet(METRICS)
    rows = []
    for g in groups:
        rs = [t["total_reward"] for t in g["trajectories"]]
        rows.append({
            "task_id": g["task_id"],
            "task_type": g["task_id"].split("-")[0],
            "rewards": rs,
            "var": float(np.var(rs)),
            "won_count": sum(1 for r in rs if r > 0),
        })
    df = pd.DataFrame(rows)
    df["zero_var"] = (df["var"] == 0).astype(int)
    df = df.merge(metrics[["task_id", f"prefix_edit_distance_mean@{K}",
                           f"termination_fraction@{K}"]], on="task_id")
    df = df.rename(columns={f"prefix_edit_distance_mean@{K}": "d_K",
                            f"termination_fraction@{K}": "t_K"})
    df["cut"] = ((df.d_K < 0.12) | (df.t_K >= 0.90)).astype(int)
    df["fp"] = ((df.cut == 1) & (df.zero_var == 0)).astype(int)
    df["fn"] = ((df.cut == 0) & (df.zero_var == 1)).astype(int)
    df["tp"] = ((df.cut == 1) & (df.zero_var == 1)).astype(int)

    fps = df[df.fp == 1].copy()
    print(f"\n=== {len(fps)} False Positives ===")
    for _, row in fps.iterrows():
        g = by_id[row.task_id]
        actions_first_K = []
        for t in g["trajectories"]:
            acts = [s["action"] for s in t["steps"][:K]]
            actions_first_K.append(acts)
        print(f"\n[FP] {row.task_id}")
        print(f"  task_type: {row.task_type}")
        print(f"  rewards: {row.rewards} (won={row.won_count}/8)")
        print(f"  d_K={row.d_K:.3f}, t_K={row.t_K:.2f}")
        print(f"  trigger: " + ("d<0.12" if row.d_K < 0.12 else "")
              + (" AND " if row.d_K < 0.12 and row.t_K >= 0.90 else "")
              + ("t≥0.90" if row.t_K >= 0.90 else ""))
        print(f"  first-{K} actions of trajectory 0: {actions_first_K[0][:5]}")

    # Save summary
    fps[["task_id", "task_type", "rewards", "won_count", "d_K", "t_K"]].to_csv(
        OUT / "false_positives.csv", index=False)
    print(f"\nSaved to {OUT / 'false_positives.csv'}")

    # Now: why did d_K go low for these mixed groups?
    # Hypothesis: trajectories converged on the same wrong prefix early,
    # then 1-2 of them recovered later.
    # Check by: for each FP, count how many trajectories share the EXACT
    # same first-K action sequence
    print("\n=== FP prefix-collapse analysis ===")
    for _, row in fps.iterrows():
        g = by_id[row.task_id]
        prefixes = [tuple(s["action"] for s in t["steps"][:K])
                    for t in g["trajectories"]]
        unique = len(set(prefixes))
        most_common = max([(prefixes.count(p), p) for p in set(prefixes)])
        print(f"  {row.task_id}: unique prefixes = {unique}/8, "
              f"most-common-prefix repeats = {most_common[0]}")


if __name__ == "__main__":
    main()
