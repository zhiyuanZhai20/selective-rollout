"""
Consume rollouts.jsonl, produce one row per group with divergence@K for all K
in {5, 10, 15, 20}, plus the final reward variance and mean.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from src.divergence import divergence_at_K, group_reward_stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Ks", default="5,10,15,20")
    args = ap.parse_args()

    Ks = [int(x) for x in args.Ks.split(",")]

    rows = []
    with open(args.inp) as f:
        for line in f:
            g = json.loads(line)
            trajs = g["trajectories"]
            actions_per_traj = [[s["action"] for s in t["steps"]] for t in trajs]
            obs_per_traj = [[s["obs"] for s in t["steps"]] for t in trajs]
            rewards = [t["total_reward"] for t in trajs]
            lens = [len(a) for a in actions_per_traj]

            row = {
                "task_id": g["task_id"],
                "task_type": g["task_id"].split("/")[-2].split("-")[0]
                if "/" in g["task_id"] else g["task_id"].split("-")[0],
                "mean_len": sum(lens) / len(lens),
            }
            row.update(group_reward_stats(rewards))
            for K in Ks:
                metrics = divergence_at_K(actions_per_traj, obs_per_traj, K=K)
                for m, v in metrics.items():
                    row[f"{m}@{K}"] = v
            rows.append(row)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.out.endswith(".parquet"):
        df.to_parquet(args.out, index=False)
    else:
        df.to_csv(args.out, index=False)
    print(f"[metrics] {len(df)} rows  ->  {args.out}")
    print(df.describe().round(3).to_string())


if __name__ == "__main__":
    main()
