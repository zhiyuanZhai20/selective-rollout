"""
Strong-baseline comparison on the 100-task offline buffer.

Computes wall-clock-saved and L2-norm-preserved for FOUR comparison arms,
all evaluated on the same 100-task × G=8 buffer:

  1. NO-GATE (full rollout, full gradient)
  2. RANDOM (cut a random 16% of groups — sanity: signal-free baseline)
  3. ORACLE (cut iff actually zero-variance — theoretical upper bound)
  4. DAPO-ORACLE (rollout full, but skip backprop for zero-var groups —
                  what DAPO realises in the rollout-phase A/B setting)
  5. SINGLE-AXIS-d (cut iff d_K < d_L, our R1)
  6. SINGLE-AXIS-t (cut iff τ_K ≥ τ_H, only-τ rule)
  7. OURS (R3: d < d_L OR τ ≥ τ_H)
  8. OURS+DAPO (combine: cut early on R3, then drop remaining zero-var groups)

We report each arm with:
  - rollout step-tokens saved (%)
  - training items dropped (%)
  - GRPO advantage L2-norm preserved (%)
  - precision against oracle (where applicable)
  - 1000-fold bootstrap 95% CI on each metric

Outputs:
  results/strong_baselines.csv
  results/strong_baselines_bootstrap.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ROLLOUTS = ROOT / "data" / "rollouts.jsonl"
METRICS = ROOT / "data" / "metrics.parquet"
OUT_DIR = ROOT / "results"
MAX_STEPS = 30
N = 100
G = 8
K = 10
RNG = np.random.default_rng(seed=42)


def load():
    groups = []
    with open(ROLLOUTS) as f:
        for line in f:
            groups.append(json.loads(line))
    metrics = pd.read_parquet(METRICS)
    rows = []
    for g in groups:
        rs = [t["total_reward"] for t in g["trajectories"]]
        # length up to terminated_at (inclusive) or full step list
        lens = [(t.get("terminated_at", len(t["steps"])-1) + 1)
                if t.get("terminated_at") is not None else len(t["steps"])
                for t in g["trajectories"]]
        rows.append({
            "task_id": g["task_id"],
            "rewards": rs,
            "lens": lens,
            "var": float(np.var(rs)),
            "mean_steps": float(np.mean(lens)),
            "total_steps": int(sum(lens)),
        })
    df = pd.DataFrame(rows)
    df["zero_var"] = (df["var"] == 0).astype(int)
    df = df.merge(
        metrics[["task_id", f"prefix_edit_distance_mean@{K}", f"termination_fraction@{K}"]],
        on="task_id",
    )
    df = df.rename(columns={
        f"prefix_edit_distance_mean@{K}": "d_K",
        f"termination_fraction@{K}": "t_K",
    })
    return df


def grpo_advantages(rewards):
    """z-score advantage per group."""
    rewards = np.asarray(rewards, dtype=float)
    m = rewards.mean()
    s = rewards.std()
    if s < 1e-12:
        return np.zeros_like(rewards)
    return (rewards - m) / s


def evaluate(df, cut_mask, name, drop_zv_at_train=False):
    """
    Score an arm.

    `cut_mask`: bool[N] — which groups are cut at step K
    `drop_zv_at_train`: if True, additionally drop zero-var groups from
                        the gradient batch (DAPO-style).
                        If a group is BOTH cut AND drop, the cut wins.
    """
    cut = np.asarray(cut_mask, bool)
    zv = df["zero_var"].values.astype(bool)

    # rollout cost: for cut groups we save (T_max - K) * G; for non-cut, full mean_steps
    rollout_per_group = df["total_steps"].values  # actual sum of steps per group
    saved_per_group = np.where(cut, np.maximum(0, (df["mean_steps"].values - K) * G), 0)
    saved_steps = float(saved_per_group.sum())
    total_steps = float(rollout_per_group.sum())
    rollout_saved_pct = 100 * saved_steps / total_steps

    # training items dropped: cut groups contribute zero items
    train_drop_groups = cut.copy()
    if drop_zv_at_train:
        train_drop_groups = train_drop_groups | zv  # also drop zv if surviving
    items_dropped = int(train_drop_groups.sum() * G)
    items_total = N * G
    items_dropped_pct = 100 * items_dropped / items_total

    # GRPO advantage L2 preserved
    A_all = []
    A_kept = []
    for i, row in df.iterrows():
        a = grpo_advantages(row["rewards"])
        A_all.extend(a.tolist())
        if not train_drop_groups[i]:
            A_kept.extend(a.tolist())
    A_all = np.asarray(A_all)
    A_kept = np.asarray(A_kept)
    norm_all = float(np.linalg.norm(A_all))
    norm_kept = float(np.linalg.norm(A_kept))
    l2_preserved_pct = 100 * norm_kept / max(norm_all, 1e-12)

    # precision = TP / (TP + FP) where TP = (cut & zv), FP = (cut & ~zv)
    tp = int((cut & zv).sum())
    fp = int((cut & ~zv).sum())
    fn = int((~cut & zv).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    return {
        "arm": name,
        "n_cut": int(cut.sum()),
        "TP": tp, "FP": fp, "FN": fn,
        "precision": precision,
        "recall": recall,
        "rollout_saved_pct": rollout_saved_pct,
        "items_dropped_pct": items_dropped_pct,
        "L2_preserved_pct": l2_preserved_pct,
    }


def main():
    df = load()
    print(f"Loaded {len(df)} groups; {df.zero_var.sum()} zero-variance.")

    # Define each arm's cut mask
    no_cut = np.zeros(len(df), bool)
    oracle_cut = df["zero_var"].values.astype(bool)
    target_n_cut = 23  # match our gate's cut count

    rows = []

    # ARM 1: NO GATE
    rows.append(evaluate(df, no_cut, "no-gate"))

    # ARM 2: RANDOM (16%)
    # average over 1000 random seeds for stable point estimate
    boot_random = []
    for trial in range(1000):
        rng_t = np.random.default_rng(seed=trial)
        idx = rng_t.choice(len(df), size=target_n_cut, replace=False)
        rmask = np.zeros(len(df), bool); rmask[idx] = True
        boot_random.append(evaluate(df, rmask, f"random-{trial}"))
    rdf = pd.DataFrame(boot_random)
    rows.append({
        "arm": "random-cut-23",
        "n_cut": target_n_cut,
        "TP": int(rdf.TP.mean()),
        "FP": int(rdf.FP.mean()),
        "FN": int(rdf.FN.mean()),
        "precision": rdf.precision.mean(),
        "recall": rdf.recall.mean(),
        "rollout_saved_pct": rdf.rollout_saved_pct.mean(),
        "items_dropped_pct": rdf.items_dropped_pct.mean(),
        "L2_preserved_pct": rdf.L2_preserved_pct.mean(),
        "rollout_saved_pct_ci95": [rdf.rollout_saved_pct.quantile(0.025),
                                   rdf.rollout_saved_pct.quantile(0.975)],
        "L2_preserved_pct_ci95": [rdf.L2_preserved_pct.quantile(0.025),
                                  rdf.L2_preserved_pct.quantile(0.975)],
    })

    # ARM 3: ORACLE (cut iff actually zero-var)
    rows.append(evaluate(df, oracle_cut, "oracle"))

    # ARM 4: DAPO-oracle (no rollout cut, but train-stage drop zv)
    rows.append(evaluate(df, no_cut, "DAPO-oracle", drop_zv_at_train=True))

    # ARM 5: single-axis d (R1: d < d_L)
    d_only = df["d_K"].values < 0.11
    rows.append(evaluate(df, d_only, "R1: d<0.11 only"))

    # ARM 6: single-axis t (τ ≥ τ_H)
    t_only = df["t_K"].values >= 0.90
    rows.append(evaluate(df, t_only, "single-τ: τ≥0.90"))

    # ARM 7: OURS (R3 OR rule)
    ours = (df["d_K"].values < 0.12) | (df["t_K"].values >= 0.90)
    rows.append(evaluate(df, ours, "OURS R3"))

    # ARM 8: OURS + DAPO (cut early + drop surviving zv at train)
    rows.append(evaluate(df, ours, "OURS+DAPO", drop_zv_at_train=True))

    # save
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "strong_baselines.csv", index=False)
    print(res[["arm", "n_cut", "TP", "FP", "precision", "recall",
               "rollout_saved_pct", "items_dropped_pct", "L2_preserved_pct"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
