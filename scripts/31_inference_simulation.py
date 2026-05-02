"""
Inference-time generalization simulation on the offline 100-rollout buffer.

For self-consistency: G samples per prompt, majority vote on answer.
We treat the binary task-success as the ``answer'' (succeed/fail) and check:

  (a) When the gate fires at step K (d_K < d_L), how many of G trajectories
      would have ended with the same outcome? — this is exactly the
      zero-variance property: if σ_r = 0 then all G have the same reward
      (success or fail) and the majority vote is that single value.
  (b) Wall-clock savings: cut G-1 of G trajectories at step K, let one
      finish to recover the answer. Saving per cut group:
          (G-1) * (T_max - K) / (G * T_max)

Compare with our GRPO saving formula:
          T_max - K) / T_max  (cuts ALL G trajectories of a zero-var group)

For G=8, T_max=30, K=10:
  GRPO saving per cut: (30-10)/30 = 66.7% of group cost
  Self-consistency saving per cut: 7*(30-10)/(8*30) = 58.3% of group cost
  ratio: 58.3 / 66.7 = 87.5%

So GRPO (cutting all G) saves more per cut, but for inference settings the
savings are still ~87% of GRPO. Total dataset-level savings:
  GRPO:           cuts × 66.7% / 100 groups = 17 * 66.7% / 100 = 11.3%
  self-consistency: cuts × 58.3% / 100 groups = 17 * 58.3% / 100 = 9.9%

Outputs:
  results/inference_simulation.csv
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ROLLOUTS = ROOT / "data" / "rollouts.jsonl"
METRICS = ROOT / "data" / "metrics.parquet"
OUT = ROOT / "results"
T_MAX = 30
K = 10
G = 8


def main():
    groups = []
    with open(ROLLOUTS) as f:
        for line in f: groups.append(json.loads(line))
    metrics = pd.read_parquet(METRICS)

    rows = []
    for g in groups:
        rs = [t["total_reward"] for t in g["trajectories"]]
        rows.append({
            "task_id": g["task_id"],
            "rewards": rs,
            "var": float(np.var(rs)),
            "won_count": sum(1 for r in rs if r > 0),
            "lost_count": sum(1 for r in rs if r == 0),
        })
    df = pd.DataFrame(rows)
    df["zero_var"] = (df["var"] == 0).astype(int)
    df = df.merge(metrics[["task_id", f"prefix_edit_distance_mean@{K}"]], on="task_id")
    df = df.rename(columns={f"prefix_edit_distance_mean@{K}": "d_K"})

    # R1 single-axis gate: cut iff d_K < 0.11 (matches our chosen operating point)
    df["cut"] = (df.d_K < 0.11).astype(int)

    n = len(df)
    n_zv = int(df.zero_var.sum())
    n_cut = int(df.cut.sum())
    n_tp = int(((df.cut == 1) & (df.zero_var == 1)).sum())
    n_fp = int(((df.cut == 1) & (df.zero_var == 0)).sum())

    # ===== Self-consistency analysis =====
    # In self-consistency, we want the majority-vote answer.
    # If all G trajectories agree (zero-var), majority vote = that single value.
    # If gate cuts a group, we keep ONE trajectory finishing — its outcome IS
    # the majority vote (since all G agree).
    # FP = group is mixed reward but gate cut anyway; we'd return ONE traj's
    # outcome, which may or may not match the true majority vote.
    # For mixed groups with reward distribution {1: a, 0: b} (a + b = G):
    #   true majority = 1 if a > G/2 else 0
    # If gate cut, we return a random trajectory's outcome; expected accuracy:
    #   P(return correct majority) = max(a, b) / G

    # Check majority-vote accuracy for our gate:
    correct_votes_tp = n_tp  # zero-var groups: trivially correct (all agree)

    correct_votes_fp = 0
    for _, row in df[(df.cut == 1) & (df.zero_var == 0)].iterrows():
        a = row.won_count; b = row.lost_count
        majority = 1 if a > b else 0  # if equal it's a tie; assume conservatively
        # if we cut, we return one rollout's value. expected correctness:
        # = max(a, b) / G  (probability sampled traj agrees with majority)
        correct_votes_fp += max(a, b) / G

    total_correct = correct_votes_tp + correct_votes_fp
    sc_vote_accuracy = total_correct / max(n_cut, 1)

    # GRPO savings
    grpo_save_per_cut = (T_MAX - K) / T_MAX * 1.0  # cuts all G traj
    grpo_total_save_pct = n_cut * grpo_save_per_cut / n * 100

    # Self-consistency savings (cut G-1, let 1 finish)
    sc_save_per_cut = (T_MAX - K) / T_MAX * (G - 1) / G
    sc_total_save_pct = n_cut * sc_save_per_cut / n * 100

    # Best-of-N
    # Same formula as self-consistency (cut G-1, let 1 finish).
    # But for best-of-N the gate's correctness criterion is "the verifier
    # would have picked from these G candidates; if all G are identical,
    # picking any one gives the same answer as best-of-N".

    print(f"=== Inference-time generalization simulation ===")
    print(f"Buffer: {n} groups, {n_zv} zero-variance, {n_cut} gate-cut")
    print(f"  TP: {n_tp}, FP: {n_fp}, precision: {n_tp/max(n_cut,1):.3f}")
    print()
    print(f"Self-consistency (G=8, T_max=30, K=10):")
    print(f"  Cut G-1=7 of 8 trajectories per cut group, let 1 finish.")
    print(f"  Saving per cut: (G-1)/G * (T_max-K)/T_max = 7/8 * 20/30 = {sc_save_per_cut*100:.1f}%")
    print(f"  Total dataset saving: {n_cut} * {sc_save_per_cut*100:.1f}% / {n} = {sc_total_save_pct:.2f}%")
    print(f"  Majority-vote accuracy preservation: {sc_vote_accuracy*100:.1f}%")
    print(f"    (TP cuts: {correct_votes_tp}/{n_cut} trivially correct,")
    print(f"     FP cuts: {correct_votes_fp:.2f}/{n_fp} expected correct via dominant-class sampling)")
    print()
    print(f"For comparison — GRPO (cuts ALL G of zero-var groups):")
    print(f"  Saving per cut: (T_max-K)/T_max = {grpo_save_per_cut*100:.1f}%")
    print(f"  Total dataset saving: {grpo_total_save_pct:.2f}%")
    print(f"  GRPO advantage L2 preservation: 96.7% (from main paper)")
    print()
    print(f"Ratio: self-consistency saving / GRPO saving = {sc_total_save_pct/grpo_total_save_pct*100:.1f}%")
    print()

    # ===== Best-of-N =====
    print(f"Best-of-N (same gate, same construction as self-consistency):")
    print(f"  Saving per cut: identical to self-consistency: {sc_save_per_cut*100:.1f}%")
    print(f"  Total dataset saving: {sc_total_save_pct:.2f}%")
    print(f"  Verifier picks from a single (representative) candidate;")
    print(f"  for zero-var (TP) groups, this is the true best-of-N answer.")

    # ===== Save outputs =====
    out_rows = [
        {"setting": "GRPO training", "save_per_cut_pct": grpo_save_per_cut*100,
         "total_save_pct": grpo_total_save_pct, "answer_preservation_pct": 96.7,
         "note": "cuts all G of zero-var; advantage L2 preserved from offline"},
        {"setting": "self-consistency", "save_per_cut_pct": sc_save_per_cut*100,
         "total_save_pct": sc_total_save_pct, "answer_preservation_pct": sc_vote_accuracy*100,
         "note": "cuts G-1, lets 1 finish; majority vote preservation"},
        {"setting": "best-of-N", "save_per_cut_pct": sc_save_per_cut*100,
         "total_save_pct": sc_total_save_pct, "answer_preservation_pct": sc_vote_accuracy*100,
         "note": "identical mechanic; verifier picks survivor"},
    ]
    pd.DataFrame(out_rows).to_csv(OUT / "inference_simulation.csv", index=False)
    print(f"\nwrote {OUT / 'inference_simulation.csv'}")

    # Generalize: try larger G to see scaling
    print(f"\n=== Scaling G ===")
    print(f"For larger G the (G-1)/G factor approaches 1, making self-consistency")
    print(f"saving approach GRPO saving:")
    for G_v in [4, 8, 16, 32, 64]:
        sc_per_cut = (T_MAX - K) / T_MAX * (G_v - 1) / G_v
        ratio = sc_per_cut / grpo_save_per_cut * 100
        print(f"  G={G_v:2d}: saving per cut = {sc_per_cut*100:.2f}%, "
              f"ratio to GRPO = {ratio:.1f}%")


if __name__ == "__main__":
    main()
