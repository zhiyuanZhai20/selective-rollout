"""
Hybrid gate analysis.

Question: among groups with HIGH prefix_edit_distance_mean (where the simple
gate fails because all-fail "chaotic" groups look like mixed groups), can a
secondary feature separate all-fail from mixed?

Strategy:
- all-fail = no traj got reward => no traj is "making progress"
- mixed    = at least one traj is on a productive path

So the discriminator should reflect "is anyone in the group actually doing
something productive vs. just wandering?". Candidate features at step K:
  * max_prod_action_count_K     — max over trajs of "productive verb" count
  * max_unique_obs_count_K      — max over trajs of unique observations
  * max_unique_action_count_K   — max over trajs of unique actions
  * intra_traj_repeat_rate_K    — mean repetition rate per traj (looped behavior)
  * navigation_fraction_K       — fraction of first-K actions that are pure 'go'/'move'

We then evaluate each as a secondary gate among high-div groups.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ROLLOUTS = ROOT / "data" / "rollouts.jsonl"
METRICS = ROOT / "data" / "metrics.parquet"
OUT_CSV = ROOT / "results" / "hybrid_gate_features.csv"

# productive verbs: things that interact with objects toward task completion
PROD = {"take", "put", "use", "clean", "heat", "cool", "slice",
        "open", "close", "examine", "look", "inventory"}
NAV = {"go", "move"}


def first_word(a: str) -> str:
    s = a.strip().split(maxsplit=1)
    return s[0].lower() if s else ""


def features_at_K(group: dict, K: int) -> dict:
    """Per-group features computed from first K steps of each trajectory."""
    trajs = group["trajectories"]
    G = len(trajs)
    per_traj_actions = []
    per_traj_obs = []
    for t in trajs:
        steps = t["steps"][:K]
        per_traj_actions.append([s["action"] for s in steps])
        per_traj_obs.append([s["obs"] for s in steps])

    # max productive action count across trajs
    prod_counts = []
    nav_counts = []
    repeat_rates = []
    unique_obs_counts = []
    unique_action_counts = []
    for acts, obs in zip(per_traj_actions, per_traj_obs):
        if not acts:
            prod_counts.append(0); nav_counts.append(0)
            repeat_rates.append(0); unique_obs_counts.append(0)
            unique_action_counts.append(0)
            continue
        verbs = [first_word(a) for a in acts]
        prod_counts.append(sum(1 for v in verbs if v in PROD))
        nav_counts.append(sum(1 for v in verbs if v in NAV))
        # repeat rate: 1 - unique/total of full action strings
        n = len(acts)
        u = len(set(acts))
        repeat_rates.append(1 - u / n if n else 0)
        unique_obs_counts.append(len(set(obs)))
        unique_action_counts.append(u)

    # group-level summaries; want to capture "is anyone productive?"
    return {
        "K": K,
        "max_prod_K": max(prod_counts),
        "mean_prod_K": np.mean(prod_counts),
        "max_unique_obs_K": max(unique_obs_counts),
        "mean_unique_obs_K": np.mean(unique_obs_counts),
        "max_unique_action_K": max(unique_action_counts),
        "min_repeat_K": min(repeat_rates),  # lowest = most progressing
        "mean_repeat_K": np.mean(repeat_rates),
        "max_repeat_K": max(repeat_rates),  # highest = most stuck
        "nav_fraction_K": sum(nav_counts) / max(1, sum(prod_counts) + sum(nav_counts)),
    }


def label_group(group: dict) -> str:
    rs = [t["total_reward"] for t in group["trajectories"]]
    n = len(rs); won = sum(1 for r in rs if r > 0)
    if won == 0: return "all_fail"
    if won == n: return "all_succeed"
    return "mixed"


def reward_var(group):
    rs = [t["total_reward"] for t in group["trajectories"]]
    return float(np.var(rs))


def main():
    groups = []
    with open(ROLLOUTS) as f:
        for line in f:
            groups.append(json.loads(line))

    metrics = pd.read_parquet(METRICS)

    rows = []
    for g in groups:
        tid = g["task_id"]
        lab = label_group(g)
        rv = reward_var(g)
        for K in (10, 15, 20):
            f = features_at_K(g, K)
            f["task_id"] = tid
            f["label"] = lab
            f["reward_var"] = rv
            rows.append(f)
    feat_df = pd.DataFrame(rows)

    # Reshape metrics from wide (col@K) to long
    div_metric_names = ["unique_action_ratio", "unique_prefix_ratio",
                        "action_bigram_jaccard_mean", "prefix_edit_distance_mean",
                        "obs_unique_ratio", "termination_fraction", "action_entropy"]
    long_rows = []
    for _, r in metrics.iterrows():
        for K in (10, 15, 20):
            d = {"task_id": r["task_id"], "K": K, "task_type": r["task_type"]}
            for m in div_metric_names:
                col = f"{m}@{K}"
                if col in metrics.columns:
                    d[m] = r[col]
            long_rows.append(d)
    metrics_long = pd.DataFrame(long_rows)
    merged = feat_df.merge(metrics_long, on=["task_id", "K"], how="left",
                            suffixes=("", "_div"))
    merged.to_csv(OUT_CSV, index=False)

    print()
    print("=== global correlations (vs reward_var) ===")
    feature_cols = [c for c in merged.columns
                    if c not in {"task_id", "K", "label", "reward_var"}
                    and pd.api.types.is_numeric_dtype(merged[c])]
    for K in (10, 15, 20):
        sub = merged[merged.K == K]
        rows = []
        for c in feature_cols:
            x = sub[c].values
            y = sub.reward_var.values
            if np.std(x) == 0:
                continue
            rho, p = spearmanr(x, y)
            try:
                # AUROC: predict zero-var (var==0) vs not. positive class = non-zero
                y_bin = (y > 0).astype(int)
                if len(set(y_bin)) > 1:
                    # higher x predicts non-zero? sign by spearman
                    auroc = roc_auc_score(y_bin, x if rho >= 0 else -x)
                else:
                    auroc = float("nan")
            except Exception:
                auroc = float("nan")
            rows.append((c, rho, p, auroc))
        df = pd.DataFrame(rows, columns=["feature", "rho", "p", "AUROC"])
        df = df.reindex(df.AUROC.abs().sort_values(ascending=False).index)
        print(f"\n--- K={K} ---")
        print(df.head(10).to_string(index=False))

    # The key: among HIGH div groups, can we separate all-fail from mixed?
    print("\n\n=== HIGH-div subset analysis (K=15) ===")
    K = 15
    sub = merged[merged.K == K].copy()
    high = sub[sub.prefix_edit_distance_mean > 0.4]
    print(f"high-div groups (prefix_edit_dist_mean>0.4): {len(high)}")
    print(high.label.value_counts().to_dict())

    # For each candidate feature, AUROC mixed vs all-fail
    print("\nAUROC: mixed (positive) vs all-fail (negative), within high-div groups")
    rows = []
    for c in feature_cols:
        h = high[high.label.isin(["mixed", "all_fail"])]
        x = h[c].values
        y = (h.label.values == "mixed").astype(int)
        if np.std(x) == 0 or len(set(y)) < 2:
            continue
        try:
            auroc = roc_auc_score(y, x)
            auroc = max(auroc, 1 - auroc)  # symmetric: pick best direction
        except Exception:
            continue
        rows.append((c, auroc))
    out = pd.DataFrame(rows, columns=["feature", "AUROC_mixed_vs_allfail"])
    out = out.sort_values("AUROC_mixed_vs_allfail", ascending=False).reset_index(drop=True)
    print(out.head(15).to_string(index=False))

    # Now propose & evaluate the hybrid gate:
    #   gate_low: prefix_edit_distance_mean < 0.10  -> CUT (mostly all_succeed + coordinated all-fail)
    #   gate_high: prefix_edit_distance_mean > 0.4 AND <best_secondary_feature> in <bad direction> -> CUT
    print("\n\n=== Hybrid gate evaluation @ K=15 ===")
    pdm = sub.prefix_edit_distance_mean.values
    is_zero = (sub.reward_var.values == 0).astype(int)

    # candidate secondary features (predict all_fail among high-div groups)
    candidates = [
        "max_prod_K", "mean_prod_K", "max_unique_obs_K", "mean_unique_obs_K",
        "max_unique_action_K", "min_repeat_K", "max_repeat_K",
        "nav_fraction_K", "termination_fraction",
    ]
    print(f"baseline single-threshold gate (cut iff prefix_edit_dist_mean < 0.1):")
    sel = pdm < 0.1
    cut = sel.sum()
    correct_cuts = (sel & (is_zero == 1)).sum()
    wrong_cuts = (sel & (is_zero == 0)).sum()
    print(f"  cut: {cut}, correctly cut zero-var: {correct_cuts}/{is_zero.sum()}, wrongly cut non-zero: {wrong_cuts}/{(is_zero==0).sum()}")

    print("\nadd a high-div secondary cut: cut iff (pdm < 0.10) OR (pdm > 0.4 AND <feat> below threshold)")
    print(f"{'feature':30s}  {'thr':>6s}  {'extra_cut':>10s}  {'precision':>10s}  {'recall_total':>12s}")
    for c in candidates:
        # find threshold maximizing precision*coverage among high-div all-fail
        h = sub[sub.prefix_edit_distance_mean > 0.4].copy()
        if c not in h.columns: continue
        # Try several thresholds, pick best F1-ish
        x = h[c].values
        y = (h.reward_var.values == 0).astype(int)
        # Sort and sweep
        order = np.argsort(x)
        best = None
        for k in range(2, len(order) - 1):
            thr = x[order[k]]
            # decide direction: if more zero-var on lower side, "cut iff x <= thr"
            low_mask = x <= thr
            high_mask = ~low_mask
            for direction in (-1, 1):
                if direction == -1:
                    pred_zero = low_mask
                else:
                    pred_zero = high_mask
                tp = ((pred_zero) & (y == 1)).sum()
                fp = ((pred_zero) & (y == 0)).sum()
                tn = ((~pred_zero) & (y == 0)).sum()
                fn = ((~pred_zero) & (y == 1)).sum()
                if tp + fp == 0: continue
                prec = tp / (tp + fp)
                cov = tp  # number cut correctly in high-div segment
                # don't accept if FP rate > 20%
                if fp + tn > 0 and fp / (fp + tn) > 0.2: continue
                score = prec * cov
                if best is None or score > best[0]:
                    best = (score, thr, direction, prec, tp, fp, fn, tn)
        if best is None:
            continue
        score, thr, direction, prec, tp, fp, fn, tn = best
        # Now evaluate full hybrid gate on full population
        # gate fires if (pdm < 0.10) OR (pdm > 0.4 AND secondary fires)
        secondary_fires = ((sub[c].values <= thr) if direction == -1
                           else (sub[c].values >= thr))
        full_pred_zero = (sub.prefix_edit_distance_mean.values < 0.1) | (
            (sub.prefix_edit_distance_mean.values > 0.4) & secondary_fires)
        TP = (full_pred_zero & (is_zero == 1)).sum()
        FP = (full_pred_zero & (is_zero == 0)).sum()
        recall = TP / max(1, is_zero.sum())
        precision = TP / max(1, TP + FP)
        sign = "<=" if direction == -1 else ">="
        print(f"{c:30s}  {thr:>6.3f}  {sign} cuts {TP - sel.sum():>+5d}  "
              f"prec={precision:>5.3f}  recall={recall:>5.3f}  "
              f"({TP}/{is_zero.sum()} zero-var, {FP} mistakes)")


if __name__ == "__main__":
    main()
