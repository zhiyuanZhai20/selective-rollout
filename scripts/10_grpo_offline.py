"""
Phase C.0 — offline GRPO gradient-loss analysis.

Question: when the gate cuts a group at step K, how much GRPO gradient signal
is lost?

GRPO advantage for trajectory i in group g (rewards r_{g,1..G}):
    A_{g,i} = (r_{g,i} - mean(r_g)) / std(r_g)
For zero-variance groups, std=0 ⇒ advantage is undefined / set to 0.
For mixed groups, advantages are finite and meaningful.

If the gate cuts a TRUE zero-var group, A is already 0 — no signal lost.
If the gate cuts a FALSE positive (mixed group), the signal contributed
by that group is lost.

We quantify three things:
  1. Compute saved (% of total rollout step-tokens):
        sum over cut groups of G * (mean_steps_in_group - K)  /  total_steps
  2. Gradient information preserved (relative L2 norm):
        ||A_kept|| / ||A_all||
     where ||·|| is sqrt of sum of squared advantages over all (g, i).
  3. Per-FP cost: for each FP group, what was its |Σ A_{g,i}|?
     (Should be 0 by GRPO definition, since advantages sum to 0 within a group.
     But the SQUARED contribution is nonzero — that's what GRPO loss uses.)

Outputs:
    results/grpo_offline.csv (one row per gate config)
    results/figures/grpo_offline_tradeoff.png
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


def load():
    groups = []
    with open(ROOT / "data" / "rollouts.jsonl") as f:
        for line in f:
            groups.append(json.loads(line))
    metrics = pd.read_parquet(ROOT / "data" / "metrics.parquet")
    metrics_by_id = metrics.set_index("task_id")
    return groups, metrics_by_id


def grpo_advantages(rewards: np.ndarray) -> np.ndarray:
    """Standard GRPO group-relative advantage: (r - mean) / (std + eps)."""
    mu = rewards.mean()
    sigma = rewards.std()
    eps = 1e-8
    return (rewards - mu) / (sigma + eps)


def total_steps(group: dict) -> int:
    """Sum of steps across all G trajectories (compute proxy)."""
    return sum(len(t["steps"]) for t in group["trajectories"])


def cut_compute_saved(group: dict, K: int) -> int:
    """If we cut this group at step K, how many step-tokens are saved?

    For each trajectory: max(len(steps) - K, 0) tokens saved (we already paid K
    steps before deciding to cut).
    """
    saved = 0
    for t in group["trajectories"]:
        n_steps = len(t["steps"])
        if n_steps > K:
            saved += n_steps - K
    return saved


def evaluate_gate(groups, metrics_by_id, K, dl, th):
    """Apply gate (div<dl OR term>=th) at step K to all groups.

    Returns dict with:
      compute_saved_frac, grad_norm_preserved, n_cut, n_TP_cut, n_FP_cut,
      grad_lost_FP, total_steps, total_steps_saved
    """
    rows = []
    grad_all = []  # all advantages across all groups
    grad_kept = []  # advantages from groups not cut by gate
    grad_lost_fp = 0.0  # squared-advantage lost from FP cuts
    total_compute = 0
    saved_compute = 0
    n_cut = n_tp = n_fp = 0

    for g in groups:
        rewards = np.array([t["total_reward"] for t in g["trajectories"]])
        is_zero_var = float(rewards.var()) == 0.0
        adv = grpo_advantages(rewards)
        grad_all.append(adv)

        m = metrics_by_id.loc[g["task_id"]]
        div_K = m[f"prefix_edit_distance_mean@{K}"]
        term_K = m[f"termination_fraction@{K}"]
        cut = (div_K < dl) or (term_K >= th)

        steps_g = total_steps(g)
        total_compute += steps_g

        if cut:
            n_cut += 1
            saved_compute += cut_compute_saved(g, K)
            if is_zero_var:
                n_tp += 1  # correct cut (no gradient loss anyway)
            else:
                n_fp += 1
                grad_lost_fp += float((adv ** 2).sum())
        else:
            grad_kept.append(adv)

    A_all = np.concatenate(grad_all)
    A_kept = np.concatenate(grad_kept) if grad_kept else np.zeros(0)
    norm_all = float(np.sqrt((A_all ** 2).sum()))
    norm_kept = float(np.sqrt((A_kept ** 2).sum()))
    norm_preserved = norm_kept / max(norm_all, 1e-8)

    return {
        "K": K, "d_low": dl, "t_high": th,
        "n_cut": n_cut, "n_TP": n_tp, "n_FP": n_fp,
        "compute_saved_frac": saved_compute / total_compute,
        "compute_total_steps": total_compute,
        "compute_saved_steps": saved_compute,
        "grad_norm_all": norm_all,
        "grad_norm_kept": norm_kept,
        "grad_norm_preserved": norm_preserved,
        "grad_lost_sqsum_fp": grad_lost_fp,
    }


def main():
    groups, metrics_by_id = load()
    print(f"Loaded {len(groups)} groups")

    # All advantages baseline
    all_adv = []
    for g in groups:
        rewards = np.array([t["total_reward"] for t in g["trajectories"]])
        all_adv.append(grpo_advantages(rewards))
    A = np.concatenate(all_adv)
    print(f"Total trajectories: {len(A)}")
    print(f"Total |A| (sum of |adv|): {np.abs(A).sum():.3f}")
    print(f"Total advantage L2 norm: {np.sqrt((A**2).sum()):.3f}")

    # Sweep gate configs
    configs = [
        # (label, K, dl, th)
        ("R1_K10_div0.11", 10, 0.11, 1e9),  # div only
        ("R1_K10_div0.12", 10, 0.12, 1e9),
        ("R3_K10_div0.12_th0.90", 10, 0.12, 0.90),  # winner
        ("R3_K15_div0.24_th0.90", 15, 0.24, 0.90),
        ("R3_K20_div0.30_th0.90", 20, 0.30, 0.90),
    ]
    rows = []
    for label, K, dl, th in configs:
        r = evaluate_gate(groups, metrics_by_id, K=K, dl=dl, th=th)
        r["label"] = label
        rows.append(r)
        print(f"\n{label}:")
        print(f"  cut {r['n_cut']} (TP={r['n_TP']}, FP={r['n_FP']})")
        print(f"  compute saved: {r['compute_saved_frac']*100:.2f}% "
              f"({r['compute_saved_steps']} / {r['compute_total_steps']} step-tokens)")
        print(f"  grad L2 preserved: {r['grad_norm_preserved']*100:.2f}%  "
              f"(kept {r['grad_norm_kept']:.3f} / all {r['grad_norm_all']:.3f})")
        print(f"  grad sqsum lost from FPs: {r['grad_lost_sqsum_fp']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "grpo_offline.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'grpo_offline.csv'}")

    # Plot: compute saved vs grad norm preserved
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cs = df["compute_saved_frac"].values * 100
    gp = df["grad_norm_preserved"].values * 100
    ax.scatter(cs, gp, s=120, c=range(len(df)), cmap="viridis", zorder=3)
    for x, y, lab in zip(cs, gp, df["label"]):
        ax.annotate(lab.replace("R3_", "").replace("R1_", ""),
                    (x, y), fontsize=8,
                    xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Compute saved (% of total rollout step-tokens)")
    ax.set_ylabel("GRPO advantage L2 norm preserved (%)")
    ax.set_title("Gate trade-off: compute saved vs gradient signal preserved")
    ax.axhline(100, ls=":", c="gray", alpha=0.5, label="ideal: 100% gradient preserved")
    ax.set_xlim(0, max(cs) * 1.2)
    ax.set_ylim(min(gp) - 2, 102)
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    safe_savefig(fig, FIG_DIR / "grpo_offline_tradeoff.png", dpi=140)
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'grpo_offline_tradeoff.png'}")


if __name__ == "__main__":
    main()
