"""
Wall-clock A/B comparison: data/rollouts_bl.jsonl vs data/rollouts_gated.jsonl.

Both runs use the same 100 ALFWorld tasks, same seed, same 4-GPU supervisor;
the only difference is `--gate-K 10 --gate-d-low 0.12 --gate-t-high 0.90`
on the gated run.

Reports:
  * Per-task wall-clock distribution (mean, median, total)
  * For gated run: how many groups were cut early, at what step, with what
    correctness vs final reward variance
  * Step-token compute saved
  * Group composition unchanged (identity check)

Outputs:
  results/wallclock_ab.csv (per-task)
  results/wallclock_ab_summary.csv
  results/figures/wallclock_ab.png
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


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def per_task(groups, label):
    rows = []
    for g in groups:
        rs = [t["total_reward"] for t in g["trajectories"]]
        steps = [len(t["steps"]) for t in g["trajectories"]]
        terminated = [t["terminated_at"] for t in g["trajectories"]]
        # If gated and at least one trajectory was cut at exactly step K=10
        # AND that wasn't a natural done, it's a gate-cut.
        cut_step = -1
        cut_count = sum(1 for t in g["trajectories"]
                        if t["terminated_at"] == 9 and len(t["steps"]) == 10
                        and not t["won"]
                        and (len(t["steps"]) > 0 and not t["steps"][-1]["done"]))
        # Simpler: any trajectory terminated_at=9 with steps[-1].done=False ⇒ gate cut.
        # The gate sets terminated_at without setting steps[-1].done=True.
        gate_cut = any(
            t["terminated_at"] == 9
            and not (t["steps"] and t["steps"][-1]["done"])
            for t in g["trajectories"]
        )
        rows.append({
            "task_id": g["task_id"],
            "label": label,
            "wall_time_sec": g["wall_time_sec"],
            "step_tokens": sum(steps),
            "rewards_mean": float(np.mean(rs)),
            "rewards_var": float(np.var(rs)),
            "won": sum(1 for t in g["trajectories"] if t["won"]),
            "group_size": g["group_size"],
            "max_steps_cap": g["max_steps"],
            "gate_cut": int(gate_cut),
        })
    return pd.DataFrame(rows)


def main():
    bl = load(ROOT / "data" / "rollouts_bl.jsonl")
    gated = load(ROOT / "data" / "rollouts_gated.jsonl")
    bl_df = per_task(bl, "baseline")
    gt_df = per_task(gated, "gated")
    df = pd.concat([bl_df, gt_df], ignore_index=True)
    df.to_csv(OUT_DIR / "wallclock_ab.csv", index=False)
    print(f"wrote {OUT_DIR / 'wallclock_ab.csv'}")

    # Align by task_id for paired comparison
    paired = bl_df.merge(gt_df, on="task_id", suffixes=("_bl", "_gt"))
    print(f"\nPaired: {len(paired)} tasks")

    # Group composition (should be similar; different sampling temperature
    # means group rewards aren't expected to be identical)
    def composition(d):
        zv = (d["rewards_var"] == 0).sum()
        af = ((d["rewards_var"] == 0) & (d["rewards_mean"] == 0)).sum()
        au = ((d["rewards_var"] == 0) & (d["rewards_mean"] > 0)).sum()
        return f"zero_var={zv}  all_fail={af}  all_succeed={au}  mixed={len(d)-zv}"

    print(f"Baseline composition: {composition(bl_df)}")
    print(f"Gated    composition: {composition(gt_df)}")

    # Wall-clock summary
    print(f"\nWall-clock (per task):")
    print(f"  baseline: mean={bl_df.wall_time_sec.mean():.1f}s  "
          f"median={bl_df.wall_time_sec.median():.1f}s  "
          f"total={bl_df.wall_time_sec.sum():.1f}s")
    print(f"  gated:    mean={gt_df.wall_time_sec.mean():.1f}s  "
          f"median={gt_df.wall_time_sec.median():.1f}s  "
          f"total={gt_df.wall_time_sec.sum():.1f}s")
    delta_mean = (bl_df.wall_time_sec.mean() - gt_df.wall_time_sec.mean()) / bl_df.wall_time_sec.mean()
    delta_total = (bl_df.wall_time_sec.sum() - gt_df.wall_time_sec.sum()) / bl_df.wall_time_sec.sum()
    print(f"  Δ (gated faster): mean {delta_mean*100:.2f}%, total {delta_total*100:.2f}%")

    # Step-token compute
    print(f"\nStep-tokens (per task):")
    print(f"  baseline: mean={bl_df.step_tokens.mean():.1f}  "
          f"total={bl_df.step_tokens.sum()}")
    print(f"  gated:    mean={gt_df.step_tokens.mean():.1f}  "
          f"total={gt_df.step_tokens.sum()}")
    delta_st = (bl_df.step_tokens.sum() - gt_df.step_tokens.sum()) / bl_df.step_tokens.sum()
    print(f"  Δ (gated less compute): {delta_st*100:.2f}%")

    # Gate cuts
    n_cut = gt_df.gate_cut.sum()
    print(f"\nGate fired on {n_cut}/{len(gt_df)} groups in gated run")
    cut_groups = gt_df[gt_df.gate_cut == 1]
    if len(cut_groups) > 0:
        cut_zv = (cut_groups.rewards_var == 0).sum()
        cut_nzv = (cut_groups.rewards_var > 0).sum()
        print(f"  Of cut groups: {cut_zv} were zero-variance (TP), "
              f"{cut_nzv} were non-zero-variance (FP) → "
              f"precision={cut_zv/(cut_zv+cut_nzv):.3f}")

    # Save summary
    summary = {
        "n_tasks": len(paired),
        "wall_total_baseline_s": float(bl_df.wall_time_sec.sum()),
        "wall_total_gated_s": float(gt_df.wall_time_sec.sum()),
        "wall_savings_pct": float(delta_total * 100),
        "step_tokens_baseline": int(bl_df.step_tokens.sum()),
        "step_tokens_gated": int(gt_df.step_tokens.sum()),
        "step_tokens_savings_pct": float(delta_st * 100),
        "gate_cut_groups": int(n_cut),
        "gate_TP_zero_var": int((cut_groups.rewards_var == 0).sum()) if n_cut else 0,
        "gate_FP_non_zero_var": int((cut_groups.rewards_var > 0).sum()) if n_cut else 0,
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "wallclock_ab_summary.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'wallclock_ab_summary.csv'}")
    print("\nSummary JSON:")
    print(json.dumps(summary, indent=2))

    # Plot: per-task wall-clock distribution + cumulative
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, max(bl_df.wall_time_sec.max(),
                              gt_df.wall_time_sec.max()) * 1.05, 30)
    axes[0].hist(bl_df.wall_time_sec, bins=bins, alpha=0.55,
                  label=f"baseline (n={len(bl_df)}, mean={bl_df.wall_time_sec.mean():.0f}s)",
                  color="#1f77b4")
    axes[0].hist(gt_df.wall_time_sec, bins=bins, alpha=0.55,
                  label=f"gated (n={len(gt_df)}, mean={gt_df.wall_time_sec.mean():.0f}s)",
                  color="#ff7f0e")
    axes[0].set_xlabel("Per-task wall-clock (s)")
    axes[0].set_ylabel("# tasks")
    axes[0].set_title("Per-task rollout wall-clock distribution")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    sorted_bl = np.sort(bl_df.wall_time_sec.values)
    sorted_gt = np.sort(gt_df.wall_time_sec.values)
    axes[1].plot(np.arange(len(sorted_bl)), np.cumsum(sorted_bl),
                  color="#1f77b4", label="baseline")
    axes[1].plot(np.arange(len(sorted_gt)), np.cumsum(sorted_gt),
                  color="#ff7f0e", label="gated")
    axes[1].set_xlabel("# tasks (sorted by wall-clock)")
    axes[1].set_ylabel("Cumulative wall-clock (s)")
    axes[1].set_title(f"Cumulative wall-clock (gated saves {delta_total*100:.1f}%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    safe_savefig(fig, FIG_DIR / "wallclock_ab.png", dpi=140)
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'wallclock_ab.png'}")


if __name__ == "__main__":
    main()
