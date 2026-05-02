"""
Multi-seed aggregation for the on-policy A/B.

Aggregates runs/onpolicy_{baseline,gated}{,_s7,_s13,...}/{train,eval}.jsonl
across all available seeds and reports:

  - mean ± std across seeds for: total wall-clock, rollout-saved %,
    final held-out eval %, gradient L2 norm
  - per-iter held-out trajectory mean ± std
  - paired-difference test (gated minus baseline) per seed

Outputs:
  results/multiseed_summary.csv
  results/figures/multiseed_eval.png
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
import sys; sys.path.insert(0, str(ROOT))
from src.figutil import safe_savefig

OUT = ROOT / "results"
FIG = OUT / "figures"
RUNS = ROOT / "runs"


def find_runs():
    """Returns dict[seed_str -> (baseline_dir, gated_dir)].

    Recognises directory names: onpolicy_{baseline,gated}[_sN][_R1].
    For R1 runs we prefer the _R1 gated dir over the original (R3) gated dir.
    The baseline is gate-agnostic so we share it across rules where present.
    """
    pairs = {}
    for d in RUNS.iterdir():
        if not d.is_dir(): continue
        m = re.match(r"onpolicy_(baseline|gated)(?:_s(\d+))?(_R1)?$", d.name)
        if not m: continue
        kind = m.group(1)
        seed = m.group(2) or "42"
        is_r1 = m.group(3) is not None
        # Skip seed=42 R3 gated (no _R1) when we have an _R1 alternative;
        # we'll prefer _R1 in main multi-seed report.
        key = (seed, kind)
        existing = pairs.get(key)
        if existing is None or (is_r1 and "_R1" not in existing.name):
            pairs[key] = d
    # combine
    seeds = sorted({s for (s, _) in pairs.keys()})
    out = {}
    for s in seeds:
        bl = pairs.get((s, "baseline"))
        gt = pairs.get((s, "gated"))
        if bl and gt:
            out[s] = (bl, gt)
    return out


def load(p, name):
    """Load a jsonl file, returns DataFrame or None if missing/empty."""
    pp = p / name
    if not pp.exists(): return None
    rows = []
    with open(pp) as f:
        for line in f: rows.append(json.loads(line))
    if not rows: return None
    return pd.DataFrame(rows)


def main():
    pairs = find_runs()
    print(f"Found {len(pairs)} (baseline,gated) pairs: {sorted(pairs.keys())}")

    rows = []
    eval_traces = {"baseline": {}, "gated": {}}
    for seed, (bl_dir, gt_dir) in sorted(pairs.items()):
        bl_t = load(bl_dir, "train.jsonl")
        gt_t = load(gt_dir, "train.jsonl")
        bl_e = load(bl_dir, "eval.jsonl")
        gt_e = load(gt_dir, "eval.jsonl")

        if bl_t is None or gt_t is None or bl_e is None or gt_e is None:
            print(f"seed={seed}: incomplete; bl_t={bl_t is not None}, "
                  f"gt_t={gt_t is not None}, bl_e={bl_e is not None}, "
                  f"gt_e={gt_e is not None}")
            continue

        bl_iters = len(bl_t)
        gt_iters = len(gt_t)
        bl_done = "cumulative_wall_clock_sec" in bl_t.columns and bl_iters >= 60
        gt_done = "cumulative_wall_clock_sec" in gt_t.columns and gt_iters >= 60
        if not (bl_done and gt_done):
            print(f"seed={seed}: bl_iters={bl_iters}, gt_iters={gt_iters} "
                  f"(needs 60 each — still running?)")
            continue

        bl_total = float(bl_t.cumulative_wall_clock_sec.iloc[-1])
        gt_total = float(gt_t.cumulative_wall_clock_sec.iloc[-1])
        wall_saved = (bl_total - gt_total) / bl_total * 100

        bl_final = float(bl_e.success_rate.iloc[-1]) * 100
        gt_final = float(gt_e.success_rate.iloc[-1]) * 100
        delta_pp = gt_final - bl_final

        rows.append({
            "seed": seed,
            "bl_total_sec": bl_total,
            "gt_total_sec": gt_total,
            "wall_saved_pct": wall_saved,
            "bl_final_eval": bl_final,
            "gt_final_eval": gt_final,
            "delta_pp": delta_pp,
            "n_groups_cut": int(gt_t.n_groups_cut.sum()),
            "cut_rate_pct": float(gt_t.n_groups_cut.sum()) / float(gt_t.n_groups.sum()) * 100,
            "bl_grad_norm_mean": float(bl_t.grad_norm.mean()),
            "gt_grad_norm_mean": float(gt_t.grad_norm.mean()),
            "grad_amp": float(gt_t.grad_norm.mean()) / float(bl_t.grad_norm.mean()),
        })

        eval_traces["baseline"][seed] = bl_e
        eval_traces["gated"][seed] = gt_e
        print(f"seed={seed}: wall_saved={wall_saved:+.2f}%, "
              f"final eval bl={bl_final:.1f}% gt={gt_final:.1f}% "
              f"(Δ={delta_pp:+.1f}pp), grad_amp={rows[-1]['grad_amp']:.2f}x")

    if not rows:
        print("\nNo complete (baseline,gated) pairs yet.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "multiseed_summary.csv", index=False)

    print("\n=== Aggregate (mean ± std across seeds) ===")
    n = len(df)
    for col in ("wall_saved_pct", "bl_final_eval", "gt_final_eval",
                "delta_pp", "cut_rate_pct", "grad_amp"):
        m, s = df[col].mean(), df[col].std()
        print(f"  {col}: {m:+.2f} ± {s:.2f}  (n={n})")

    # Plot held-out trajectories
    if len(eval_traces["baseline"]) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        for seed, ev in eval_traces["baseline"].items():
            ax.plot(ev.iter, ev.success_rate * 100, color="#1f77b4", alpha=0.4,
                     marker="o", linewidth=1, markersize=4)
        for seed, ev in eval_traces["gated"].items():
            ax.plot(ev.iter, ev.success_rate * 100, color="#ff7f0e", alpha=0.4,
                     marker="o", linewidth=1, markersize=4)

        # mean trajectory
        if len(eval_traces["baseline"]) >= 1:
            iters = next(iter(eval_traces["baseline"].values())).iter.values
            bl_traj = np.stack([ev.success_rate.values * 100
                                for ev in eval_traces["baseline"].values()])
            gt_traj = np.stack([ev.success_rate.values * 100
                                for ev in eval_traces["gated"].values()])
            ax.plot(iters, bl_traj.mean(0), color="#1f77b4", linewidth=3,
                    marker="o", markersize=8, label=f"baseline mean (n={len(bl_traj)})")
            ax.plot(iters, gt_traj.mean(0), color="#ff7f0e", linewidth=3,
                    marker="o", markersize=8, label=f"gated mean (n={len(gt_traj)})")
            if len(bl_traj) > 1:
                ax.fill_between(iters,
                    bl_traj.mean(0) - bl_traj.std(0),
                    bl_traj.mean(0) + bl_traj.std(0),
                    color="#1f77b4", alpha=0.2)
                ax.fill_between(iters,
                    gt_traj.mean(0) - gt_traj.std(0),
                    gt_traj.mean(0) + gt_traj.std(0),
                    color="#ff7f0e", alpha=0.2)

        ax.set_xlabel("training iteration")
        ax.set_ylabel("held-out success rate (%)")
        ax.set_title(f"Multi-seed on-policy A/B: held-out eval trajectory (n={n} seeds)")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        safe_savefig(fig, FIG / "multiseed_eval.png", dpi=140)
        plt.close(fig)
        print(f"\nwrote {FIG / 'multiseed_eval.png'}")


if __name__ == "__main__":
    main()
