# Selective Rollout: Mid-Trajectory Termination for Multi-Sample Agent RL

A one-parameter mid-rollout gate for **GRPO-style agent RL** that detects
zero-variance groups partway through their rollout and stops them at step `K`,
recovering the post-`K` rollout compute and a smaller training-step cost.

## Headline result

On **100 ALFWorld tasks** with **Qwen2.5-7B-Instruct** at `G=8`:

- The gate `Cut(K=10, d_L<0.12)` recovers **14.0%** of rollout step-tokens
  (**11.3%** losslessly from true-positive cuts) at offline precision
  **0.81**, while preserving **96.7%** of the GRPO advantage L²-norm.
- In a 60-iteration on-policy GRPO training A/B averaged over **n=4 seeds**,
  the gated arm finishes **10.7%** faster wall-clock (bootstrap 95% CI excludes 0)
  and shifts held-out success on 50 unseen tasks by **+2.5 pp**.
- The held-out improvement traces to a measurable reduction in
  zero-advantage gradient-batch dilution: gated runs raise the effective
  gradient L² by **1.16×**, in agreement with the dilution prediction.

## Repo layout

```
src/          library code: GRPO loss, gate enforcement, divergence metrics
scripts/      numbered entry points (01..32) and dispatcher
paper/        LaTeX sources (paper.tex, checklist.tex)
results/      reproducible CSVs and figures from the paper
explanations/ auxiliary derivations / notes
```

External data (rollout buffer, ALFWorld game files, training logs) is
**not in this repo** — it's large and lives outside as the symlinks
`data/` and `alfworld_data/` (gitignored). The conference style file
and compiled PDF are also gitignored (they live with the local working
copy). To reproduce, point `data/` and `alfworld_data/` at your local
copies of the rollout buffer and ALFWorld distribution.

## Reproducing the predictive analysis (no GPU)

After unpacking the rollout buffer (`data/rollouts.jsonl`,
`data/metrics.parquet`):

```bash
python scripts/03_analyze.py --in data/metrics.parquet --out results
python scripts/06_gate_sweep.py
python scripts/10_grpo_offline.py
python scripts/24_mechanism.py
python scripts/30_compare_multiseed.py
python scripts/32_single_axis_fig.py
```

These regenerate the predictive heatmap, the `d_K=10` stratified plot,
the on-policy A/B figure, the gate ablation table, and the
mechanism-evolution figure.

## Reproducing the online integration A/Bs (GPU required)

Each tier takes 4–5 GPU-hours on a single RTX 6000 Ada (48 GB):

- `scripts/14_wallclock_ab.py`  — Tier 1 (rollout-only A/B)
- `scripts/17_grpo_static.py`   — Tier 2 (off-policy training A/B)
- `scripts/19_grpo_onpolicy.py` — Tier 3 (on-policy training A/B)

Set `ALFWORLD_DATA` to your local ALFWorld game directory before running.

## License

Code released under MIT. Trajectory buffer (when distributed separately)
under CC-BY 4.0. Upstream: ALFWorld (MIT), Qwen2.5 (Apache 2.0).
