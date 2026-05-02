"""
Correlation analysis + plots for the gate experiment.

Inputs: a pandas.DataFrame where each row is one task's group, with columns:
  - task_id, task_type, reward_mean, reward_variance, is_zero_variance
  - for each (metric, K): <metric>@<K>

Outputs (into `out_dir`):
  - correlation_table.csv
  - scatter_K{K}.png   (one subplot per metric, x=metric, y=reward_variance)
  - hist_zero_vs_nonzero.png  (one subplot per (metric, K))
  - report.md  (one-paragraph pass/fail verdict)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .figutil import safe_savefig


METRICS = [
    "unique_action_ratio",
    "unique_prefix_ratio",
    "action_bigram_jaccard_mean",
    "prefix_edit_distance_mean",
    "obs_unique_ratio",
    "termination_fraction",
    "action_entropy",
]

Ks = [5, 10, 15, 20]


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan"), float("nan")
    rho, p = spearmanr(x[mask], y[mask])
    return rho, p


def _safe_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC for using `scores` (higher = more likely to be positive) to predict `labels`.
    We want to predict is_nonzero_variance (= 1 - is_zero_variance) from divergence."""
    mask = ~(np.isnan(scores) | np.isnan(labels))
    if mask.sum() < 5 or np.unique(labels[mask]).size < 2:
        return float("nan")
    return float(roc_auc_score(labels[mask], scores[mask]))


def spearman_table(df: pd.DataFrame, Ks: List[int] = Ks, metrics: List[str] = METRICS) -> pd.DataFrame:
    """Spearman(div@K, reward_variance) + AUROC(div@K → non-zero-variance)."""
    rows = []
    is_nonzero = (df["is_zero_variance"] == 0).astype(float).to_numpy()
    rv = df["reward_variance"].to_numpy()
    for K in Ks:
        for m in metrics:
            col = f"{m}@{K}"
            if col not in df.columns:
                continue
            x = df[col].to_numpy()
            rho, p = _safe_spearman(x, rv)
            auc = _safe_auroc(x, is_nonzero)
            rows.append({
                "K": K, "metric": m,
                "n": int(np.isfinite(x).sum()),
                "spearman_rho": rho, "p_value": p, "auroc_nonzero_var": auc,
            })
    return pd.DataFrame(rows).sort_values(
        ["K", "spearman_rho"], ascending=[True, False]
    ).reset_index(drop=True)


def spearman_by_task_type(df: pd.DataFrame, Ks: List[int] = Ks, metrics: List[str] = METRICS) -> pd.DataFrame:
    rows = []
    for tt, g in df.groupby("task_type"):
        if len(g) < 5:
            continue
        rv = g["reward_variance"].to_numpy()
        is_nonzero = (g["is_zero_variance"] == 0).astype(float).to_numpy()
        for K in Ks:
            for m in metrics:
                col = f"{m}@{K}"
                if col not in g.columns:
                    continue
                x = g[col].to_numpy()
                rho, p = _safe_spearman(x, rv)
                auc = _safe_auroc(x, is_nonzero)
                rows.append({
                    "task_type": tt, "K": K, "metric": m, "n": len(g),
                    "spearman_rho": rho, "p_value": p, "auroc_nonzero_var": auc,
                })
    return pd.DataFrame(rows)


def scatter_grid_per_K(df: pd.DataFrame, K: int, out_path: str, metrics: List[str] = METRICS):
    cols = 3
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = axes.flatten()
    sc = None
    for idx, m in enumerate(metrics):
        ax = axes[idx]
        col = f"{m}@{K}"
        if col not in df.columns:
            ax.set_visible(False)
            continue
        x = df[col]
        y = df["reward_variance"]
        colors = df["reward_mean"]
        sc = ax.scatter(x, y, c=colors, cmap="coolwarm", alpha=0.7, s=25,
                        edgecolor="k", linewidth=0.3, vmin=0, vmax=1)
        mask = ~(x.isna() | y.isna())
        if mask.sum() >= 3 and x[mask].nunique() > 1 and y[mask].nunique() > 1:
            rho, p = spearmanr(x[mask], y[mask])
            title = f"{m}\nρ={rho:.3f}  p={p:.2g}"
        else:
            title = f"{m}\n(insufficient variance)"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(f"{m}@K={K}")
        ax.set_ylabel("final reward var")
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(f"Divergence at step K={K}  vs  final reward variance", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if sc is not None:
        fig.colorbar(sc, ax=axes[:len(metrics)], label="reward mean", shrink=0.6)
    safe_savefig(fig, out_path, dpi=140)
    plt.close(fig)


def hist_zero_vs_nonzero(df: pd.DataFrame, out_path: str, Ks: List[int] = Ks, metrics: List[str] = METRICS):
    n_metrics = len(metrics)
    n_Ks = len(Ks)
    fig, axes = plt.subplots(n_metrics, n_Ks, figsize=(3 * n_Ks, 2.2 * n_metrics))
    if n_metrics == 1:
        axes = np.array([axes])
    for i, m in enumerate(metrics):
        for j, K in enumerate(Ks):
            ax = axes[i, j]
            col = f"{m}@{K}"
            if col not in df.columns:
                ax.set_visible(False)
                continue
            zv = df.loc[df["is_zero_variance"] == 1, col].dropna()
            nz = df.loc[df["is_zero_variance"] == 0, col].dropna()
            bins = np.linspace(0, 1, 21)
            ax.hist(zv, bins=bins, alpha=0.55, label=f"zero-var (n={len(zv)})", color="tab:red")
            ax.hist(nz, bins=bins, alpha=0.55, label=f"nonzero-var (n={len(nz)})", color="tab:blue")
            if i == 0:
                ax.set_title(f"K={K}")
            if j == 0:
                ax.set_ylabel(m, fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0 and j == n_Ks - 1:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Divergence distribution: zero-var vs non-zero-var groups", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    safe_savefig(fig, out_path, dpi=140)
    plt.close(fig)


def verdict(corr_df: pd.DataFrame) -> tuple[str, float, float]:
    """Return (verdict, best_|rho|, best_auroc). Verdict is driven by whichever
    of |ρ| or (AUROC - 0.5) × 2 is larger — AUROC gives credit for threshold-style
    signals the Spearman correlation might miss."""
    if corr_df.empty:
        return "FAIL (no data)", float("nan"), float("nan")
    best_rho = corr_df["spearman_rho"].abs().max(skipna=True)
    best_auc = corr_df["auroc_nonzero_var"].max(skipna=True)
    # map AUROC ∈ [0.5, 1] → [0, 1]
    best_score = max(best_rho if np.isfinite(best_rho) else 0,
                     2 * (best_auc - 0.5) if np.isfinite(best_auc) else 0)
    if best_score >= 0.4:
        v = "PASS"
    elif best_score >= 0.2:
        v = "MARGINAL"
    else:
        v = "FAIL"
    return v, best_rho, best_auc


def plot_auroc_heatmap(corr_df: pd.DataFrame, out_path: str,
                       Ks: List[int] = Ks, metrics: List[str] = METRICS):
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3, 0.35 * len(metrics) + 1)))
    for ax, col, title, cmap, vlim in (
        (axes[0], "spearman_rho", "Spearman ρ(div, reward_var)", "RdBu_r", (-0.6, 0.6)),
        (axes[1], "auroc_nonzero_var", "AUROC: div → non-zero-var", "viridis", (0.3, 0.9)),
    ):
        mat = corr_df.pivot(index="metric", columns="K", values=col).reindex(
            index=metrics, columns=Ks)
        im = ax.imshow(mat.to_numpy(), aspect="auto", cmap=cmap,
                       vmin=vlim[0], vmax=vlim[1])
        ax.set_xticks(range(len(Ks)), [f"K={k}" for k in Ks])
        ax.set_yticks(range(len(metrics)), metrics, fontsize=8)
        ax.set_title(title)
        for i in range(len(metrics)):
            for j in range(len(Ks)):
                v = mat.iat[i, j] if i < mat.shape[0] and j < mat.shape[1] else np.nan
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if abs(v) < 0.3 else "white")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    safe_savefig(fig, out_path, dpi=140)
    plt.close(fig)


def plot_stratified_scatter(df: pd.DataFrame, metric: str, K: int, out_path: str):
    """Color-coded scatter: all-fail (blue), all-succeed (green), mixed (orange).
    Makes the two zero-variance sub-populations visible."""
    col = f"{metric}@{K}"
    if col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub_af = df[df["reward_mean"] == 0]
    sub_as = df[df["reward_mean"] == 1]
    sub_mx = df[(df["reward_mean"] > 0) & (df["reward_mean"] < 1)]
    ax.scatter(sub_af[col], sub_af["reward_variance"] - 0.01,
               c="tab:blue", alpha=0.65, s=34, label=f"all-fail (n={len(sub_af)})",
               edgecolor="k", linewidth=0.3)
    ax.scatter(sub_as[col], sub_as["reward_variance"] + 0.01,
               c="tab:green", alpha=0.65, s=34, label=f"all-succeed (n={len(sub_as)})",
               edgecolor="k", linewidth=0.3)
    ax.scatter(sub_mx[col], sub_mx["reward_variance"],
               c="tab:orange", alpha=0.85, s=45, label=f"mixed (n={len(sub_mx)})",
               edgecolor="k", linewidth=0.4, marker="D")
    ax.set_xlabel(f"{metric}@K={K}")
    ax.set_ylabel("final reward variance")
    ax.set_title(f"{metric}@K={K}  stratified by group outcome")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    safe_savefig(fig, out_path, dpi=140)
    plt.close(fig)


def write_report(df: pd.DataFrame, corr_df: pd.DataFrame, out_path: str,
                 per_task_df: pd.DataFrame | None = None):
    v, best_rho, best_auc = verdict(corr_df)
    top_rho = corr_df.reindex(
        corr_df["spearman_rho"].abs().sort_values(ascending=False).index
    ).head(5)
    top_auc = corr_df.sort_values("auroc_nonzero_var", ascending=False).head(5)

    n_total = len(df)
    n_zero = int(df["is_zero_variance"].sum())
    n_all_fail = int((df["reward_mean"] == 0).sum())
    n_all_succ = int((df["reward_mean"] == 1).sum())
    n_mixed = n_total - n_all_fail - n_all_succ

    lines = [
        "# Gate experiment — result",
        "",
        f"- Tasks rolled out: **{n_total}**",
        f"- Mean success rate: **{df['reward_mean'].mean():.2f}**",
        f"- Group composition: all-fail **{n_all_fail}** / mixed **{n_mixed}** / "
        f"all-succeed **{n_all_succ}** → zero-variance fraction "
        f"**{n_zero}/{n_total} ({n_zero / n_total:.0%})**",
        "",
        f"## Verdict: **{v}**  best |ρ|={best_rho:.3f}  best AUROC={best_auc:.3f}",
        "",
        "Pass criterion: max(|ρ|, 2·(AUROC − 0.5)) ≥ 0.4 on any (metric, K) pair.",
        "AUROC ≥ 0.7 ≈ strong threshold signal; AUROC ≈ 0.5 → no signal.",
        "",
        "## Top (metric, K) pairs by |Spearman ρ|",
        "",
        "| K | metric | n | ρ | p | AUROC |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in top_rho.iterrows():
        lines.append(f"| {int(r['K'])} | {r['metric']} | {int(r['n'])} | "
                     f"{r['spearman_rho']:.3f} | {r['p_value']:.2g} | "
                     f"{r['auroc_nonzero_var']:.3f} |")

    lines += [
        "",
        "## Top (metric, K) pairs by AUROC (non-zero-var classification)",
        "",
        "| K | metric | n | AUROC | ρ |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, r in top_auc.iterrows():
        lines.append(f"| {int(r['K'])} | {r['metric']} | {int(r['n'])} | "
                     f"{r['auroc_nonzero_var']:.3f} | {r['spearman_rho']:.3f} |")

    if per_task_df is not None and not per_task_df.empty:
        lines += [
            "",
            "## Strongest signal per task type (best metric × K per type)",
            "",
            "| task_type | n | K | metric | ρ | AUROC |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        idx = per_task_df.groupby("task_type")["spearman_rho"].apply(
            lambda s: s.abs().idxmax() if s.abs().notna().any() else None
        )
        for tt, ix in idx.items():
            if ix is None:
                continue
            r = per_task_df.loc[ix]
            lines.append(f"| {tt} | {int(r['n'])} | {int(r['K'])} | {r['metric']} | "
                         f"{r['spearman_rho']:.3f} | {r['auroc_nonzero_var']:.3f} |")

    Path(out_path).write_text("\n".join(lines))
