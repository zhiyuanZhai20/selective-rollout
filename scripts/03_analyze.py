"""
Produce the final deliverables:
  - results/correlation_table.csv
  - results/figures/scatter_K{K}.png for K in 5,10,15,20
  - results/figures/hist_zero_vs_nonzero.png
  - results/report.md   (one-paragraph pass/fail verdict)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from src.analysis import (  # noqa: E402
    spearman_table,
    spearman_by_task_type,
    scatter_grid_per_K,
    hist_zero_vs_nonzero,
    plot_auroc_heatmap,
    plot_stratified_scatter,
    write_report,
    Ks as DEFAULT_Ks,
    METRICS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.inp.endswith(".parquet"):
        df = pd.read_parquet(args.inp)
    else:
        df = pd.read_csv(args.inp)

    outdir = Path(args.out)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    corr = spearman_table(df)
    corr_path = outdir / "correlation_table.csv"
    corr.to_csv(corr_path, index=False)
    print(f"[analyze] wrote {corr_path}")
    print(corr.to_string(index=False))

    per_tt = spearman_by_task_type(df)
    per_tt_path = outdir / "correlation_by_task_type.csv"
    per_tt.to_csv(per_tt_path, index=False)
    print(f"[analyze] wrote {per_tt_path}")

    for K in DEFAULT_Ks:
        sp = outdir / "figures" / f"scatter_K{K}.png"
        scatter_grid_per_K(df, K, str(sp))
        print(f"[analyze] wrote {sp}")

    hp = outdir / "figures" / "hist_zero_vs_nonzero.png"
    hist_zero_vs_nonzero(df, str(hp))
    print(f"[analyze] wrote {hp}")

    heat = outdir / "figures" / "heatmap_rho_auroc.png"
    plot_auroc_heatmap(corr, str(heat))
    print(f"[analyze] wrote {heat}")

    # Stratified scatter for the best (metric, K) by AUROC
    if not corr.empty and corr["auroc_nonzero_var"].notna().any():
        r = corr.loc[corr["auroc_nonzero_var"].idxmax()]
        strat = outdir / "figures" / f"stratified_{r['metric']}_K{int(r['K'])}.png"
        plot_stratified_scatter(df, r["metric"], int(r["K"]), str(strat))
        print(f"[analyze] wrote {strat}")

    rp = outdir / "report.md"
    write_report(df, corr, str(rp), per_task_df=per_tt)
    print(f"[analyze] wrote {rp}")
    print("\n--- report.md ---")
    print(rp.read_text())


if __name__ == "__main__":
    main()
