#!/usr/bin/env python3
"""Scatter: success vs diagnostic metrics for Cube and Scene.

For each env (cube, scene): plots success vs FR-AUC, gap_mean, MRR, ESS,
top5_mass. Points coloured by method. Reports Pearson/Spearman r per method
and pooled.

Data joined per (algo, config) using final-phase success averaged over seeds,
and diagnostic metrics averaged over all batches/seeds.

Outputs
-------
  metric_summaries/diagnostic_scatter_{env}.png   — 5-panel scatter per env
  metric_summaries/diagnostic_correlations.csv    — per-method + pooled r

Usage
-----
  python ANALYSIS/diagnostic_scatter.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

_ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "metric_summaries"
OUT.mkdir(exist_ok=True)

AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]
COLORS = {"CRL": "#1f77b4", "GCIQL": "#ff7f0e", "GCIVL": "#2ca02c", "QRL": "#d62728"}
ENVS = ["cube", "scene"]
METRICS = ["fr_auc", "gap_mean", "mrr", "ess", "top5_mass"]
METRIC_LABELS = {
    "fr_auc": "FR-AUC",
    "gap_mean": "Gap mean",
    "mrr": "MRR",
    "ess": "ESS",
    "top5_mass": "Top-5 mass",
}


def load_diagnostics(env: str) -> pd.DataFrame:
    env_dir = _ROOT / "ENVS" / env

    # FR-AUC, gap_mean, MRR from matrix metrics
    mat = pd.read_parquet(env_dir / "advantage_matrix_metrics.parquet")
    # per-config version
    mat_cfg = (
        mat.groupby(["agent", "config"])[["fr_auc", "gap_mean", "mrr"]]
        .mean()
        .reset_index()
        .rename(columns={"agent": "algo"})
    )

    # ESS from ess_per_batch
    ess_df = pd.read_parquet(env_dir / "ess_per_batch.parquet")
    ess_cfg = (
        ess_df.groupby(["agent", "config"])["ess"]
        .mean()
        .reset_index()
        .rename(columns={"agent": "algo"})
    )

    # top5_mass from awr_concentration
    conc = pd.read_csv(env_dir / "awr_concentration.csv")
    if "has_real_alpha" in conc.columns:
        conc = conc[conc["has_real_alpha"]]
    col = "top5_mass_mean" if "top5_mass_mean" in conc.columns else "top5_mass"
    conc_col = "configuration" if "configuration" in conc.columns else "config"
    conc_cfg = (
        conc.groupby(["agent", conc_col])[col]
        .mean()
        .reset_index()
        .rename(columns={"agent": "algo", conc_col: "config", col: "top5_mass"})
    )

    diag = mat_cfg.merge(ess_cfg, on=["algo", "config"]).merge(
        conc_cfg, on=["algo", "config"]
    )
    return diag


def load_success(env: str) -> pd.DataFrame:
    p = _ROOT / "ENVS" / env / "eval_stats.csv"
    df = pd.read_csv(p)
    final = df[df["phase"] == df["phase"].max()]
    return final.groupby(["algo", "config"])["success"].mean().reset_index()


def correlations(df: pd.DataFrame, metric: str, env: str) -> list[dict]:
    rows = []
    x, y = df[metric].values, df["success"].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 3:
        pr, _ = pearsonr(x[mask], y[mask])
        sr, _ = spearmanr(x[mask], y[mask])
        rows.append(
            dict(
                env=env,
                method="pooled",
                metric=metric,
                pearson=round(pr, 3),
                spearman=round(sr, 3),
                n=int(mask.sum()),
            )
        )
    for agent in AGENTS:
        sub = df[df["algo"] == agent]
        x2, y2 = sub[metric].values, sub["success"].values
        mask2 = np.isfinite(x2) & np.isfinite(y2)
        if mask2.sum() > 3:
            pr, _ = pearsonr(x2[mask2], y2[mask2])
            sr, _ = spearmanr(x2[mask2], y2[mask2])
        else:
            pr = sr = float("nan")
        rows.append(
            dict(
                env=env,
                method=agent,
                metric=metric,
                pearson=round(pr, 3) if np.isfinite(pr) else float("nan"),
                spearman=round(sr, 3) if np.isfinite(sr) else float("nan"),
                n=int(mask2.sum()),
            )
        )
    return rows


def main() -> None:
    all_corr = []

    for env in ENVS:
        diag = load_diagnostics(env)
        success = load_success(env)
        df = diag.merge(success, on=["algo", "config"])

        fig, axes = plt.subplots(1, len(METRICS), figsize=(4.5 * len(METRICS), 4.5))
        fig.suptitle(f"{env} — Success vs diagnostic metrics", fontsize=13)

        for ax, metric in zip(axes, METRICS):
            for agent in AGENTS:
                sub = df[df["algo"] == agent]
                ax.scatter(
                    sub[metric],
                    sub["success"],
                    label=agent,
                    color=COLORS[agent],
                    alpha=0.75,
                    s=45,
                    edgecolors="none",
                )

            # Pooled trend line
            x_all = df[metric].values
            y_all = df["success"].values
            mask = np.isfinite(x_all) & np.isfinite(y_all)
            if mask.sum() > 3:
                m, b = np.polyfit(x_all[mask], y_all[mask], 1)
                xs = np.linspace(x_all[mask].min(), x_all[mask].max(), 60)
                ax.plot(xs, m * xs + b, "k--", lw=1, alpha=0.5)

            ax.set_xlabel(METRIC_LABELS[metric], fontsize=10)
            ax.set_ylabel("Success (final phase)", fontsize=10)
            ax.legend(fontsize=8)

            all_corr.extend(correlations(df, metric, env))

        plt.tight_layout()
        for fmt in ("png", "pdf"):
            out = OUT / f"diagnostic_scatter_{env}.{fmt}"
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  → {out.name}")
        plt.close()

    corr_df = pd.DataFrame(all_corr)
    corr_df.to_csv(OUT / "diagnostic_correlations.csv", index=False)
    print("  → diagnostic_correlations.csv")

    print("\n=== Pooled Pearson / Spearman (success vs metric) ===")
    print(
        corr_df[corr_df["method"] == "pooled"][
            ["metric", "pearson", "spearman", "n"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
