#!/usr/bin/env python3
"""Seed variance vs landscape breadth.

From eval_stats.csv:
  1. Per-config seed variance of success (final phase)
  2. Mean seed variance per method/env
  3. Correlation between config mean success and seed variance
  4. Relation between rho_0.9 and average seed variance

Outputs
-------
  metric_summaries/seed_variance_summary.csv      — mean seed var per algo×env
  metric_summaries/seed_variance_correlations.csv — r(success, seed_var) per algo×env
  metric_summaries/seed_variance_vs_rho.csv       — rho_0.9 vs mean seed var

Usage
-----
  python ANALYSIS/seed_variance_analysis.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "metric_summaries"
OUT.mkdir(exist_ok=True)

AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]
STATS_CSV = Path(
    "/Users/adityamohan/git/GCRL/plots/new-plots/"
    "advantage-dist-phases/basin-analysis/additional_stats_raw.csv"
)
EVAL_ENVS = {
    "antmaze-large": _ROOT / "ENVS" / "antmaze-large" / "eval_stats.csv",
    "cube": _ROOT / "ENVS" / "cube" / "eval_stats.csv",
    "scene": _ROOT / "ENVS" / "scene" / "eval_stats.csv",
}


def load_all() -> pd.DataFrame:
    parts = []
    if STATS_CSV.exists():
        df = pd.read_csv(STATS_CSV)
        df = df[df["env"] == "antmaze-medium-navigate-v0"].copy()
        df["env_short"] = "antmaze-medium"
        parts.append(df[["algo", "env_short", "config", "seed", "phase", "success"]])
    for env_short, p in EVAL_ENVS.items():
        if p.exists():
            df = pd.read_csv(p)
            df["env_short"] = env_short
            parts.append(
                df[["algo", "env_short", "config", "seed", "phase", "success"]]
            )
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    raw = load_all()
    # Final phase only
    df = raw[
        raw["phase"] == raw.groupby(["algo", "env_short"])["phase"].transform("max")
    ]

    # ── 1 & 2. Seed variance per config, then mean per (algo, env) ─────────────
    config_stats = (
        df.groupby(["algo", "env_short", "config"])["success"]
        .agg(mean_success="mean", seed_var="var", n_seeds="count")
        .reset_index()
    )

    summary = (
        config_stats.groupby(["algo", "env_short"])
        .agg(
            mean_seed_var=("seed_var", "mean"),
            std_seed_var=("seed_var", "std"),
            mean_success=("mean_success", "mean"),
            n_configs=("config", "count"),
        )
        .reset_index()
        .round(4)
    )
    summary.to_csv(OUT / "seed_variance_summary.csv", index=False)
    print("Saved → seed_variance_summary.csv")
    print(summary.to_string(index=False))

    # ── 3. Correlation: config mean success vs seed variance ───────────────────
    corr_rows = []
    for (algo, env), grp in config_stats.groupby(["algo", "env_short"]):
        x = grp["mean_success"].values
        y = grp["seed_var"].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 3:
            pr, pp = pearsonr(x[mask], y[mask])
            sr, sp = spearmanr(x[mask], y[mask])
        else:
            pr = sr = pp = sp = float("nan")
        corr_rows.append(
            dict(
                algo=algo,
                env=env,
                pearson=round(pr, 3),
                pearson_p=round(pp, 3),
                spearman=round(sr, 3),
                spearman_p=round(sp, 3),
                n=int(mask.sum()),
            )
        )

    # Pooled per env
    for env, grp in config_stats.groupby("env_short"):
        x = grp["mean_success"].values
        y = grp["seed_var"].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 3:
            pr, pp = pearsonr(x[mask], y[mask])
            sr, sp = spearmanr(x[mask], y[mask])
        else:
            pr = sr = pp = sp = float("nan")
        corr_rows.append(
            dict(
                algo="pooled",
                env=env,
                pearson=round(pr, 3),
                pearson_p=round(pp, 3),
                spearman=round(sr, 3),
                spearman_p=round(sp, 3),
                n=int(mask.sum()),
            )
        )

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT / "seed_variance_correlations.csv", index=False)
    print("\nSaved → seed_variance_correlations.csv")
    print("\n=== r(mean_success, seed_var) per algo/env ===")
    print(corr_df.to_string(index=False))

    # ── 4. rho_0.9 vs mean seed variance ──────────────────────────────────────
    rho_rows = []
    for (algo, env), grp in config_stats.groupby(["algo", "env_short"]):
        means = grp["mean_success"].values
        vars_ = grp["seed_var"].values
        mx = means.max()
        rho09 = float(np.mean(means >= 0.9 * mx)) if mx > 0 else float("nan")
        rho08 = float(np.mean(means >= 0.8 * mx)) if mx > 0 else float("nan")
        mean_var_top10 = (
            float(vars_[means >= 0.9 * mx].mean())
            if (means >= 0.9 * mx).any()
            else float("nan")
        )
        rho_rows.append(
            dict(
                algo=algo,
                env=env,
                rho_0_9=round(rho09, 4),
                rho_0_8=round(rho08, 4),
                mean_seed_var=round(vars_.mean(), 4),
                mean_seed_var_top10pct=round(mean_var_top10, 4),
            )
        )

    rho_df = pd.DataFrame(rho_rows)
    rho_df.to_csv(OUT / "seed_variance_vs_rho.csv", index=False)
    print("\nSaved → seed_variance_vs_rho.csv")
    print("\n=== rho_0.9 vs mean seed variance ===")
    print(rho_df.to_string(index=False))


if __name__ == "__main__":
    main()
