#!/usr/bin/env python3
"""Reproduce all tables in findings_antmaze.md.

Usage
-----
    cd RAW-ADV
    python ANALYSIS/analyse_framing.py
"""

from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr

BASE = Path("ENVS")
STATS = Path(
    "/Users/adityamohan/git/GCRL/plots/new-plots/"
    "advantage-dist-phases/basin-analysis/additional_stats_raw.csv"
)

AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]
ENVS = ["antmaze-medium", "antmaze-large", "cube"]


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ── Finding 1 — FR-AUC / gap by agent across environments ──────────────────
section("Finding 1 — FR-AUC, gap_mean, MRR, prec@5 by agent")
for env in ENVS:
    mat = pd.read_parquet(BASE / env / "advantage_matrix_metrics.parquet")
    cols = [c for c in ["fr_auc", "gap_mean", "mrr", "prec_at_5"] if c in mat.columns]
    print(f"\n{env}:")
    print(mat.groupby("agent")[cols].mean().round(3).to_string())

# ── Batch stability ─────────────────────────────────────────────────────────
section("Batch stability — std of fr_auc across 10 batches per (config, seed)")
for env in ENVS:
    mat = pd.read_parquet(BASE / env / "advantage_matrix_metrics.parquet")
    stab = (
        mat.groupby(["agent", "config", "seed"])["fr_auc"]
        .std()
        .reset_index()
        .groupby("agent")["fr_auc"]
        .agg(["mean", "median"])
        .round(4)
    )
    print(f"\n{env}:\n{stab.to_string()}")

# ── Finding 5 — ESS ordering ─────────────────────────────────────────────────
section("Finding 5 — ESS by agent")
for env in ENVS:
    ess_path = BASE / env / "ess_per_batch.parquet"
    if not ess_path.exists():
        print(f"\n{env}: ess_per_batch.parquet not available")
        continue
    ess = pd.read_parquet(ess_path)
    print(f"\n{env}:")
    print(ess.groupby("agent")["ess"].agg(["mean", "std"]).round(3).to_string())

# ── Findings 2–4 — correlations (antmaze-medium only, needs success data) ───
section("Findings 2–4 — Correlations (antmaze-medium)")

if not STATS.exists():
    print(f"  [skip] {STATS} not found")
else:
    mat_m = pd.read_parquet(
        BASE / "antmaze-medium" / "advantage_matrix_metrics.parquet"
    )
    ess_m = pd.read_parquet(BASE / "antmaze-medium" / "ess_per_batch.parquet")
    stats = pd.read_csv(STATS)

    mat_agg = (
        mat_m.groupby(["agent", "config", "seed"])[["fr_auc", "gap_mean", "mrr"]]
        .mean()
        .reset_index()
    )
    ess_agg = ess_m.groupby(["agent", "config", "seed"])["ess"].mean().reset_index()

    nav = (
        stats[
            stats["env"].str.contains("navigate-v0") & ~stats["env"].str.contains(" ")
        ]
        .groupby(["algo", "config", "seed"])["success"]
        .mean()
        .reset_index()
        .rename(columns={"algo": "agent"})
    )

    merged_ess = mat_agg.merge(ess_agg, on=["agent", "config", "seed"])
    merged_suc = mat_agg.merge(nav, on=["agent", "config", "seed"])

    print(
        f"\n{'Agent':<8} {'fr_auc↔ess':>12} {'fr_auc↔success':>16} {'gap↔success':>13}"
    )
    print("-" * 55)
    for agent in AGENTS:
        g_e = merged_ess[merged_ess["agent"] == agent]
        g_s = merged_suc[merged_suc["agent"] == agent]

        r_fe, p_fe = pearsonr(g_e["fr_auc"], g_e["ess"])
        r_fs, p_fs = pearsonr(g_s["fr_auc"], g_s["success"])
        r_gs, p_gs = pearsonr(g_s["gap_mean"], g_s["success"])

        def fmt(r, p):
            star = (
                "***"
                if p < 0.001
                else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            )
            return f"{r:+.3f}{star}"

        print(
            f"{agent:<8} {fmt(r_fe, p_fe):>12} {fmt(r_fs, p_fs):>16} {fmt(r_gs, p_gs):>13}"
        )

    print("\n  *** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")

    # GCIQL inversion: fraction of batches below 0.5
    section("GCIQL inversion — fraction of batches with fr_auc < 0.5")
    for env in ["antmaze-medium", "antmaze-large"]:
        mat = pd.read_parquet(BASE / env / "advantage_matrix_metrics.parquet")
        gciql = mat[mat["agent"] == "GCIQL"]["fr_auc"]
        print(f"  {env}: {(gciql < 0.5).mean():.3f} of {len(gciql)} batches below 0.5")
