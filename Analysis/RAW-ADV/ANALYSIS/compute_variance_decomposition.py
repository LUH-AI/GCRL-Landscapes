#!/usr/bin/env python3
"""Decompose performance variance explained by learning rate, alpha, and their interaction.

For each (agent, env) combination:
  success ~ log(lr) + log(alpha) + log(lr)*log(alpha)

Reports:
  R2_lr        variance explained by lr alone
  R2_alpha     variance explained by alpha alone
  R2_main      variance explained by lr + alpha (main effects)
  R2_full      variance explained by full model (+ interaction)
  dR2_lr       unique contribution of lr  = R2_main - R2_alpha
  dR2_alpha    unique contribution of alpha = R2_main - R2_lr
  dR2_inter    unique contribution of interaction = R2_full - R2_main
  R2_shared    shared variance between lr and alpha = R2_main - dR2_lr - dR2_alpha

Data sources (merged automatically if present):
  1. additional_stats_raw.csv          — antmaze-medium (pre-existing)
  2. ENVS/antmaze-large/eval_stats.csv — antmaze-large  (built by build_eval_stats.py)
  3. ENVS/cube/eval_stats.csv          — cube           (built by build_eval_stats.py)

Uses final phase (phase=4) averaged over seeds for the primary table.
A second table uses all phases to check stability.

Outputs
-------
  ANALYSIS/metric_summaries/variance_decomposition_final_phase.csv
  ANALYSIS/metric_summaries/variance_decomposition_allphases.csv

Usage
-----
  python ANALYSIS/compute_variance_decomposition.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

_ROOT = Path(__file__).parent.parent

STATS_CSV = Path(
    "/Users/adityamohan/git/GCRL/plots/new-plots/"
    "advantage-dist-phases/basin-analysis/additional_stats_raw.csv"
)
EXTRA_CSVS = [
    _ROOT / "ENVS" / "antmaze-large" / "eval_stats.csv",
    _ROOT / "ENVS" / "cube" / "eval_stats.csv",
    _ROOT / "ENVS" / "scene" / "eval_stats.csv",
]
OUT = Path(__file__).parent / "metric_summaries"
OUT.mkdir(exist_ok=True)

AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]


def r2_ols(X: np.ndarray, y: np.ndarray) -> float:
    if X.ndim == 1:
        X = X[:, None]
    reg = LinearRegression().fit(X, y)
    ss_res = ((y - reg.predict(X)) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def decompose(df: pd.DataFrame) -> dict:
    log_lr = np.log(df["lr"].values)
    log_alpha = np.log(df["alpha"].values)
    y = df["success"].values

    ll = (log_lr - log_lr.mean()) / (log_lr.std() + 1e-12)
    la = (log_alpha - log_alpha.mean()) / (log_alpha.std() + 1e-12)
    inter = ll * la

    r2_lr = r2_ols(ll, y)
    r2_alpha = r2_ols(la, y)
    r2_main = r2_ols(np.column_stack([ll, la]), y)
    r2_full = r2_ols(np.column_stack([ll, la, inter]), y)

    d_lr = r2_main - r2_alpha
    d_alpha = r2_main - r2_lr
    d_inter = r2_full - r2_main
    shared = r2_main - d_lr - d_alpha

    return {
        "R2_lr": round(r2_lr, 4),
        "R2_alpha": round(r2_alpha, 4),
        "R2_main": round(r2_main, 4),
        "R2_full": round(r2_full, 4),
        "dR2_lr": round(d_lr, 4),
        "dR2_alpha": round(d_alpha, 4),
        "dR2_inter": round(d_inter, 4),
        "R2_shared": round(shared, 4),
        "n": len(df),
    }


def run(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for env in sorted(df["env"].unique()):
        for agent in AGENTS:
            sub = df[(df["algo"] == agent) & (df["env"] == env)]
            if len(sub) < 5:
                continue
            rec = {"agent": agent, "env": env}
            rec.update(decompose(sub))
            rows.append(rec)

    result = pd.DataFrame(rows)
    out_path = OUT / f"variance_decomposition_{label}.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path.name}  ({len(result)} rows)")
    return result


def load_stats() -> pd.DataFrame:
    parts = []
    if STATS_CSV.exists():
        parts.append(pd.read_csv(STATS_CSV))
        print(
            f"Loaded {STATS_CSV.name}  ({len(parts[-1])} rows, envs: {sorted(parts[-1]['env'].unique())})"
        )
    else:
        print(f"WARNING: {STATS_CSV} not found — skipping antmaze-medium baseline")

    for p in EXTRA_CSVS:
        if p.exists():
            df = pd.read_csv(p)
            parts.append(df)
            print(f"Loaded {p}  ({len(df)} rows, envs: {sorted(df['env'].unique())})")
        else:
            print(f"  [skip] {p} not found (run build_eval_stats.py first)")

    if not parts:
        raise RuntimeError("No input CSVs found.")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    stats = load_stats()
    print(f"\nTotal: {len(stats)} rows across envs: {sorted(stats['env'].unique())}")

    # ── Primary: final phase, mean over seeds ────────────────────────────────
    final = (
        stats[stats["phase"] == stats["phase"].max()]
        .groupby(["algo", "env", "config", "lr", "alpha"])["success"]
        .mean()
        .reset_index()
    )
    primary = run(final, "final_phase")

    print("\n=== Final phase — all envs ===")
    print(
        f"{'Agent':<8} {'Env':<30} {'R2_lr':>7} {'R2_alpha':>9} {'R2_main':>8} "
        f"{'R2_full':>8} {'dR2_lr':>8} {'dR2_alpha':>10} {'dR2_inter':>10}"
    )
    print("-" * 102)
    for _, row in primary.iterrows():
        print(
            f"{row['agent']:<8} {row['env']:<30} {row['R2_lr']:>7.3f} {row['R2_alpha']:>9.3f} "
            f"{row['R2_main']:>8.3f} {row['R2_full']:>8.3f} "
            f"{row['dR2_lr']:>8.3f} {row['dR2_alpha']:>10.3f} "
            f"{row['dR2_inter']:>10.3f}"
        )

    # ── All phases ────────────────────────────────────────────────────────────
    phase_rows = []
    for phase in sorted(stats["phase"].unique()):
        sub_agg = (
            stats[stats["phase"] == phase]
            .groupby(["algo", "env", "config", "lr", "alpha"])["success"]
            .mean()
            .reset_index()
        )
        tmp = run(sub_agg, f"phase{phase}")
        tmp["phase"] = phase
        phase_rows.append(tmp)

    all_phase_df = pd.concat(phase_rows, ignore_index=True)
    all_phase_df.to_csv(OUT / "variance_decomposition_allphases.csv", index=False)
    print("\nSaved → variance_decomposition_allphases.csv")

    print("\n=== dR2_alpha vs dR2_lr by phase (all envs) ===")
    for env in sorted(all_phase_df["env"].unique()):
        for agent in AGENTS:
            g = all_phase_df[
                (all_phase_df["env"] == env) & (all_phase_df["agent"] == agent)
            ][["phase", "dR2_lr", "dR2_alpha", "dR2_inter", "R2_full"]].sort_values(
                "phase"
            )
            if g.empty:
                continue
            print(f"\n{agent} / {env}:")
            print(g.to_string(index=False))


if __name__ == "__main__":
    main()
