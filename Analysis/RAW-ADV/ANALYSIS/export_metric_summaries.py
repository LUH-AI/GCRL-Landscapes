#!/usr/bin/env python3
"""Export one CSV per metric, summarised per (agent, environment).

Outputs written to ANALYSIS/metric_summaries/
Each CSV has agents as rows and environments as columns (mean ± std).

Usage
-----
    python ANALYSIS/export_metric_summaries.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

ENVS = ["antmaze-medium", "antmaze-large", "cube", "scene"]
AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]

_root = Path(__file__).parent.parent
OUT = Path(__file__).parent / "metric_summaries"
OUT.mkdir(exist_ok=True)

STATS_CSV = Path(
    "/Users/adityamohan/git/GCRL/plots/new-plots/"
    "advantage-dist-phases/basin-analysis/additional_stats_raw.csv"
)


# ── helpers ───────────────────────────────────────────────────────────────────


def env_dir(env: str) -> Path:
    return _root / "ENVS" / env


def mean_std_cell(s: pd.Series) -> str:
    return f"{s.mean():.3f} ± {s.std():.3f}"


def pivot_metric(
    metric: str,
    source_fn,  # callable(env) -> pd.DataFrame with columns [agent, <metric>]
    agents=AGENTS,
    envs=ENVS,
) -> pd.DataFrame:
    """Build agent×env pivot table with 'mean ± std' cells."""
    rows = []
    for env in envs:
        df = source_fn(env)
        if df is None or df.empty:
            for agent in agents:
                rows.append({"agent": agent, "env": env, "value": float("nan")})
            continue
        for agent in agents:
            vals = df.loc[df["agent"] == agent, metric].dropna()
            rows.append(
                {
                    "agent": agent,
                    "env": env,
                    "mean": vals.mean() if len(vals) else float("nan"),
                    "std": vals.std() if len(vals) else float("nan"),
                }
            )

    long = pd.DataFrame(rows)
    long["cell"] = long.apply(
        lambda r: (
            f"{r['mean']:.3f} ± {r['std']:.3f}" if not np.isnan(r["mean"]) else "n/a"
        ),
        axis=1,
    )
    pivot = long.pivot(index="agent", columns="env", values="cell")
    # Reorder
    pivot = pivot.reindex(index=agents, columns=[e for e in envs if e in pivot.columns])
    pivot.index.name = "agent"
    pivot.columns.name = None
    return pivot


def save(df: pd.DataFrame, name: str) -> None:
    path = OUT / f"{name}.csv"
    df.to_csv(path)
    print(f"  → {path.name}  ({df.shape[0]}×{df.shape[1]})")


# ── data loaders ─────────────────────────────────────────────────────────────


def load_matrix(env: str) -> pd.DataFrame | None:
    p = env_dir(env) / "advantage_matrix_metrics.parquet"
    return pd.read_parquet(p) if p.exists() else None


def load_conc(env: str) -> pd.DataFrame | None:
    p = env_dir(env) / "awr_concentration.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # Keep only rows with real alpha; rename _mean columns to bare names for pivot
    df = df[df["has_real_alpha"]].copy()
    for col in ["ess", "saturation_mass", "fraction_clipped", "top5_mass", "entropy"]:
        if f"{col}_mean" in df.columns:
            df[col] = df[f"{col}_mean"]
    return df


def load_ess(env: str) -> pd.DataFrame | None:
    p = env_dir(env) / "ess_per_batch.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


# ── 1. Semantic reliability metrics (from cross-goal matrix) ─────────────────
print("\nSemantic reliability metrics …")

for metric in [
    "fr_auc",
    "gap_mean",
    "gap_std",
    "mrr",
    "prec_at_1",
    "prec_at_5",
    "Aplus_mean",
    "Aplus_std",
    "Aminus_std",
]:
    save(pivot_metric(metric, load_matrix), metric.lower())


# extractability_index lives in afr_metrics.csv (aggregated across batches)
def load_afr(env: str) -> pd.DataFrame | None:
    p = env_dir(env) / "afr_metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df.rename(columns={"extractability_index_mean": "extractability_index"})
    return df


save(pivot_metric("extractability_index", load_afr), "extractability_index")

# ── 2. AWR concentration metrics ─────────────────────────────────────────────
print("\nAWR concentration metrics …")

for metric in ["ess", "saturation_mass", "fraction_clipped", "top5_mass", "entropy"]:
    save(pivot_metric(metric, load_conc), metric)

# ESS from ess_per_batch (the paper's primary ESS table)
save(pivot_metric("ess", load_ess), "ess_per_batch")

# ── 3. Alpha-sensitivity summary ─────────────────────────────────────────────
print("\nAlpha-sensitivity summary …")

# For each env: mean ESS at each alpha, per agent — save as agent×alpha for each env
for env in ENVS:
    p = env_dir(env) / "awr_alpha_sweep.csv"
    if not p.exists():
        print(f"  [skip] {env}: awr_alpha_sweep.csv not found")
        continue
    sweep = pd.read_csv(p)
    # Pivot: agents as rows, alpha values as columns (mean ESS across configs/seeds/batches)
    agg = sweep.groupby(["agent", "alpha_sweep"])["ess"].mean().reset_index()
    pivot = agg.pivot(index="agent", columns="alpha_sweep", values="ess")
    pivot = pivot.reindex(AGENTS)
    pivot.columns = [f"alpha={a:.3f}" for a in pivot.columns]
    pivot.index.name = "agent"
    pivot.columns.name = None
    save(pivot, f"alpha_sensitivity_{env.replace('-', '_')}")

# ── 4. Return correlation table (antmaze-medium only) ────────────────────────
print("\nReturn correlation table …")

corr_p = env_dir("antmaze-medium") / "awr_vs_return.csv"
if corr_p.exists():
    corr = pd.read_csv(corr_p)
    # Keep only r_ columns and reshape to readable table
    r_cols = [c for c in corr.columns if c.startswith("r_")]
    sig_cols = [c for c in corr.columns if c.startswith("sig_")]
    rows = []
    for _, row in corr.iterrows():
        r = {"agent": row["agent"], "n": row["n"]}
        for rc in r_cols:
            metric = rc[2:]  # strip "r_"
            sc = f"sig_{metric}"
            r_val = row.get(rc, float("nan"))
            sig = row.get(sc, "")
            r[metric] = f"{r_val:+.3f}{sig}" if not np.isnan(r_val) else "n/a"
        rows.append(r)
    corr_clean = pd.DataFrame(rows).set_index("agent")
    # Rename columns to drop trailing _mean
    corr_clean.columns = [
        c.replace("_mean", "") if c != "n" else c for c in corr_clean.columns
    ]
    save(corr_clean, "return_correlations_antmaze_medium")

# ── 5. Combined cross-env summary table ──────────────────────────────────────
print("\nCombined summary …")

rows = []
for env in ENVS:
    mat = load_matrix(env)
    conc = load_conc(env)
    ess = load_ess(env)
    for agent in AGENTS:
        r: dict = {"env": env, "agent": agent}
        if mat is not None:
            g = mat[mat["agent"] == agent]
            for col in [
                "fr_auc",
                "gap_mean",
                "extractability_index",
                "mrr",
                "prec_at_1",
                "prec_at_5",
            ]:
                if col in g.columns:
                    r[col] = round(g[col].mean(), 4)
        if ess is not None:
            g = ess[ess["agent"] == agent]
            r["ess"] = round(g["ess"].mean(), 4) if len(g) else float("nan")
        if conc is not None:
            g = conc[conc["agent"] == agent]
            for col in ["saturation_mass", "fraction_clipped", "top5_mass", "entropy"]:
                if col in g.columns:
                    r[col] = round(g[col].mean(), 4)
        rows.append(r)

combined = pd.DataFrame(rows).set_index(["env", "agent"])
combined.to_csv(OUT / "all_metrics_combined.csv")
print(f"  → all_metrics_combined.csv  ({combined.shape[0]}×{combined.shape[1]})")

print(f"\nDone. {len(list(OUT.glob('*.csv')))} CSVs in {OUT}/")
