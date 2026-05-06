#!/usr/bin/env python3
"""Bootstrap 95% CIs for key diagnostic tables.

Targets:
  A. Landscape breadth (ρ_0.9, ρ_0.8, mean_success) — bootstrap over configs
  B. ESS / top5_mass / saturation_mass          — bootstrap over batches
  C. Variance decomposition R²_lr, R²_alpha      — bootstrap over configs

For each: 2000 bootstrap resamples, 95% CI = [2.5, 97.5] percentiles.

Outputs (all in ANALYSIS/metric_summaries/)
  bootstrap_landscape.csv
  bootstrap_awr.csv
  bootstrap_variance.csv

Usage
-----
  python ANALYSIS/bootstrap_confidence_intervals.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

_ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "metric_summaries"
OUT.mkdir(exist_ok=True)

AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]
N_BOOT = 2000
RNG = np.random.default_rng(42)

STATS_CSV = Path(
    "/Users/adityamohan/git/GCRL/plots/new-plots/"
    "advantage-dist-phases/basin-analysis/additional_stats_raw.csv"
)
EVAL_ENVS = {
    "antmaze-large": _ROOT / "ENVS" / "antmaze-large" / "eval_stats.csv",
    "cube": _ROOT / "ENVS" / "cube" / "eval_stats.csv",
    "scene": _ROOT / "ENVS" / "scene" / "eval_stats.csv",
}
AWR_ENVS = ["antmaze-medium", "antmaze-large", "cube", "scene"]


def boot_ci(values: np.ndarray, stat_fn, n=N_BOOT):
    stats = [
        stat_fn(RNG.choice(values, size=len(values), replace=True)) for _ in range(n)
    ]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(np.mean(stats)), float(lo), float(hi)


# ── A. Landscape breadth ─────────────────────────────────────────────────────


def load_success() -> pd.DataFrame:
    parts = []
    if STATS_CSV.exists():
        df = pd.read_csv(STATS_CSV)
        df = df[df["env"] == "antmaze-medium-navigate-v0"].copy()
        df["env_short"] = "antmaze-medium"
        parts.append(
            df[
                [
                    "algo",
                    "env_short",
                    "config",
                    "seed",
                    "phase",
                    "lr",
                    "alpha",
                    "success",
                ]
            ]
        )
    for env_short, p in EVAL_ENVS.items():
        if p.exists():
            df = pd.read_csv(p)
            df["env_short"] = env_short
            parts.append(
                df[
                    [
                        "algo",
                        "env_short",
                        "config",
                        "seed",
                        "phase",
                        "lr",
                        "alpha",
                        "success",
                    ]
                ]
            )
    return pd.concat(parts, ignore_index=True)


def frac09(v):
    return np.mean(v >= 0.9 * v.max()) if v.max() > 0 else np.nan


def frac08(v):
    return np.mean(v >= 0.8 * v.max()) if v.max() > 0 else np.nan


print("A. Landscape breadth bootstrap …")
sdf = load_success()
rows_A = []
for (algo, env, phase), grp in sdf.groupby(["algo", "env_short", "phase"]):
    cfg_means = grp.groupby("config")["success"].mean().values
    if len(cfg_means) < 5:
        continue
    mx = cfg_means.max()
    if mx <= 0:
        continue

    m09, lo09, hi09 = boot_ci(cfg_means, frac09)
    m08, lo08, hi08 = boot_ci(cfg_means, frac08)
    mm, lomm, himm = boot_ci(cfg_means, np.mean)
    rows_A.append(
        dict(
            algo=algo,
            env=env,
            phase=phase,
            rho09=round(m09, 4),
            rho09_lo=round(lo09, 4),
            rho09_hi=round(hi09, 4),
            rho08=round(m08, 4),
            rho08_lo=round(lo08, 4),
            rho08_hi=round(hi08, 4),
            mean_success=round(mm, 4),
            mean_lo=round(lomm, 4),
            mean_hi=round(himm, 4),
        )
    )

df_A = pd.DataFrame(rows_A)
df_A.to_csv(OUT / "bootstrap_landscape.csv", index=False)
print(f"  → bootstrap_landscape.csv  ({len(df_A)} rows)")


# ── B. AWR metrics (ESS, top5_mass, saturation_mass) ─────────────────────────

print("B. AWR bootstrap …")
rows_B = []
for env in AWR_ENVS:
    conc_p = _ROOT / "ENVS" / env / "awr_concentration.csv"
    ess_p = _ROOT / "ENVS" / env / "ess_per_batch.parquet"
    if not conc_p.exists() and not ess_p.exists():
        continue

    if conc_p.exists():
        conc = pd.read_csv(conc_p)
        if "has_real_alpha" in conc.columns:
            conc = conc[conc["has_real_alpha"]]
        for col_raw, col_out in [
            ("ess_mean", "ess"),
            ("top5_mass_mean", "top5_mass"),
            ("saturation_mass_mean", "saturation_mass"),
            ("fraction_clipped_mean", "fraction_clipped"),
        ]:
            col = col_raw if col_raw in conc.columns else col_out
            if col not in conc.columns:
                continue
            for agent in AGENTS:
                vals = conc.loc[conc["agent"] == agent, col].dropna().values
                if len(vals) < 5:
                    continue
                m, lo, hi = boot_ci(vals, np.mean)
                rows_B.append(
                    dict(
                        env=env,
                        algo=agent,
                        metric=col_out,
                        mean=round(m, 4),
                        ci_lo=round(lo, 4),
                        ci_hi=round(hi, 4),
                    )
                )

    if ess_p.exists():
        ess = pd.read_parquet(ess_p)
        for agent in AGENTS:
            vals = ess.loc[ess["agent"] == agent, "ess"].dropna().values
            if len(vals) < 5:
                continue
            m, lo, hi = boot_ci(vals, np.mean)
            rows_B.append(
                dict(
                    env=env,
                    algo=agent,
                    metric="ess_per_batch",
                    mean=round(m, 4),
                    ci_lo=round(lo, 4),
                    ci_hi=round(hi, 4),
                )
            )

df_B = pd.DataFrame(rows_B)
df_B.to_csv(OUT / "bootstrap_awr.csv", index=False)
print(f"  → bootstrap_awr.csv  ({len(df_B)} rows)")


# ── C. Variance decomposition ─────────────────────────────────────────────────


def r2_ols(X, y):
    if X.ndim == 1:
        X = X[:, None]
    reg = LinearRegression().fit(X, y)
    ss_res = ((y - reg.predict(X)) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def decompose_r2(lr_vals, alpha_vals, y):
    ll = np.log(lr_vals)
    ll = (ll - ll.mean()) / (ll.std() + 1e-12)
    la = np.log(alpha_vals)
    la = (la - la.mean()) / (la.std() + 1e-12)
    r2_lr = r2_ols(ll, y)
    r2_alpha = r2_ols(la, y)
    r2_main = r2_ols(np.column_stack([ll, la]), y)
    r2_full = r2_ols(np.column_stack([ll, la, ll * la]), y)
    return r2_lr, r2_alpha, r2_main - r2_alpha, r2_main - r2_lr, r2_full - r2_main


print("C. Variance decomposition bootstrap …")
all_eval = load_success()
rows_C = []
for (algo, env), grp in all_eval.groupby(["algo", "env_short"]):
    final_phase = grp["phase"].max()
    sub = (
        grp[grp["phase"] == final_phase]
        .groupby(["config", "lr", "alpha"] if "lr" in grp.columns else ["config"])[
            "success"
        ]
        .mean()
        .reset_index()
    )
    # need lr/alpha — merge back
    sub = (
        grp[grp["phase"] == final_phase]
        .groupby(["config"])
        .agg(success=("success", "mean"), lr=("lr", "first"), alpha=("alpha", "first"))
        .reset_index()
    )
    if len(sub) < 10:
        continue
    lr_v = sub["lr"].values
    al_v = sub["alpha"].values
    y = sub["success"].values
    idx = np.arange(len(sub))

    boot_r2lr, boot_r2al, boot_dlr, boot_dal, boot_dint = [], [], [], [], []
    for _ in range(N_BOOT):
        s = RNG.choice(idx, size=len(idx), replace=True)
        r2lr, r2al, dlr, dal, dint = decompose_r2(lr_v[s], al_v[s], y[s])
        boot_r2lr.append(r2lr)
        boot_r2al.append(r2al)
        boot_dlr.append(dlr)
        boot_dal.append(dal)
        boot_dint.append(dint)

    def ci(b):
        lo, hi = np.percentile(b, [2.5, 97.5])
        return round(float(np.mean(b)), 4), round(float(lo), 4), round(float(hi), 4)

    m_lr, lo_lr, hi_lr = ci(boot_r2lr)
    m_al, lo_al, hi_al = ci(boot_r2al)
    m_dlr, lo_dlr, hi_dlr = ci(boot_dlr)
    m_dal, lo_dal, hi_dal = ci(boot_dal)
    m_di, lo_di, hi_di = ci(boot_dint)
    rows_C.append(
        dict(
            algo=algo,
            env=env,
            R2_lr=m_lr,
            R2_lr_lo=lo_lr,
            R2_lr_hi=hi_lr,
            R2_alpha=m_al,
            R2_alpha_lo=lo_al,
            R2_alpha_hi=hi_al,
            dR2_lr=m_dlr,
            dR2_lr_lo=lo_dlr,
            dR2_lr_hi=hi_dlr,
            dR2_alpha=m_dal,
            dR2_alpha_lo=lo_dal,
            dR2_alpha_hi=hi_dal,
            dR2_inter=m_di,
            dR2_inter_lo=lo_di,
            dR2_inter_hi=hi_di,
        )
    )

df_C = pd.DataFrame(rows_C)
df_C.to_csv(OUT / "bootstrap_variance.csv", index=False)
print(f"  → bootstrap_variance.csv  ({len(df_C)} rows)")
print("\nDone.")
