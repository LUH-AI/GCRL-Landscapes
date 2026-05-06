#!/usr/bin/env python3
"""fANOVA-style sensitivity analysis via random forest + partial dependence.

For each (algo, env): fit RF(success ~ log_lr, log_alpha) on final-phase
config means, then report:
  - RF feature importance (MDI)
  - Permutation importance (more reliable)
  - Partial dependence range: max(PD) - min(PD) for each hyperparameter
  - Compare against linear R² from variance decomposition

Outputs
-------
  ANALYSIS/metric_summaries/fanова_sensitivity.csv      — per algo×env
  ANALYSIS/metric_summaries/fanова_pdp_{env}.png        — partial dependence plots

Usage
-----
  python ANALYSIS/fanова_sensitivity.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

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


def load_final_phase() -> pd.DataFrame:
    parts = []
    if STATS_CSV.exists():
        df = pd.read_csv(STATS_CSV)
        df = df[df["env"] == "antmaze-medium-navigate-v0"].copy()
        df["env_short"] = "antmaze-medium"
        parts.append(df)
    for env_short, p in EVAL_ENVS.items():
        if p.exists():
            df = pd.read_csv(p)
            df["env_short"] = env_short
            parts.append(df)
    raw = pd.concat(parts, ignore_index=True)
    return (
        raw[
            raw["phase"] == raw.groupby(["algo", "env_short"])["phase"].transform("max")
        ]
        .groupby(["algo", "env_short", "config", "lr", "alpha"])["success"]
        .mean()
        .reset_index()
    )


def r2_linear(ll, la, y):
    ll = (ll - ll.mean()) / (ll.std() + 1e-12)
    la = (la - la.mean()) / (la.std() + 1e-12)

    def r2(X):
        if X.ndim == 1:
            X = X[:, None]
        reg = LinearRegression().fit(X, y)
        ss_res = ((y - reg.predict(X)) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return r2(ll), r2(la), r2(np.column_stack([ll, la]))


def main() -> None:
    df = load_final_phase()
    print(f"Loaded {len(df):,} rows (final phase, seed-averaged)")

    rows = []

    for env in sorted(df["env_short"].unique()):
        fig, axes = plt.subplots(len(AGENTS), 2, figsize=(10, 3.5 * len(AGENTS)))
        fig.suptitle(f"Partial Dependence — {env}", fontsize=13)

        for i, agent in enumerate(AGENTS):
            sub = df[(df["algo"] == agent) & (df["env_short"] == env)].copy()
            if len(sub) < 10:
                continue

            X = np.column_stack([np.log(sub["lr"].values), np.log(sub["alpha"].values)])
            y = sub["success"].values

            # Random forest
            rf = RandomForestRegressor(
                n_estimators=500, max_features="sqrt", random_state=42, n_jobs=-1
            )
            rf.fit(X, y)
            r2_rf = float(rf.score(X, y))

            # MDI importance
            mdi_lr, mdi_al = rf.feature_importances_

            # Permutation importance (on training set — 64 points too few for OOB)
            perm = permutation_importance(
                rf, X, y, n_repeats=50, random_state=42, n_jobs=-1
            )
            perm_lr = float(perm.importances_mean[0])
            perm_al = float(perm.importances_mean[1])

            # Partial dependence range
            grid_lr = np.linspace(X[:, 0].min(), X[:, 0].max(), 40)
            grid_al = np.linspace(X[:, 1].min(), X[:, 1].max(), 40)

            pd_lr = np.array(
                [
                    rf.predict(np.column_stack([np.full(len(sub), v), X[:, 1]])).mean()
                    for v in grid_lr
                ]
            )
            pd_al = np.array(
                [
                    rf.predict(np.column_stack([X[:, 0], np.full(len(sub), v)])).mean()
                    for v in grid_al
                ]
            )

            pd_range_lr = float(pd_lr.max() - pd_lr.min())
            pd_range_al = float(pd_al.max() - pd_al.min())

            # Linear R² for comparison
            r2l_lr, r2l_al, r2l_main = r2_linear(
                sub["lr"].values, sub["alpha"].values, y
            )

            rows.append(
                dict(
                    algo=agent,
                    env=env,
                    rf_r2=round(r2_rf, 4),
                    mdi_lr=round(mdi_lr, 4),
                    mdi_alpha=round(mdi_al, 4),
                    perm_lr=round(perm_lr, 4),
                    perm_alpha=round(perm_al, 4),
                    pd_range_lr=round(pd_range_lr, 4),
                    pd_range_alpha=round(pd_range_al, 4),
                    linear_R2_lr=round(r2l_lr, 4),
                    linear_R2_alpha=round(r2l_al, 4),
                    linear_R2_main=round(r2l_main, 4),
                )
            )

            # PDP subplots
            ax_lr, ax_al = axes[i, 0], axes[i, 1]
            ax_lr.plot(grid_lr, pd_lr, color="steelblue", lw=2)
            ax_lr.set_title(f"{agent} — log(lr)", fontsize=10)
            ax_lr.set_xlabel("log(lr)")
            ax_lr.set_ylabel("E[success]")

            ax_al.plot(grid_al, pd_al, color="darkorange", lw=2)
            ax_al.set_title(f"{agent} — log(α)", fontsize=10)
            ax_al.set_xlabel("log(α)")
            ax_al.set_ylabel("E[success]")

        plt.tight_layout()
        out_fig = OUT / f"fanова_pdp_{env}.png"
        plt.savefig(out_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → {out_fig.name}")

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "fanова_sensitivity.csv", index=False)
    print(f"\nSaved → fanова_sensitivity.csv  ({len(result)} rows)")

    # Summary print
    print("\n=== RF vs Linear: importance of lr vs alpha ===")
    print(
        f"{'Algo':<8} {'Env':<18} {'RF_R2':>6} {'perm_lr':>8} {'perm_al':>8} "
        f"{'PD_rng_lr':>10} {'PD_rng_al':>10} {'lin_R2_lr':>10} {'lin_R2_al':>10}"
    )
    print("-" * 92)
    for _, r in result.iterrows():
        print(
            f"{r['algo']:<8} {r['env']:<18} {r['rf_r2']:>6.3f} "
            f"{r['perm_lr']:>8.3f} {r['perm_alpha']:>8.3f} "
            f"{r['pd_range_lr']:>10.3f} {r['pd_range_alpha']:>10.3f} "
            f"{r['linear_R2_lr']:>10.3f} {r['linear_R2_alpha']:>10.3f}"
        )


if __name__ == "__main__":
    main()
