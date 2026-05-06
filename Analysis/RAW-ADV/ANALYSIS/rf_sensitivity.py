#!/usr/bin/env python3
"""Random-forest sensitivity analysis with partial dependence.

For each (algo, env): fit RF(success ~ log_lr, log_alpha) on final-phase
config means, then report:
  - OOB R²  (out-of-bag, unbiased with 64-point training sets)
  - 5-fold CV R²
  - MDI feature importance
  - Permutation importance evaluated on held-out CV folds
  - Partial dependence range: max(PD) - min(PD) along log(lr) and log(alpha)
  - Linear log-space R² (correctly using log-transformed inputs) for comparison

RF importances are computed OOB / cross-validated to avoid training-set
inflation on the small 64-config design.

PD ranges are the main paper-facing result; importances go to appendix.

Outputs
-------
  ANALYSIS/metric_summaries/rf_sensitivity.csv   — per algo x env
  ANALYSIS/metric_summaries/rf_pdp_{env}.png     — partial dependence plots

Usage
-----
  python ANALYSIS/rf_sensitivity.py
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
from sklearn.model_selection import cross_val_score, KFold

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


def r2_linear_logspace(log_lr: np.ndarray, log_alpha: np.ndarray, y: np.ndarray):
    """OLS R² using log-transformed, standardised inputs."""
    ll = (log_lr - log_lr.mean()) / (log_lr.std() + 1e-12)
    la = (log_alpha - log_alpha.mean()) / (log_alpha.std() + 1e-12)

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
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for env in sorted(df["env_short"].unique()):
        fig, axes = plt.subplots(len(AGENTS), 2, figsize=(10, 3.5 * len(AGENTS)))
        fig.suptitle(f"Partial Dependence (RF) — {env}", fontsize=13)

        for i, agent in enumerate(AGENTS):
            sub = df[(df["algo"] == agent) & (df["env_short"] == env)].copy()
            if len(sub) < 10:
                for ax in axes[i]:
                    ax.axis("off")
                continue

            log_lr = np.log(sub["lr"].values)
            log_alpha = np.log(sub["alpha"].values)
            X = np.column_stack([log_lr, log_alpha])
            y = sub["success"].values

            # ── Random forest (OOB + CV) ──────────────────────────────────────
            rf = RandomForestRegressor(
                n_estimators=500,
                max_features="sqrt",
                oob_score=True,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X, y)
            r2_oob = float(rf.oob_score_)
            r2_cv = float(cross_val_score(rf, X, y, cv=cv, scoring="r2").mean())

            # MDI (in-bag, for reference only)
            mdi_lr, mdi_al = rf.feature_importances_

            # Permutation importance on CV held-out folds
            perm_lr_scores, perm_al_scores = [], []
            for train_idx, test_idx in cv.split(X):
                rf_cv = RandomForestRegressor(
                    n_estimators=300,
                    max_features="sqrt",
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1,
                )
                rf_cv.fit(X[train_idx], y[train_idx])
                pi = permutation_importance(
                    rf_cv,
                    X[test_idx],
                    y[test_idx],
                    n_repeats=20,
                    random_state=42,
                    n_jobs=-1,
                )
                perm_lr_scores.append(pi.importances_mean[0])
                perm_al_scores.append(pi.importances_mean[1])
            perm_lr = float(np.mean(perm_lr_scores))
            perm_al = float(np.mean(perm_al_scores))

            # ── Partial dependence ranges ─────────────────────────────────────
            grid_lr = np.linspace(log_lr.min(), log_lr.max(), 40)
            grid_al = np.linspace(log_alpha.min(), log_alpha.max(), 40)

            pd_lr = np.array(
                [
                    rf.predict(
                        np.column_stack([np.full(len(sub), v), log_alpha])
                    ).mean()
                    for v in grid_lr
                ]
            )
            pd_al = np.array(
                [
                    rf.predict(np.column_stack([log_lr, np.full(len(sub), v)])).mean()
                    for v in grid_al
                ]
            )

            pd_range_lr = float(pd_lr.max() - pd_lr.min())
            pd_range_al = float(pd_al.max() - pd_al.min())

            # ── Linear log-space R² (fixed: log inputs passed correctly) ─────
            lin_r2_lr, lin_r2_al, lin_r2_main = r2_linear_logspace(log_lr, log_alpha, y)

            rows.append(
                dict(
                    algo=agent,
                    env=env,
                    rf_r2_oob=round(r2_oob, 4),
                    rf_r2_cv5=round(r2_cv, 4),
                    mdi_lr=round(mdi_lr, 4),
                    mdi_alpha=round(mdi_al, 4),
                    perm_lr=round(perm_lr, 4),
                    perm_alpha=round(perm_al, 4),
                    pd_range_lr=round(pd_range_lr, 4),
                    pd_range_alpha=round(pd_range_al, 4),
                    linear_R2_lr=round(lin_r2_lr, 4),
                    linear_R2_alpha=round(lin_r2_al, 4),
                    linear_R2_main=round(lin_r2_main, 4),
                )
            )

            # ── PDP subplots ──────────────────────────────────────────────────
            ax_lr, ax_al = axes[i, 0], axes[i, 1]

            ax_lr.plot(grid_lr, pd_lr, color="steelblue", lw=2)
            ax_lr.fill_between(
                grid_lr, pd_lr, pd_lr.mean(), alpha=0.15, color="steelblue"
            )
            ax_lr.set_title(
                f"{agent} — log(η)  [PD range={pd_range_lr:.3f}]", fontsize=10
            )
            ax_lr.set_xlabel("log(η)")
            ax_lr.set_ylabel("E[success]")

            ax_al.plot(grid_al, pd_al, color="darkorange", lw=2)
            ax_al.fill_between(
                grid_al, pd_al, pd_al.mean(), alpha=0.15, color="darkorange"
            )
            ax_al.set_title(
                f"{agent} — log(α)  [PD range={pd_range_al:.3f}]", fontsize=10
            )
            ax_al.set_xlabel("log(α)")
            ax_al.set_ylabel("E[success]")

        plt.tight_layout()
        out_fig = OUT / f"rf_pdp_{env}.png"
        plt.savefig(out_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → {out_fig.name}")

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "rf_sensitivity.csv", index=False)
    print(f"\nSaved → rf_sensitivity.csv  ({len(result)} rows)")

    print("\n=== RF sensitivity (OOB R², CV R², PD ranges, linear R²) ===")
    print(
        f"{'Algo':<8} {'Env':<18} {'OOB_R2':>7} {'CV_R2':>6} "
        f"{'perm_lr':>8} {'perm_al':>8} "
        f"{'PD_lr':>7} {'PD_al':>7} "
        f"{'lin_lr':>7} {'lin_al':>7}"
    )
    print("-" * 88)
    for _, r in result.iterrows():
        print(
            f"{r['algo']:<8} {r['env']:<18} {r['rf_r2_oob']:>7.3f} {r['rf_r2_cv5']:>6.3f} "
            f"{r['perm_lr']:>8.3f} {r['perm_alpha']:>8.3f} "
            f"{r['pd_range_lr']:>7.3f} {r['pd_range_alpha']:>7.3f} "
            f"{r['linear_R2_lr']:>7.3f} {r['linear_R2_alpha']:>7.3f}"
        )


if __name__ == "__main__":
    main()
