"""Reproduce `tab:sensitivity_summary` from `PAPER/APPENDIX/robustness.tex`.

Per (env, algo) reports four numbers:
  - R^2_eta   : R^2 of OLS(success ~ z(log lr)),  bootstrap-CI half-width as ±
  - R^2_alpha : R^2 of OLS(success ~ z(log alpha)),  same
  - PD_lr     : range of partial-dependence of a RandomForestRegressor over log lr
  - PD_alpha  : range over log alpha

Inputs:
  data/eval_stats/{antmaze-medium,antmaze-large,cube,scene}.csv
Output:
  outputs/a7_sensitivity_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from _common import AGENTS, ENVS, load_all_eval_stats, save_csv

N_BOOT = 2000
RNG = np.random.default_rng(42)


def r2_ols(X: np.ndarray, y: np.ndarray) -> float:
    if X.ndim == 1:
        X = X[:, None]
    reg = LinearRegression().fit(X, y)
    ss_res = ((y - reg.predict(X)) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def boot_r2(
    lr_v: np.ndarray, al_v: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float]:
    n = len(y)
    boot_lr, boot_al = [], []
    for _ in range(N_BOOT):
        s = RNG.choice(n, size=n, replace=True)
        ll = np.log(lr_v[s])
        ll = (ll - ll.mean()) / (ll.std() + 1e-12)
        la = np.log(al_v[s])
        la = (la - la.mean()) / (la.std() + 1e-12)
        boot_lr.append(r2_ols(ll, y[s]))
        boot_al.append(r2_ols(la, y[s]))
    lr_mean = float(np.mean(boot_lr))
    lr_hw = float((np.percentile(boot_lr, 97.5) - np.percentile(boot_lr, 2.5)) / 2)
    al_mean = float(np.mean(boot_al))
    al_hw = float((np.percentile(boot_al, 97.5) - np.percentile(boot_al, 2.5)) / 2)
    return lr_mean, lr_hw, al_mean, al_hw


def pd_ranges(lr_v: np.ndarray, al_v: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    log_lr, log_al = np.log(lr_v), np.log(al_v)
    X = np.column_stack([log_lr, log_al])
    rf = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
    grid_lr = np.linspace(log_lr.min(), log_lr.max(), 40)
    grid_al = np.linspace(log_al.min(), log_al.max(), 40)
    pd_lr = np.array(
        [
            rf.predict(np.column_stack([np.full(len(y), v), log_al])).mean()
            for v in grid_lr
        ]
    )
    pd_al = np.array(
        [
            rf.predict(np.column_stack([log_lr, np.full(len(y), v)])).mean()
            for v in grid_al
        ]
    )
    return float(pd_lr.max() - pd_lr.min()), float(pd_al.max() - pd_al.min())


def main() -> None:
    raw = load_all_eval_stats()
    rows = []
    for env in ENVS:
        sub = raw[raw["env_short"] == env]
        last = sub[sub["phase"] == sub["phase"].max()]
        for agent in AGENTS:
            g = (
                last[last["algo"] == agent]
                .groupby("config")
                .agg(
                    success=("success", "mean"),
                    lr=("lr", "first"),
                    alpha=("alpha", "first"),
                )
                .reset_index()
            )
            if len(g) < 10:
                continue
            lr_v, al_v, y = g["lr"].values, g["alpha"].values, g["success"].values
            r2lr, r2lr_hw, r2al, r2al_hw = boot_r2(lr_v, al_v, y)
            pd_lr, pd_al = pd_ranges(lr_v, al_v, y)
            rows.append(
                {
                    "env": env,
                    "agent": agent,
                    "R2_eta": round(r2lr, 2),
                    "R2_eta_hw": round(r2lr_hw, 2),
                    "R2_alpha": round(r2al, 2),
                    "R2_alpha_hw": round(r2al_hw, 2),
                    "PD_eta": round(pd_lr, 3),
                    "PD_alpha": round(pd_al, 3),
                }
            )

    out = pd.DataFrame(rows)
    out["env_order"] = out["env"].map({e: i for i, e in enumerate(ENVS)})
    out["agent_order"] = out["agent"].map({a: i for i, a in enumerate(AGENTS)})
    out = out.sort_values(["env_order", "agent_order"]).drop(
        columns=["env_order", "agent_order"]
    )

    save_csv(out, "a7_sensitivity_summary")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
