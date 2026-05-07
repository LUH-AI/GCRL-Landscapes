"""Reproduce `tab:phase_mobility` from `PAPER/APPENDIX/robustness.tex`.

Per env reports the mean Jaccard overlap and mean centroid drift of the
top-10% configuration set across adjacent phases, averaged across the 4
algorithms.

Inputs:
  data/eval_stats/{antmaze-medium,antmaze-large,cube,scene}.csv
Output:
  outputs/a3_phase_mobility.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from _common import ENVS, load_all_eval_stats, save_csv

TOP_FRAC = 0.10


def sobol_coords(lr_vals: np.ndarray, alpha_vals: np.ndarray) -> np.ndarray:
    ll = np.log(lr_vals)
    u_lr = (ll - ll.min()) / (ll.max() - ll.min() + 1e-12)
    u_al = (alpha_vals - alpha_vals.min()) / (
        alpha_vals.max() - alpha_vals.min() + 1e-12
    )
    return np.column_stack([u_lr, u_al])


def top_k_set(s: pd.Series, frac: float) -> set:
    k = max(1, int(np.ceil(frac * len(s))))
    return set(s.nlargest(k).index)


def main() -> None:
    raw = load_all_eval_stats()
    cfg = (
        raw.groupby(["algo", "env_short", "config", "phase", "lr", "alpha"])["success"]
        .mean()
        .reset_index()
    )

    rows = []
    for (algo, env), grp in cfg.groupby(["algo", "env_short"]):
        phases = sorted(grp["phase"].unique())
        if len(phases) < 2:
            continue
        cfg_base = grp[grp["phase"] == phases[0]][["config", "lr", "alpha"]].set_index(
            "config"
        )
        U = sobol_coords(cfg_base["lr"].values, cfg_base["alpha"].values)
        coord_map = {c: U[i] for i, c in enumerate(cfg_base.index)}
        for t_idx in range(len(phases) - 1):
            t, t1 = phases[t_idx], phases[t_idx + 1]
            s_t = grp[grp["phase"] == t].set_index("config")["success"]
            s_t1 = grp[grp["phase"] == t1].set_index("config")["success"]
            T_t, T_t1 = top_k_set(s_t, TOP_FRAC), top_k_set(s_t1, TOP_FRAC)
            inter, union = T_t & T_t1, T_t | T_t1
            jaccard = len(inter) / len(union) if union else float("nan")
            c_t = np.mean([coord_map[c] for c in T_t if c in coord_map], axis=0)
            c_t1 = np.mean([coord_map[c] for c in T_t1 if c in coord_map], axis=0)
            drift = float(np.linalg.norm(c_t - c_t1))
            rows.append(dict(algo=algo, env=env, jaccard=jaccard, drift=drift))

    per_pair = pd.DataFrame(rows)
    per_algo_env = (
        per_pair.groupby(["algo", "env"])[["jaccard", "drift"]].mean().reset_index()
    )

    summary = (
        per_algo_env.groupby("env")[["jaccard", "drift"]]
        .mean()
        .round(3)
        .reset_index()
        .rename(columns={"jaccard": "mean_jaccard", "drift": "mean_centroid_drift"})
    )
    summary["env_order"] = summary["env"].map({e: i for i, e in enumerate(ENVS)})
    summary = summary.sort_values("env_order").drop(columns=["env_order"])

    save_csv(summary, "a3_phase_mobility")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
