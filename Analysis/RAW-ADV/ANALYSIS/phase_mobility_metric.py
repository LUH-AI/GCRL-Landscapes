#!/usr/bin/env python3
"""Phase mobility: Jaccard overlap and centroid drift of top-10% config sets.

For each (algo, env, phase-pair t→t+1):
  T_t   = set of configs in top 10% by final-phase mean success
  J_t   = |T_t ∩ T_{t+1}| / |T_t ∪ T_{t+1}|   (Jaccard, higher = more stable)
  d_t   = ||ū_t - ū_{t+1}||_2                   (centroid drift in normalized Sobol coords: log-lr, linear-alpha)

Sobol coordinates: lr and alpha are mapped to [0,1] by min-max scaling in
log-space, matching the Sobol sweep parameterisation.

Outputs
-------
  metric_summaries/phase_mobility.csv       — J and d per algo×env×phase-pair
  metric_summaries/phase_mobility_mean.csv  — mean J and d per algo×env

Usage
-----
  python ANALYSIS/phase_mobility_metric.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

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
TOP_FRAC = 0.10


def load_all() -> pd.DataFrame:
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


def sobol_coords(lr_vals: np.ndarray, alpha_vals: np.ndarray) -> np.ndarray:
    """Map log(lr) and alpha to [0,1] by global min-max (true Sobol search coordinates)."""
    ll = np.log(lr_vals)
    u_lr = (ll - ll.min()) / (ll.max() - ll.min() + 1e-12)
    u_al = (alpha_vals - alpha_vals.min()) / (
        alpha_vals.max() - alpha_vals.min() + 1e-12
    )
    return np.column_stack([u_lr, u_al])


def top_k_set(success_series: pd.Series, frac: float) -> set:
    k = max(1, int(np.ceil(frac * len(success_series))))
    return set(success_series.nlargest(k).index)


def main() -> None:
    raw = load_all()
    # Per-config, per-phase: mean success over seeds
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

        # Global Sobol coords (same for all phases — same configs)
        cfg_base = grp[grp["phase"] == phases[0]][["config", "lr", "alpha"]].set_index(
            "config"
        )
        U = sobol_coords(cfg_base["lr"].values, cfg_base["alpha"].values)
        coord_map = {c: U[i] for i, c in enumerate(cfg_base.index)}

        for t_idx in range(len(phases) - 1):
            t, t1 = phases[t_idx], phases[t_idx + 1]
            s_t = grp[grp["phase"] == t].set_index("config")["success"]
            s_t1 = grp[grp["phase"] == t1].set_index("config")["success"]

            T_t = top_k_set(s_t, TOP_FRAC)
            T_t1 = top_k_set(s_t1, TOP_FRAC)

            inter = T_t & T_t1
            union = T_t | T_t1
            jaccard = len(inter) / len(union) if union else float("nan")

            # Centroid drift in Sobol coords
            c_t = np.mean([coord_map[c] for c in T_t if c in coord_map], axis=0)
            c_t1 = np.mean([coord_map[c] for c in T_t1 if c in coord_map], axis=0)
            drift = float(np.linalg.norm(c_t - c_t1))

            rows.append(
                dict(
                    algo=algo,
                    env=env,
                    phase_from=t,
                    phase_to=t1,
                    top_k=len(T_t),
                    jaccard=round(jaccard, 4),
                    centroid_drift=round(drift, 4),
                    n_stable=len(inter),
                )
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "phase_mobility.csv", index=False)
    print("Saved → phase_mobility.csv")

    mean_df = (
        df.groupby(["algo", "env"])[["jaccard", "centroid_drift"]]
        .mean()
        .round(4)
        .reset_index()
    )
    mean_df.to_csv(OUT / "phase_mobility_mean.csv", index=False)
    print("Saved → phase_mobility_mean.csv")

    print("\n=== Mean Jaccard and centroid drift per algo/env ===")
    print(
        f"{'Algo':<8} {'Env':<18} {'Jaccard':>8} {'Drift':>8}  (1=stable, 0=full turnover)"
    )
    print("-" * 50)
    for _, r in mean_df.sort_values(["env", "algo"]).iterrows():
        print(
            f"{r['algo']:<8} {r['env']:<18} {r['jaccard']:>8.3f} {r['centroid_drift']:>8.3f}"
        )


if __name__ == "__main__":
    main()
