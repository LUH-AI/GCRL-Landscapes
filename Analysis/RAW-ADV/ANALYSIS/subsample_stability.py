#!/usr/bin/env python3
"""Subsampling stability of rho_0.9, rho_0.8, mean success, and method ordering.

Repeatedly draws 32 or 48 of the 64 configs, recomputes metrics, and reports:
  - Mean and std of rho_0.9/rho_0.8/mean_success across subsamples
  - Ordering stability: fraction of resamples where pairwise method ranking
    matches the full-64 ranking

Outputs
-------
  metric_summaries/subsample_stability.csv        — metric distributions
  metric_summaries/subsample_ordering.csv         — ordering stability

Usage
-----
  python ANALYSIS/subsample_stability.py
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "metric_summaries"
OUT.mkdir(exist_ok=True)

AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]
N_BOOT = 1000
SUB_SIZES = [32, 48]
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


def load_final() -> pd.DataFrame:
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
    raw = pd.concat(parts, ignore_index=True)
    final = raw[
        raw["phase"] == raw.groupby(["algo", "env_short"])["phase"].transform("max")
    ]
    return (
        final.groupby(["algo", "env_short", "config"])["success"].mean().reset_index()
    )


def compute_metrics(success: np.ndarray) -> dict:
    mx = success.max()
    return dict(
        mean=float(success.mean()),
        rho09=float((success >= 0.9 * mx).mean()) if mx > 0 else float("nan"),
        rho08=float((success >= 0.8 * mx).mean()) if mx > 0 else float("nan"),
    )


def main() -> None:
    cfg = load_final()

    stab_rows = []
    order_rows = []

    for (env, algo), grp in cfg.groupby(["env_short", "algo"]):
        configs = grp["config"].values
        success = grp["success"].values
        n_cfg = len(configs)
        if n_cfg < 10:
            continue

        full = compute_metrics(success)

        for k in SUB_SIZES:
            if k >= n_cfg:
                continue
            boot_metrics = {"mean": [], "rho09": [], "rho08": []}
            for _ in range(N_BOOT):
                idx = RNG.choice(n_cfg, size=k, replace=False)
                m = compute_metrics(success[idx])
                for key in boot_metrics:
                    boot_metrics[key].append(m[key])

            for metric, vals in boot_metrics.items():
                vals = np.array([v for v in vals if np.isfinite(v)])
                lo, hi = np.percentile(vals, [2.5, 97.5])
                stab_rows.append(
                    dict(
                        algo=algo,
                        env=env,
                        subsample_k=k,
                        metric=metric,
                        full_value=round(full[metric], 4),
                        boot_mean=round(float(vals.mean()), 4),
                        boot_std=round(float(vals.std()), 4),
                        ci_lo=round(float(lo), 4),
                        ci_hi=round(float(hi), 4),
                    )
                )

    pd.DataFrame(stab_rows).to_csv(OUT / "subsample_stability.csv", index=False)
    print("Saved → subsample_stability.csv")

    # ── Ordering stability across methods ─────────────────────────────────────
    for env, grp_env in cfg.groupby("env_short"):
        # Full-64 mean success per method
        full_means = {
            a: grp_env[grp_env["algo"] == a]["success"].mean()
            for a in AGENTS
            if (grp_env["algo"] == a).any()
        }
        full_rank = sorted(full_means, key=full_means.get, reverse=True)

        for k in SUB_SIZES:
            agree_counts = {pair: 0 for pair in combinations(full_rank, 2)}
            n_valid = 0
            for _ in range(N_BOOT):
                boot_means = {}
                for algo, g in grp_env.groupby("algo"):
                    configs = g["config"].values
                    n = len(configs)
                    if n < k:
                        continue
                    idx = RNG.choice(n, size=k, replace=False)
                    boot_means[algo] = g["success"].values[idx].mean()

                if len(boot_means) < 2:
                    continue
                n_valid += 1
                for a1, a2 in combinations(full_rank, 2):
                    if a1 in boot_means and a2 in boot_means:
                        if boot_means[a1] > boot_means[a2]:
                            agree_counts[(a1, a2)] += 1

            for (a1, a2), cnt in agree_counts.items():
                order_rows.append(
                    dict(
                        env=env,
                        subsample_k=k,
                        method_higher=a1,
                        method_lower=a2,
                        pct_agree=round(cnt / n_valid * 100, 1)
                        if n_valid > 0
                        else float("nan"),
                        n_valid=n_valid,
                    )
                )

    order_df = pd.DataFrame(order_rows)
    order_df.to_csv(OUT / "subsample_ordering.csv", index=False)
    print("Saved → subsample_ordering.csv")

    print("\n=== Ordering stability (% resamples where full ranking holds) ===")
    for env in order_df["env"].unique():
        print(f"\n{env}")
        sub = order_df[order_df["env"] == env]
        pivot = sub.pivot_table(
            index=["method_higher", "method_lower"],
            columns="subsample_k",
            values="pct_agree",
        )
        pivot.columns = [f"k={c}" for c in pivot.columns]
        print(pivot.to_string())


if __name__ == "__main__":
    main()
