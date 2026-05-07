"""Reproduce `tab:seed_variance_summary` from `PAPER/APPENDIX/robustness.tex`.

Per env reports min / max / mean (across the 4 algorithms) of the
mean-per-config seed variance, computed at the final phase using arithmetic
seed variance (Bessel-corrected, ddof=1).

Inputs:
  data/eval_stats/{antmaze-medium,antmaze-large,cube,scene}.csv
Output:
  outputs/a2_seed_variance.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


from _common import ENVS, load_all_eval_stats, save_csv


def main() -> None:
    df = load_all_eval_stats()
    last = df[
        df["phase"] == df.groupby(["algo", "env_short"])["phase"].transform("max")
    ]

    # Per-config seed variance (ddof=1)
    config_var = (
        last.groupby(["algo", "env_short", "config"])["success"]
        .var()
        .reset_index(name="seed_var")
    )
    # Mean per (algo, env)
    mean_var = (
        config_var.groupby(["algo", "env_short"])["seed_var"]
        .mean()
        .reset_index(name="mean_seed_var")
    )

    # Per-env summary across the 4 algorithms
    out = (
        mean_var.groupby("env_short")["mean_seed_var"]
        .agg(["min", "max", "mean"])
        .round(4)
        .reset_index()
        .rename(
            columns={
                "env_short": "env",
                "min": "min_mean_seed_var",
                "max": "max_mean_seed_var",
                "mean": "mean_mean_seed_var",
            }
        )
    )
    out["env_order"] = out["env"].map({e: i for i, e in enumerate(ENVS)})
    out = out.sort_values("env_order").drop(columns=["env_order"])

    save_csv(out, "a2_seed_variance")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
