"""Reproduce `tab:subsample_stability_close_pairs` from `PAPER/APPENDIX/robustness.tex`.

For each (env, algo-pair), report the percentage of bootstrap subsamples in
which the algorithm ranking matches the full-grid ranking. We list only
*close pairs* — those with agreement < 95% in either subsample size.

Inputs:
  data/eval_stats/{antmaze-medium,antmaze-large,cube,scene}.csv
Output:
  outputs/a4_subsample_stability.csv
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from _common import ENVS, load_all_eval_stats, save_csv

N_BOOT = 1000
SUB_SIZES = [32, 48]
SEED = 42


def main() -> None:
    raw = load_all_eval_stats()
    last = raw[
        raw["phase"] == raw.groupby(["algo", "env_short"])["phase"].transform("max")
    ]
    final_cfg = (
        last.groupby(["algo", "env_short", "config"])["success"].mean().reset_index()
    )

    rng = np.random.default_rng(SEED)
    rows = []
    for env, grp in final_cfg.groupby("env_short"):
        agents_present = sorted(grp["algo"].unique())
        full_means = {
            a: grp[grp["algo"] == a]["success"].mean() for a in agents_present
        }
        full_rank = sorted(full_means, key=full_means.get, reverse=True)

        for k in SUB_SIZES:
            agree = {p: 0 for p in combinations(full_rank, 2)}
            valid = 0
            for _ in range(N_BOOT):
                bm = {}
                for a, g in grp.groupby("algo"):
                    vals = g["success"].values
                    if len(vals) < k:
                        continue
                    idx = rng.choice(len(vals), size=k, replace=False)
                    bm[a] = vals[idx].mean()
                if len(bm) < 2:
                    continue
                valid += 1
                for a1, a2 in combinations(full_rank, 2):
                    if a1 in bm and a2 in bm and bm[a1] > bm[a2]:
                        agree[(a1, a2)] += 1
            for (a1, a2), c in agree.items():
                rows.append(
                    dict(
                        env=env,
                        k=k,
                        a1=a1,
                        a2=a2,
                        pct_agree=round(c / valid * 100, 1) if valid else float("nan"),
                    )
                )

    df = pd.DataFrame(rows)
    piv = df.pivot_table(
        index=["env", "a1", "a2"], columns="k", values="pct_agree"
    ).reset_index()
    piv.columns = ["env", "a1", "a2", "agreement_k32", "agreement_k48"]
    piv["env_order"] = piv["env"].map({e: i for i, e in enumerate(ENVS)})
    close = (
        piv[(piv["agreement_k32"] < 95) | (piv["agreement_k48"] < 95)]
        .sort_values(["env_order", "a1", "a2"])
        .drop(columns=["env_order"])
    )

    save_csv(close, "a4_subsample_stability")
    print()
    print(close.to_string(index=False))


if __name__ == "__main__":
    main()
