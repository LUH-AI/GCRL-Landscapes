"""R-L4: epsilon-sensitivity of the relative breadth metric rho_eps.

Appendix-only robustness check (risk-audit point A): the paper's broad/narrow
distinction uses rho_0.9 / rho_0.8.  This sweeps eps in {0.70, 0.75, ...,
0.95} on final-phase seed-IQM success (t1 conventions) with a 2000-draw
config bootstrap CI per cell, and checks whether the broad-vs-narrow
separation between methods is stable in eps.

Inputs:  data/eval_stats/{env}.csv
Outputs: outputs/rl4_eps_sensitivity.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from _common import AGENTS, ENVS, load_all_eval_stats, save_csv

EPS_GRID = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
N_BOOT = 2000
RNG_SEED = 0


def main() -> None:
    df = load_all_eval_stats()
    last = df[
        df["phase"] == df.groupby(["algo", "env_short"])["phase"].transform("max")
    ]
    iqm = (
        last.groupby(["algo", "env_short", "config"])["success"]
        .apply(lambda v: trim_mean(v, proportiontocut=0.25))
        .reset_index(name="success_iqm")
    )

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for env in ENVS:
        for agent in AGENTS:
            s = iqm[(iqm["env_short"] == env) & (iqm["algo"] == agent)][
                "success_iqm"
            ].values
            if len(s) == 0:
                continue
            idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
            boot = s[idx]
            row = {"env": env, "agent": agent, "max_success": round(float(s.max()), 3)}
            for eps in EPS_GRID:
                obs = (
                    float((s >= eps * s.max()).mean()) if s.max() > 0 else float("nan")
                )
                bmax = boot.max(axis=1)
                bstat = np.where(
                    bmax > 0, (boot >= eps * bmax[:, None]).mean(axis=1), np.nan
                )
                lo, hi = np.nanpercentile(bstat, [2.5, 97.5])
                row[f"rho_{eps:.2f}"] = round(obs, 3)
                row[f"rho_{eps:.2f}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
            rows.append(row)

    out = pd.DataFrame(rows)
    save_csv(out, "rl4_eps_sensitivity")

    # Stability check: per env, Spearman-style order agreement between the
    # agent ranking at each eps and the ranking at eps=0.90 (paper headline).
    print()
    point_cols = [f"rho_{e:.2f}" for e in EPS_GRID]
    for env in ENVS:
        sub = out[out["env"] == env].set_index("agent")
        ref = sub["rho_0.90"].rank()
        agree = {
            c: float((sub[c].rank().corr(ref, method="spearman"))) for c in point_cols
        }
        print(
            f"{env}: agent-ranking agreement with eps=0.90 → "
            + ", ".join(f"{c.split('_')[1]}: {v:.2f}" for c, v in agree.items())
        )
    print()
    print(out[["env", "agent", "max_success"] + point_cols].to_string(index=False))


if __name__ == "__main__":
    main()
