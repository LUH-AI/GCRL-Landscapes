"""Reproduce `tab:final_landscape_ci` from `PAPER/experiments.tex`.

Per (env, agent) row reports:
  - Max success      = max over configs of (per-config IQM seed-aggregated success)
  - Mean success     = mean over configs of the same
  - rho_0.9          = fraction of configs with seed-IQM success >= 0.9 * max
  - rho_0.8          = fraction of configs with seed-IQM success >= 0.8 * max

Last phase only (per (algo, env_short)).

Inputs:
  data/eval_stats/{antmaze-medium,antmaze-large,cube,scene}.csv
Output:
  outputs/t1_final_landscape_ci.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from scipy.stats import trim_mean

from _common import AGENTS, ENVS, load_all_eval_stats, save_csv


def main() -> None:
    df = load_all_eval_stats()
    last = df[
        df["phase"] == df.groupby(["algo", "env_short"])["phase"].transform("max")
    ]

    # Seed-IQM per (agent, env, config) — drops the highest and lowest of the 5 seeds
    iqm = (
        last.groupby(["algo", "env_short", "config"])["success"]
        .apply(lambda v: trim_mean(v, proportiontocut=0.25))
        .reset_index(name="success_iqm")
    )

    rows = []
    for (agent, env), g in iqm.groupby(["algo", "env_short"]):
        s = g["success_iqm"].values
        mx = s.max()
        rows.append(
            {
                "env": env,
                "agent": agent,
                "max_success": round(float(mx), 3),
                "mean_success": round(float(s.mean()), 3),
                "rho_0.9": round(float((s >= 0.9 * mx).mean()), 3)
                if mx > 0
                else float("nan"),
                "rho_0.8": round(float((s >= 0.8 * mx).mean()), 3)
                if mx > 0
                else float("nan"),
            }
        )

    out = pd.DataFrame(rows)
    # Order by env then agent for the paper layout
    out["env_order"] = out["env"].map({e: i for i, e in enumerate(ENVS)})
    out["agent_order"] = out["agent"].map({a: i for i, a in enumerate(AGENTS)})
    out = out.sort_values(["env_order", "agent_order"]).drop(
        columns=["env_order", "agent_order"]
    )

    save_csv(out, "t1_final_landscape_ci")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
