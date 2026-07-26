"""R-L2: restricted-domain re-analysis (7VBe-Q1, fills {RESULT-ALPHA}).

Recompute the final-phase landscape summary after restricting the config
domain two ways, separately:
  (a) alpha <= 10           (drop the high-temperature tail of the sweep)
  (b) saturation_mass < 0.2 (drop configs whose AWR weights are saturated,
                             from awr_concentration.csv at the final phase,
                             per-config seed-mean of saturation_mass_mean)

For each variant reports remaining n_configs, mean/max seed-IQM success,
rho_0.9 / rho_0.8 relative to the *restricted* max, and absolute breadth
rho_abs(0.25) / rho_abs(0.5), next to the unrestricted baseline; a `flip`
column marks cells where the qualitative broad/narrow call changes
(rho_0.8 crossing 0.25, or the max moving by more than 0.1).

Inputs:  data/eval_stats/{env}.csv
         RAW-ADV/ENVS/{env}/awr_concentration.csv
Outputs: outputs/rl2_restricted_domain.csv, outputs/rl2_restricted_domain.tex
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from _common import AGENTS, ENVS, OUTPUTS, load_all_eval_stats, save_csv

RAW_ADV_ENVS = Path(
    os.environ.get(
        "RAW_ADV_ENVS",
        str(Path(__file__).resolve().parents[2] / "Analysis" / "RAW-ADV" / "ENVS"),
    )
)
ALPHA_MAX = 10.0
SAT_MAX = 0.2
RHO08_FLIP = 0.25  # broad/narrow call boundary on rho_0.8
MAX_FLIP = 0.10  # meaningful shift in max success


def _iqm(v: np.ndarray) -> float:
    return float(trim_mean(v, proportiontocut=0.25))


def load_saturation(env: str) -> pd.DataFrame:
    """Per (agent, config) seed-mean saturation_mass at the final phase.

    awr_concentration.csv holds exactly one checkpoint step per agent (that
    agent's final phase), so no phase filtering is needed — and filtering on
    the global max step would silently keep only one agent.
    """
    c = pd.read_csv(RAW_ADV_ENVS / env / "awr_concentration.csv")
    return (
        c.groupby(["agent", "configuration"])["saturation_mass_mean"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "agent": "algo",
                "configuration": "config",
                "saturation_mass_mean": "saturation",
            }
        )
    )


def summarize(iqm: pd.DataFrame) -> dict:
    s = iqm["success_iqm"].values
    mx = float(s.max()) if len(s) else float("nan")
    return {
        "n_configs": len(s),
        "mean_success": round(float(s.mean()), 3) if len(s) else float("nan"),
        "max_success": round(mx, 3),
        "rho_0.9": round(float((s >= 0.9 * mx).mean()), 3) if mx > 0 else float("nan"),
        "rho_0.8": round(float((s >= 0.8 * mx).mean()), 3) if mx > 0 else float("nan"),
        "rho_abs_0.25": round(float((s >= 0.25).mean()), 3) if len(s) else float("nan"),
        "rho_abs_0.5": round(float((s >= 0.5).mean()), 3) if len(s) else float("nan"),
    }


def main() -> None:
    df = load_all_eval_stats()
    last = df[
        df["phase"] == df.groupby(["algo", "env_short"])["phase"].transform("max")
    ]

    rows = []
    for env in ENVS:
        sat = load_saturation(env)
        for agent in AGENTS:
            g = last[(last["env_short"] == env) & (last["algo"] == agent)]
            if g.empty:
                continue
            iqm = (
                g.groupby(["config", "alpha"])["success"]
                .apply(lambda v: _iqm(v.values))
                .reset_index(name="success_iqm")
            )
            base = summarize(iqm)

            variants = {
                "unrestricted": iqm,
                "alpha_le_10": iqm[iqm["alpha"] <= ALPHA_MAX],
            }
            s_ag = sat[sat["algo"] == agent]
            keep = set(s_ag[s_ag["saturation"] < SAT_MAX]["config"])
            variants["low_saturation"] = iqm[iqm["config"].isin(keep)]

            for name, sub in variants.items():
                summ = summarize(sub)
                flip = ""
                if name != "unrestricted" and summ["n_configs"] > 0:
                    flags = []
                    if (base["rho_0.8"] >= RHO08_FLIP) != (
                        summ["rho_0.8"] >= RHO08_FLIP
                    ):
                        flags.append("rho0.8")
                    if abs(base["max_success"] - summ["max_success"]) > MAX_FLIP:
                        flags.append("max")
                    flip = "+".join(flags)
                rows.append(
                    {"env": env, "agent": agent, "variant": name, **summ, "flip": flip}
                )

    out = pd.DataFrame(rows)
    save_csv(out, "rl2_restricted_domain")

    lines = [
        r"\begin{tabular}{lllrrrrrrr}",
        r"\toprule",
        r"Env & Agent & Domain & $n$ & Mean & Max & $\rho_{0.9}$ & $\rho_{0.8}$ & $\rho^{\mathrm{abs}}(0.25)$ & $\rho^{\mathrm{abs}}(0.5)$ \\",
        r"\midrule",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"{r['env']} & {r['agent']} & {r['variant'].replace('_', ' ')} & {r['n_configs']} & "
            f"{r['mean_success']:.2f} & {r['max_success']:.2f} & {r['rho_0.9']:.2f} & "
            f"{r['rho_0.8']:.2f} & {r['rho_abs_0.25']:.2f} & {r['rho_abs_0.5']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = OUTPUTS / "rl2_restricted_domain.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"  → {tex}")

    flips = out[out["flip"] != ""]
    print()
    print(out.to_string(index=False))
    print()
    if flips.empty:
        print("No qualitative flips under either restriction.")
    else:
        print("FLIPPED CELLS (need honest reporting before drafting):")
        print(
            flips[
                [
                    "env",
                    "agent",
                    "variant",
                    "n_configs",
                    "max_success",
                    "rho_0.8",
                    "flip",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
