"""R-L3: realized-selectivity analysis (7VBe-Q1, fills {RESULT-SELECTIVITY}).

Two parts, final phase only:

1. Curves: per (env, agent), ESS(alpha) and saturation-mass(alpha) from the
   precomputed 60-point alpha sweep (awr_alpha_sweep.csv), median across
   configs of the per-config seed-mean, with IQR band.  Shows what
   selectivity the sweep's alpha range actually realizes.

2. Binning: configs binned by their *realized* ESS (awr_concentration.csv,
   seed-mean ess_mean at the config's own alpha); per bin reports n, mean and
   max seed-IQM success, and rho_abs(0.25).  Links realized selectivity to
   success breadth without any causal claim.

Inputs:  RAW-ADV/ENVS/{env}/awr_alpha_sweep.csv, awr_concentration.csv,
         data/eval_stats/{env}.csv
Outputs: outputs/rl3_selectivity_curves.png/.pdf
         outputs/rl3_ess_bins.csv, outputs/rl3_ess_bins.tex
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# Okabe-Ito (CVD-safe), fixed agent order; line styles as secondary encoding
AGENT_COLOR = {
    "CRL": "#0072B2",
    "GCIQL": "#E69F00",
    "GCIVL": "#009E73",
    "QRL": "#D55E00",
}
AGENT_STYLE = {"CRL": "-", "GCIQL": "--", "GCIVL": "-.", "QRL": ":"}

ESS_BINS = [0.0, 0.1, 0.3, 1.0]
ESS_LABELS = ["ESS<0.1", "0.1-0.3", ">=0.3"]


def _iqm(v: np.ndarray) -> float:
    return float(trim_mean(v, proportiontocut=0.25))


def plot_curves() -> None:
    fig, axes = plt.subplots(
        2, len(ENVS), figsize=(3.4 * len(ENVS), 5.6), sharex=True, sharey="row"
    )
    for j, env in enumerate(ENVS):
        sweep = pd.read_csv(RAW_ADV_ENVS / env / "awr_alpha_sweep.csv")
        for metric, i in (("ess", 0), ("saturation_mass", 1)):
            ax = axes[i, j]
            for agent in AGENTS:
                s = sweep[sweep["agent"] == agent]
                if s.empty:
                    continue
                per_cfg = (
                    s.groupby(["configuration", "alpha_sweep"])[metric]
                    .mean()
                    .reset_index()
                )
                q = (
                    per_cfg.groupby("alpha_sweep")[metric]
                    .quantile([0.25, 0.5, 0.75])
                    .unstack()
                )
                ax.plot(
                    q.index,
                    q[0.5],
                    AGENT_STYLE[agent],
                    color=AGENT_COLOR[agent],
                    lw=2,
                    label=agent,
                )
                ax.fill_between(
                    q.index,
                    q[0.25],
                    q[0.75],
                    color=AGENT_COLOR[agent],
                    alpha=0.12,
                    lw=0,
                )
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(env, fontsize=11)
            if i == 1:
                ax.set_xlabel(r"AWR temperature $\alpha$", fontsize=9)
            if j == 0:
                ax.set_ylabel(
                    "ESS (Kish, normalized)" if i == 0 else "Saturation mass",
                    fontsize=9,
                )
            ax.grid(True, which="both", alpha=0.15, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUTPUTS / f"rl3_selectivity_curves.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  → {p}")
    plt.close(fig)


def bin_table() -> pd.DataFrame:
    df = load_all_eval_stats()
    last = df[
        df["phase"] == df.groupby(["algo", "env_short"])["phase"].transform("max")
    ]

    rows = []
    for env in ENVS:
        conc = pd.read_csv(RAW_ADV_ENVS / env / "awr_concentration.csv")
        ess = (
            conc.groupby(["agent", "configuration"])["ess_mean"]
            .mean()
            .reset_index()
            .rename(columns={"agent": "algo", "configuration": "config"})
        )
        for agent in AGENTS:
            g = last[(last["env_short"] == env) & (last["algo"] == agent)]
            iqm = (
                g.groupby("config")["success"]
                .apply(lambda v: _iqm(v.values))
                .reset_index(name="success_iqm")
            )
            merged = iqm.merge(ess[ess["algo"] == agent], on="config", how="inner")
            merged["bin"] = pd.cut(
                merged["ess_mean"], ESS_BINS, labels=ESS_LABELS, include_lowest=True
            )
            for lab in ESS_LABELS:
                sub = merged[merged["bin"] == lab]
                rows.append(
                    {
                        "env": env,
                        "agent": agent,
                        "ess_bin": lab,
                        "n_configs": len(sub),
                        "mean_success": round(float(sub["success_iqm"].mean()), 3)
                        if len(sub)
                        else float("nan"),
                        "max_success": round(float(sub["success_iqm"].max()), 3)
                        if len(sub)
                        else float("nan"),
                        "rho_abs_0.25": round(
                            float((sub["success_iqm"] >= 0.25).mean()), 3
                        )
                        if len(sub)
                        else float("nan"),
                    }
                )
    out = pd.DataFrame(rows)
    save_csv(out, "rl3_ess_bins")

    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Env & Agent & ESS bin & $n$ & Mean & Max & $\rho^{\mathrm{abs}}(0.25)$ \\",
        r"\midrule",
    ]
    for _, r in out.iterrows():
        if r["n_configs"] == 0:
            continue
        lines.append(
            f"{r['env']} & {r['agent']} & {r['ess_bin']} & {r['n_configs']} & "
            f"{r['mean_success']:.2f} & {r['max_success']:.2f} & {r['rho_abs_0.25']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = OUTPUTS / "rl3_ess_bins.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"  → {tex}")
    return out


def main() -> None:
    plot_curves()
    out = bin_table()
    print()
    print(out[out["n_configs"] > 0].to_string(index=False))


if __name__ == "__main__":
    main()
