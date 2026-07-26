"""R-L1: absolute-threshold breadth rho^abs(0.25), rho^abs(0.5).

Rebuttal analysis (GRtV-Q3, fills {RESULT-ABS-BREADTH}).

Relative breadth rho_eps = P(success >= eps * max) is vulnerable to floor
effects when max is small.  This reports the fraction of configs whose
seed-IQM success clears an *absolute* bar (0.25, 0.5) in the final phase,
with 2000-draw bootstrap CIs computed two ways:
  - over configs (resample the 64 configs with replacement)
  - over seeds   (resample each config's seeds with replacement, re-IQM)

Conventions match t1_final_landscape_ci.py: final phase, seed reduction via
scipy.stats.trim_mean(v, 0.25) per config.

Inputs:  data/eval_stats/{env}.csv
Outputs: outputs/rl1_abs_breadth.csv, outputs/rl1_abs_breadth.tex
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from _common import AGENTS, ENVS, OUTPUTS, load_all_eval_stats, save_csv

THRESHOLDS = [0.25, 0.5]
N_BOOT = 2000
RNG_SEED = 0


def _iqm(v: np.ndarray) -> float:
    return float(trim_mean(v, proportiontocut=0.25))


def bootstrap_configs(
    iqm_vals: np.ndarray, thr: float, rng: np.random.Generator
) -> tuple[float, float]:
    n = len(iqm_vals)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    stats = (iqm_vals[idx] >= thr).mean(axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def bootstrap_seeds(
    seed_mat: list[np.ndarray], thr: float, rng: np.random.Generator
) -> tuple[float, float]:
    """seed_mat: list (per config) of per-seed success arrays."""
    stats = np.empty(N_BOOT)
    for b in range(N_BOOT):
        frac = 0
        for v in seed_mat:
            res = v[rng.integers(0, len(v), size=len(v))]
            if _iqm(res) >= thr:
                frac += 1
        stats[b] = frac / len(seed_mat)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main() -> None:
    df = load_all_eval_stats()
    last = df[
        df["phase"] == df.groupby(["algo", "env_short"])["phase"].transform("max")
    ]

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for env in ENVS:
        for agent in AGENTS:
            g = last[(last["env_short"] == env) & (last["algo"] == agent)]
            if g.empty:
                continue
            per_cfg = g.groupby("config")["success"].apply(lambda v: v.values)
            seed_mat = list(per_cfg.values)
            iqm_vals = np.array([_iqm(v) for v in seed_mat])
            row = {
                "env": env,
                "agent": agent,
                "n_configs": len(iqm_vals),
                "max_success": round(float(iqm_vals.max()), 3),
                "mean_success": round(float(iqm_vals.mean()), 3),
            }
            for thr in THRESHOLDS:
                key = f"rho_abs_{thr}"
                row[key] = round(float((iqm_vals >= thr).mean()), 3)
                lo_c, hi_c = bootstrap_configs(iqm_vals, thr, rng)
                lo_s, hi_s = bootstrap_seeds(seed_mat, thr, rng)
                row[f"{key}_ci_cfg"] = f"[{lo_c:.3f}, {hi_c:.3f}]"
                row[f"{key}_ci_seed"] = f"[{lo_s:.3f}, {hi_s:.3f}]"
            rows.append(row)

    out = pd.DataFrame(rows)
    save_csv(out, "rl1_abs_breadth")

    # LaTeX table
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Env & Agent & $\rho^{\mathrm{abs}}(0.25)$ & 95\% CI (cfg) & $\rho^{\mathrm{abs}}(0.5)$ & 95\% CI (cfg) \\",
        r"\midrule",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"{r['env']} & {r['agent']} & {r['rho_abs_0.25']:.2f} & {r['rho_abs_0.25_ci_cfg']} & "
            f"{r['rho_abs_0.5']:.2f} & {r['rho_abs_0.5_ci_cfg']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = OUTPUTS / "rl1_abs_breadth.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"  → {tex}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
