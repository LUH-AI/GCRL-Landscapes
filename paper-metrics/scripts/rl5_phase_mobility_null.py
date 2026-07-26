"""R-L5: phase-mobility null baseline (7VBe-W1, fills {RESULT-PHASE-NULL}).

The paper reports mean adjacent-phase Jaccard overlap of top-10% config sets
(a3_phase_mobility.py).  A reviewer can object that low Jaccard is expected
even for random sets.  This computes:

  1. The exact null: Jaccard of two independent uniform random k-of-n sets
     (n=64, k=ceil(0.1*64)=7) — analytic expectation via the hypergeometric
     intersection law, plus a 200k-draw permutation distribution for the
     95% interval.
  2. Observed per-env mean Jaccard (identical recipe to a3) with a
     1000-draw within-config seed bootstrap CI.
  3. A set-size-free alternative: Spearman rank correlation between
     adjacent-phase config orderings, same aggregation and bootstrap.

Verdict per env: does the observed mean Jaccard sit above the null 97.5th
percentile of the *mean over the same number of (algo, pair) cells*?

Inputs:  data/eval_stats/{env}.csv
Outputs: outputs/rl5_phase_null.csv
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import AGENTS, ENVS, load_all_eval_stats, save_csv

TOP_FRAC = 0.10
N_PERM = 200_000
N_BOOT = 1000
RNG_SEED = 0


def top_k_set(s: pd.Series, frac: float) -> set:
    k = max(1, int(np.ceil(frac * len(s))))
    return set(s.nlargest(k).index)


def analytic_null_jaccard(n: int, k: int) -> float:
    """E[J] for two independent uniform k-subsets of [n]."""
    total = comb(n, k)
    e = 0.0
    for i in range(0, k + 1):
        p = comb(k, i) * comb(n - k, k - i) / total
        e += p * (i / (2 * k - i))
    return e


def perm_null_mean(
    n: int, k: int, n_cells: int, rng: np.random.Generator
) -> np.ndarray:
    """Null distribution of the mean Jaccard over n_cells independent pairs."""
    draws = np.empty(N_PERM)
    for b in range(N_PERM):
        js = 0.0
        for _ in range(n_cells):
            a = set(rng.choice(n, size=k, replace=False))
            c = set(rng.choice(n, size=k, replace=False))
            js += len(a & c) / len(a | c)
        draws[b] = js / n_cells
    return draws


def observed_stats(cfg: pd.DataFrame, env: str) -> tuple[float, float, int]:
    """(mean jaccard, mean spearman, n_cells) over (algo, adjacent pair)."""
    js, rs = [], []
    for algo in AGENTS:
        grp = cfg[(cfg["algo"] == algo) & (cfg["env_short"] == env)]
        phases = sorted(grp["phase"].unique())
        for t_idx in range(len(phases) - 1):
            s_t = grp[grp["phase"] == phases[t_idx]].set_index("config")["success"]
            s_t1 = grp[grp["phase"] == phases[t_idx + 1]].set_index("config")["success"]
            T_t, T_t1 = top_k_set(s_t, TOP_FRAC), top_k_set(s_t1, TOP_FRAC)
            js.append(len(T_t & T_t1) / len(T_t | T_t1))
            common = s_t.index.intersection(s_t1.index)
            rs.append(spearmanr(s_t[common], s_t1[common]).statistic)
    return float(np.mean(js)), float(np.mean(rs)), len(js)


def seed_bootstrap(
    raw_env: pd.DataFrame, env: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap (mean jaccard, mean spearman) by resampling seeds within config."""
    j_draws, r_draws = np.empty(N_BOOT), np.empty(N_BOOT)
    # Pre-split per (algo, config, phase) seed arrays
    grouped = {
        key: g["success"].values
        for key, g in raw_env.groupby(["algo", "config", "phase"])
    }
    for b in range(N_BOOT):
        recs = []
        for (algo, config, phase), v in grouped.items():
            res = v[rng.integers(0, len(v), size=len(v))]
            recs.append((algo, config, phase, res.mean()))
        cfg_b = pd.DataFrame(recs, columns=["algo", "config", "phase", "success"])
        cfg_b["env_short"] = env
        j, r, _ = observed_stats(cfg_b, env)
        j_draws[b], r_draws[b] = j, r
    return j_draws, r_draws


def main() -> None:
    raw = load_all_eval_stats()
    cfg = (
        raw.groupby(["algo", "env_short", "config", "phase"])["success"]
        .mean()
        .reset_index()
    )

    n, k = 64, max(1, int(np.ceil(TOP_FRAC * 64)))
    e_null = analytic_null_jaccard(n, k)
    rng = np.random.default_rng(RNG_SEED)

    rows = []
    for env in ENVS:
        j_obs, r_obs, n_cells = observed_stats(cfg, env)
        null = perm_null_mean(n, k, n_cells, rng)
        j_lo, j_hi = np.percentile(null, [2.5, 97.5])
        jb, rb = seed_bootstrap(raw[raw["env_short"] == env], env, rng)
        rows.append(
            {
                "env": env,
                "n_cells": n_cells,
                "jaccard_obs": round(j_obs, 3),
                "jaccard_obs_ci": f"[{np.percentile(jb, 2.5):.3f}, {np.percentile(jb, 97.5):.3f}]",
                "null_mean": round(e_null, 3),
                "null_95": f"[{j_lo:.3f}, {j_hi:.3f}]",
                "clears_null": bool(j_obs > j_hi),
                "spearman_obs": round(r_obs, 3),
                "spearman_obs_ci": f"[{np.percentile(rb, 2.5):.3f}, {np.percentile(rb, 97.5):.3f}]",
            }
        )

    out = pd.DataFrame(rows)
    save_csv(out, "rl5_phase_null")
    print()
    print(f"n = {n}, k = {k}, analytic E[J] under independence = {e_null:.4f}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
