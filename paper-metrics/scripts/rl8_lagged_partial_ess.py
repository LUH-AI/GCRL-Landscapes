"""R-L8: phase-lagged and partial ESS correlations on AntMaze-Medium (7VBe-Q2).

Descriptive only — no causal language.  Two additions to the t3 ESS table:

1. Lagged: Pearson/Spearman corr between ESS at phase t (per (config, seed),
   mean over eval steps within the phase, same weight construction as t3)
   and eval success at phase t+1, next to the contemporaneous corr at t.
2. Partial: per (agent, phase), partial corr of ESS with success controlling
   for log10(lr), by residualizing both on log10(lr) within agent.

Inputs:  data/zips/2026-04-26-...antmaze-medium-fixed-batch.zip (via the
         shared merged-DataFrame cache), data/eval_stats/antmaze-medium.csv
Outputs: outputs/rl8_lagged_partial_ess.csv
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

import gcrl_landscapes.evaluation.tabular as _tabular  # noqa: E402

_tabular.args = types.SimpleNamespace(no_multiprocessing=True)

from gcrl_landscapes.evaluation.tabular import compute_merged_df  # noqa: E402
from gcrl_landscapes.util.data import load_or_compute  # noqa: E402

from _common import eval_stats_path, save_csv, zip_path  # noqa: E402

CLIP = 100.0
ENV = "antmaze-medium"
AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]


def _ess(weights: np.ndarray) -> float:
    w = weights.clip(min=1e-9, max=CLIP)
    if len(w) == 0:
        return float("nan")
    w_sc = w / w.max()
    return float(w_sc.sum() ** 2 / (len(w_sc) * (w_sc**2).sum()))


def _to_array(s):
    try:
        return np.array(ast.literal_eval(s), dtype=np.float32)
    except Exception:
        return None


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Pearson corr of x and y after residualizing both on z (with intercept)."""
    Z = np.column_stack([np.ones_like(z), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(pearsonr(rx, ry).statistic)


def build_ess() -> pd.DataFrame:
    """ESS per (agent, config, seed, phase): mean over eval steps in phase."""
    zp = zip_path(ENV)
    merged_results, merged_training = load_or_compute([zp], compute_merged_df)

    adv_col = "advantage/actor"
    merged_training[adv_col] = merged_training[adv_col].apply(_to_array)

    rows = []
    for _, row in merged_training.iterrows():
        adv = row.get(adv_col)
        if adv is None or not isinstance(adv, np.ndarray):
            continue
        alpha = float(row.get("hp.alpha", np.nan))
        if not np.isfinite(alpha):
            continue
        w = np.exp(alpha * adv.astype(np.float64))
        rows.append(
            {
                "agent": row["hp.agent_name"].upper(),
                "dataset": row["dataset"],
                "config": row["config_index"],
                "seed": row["seed"],
                "eval_step": row["eval_step"],
                "ess": _ess(w),
            }
        )
    df = pd.DataFrame(rows)

    phase_map = (
        merged_results[
            ["hp.agent_name", "dataset", "config_index", "eval_step", "phase_num"]
        ]
        .assign(**{"hp.agent_name": lambda d: d["hp.agent_name"].str.upper()})
        .rename(columns={"hp.agent_name": "agent", "config_index": "config"})
    )
    df = df.merge(
        phase_map, on=["agent", "dataset", "config", "eval_step"], how="inner"
    )
    return (
        df.groupby(["agent", "config", "seed", "phase_num"])["ess"]
        .mean()
        .reset_index()
        .rename(columns={"phase_num": "phase"})
    )


def main() -> None:
    ess = build_ess()
    ess["phase"] = ess["phase"].astype(int)

    ev = pd.read_csv(eval_stats_path(ENV)).rename(columns={"algo": "agent"})
    ev = ev[["agent", "config", "seed", "phase", "lr", "success"]]

    cur = ess.merge(ev, on=["agent", "config", "seed", "phase"], how="inner")
    nxt = ess.merge(
        ev.assign(phase=ev["phase"] - 1),
        on=["agent", "config", "seed", "phase"],
        how="inner",
    )

    rows = []
    for agent in AGENTS:
        for phase in sorted(cur["phase"].unique()):
            g = cur[(cur["agent"] == agent) & (cur["phase"] == phase)].dropna(
                subset=["ess", "success"]
            )
            gn = nxt[(nxt["agent"] == agent) & (nxt["phase"] == phase)].dropna(
                subset=["ess", "success"]
            )
            if len(g) < 10:
                continue
            r_now = pearsonr(g["ess"], g["success"]).statistic
            s_now = spearmanr(g["ess"], g["success"]).statistic
            r_par = _partial_corr(
                g["ess"].values, g["success"].values, np.log10(g["lr"].values)
            )
            row = {
                "agent": agent,
                "phase": phase,
                "n": len(g),
                "r_ess_success_t": round(float(r_now), 3),
                "spearman_t": round(float(s_now), 3),
                "r_partial_given_log_lr": round(float(r_par), 3),
            }
            if len(gn) >= 10:
                row["r_ess_t_success_t+1"] = round(
                    float(pearsonr(gn["ess"], gn["success"]).statistic), 3
                )
                row["spearman_t+1"] = round(
                    float(spearmanr(gn["ess"], gn["success"]).statistic), 3
                )
                row["n_lagged"] = len(gn)
            rows.append(row)

    out = pd.DataFrame(rows)
    save_csv(out, "rl8_lagged_partial_ess")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
