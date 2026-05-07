"""Reproduce `tab:antmaze_ess_phase4` (main) and `tab:antmaze_ess_full` (appendix)
from `PAPER/experiments.tex` and `PAPER/APPENDIX/robustness.tex`.

Both tables share a single computation:
  - ESS per (algo, config, seed, phase) = Kish on training-time AWR weights
    (clipped at w_max=100), where the weights come from the per-step
    `advantage/actor` array logged inside `train_log.csv`.
  - Reported as Mean ± Std across (config, seed) and Pearson r between ESS
    and per-(algo, env, phase) normalised return.

`tab:antmaze_ess_phase4` reports the row for Phase 4 only.
`tab:antmaze_ess_full` reports all 4 phases.

Inputs:
  data/zips/2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip
Output:
  outputs/t3_antmaze_ess.csv  (one row per (agent, phase))

Requires the gcrl_landscapes package + the project venv (because we use
`gcrl_landscapes.evaluation.tabular::compute_merged_df` to pre-process the zip).
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import gcrl_landscapes.evaluation.tabular as _tabular

_tabular.args = types.SimpleNamespace(no_multiprocessing=True)

from gcrl_landscapes.evaluation.tabular import compute_merged_df  # noqa: E402
from gcrl_landscapes.util.data import load_or_compute  # noqa: E402

from _common import save_csv, zip_path  # noqa: E402

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


def main() -> None:
    zp = zip_path(ENV)
    if not zp.exists():
        raise FileNotFoundError(zp)
    print(f"Loading {zp.name} (cached on first run)…")
    merged_results, merged_training = load_or_compute([zp], compute_merged_df)

    adv_col = "advantage/actor"
    merged_training[adv_col] = merged_training[adv_col].apply(_to_array)

    # Build per-row ESS
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
                "agent": row["hp.agent_name"],
                "dataset": row["dataset"],
                "config": row["config_index"],
                "seed": row["seed"],
                "eval_step": row["eval_step"],
                "ess": _ess(w),
            }
        )
    df = pd.DataFrame(rows)

    # Attach phase_num via merge with merged_results_df
    phase_map = merged_results[
        [
            "hp.agent_name",
            "dataset",
            "config_index",
            "eval_step",
            "phase_num",
            "mean_normalized_goal_distance_return",
        ]
    ].rename(
        columns={
            "hp.agent_name": "agent",
            "config_index": "config",
            "mean_normalized_goal_distance_return": "R",
        }
    )
    df = df.merge(
        phase_map, on=["agent", "dataset", "config", "eval_step"], how="inner"
    )

    # Normalise return per (agent, dataset, phase)
    grp_max = df.groupby(["agent", "dataset", "phase_num"])["R"].transform("max")
    df["R_norm"] = df["R"] / grp_max.replace(0, np.nan)

    # Aggregate per (agent, phase)
    out_rows = []
    for (agent, phase), g in df.groupby(["agent", "phase_num"]):
        valid = g[["ess", "R_norm"]].dropna()
        if len(valid) < 3:
            continue
        r, p = pearsonr(valid["ess"].values, valid["R_norm"].values)
        out_rows.append(
            {
                "agent": agent.upper(),
                "phase": int(phase),
                "ess_mean": round(float(g["ess"].mean()), 3),
                "ess_std": round(float(g["ess"].std()), 3),
                "pearson_r": round(float(r), 3),
                "p_value": round(float(p), 4),
            }
        )
    out = pd.DataFrame(out_rows).sort_values(["agent", "phase"]).reset_index(drop=True)
    out["agent"] = pd.Categorical(out["agent"], categories=AGENTS, ordered=True)
    out = out.sort_values(["agent", "phase"])

    save_csv(out, "t3_antmaze_ess")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
