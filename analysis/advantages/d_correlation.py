# ruff: noqa
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from gcrl_landscapes.util.data import load_or_compute
from gcrl_landscapes.evaluation.tabular import compute_merged_df
from _common import parse_args, build_adv_df, _pearson_r

zipfiles, plot_dir, top_k = parse_args()
os.makedirs(plot_dir, exist_ok=True)

merged_results_df, merged_training_df = load_or_compute(zipfiles, compute_merged_df)
adv_data = load_or_compute(zipfiles, lambda: build_adv_df(zipfiles, top_k=top_k))
adv_all_df = adv_data["adv_all_df"]
adv_last_phase_df = adv_data["adv_last_phase_df"]
ADV_COLS = adv_data["ADV_COLS"]

# --- D: High-weight ratio correlation with performance, per phase ---

if ADV_COLS and merged_results_df is not None:
    high_ratio_phased = (
        adv_all_df.groupby(
            ["hp.agent_name", "dataset", "actor", "config_index", "seed", "phase_num"]
        )["weight"]
        .apply(lambda w: (w > 100).sum() / len(w))  # type: ignore[arg-type]
        .reset_index(name="high_ratio")
    )

    perf_per_phase = merged_results_df[
        [
            "hp.agent_name",
            "dataset",
            "config_index",
            "seed",
            "phase_num",
            "mean_normalized_goal_distance_return",
        ]
    ].copy()
    # Normalize return to [0, 1] per (agent, phase, env) using max scaling
    grp_keys = ["hp.agent_name", "dataset", "phase_num", "dataset"]
    _max = perf_per_phase.groupby(grp_keys)[
        "mean_normalized_goal_distance_return"
    ].transform("max")
    perf_per_phase["return_normalized"] = perf_per_phase[
        "mean_normalized_goal_distance_return"
    ] / _max.replace(0, float("nan"))

    corr_phased_df = high_ratio_phased.merge(
        perf_per_phase,
        on=["hp.agent_name", "config_index", "seed", "phase_num"],
        how="inner",
    )

    print(
        "\nCorrelation between fraction of weights > 100 and normalized return, per phase:"
    )
    print(
        corr_phased_df.groupby(["hp.agent_name", "actor", "phase_num"])
        .apply(lambda g: _pearson_r(g, "high_ratio", "return_normalized"))
        .reset_index()
        .sort_values(["hp.agent_name", "actor", "phase_num"])
        .to_string(index=False)
    )
