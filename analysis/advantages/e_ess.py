# ruff: noqa
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from gcrl_landscapes.util.data import load_or_compute
from gcrl_landscapes.evaluation.tabular import compute_merged_df
from _common import parse_args, build_adv_df, _pearson_r, _ess

zipfiles, plot_dir = parse_args()
os.makedirs(plot_dir, exist_ok=True)

merged_results_df, merged_training_df = load_or_compute(zipfiles, compute_merged_df)
adv_data = load_or_compute(zipfiles, build_adv_df)
adv_all_df = adv_data["adv_all_df"]
ADV_COLS = adv_data["ADV_COLS"]


def _save_tex(filename: Path, caption: str, label: str, df: pd.DataFrame) -> None:
    """Write a DataFrame to a booktabs-style .tex table."""
    col_format = "l" + "r" * len(df.columns)
    header = " & ".join(df.columns.tolist()) + " \\midrule\n"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        rows.append(" & ".join(vals) + " \\")
    body = "\n".join(rows)

    tex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\begin{{tabular}}{{{col_format}}}\n"
        "\\toprule\n" + header + body + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}"
    )
    with open(filename, "w") as f:
        f.write(tex)
    print(f"Saved {filename}")


# --- E: ESS per phase + correlation ---

if ADV_COLS and merged_results_df is not None:
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
    grp_keys = ["hp.agent_name", "dataset", "phase_num"]
    _max = perf_per_phase.groupby(grp_keys)[
        "mean_normalized_goal_distance_return"
    ].transform("max")
    perf_per_phase["return_normalized"] = perf_per_phase[
        "mean_normalized_goal_distance_return"
    ] / _max.replace(0, float("nan"))

    ess_phased = (
        adv_all_df.groupby(
            ["hp.agent_name", "dataset", "actor", "config_index", "seed", "phase_num"]
        )["weight"]
        .apply(_ess)  # type: ignore[arg-type]
        .reset_index(name="ess")
    )
    print("\nNormalized ESS per config per phase -- statistics:")
    print(ess_phased.groupby(["hp.agent_name", "actor", "phase_num"])["ess"].describe())

    corr_ess_phased_df = ess_phased.merge(
        perf_per_phase,
        on=["hp.agent_name", "dataset", "config_index", "seed", "phase_num"],
        how="inner",
    )

    corr_summary = (
        corr_ess_phased_df.groupby(["hp.agent_name", "actor", "phase_num"])
        .apply(lambda g: _pearson_r(g, "ess", "return_normalized"))
        .reset_index()
        .sort_values(["hp.agent_name", "actor", "phase_num"])
    )
    corr_summary.rename(columns={"pearson_r": "r"}, inplace=True)
    print("\nCorrelation between ESS and normalized return, per phase:")
    print(corr_summary.to_string(index=False))

    # --- .tex output ---
    ess_stats = (
        ess_phased.groupby(["hp.agent_name", "actor", "phase_num"])["ess"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={"hp.agent_name": "Agent", "actor": "Actor", "phase_num": "Phase"}
        )
        .sort_values(["Agent", "Actor", "Phase"])
    )
    ess_stats.columns = ["Agent", "Actor", "Phase", "Mean", "Std"]

    corr_summary_r = corr_summary[
        ["hp.agent_name", "actor", "phase_num", "r", "p_value"]
    ].copy()
    corr_summary_r.columns = ["Agent", "Actor", "Phase", "Corr", "p-value"]

    combined = ess_stats.merge(corr_summary_r, on=["Agent", "Actor", "Phase"])
    combined = combined[["Agent", "Actor", "Phase", "Mean", "Std", "Corr", "p-value"]]
    combined["ESS"] = (
        combined["Mean"].apply(lambda x: f"{x:.3f}")
        + r" \$\pm\$ "
        + combined["Std"].apply(lambda x: f"{x:.3f}")
    )
    combined = combined[["Agent", "Actor", "Phase", "ESS", "Corr", "p-value"]]

    _save_tex(
        plot_dir / "ess.tex",
        "Effective sample size statistics with correlation between ESS and normalized return per agent, actor, and phase.",
        "tab:ess",
        combined,
    )
