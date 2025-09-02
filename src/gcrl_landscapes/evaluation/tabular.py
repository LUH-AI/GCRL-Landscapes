import argparse
from gcrl_landscapes.util.data import (
    read_results_from_zip,
)
from gcrl_landscapes.util.eval import fit_model
from gcrl_landscapes.util.data import phase_results_to_pandas
from pathlib import Path
import pandas as pd
import numpy as np
import os
from .common import (
    compute_additional_information,
    merge_experiments,
    calculate_regret_for_experiment,
    CVAR_CONFIDENCE_LEVELS,
)
from gcrl_landscapes.configurations import hp_to_sobol_codomain, get_bounds
from scipy.stats import trim_mean
from scipy.optimize import shgo
from typing import Any


def create_phased_tables(results_pandas: pd.DataFrame, output_folder: Path):
    """Main code to generate the tabular data for data grouped by phases

    Args:
        results_pandas: pandas dataframe containing all results (all phases)
        output_folder: folder to save tables in
    """
    per_config_folder = output_folder / "per_config"
    per_config_folder.mkdir(exist_ok=True)

    phases = sorted(results_pandas["phase"].unique().tolist())
    phase_results = [
        (
            phase,
            results_pandas[
                (results_pandas["eval_step"] == phase)
                & (results_pandas["phase"] == phase)
            ],
        )
        for phase in phases
    ]

    results_only_final_eval_df = pd.concat(
        (results_df for _, results_df in phase_results)
    )

    table = results_only_final_eval_df.groupby(by=["agent", "dataset", "phase"]).agg(
        **{
            "Goal Distance Score": pd.NamedAgg(
                column="mean_normalized_goal_distance_return", aggfunc="mean"
            ),
            "Dispersion Score": pd.NamedAgg(
                column="disp_normalized_goal_distance_score", aggfunc="mean"
            ),
            **{
                f"CVaR{cvar_level} score": pd.NamedAgg(
                    column=f"cvar{cvar_level}_normalized_goal_distance_return",
                    aggfunc="mean",
                )
                for cvar_level in CVAR_CONFIDENCE_LEVELS
            },
        }
    )

    with open(output_folder / "table.md", "w") as f:
        f.write(table.to_markdown())

    with open(output_folder / "table.tex", "w") as f:
        f.write(table.to_latex())

    with open(output_folder / "table.csv", "w") as f:
        f.write(table.to_csv())


def create_igprfit_tables(results_pandas: pd.DataFrame, output_folder: Path):
    """Compute additional tabular data like IGPR fit

    Args:
        results_pandas: pandas dataframe containing all results (all phases)
        output_folder: folder to save tables in
    """
    # Compute k-fold cross validation fit per agent-dataset-phase combination for the IGPR model
    final_results_pandas = results_pandas[
        results_pandas["eval_step"] == results_pandas["phase"]
    ]
    table = final_results_pandas.groupby(by=["agent", "dataset", "phase"]).apply(
        lambda df: fit_model(
            df.reset_index(drop=True),
            "mean_normalized_goal_distance_return",
            [f"hp.{hp_name}" for hp_name in df["hps"].iloc[0]],
        )
        .estimate_iqm_fit()
        .drop(axis="columns", labels="fold")
        .mean(axis=0),
    )

    with open(output_folder / "additional_table.md", "w") as f:
        f.write(table.to_markdown())

    with open(output_folder / "additional_table.tex", "w") as f:
        f.write(table.to_latex())

    with open(output_folder / "additional_table.csv", "w") as f:
        f.write(table.to_csv())


def create_regret_table(results_pandas: pd.DataFrame, output_folder: Path):
    """Create regret tables

    Args:
        results_pandas: pandas dataframe containing all results (all phases)
        output_folder: folder to save tables in
    """
    final_results_pandas = results_pandas[
        results_pandas["eval_step"] == results_pandas["phase"]
    ]

    for regret_col, base_col in [
        (
            "mean_normalized_goal_distance_return_normalized_regret",
            "mean_normalized_goal_distance_return",
        ),
        (
            "mean_normalized_goal_distance_return_normalized_regret",
            "mean_normalized_goal_distance_return",
        ),
        ("success_regret", "success"),
        ("success_normalized_regret", "success"),
    ]:

        def calculate_regret(df: pd.DataFrame) -> pd.DataFrame:
            return calculate_regret_for_experiment(df, regret_col, base_col)

        regret_df = (
            final_results_pandas.groupby(by=["agent", "dataset"])
            .apply(calculate_regret)
            .reset_index(level="phase")
        )

        table_pick_first_phase = (
            regret_df.sort_values(["phase"], ascending=True)
            .groupby(by=regret_df.index.names)
            .nth(0)
        )
        table_pick_second_phase = (
            regret_df.sort_values(["phase"], ascending=True)
            .groupby(by=regret_df.index.names)
            .nth(1)
        )

        with open(output_folder / f"{regret_col}_table_first_phase.md", "w") as f:
            f.write(table_pick_first_phase.to_markdown())

        with open(output_folder / f"{regret_col}_table_first_phase.tex", "w") as f:
            f.write(table_pick_first_phase.to_latex())

        with open(output_folder / f"{regret_col}_table_first_phase.csv", "w") as f:
            f.write(table_pick_first_phase.to_csv())

        with open(output_folder / f"{regret_col}_table_second_phase.md", "w") as f:
            f.write(table_pick_second_phase.to_markdown())

        with open(output_folder / f"{regret_col}_table_second_phase.tex", "w") as f:
            f.write(table_pick_second_phase.to_latex())

        with open(output_folder / f"{regret_col}_table_second_phase.csv", "w") as f:
            f.write(table_pick_second_phase.to_csv())


def create_optimum_shift_table(results_pandas: pd.DataFrame, out):
    """Create table to compute optimum-shift over phases

    Args:
        results_pandas: pandas dataframe containing all results (all phases)
        out: folder to save tables in
    """
    final_results_pandas = results_pandas[
        results_pandas["eval_step"] == results_pandas["phase"]
    ]

    def calculate_optimum_shift(df: pd.DataFrame) -> pd.DataFrame:
        def retain_hp(values: pd.Series) -> Any:
            if not all([v == values.iloc[0] for v in values]):
                raise ValueError("all hps must be the same for simple retaining")
            return values.iloc[0]

        # Marginalize seed
        temp_df = (
            df.groupby(by=["phase", "config_index"])
            .agg(
                {
                    "success": lambda column_values: trim_mean(
                        column_values, proportiontocut=0.25
                    ),
                    "mean_normalized_goal_distance_return": lambda column_values: trim_mean(
                        column_values, proportiontocut=0.25
                    ),
                }
                | {f"hp.{hp_name}": retain_hp for hp_name in df["hps"].iloc[0]}
            )
            .reset_index()
        )
        optimum_per_phase_df = temp_df.groupby(by=["phase"]).apply(
            lambda df: df.sort_values(["success"]).iloc[-1],
        )
        for hp_name in [hp_name for hp_name in df["hps"].iloc[0]]:
            optimum_per_phase_df[f"hp.{hp_name}_sobol_codomain"] = hp_to_sobol_codomain(
                optimum_per_phase_df[f"hp.{hp_name}"],
                *get_bounds(hp_name, df["hp.agent_name"].iloc[0]),
            )

        # Train IGPR model
        models_per_phase = [
            (
                fit_model(
                    final_results_pandas[final_results_pandas["phase"] == phase],
                    "success",
                    [f"hp.{hp_name}" for hp_name in df["hps"].iloc[0]],
                ),
                phase,
            )
            for phase in sorted(temp_df["phase"].unique())
        ]
        optimum_per_phase = {
            phase: shgo(
                func=lambda x: -model.get_middle([x]),
                bounds=[(0, 1), (0, 1)],
            )
            for model, phase in models_per_phase
        }

        optimum_per_phase_df = pd.DataFrame(
            [value.x for value in optimum_per_phase.values()],
            columns=["hp0_opt", "hp1_opt"],
        )
        optimum_per_phase_df["phase"] = optimum_per_phase.keys()
        optimum_per_phase_df = optimum_per_phase_df.set_index("phase")
        optimum_per_phase_diff_df = optimum_per_phase_df.diff()
        optimum_per_phase_diff_df["optimum_shift"] = optimum_per_phase_diff_df.apply(
            lambda row: np.sqrt(row["hp0_opt"] ** 2 + row["hp1_opt"] ** 2), axis=1
        )
        return optimum_per_phase_diff_df

    table = final_results_pandas.groupby(by=["agent", "dataset"]).apply(
        calculate_optimum_shift
    )

    with open(out / "optimum_shift_table.md", "w") as f:
        f.write(table.to_markdown())

    with open(out / "optimum_shift_table.tex", "w") as f:
        f.write(table.to_latex())

    with open(out / "optimum_shift_table.csv", "w") as f:
        f.write(table.to_csv())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zipfile", type=Path, required=True)
    args = parser.parse_args()

    # Parse results
    output_folder = Path("tables") / os.path.basename(args.zipfile)
    output_folder.mkdir(exist_ok=True, parents=True)
    merged_results_df = merge_experiments(
        {
            prefix: (
                run_info,
                compute_additional_information(phase_results_to_pandas(phase_results)),
            )
            for prefix, (run_info, phase_results) in read_results_from_zip(
                args.zipfile
            ).items()
        }
    )

    create_phased_tables(merged_results_df, output_folder)
    create_igprfit_tables(merged_results_df, output_folder)
    create_regret_table(merged_results_df, output_folder)
    create_optimum_shift_table(merged_results_df, output_folder)
