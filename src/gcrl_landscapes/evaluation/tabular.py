import argparse
from gcrl_landscapes.util.data import (
    read_results_from_zip,
)
from gcrl_landscapes.util.eval import fit_model
from pathlib import Path
import pandas as pd
import os
from .common import (
    compute_additional_information,
    merge_experiments,
    phase_results_to_pandas,
    calculate_regret_for_experiment,
    CVAR_CONFIDENCE_LEVELS,
)


def create_tables(results_pandas: pd.DataFrame, output_folder: Path):
    """Main code to generate the tabular data

    Args:
        results_pandas: pandas dataframe containing all results (all phases) for one experiment
        output_folder: folder to save plots in
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


def create_additional_tables(results_pandas: pd.DataFrame, output_folder: Path):
    """Compute additional tabular data like IGPR fit

    Args:
        results_pandas: pandas dataframe containing all results (all phases) for one experiment
        output_folder: folder to save plots in
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
    final_results_pandas = results_pandas[
        results_pandas["eval_step"] == results_pandas["phase"]
    ]

    regret_df = (
        final_results_pandas.groupby(by=["agent", "dataset"])
        .apply(calculate_regret_for_experiment)
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

    with open(output_folder / "regret_table_first_phase.md", "w") as f:
        f.write(table_pick_first_phase.to_markdown())

    with open(output_folder / "regret_table_first_phase.tex", "w") as f:
        f.write(table_pick_first_phase.to_latex())

    with open(output_folder / "regret_table_first_phase.csv", "w") as f:
        f.write(table_pick_first_phase.to_csv())

    with open(output_folder / "regret_table_second_phase.md", "w") as f:
        f.write(table_pick_second_phase.to_markdown())

    with open(output_folder / "regret_table_second_phase.tex", "w") as f:
        f.write(table_pick_second_phase.to_latex())

    with open(output_folder / "regret_table_second_phase.csv", "w") as f:
        f.write(table_pick_second_phase.to_csv())


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

    create_tables(merged_results_df, output_folder)
    create_additional_tables(merged_results_df, output_folder)
    create_regret_table(merged_results_df, output_folder)
