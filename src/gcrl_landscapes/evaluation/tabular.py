import argparse
from gcrl_landscapes.util.data import (
    read_results_from_zip,
)
from pathlib import Path
import pandas as pd
import os
from .common import (
    compute_additional_information,
    merge_experiments,
    CVAR_CONFIDENCE_LEVELS,
)


def create_tables(results_pandas: pd.DataFrame, output_folder: Path):
    """Main code to generate the tabular data

    Args:
        results_pandas: pandas dataframe containing all results (all phases) for one experiment
        output_folder: folder to save plots in
    """
    results_pandas = compute_additional_information(results_pandas)  # type: ignore

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zipfile", type=Path, required=True)
    args = parser.parse_args()

    # Parse results
    output_folder = Path("tables") / os.path.basename(args.zipfile)
    output_folder.mkdir(exist_ok=True, parents=True)
    merged_results_df = merge_experiments(read_results_from_zip(args.zipfile))

    create_tables(merged_results_df, output_folder)
