import argparse
from gcrl_landscapes.util.data import (
    read_results_from_zip,
    load_or_compute,
)
from gcrl_landscapes.util.eval import fit_model
from gcrl_landscapes.util.data import phase_results_to_pandas
from pathlib import Path
import pandas as pd
import numpy as np
from .common import (
    compute_additional_information,
    merge_experiments,
    calculate_regret_for_experiment,
)
from gcrl_landscapes.configurations import hp_to_sobol_codomain, get_bounds
from scipy.stats import trim_mean
from scipy.optimize import shgo
from scipy.spatial import distance
from typing import Any
import zipfile
import re
import json
import multiprocessing
import os
from itertools import combinations, chain


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

    def aggregation(df: pd.DataFrame):
        return df.groupby(
            by=["agent", "dataset", "constant_dataset", "hps", "phase_num"]
        ).agg(
            **{
                "Goal Distance Score Normalized Regret $< 0.1$ Ratio": pd.NamedAgg(
                    column="mean_normalized_goal_distance_return_normalized_regret",
                    aggfunc=lambda x: (x < 0.1).mean(),
                ),
                "Goal Distance Score Normalized Regret $< 0.2$ Ratio": pd.NamedAgg(
                    column="mean_normalized_goal_distance_return_normalized_regret",
                    aggfunc=lambda x: (x < 0.2).mean(),
                ),
                "Goal Distance Score": pd.NamedAgg(
                    column="mean_normalized_goal_distance_return", aggfunc="mean"
                ),
                "Max Goal Distance Score": pd.NamedAgg(
                    column="mean_normalized_goal_distance_return", aggfunc="max"
                ),
                "Success": pd.NamedAgg(column="success", aggfunc="mean"),
            }
        )

    table = aggregation(results_only_final_eval_df)
    table_last_phase = aggregation(
        results_only_final_eval_df[
            results_only_final_eval_df["phase_num"]
            == max(results_only_final_eval_df["phase_num"])
        ]
    )

    with open(output_folder / "table_all_phases.md", "w") as f:
        f.write(table.to_markdown())

    with open(output_folder / "table_all_phases.tex", "w") as f:
        f.write(table.to_latex())

    with open(output_folder / "table_all_phases.csv", "w") as f:
        f.write(table.to_csv())

    with open(output_folder / "table_last_phase.md", "w") as f:
        f.write(table_last_phase.to_markdown())

    with open(output_folder / "table_last_phase.tex", "w") as f:
        f.write(table_last_phase.to_latex())

    with open(output_folder / "table_last_phase.csv", "w") as f:
        f.write(table_last_phase.to_csv())

    aggregation_columns = ["agent", "dataset", "constant_dataset", "phase_num", "hps"]
    extra_combinations = [
        ("constant_dataset", "agent"),
        ("constant_dataset", "agent", "phase_num"),
    ]
    for combination in chain(
        combinations(aggregation_columns, 2),
        [[column] for column in aggregation_columns],
        extra_combinations,
    ):
        aggregate_and_save_results(
            table, list(combination), output_folder / "table_all_phases"
        )
        aggregate_and_save_results(
            table_last_phase, list(combination), output_folder / "table_last_phase"
        )


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
    table = final_results_pandas.groupby(
        by=["agent", "dataset", "constant_dataset", "hps", "phase_num"]
    ).apply(
        lambda df: fit_model(
            df.reset_index(drop=True),
            "mean_normalized_goal_distance_return",
            [f"hp.{hp_name}" for hp_name in df["hps"].iloc[0]],
        )
        .estimate_iqm_fit()
        .drop(axis="columns", labels="fold")
        .mean(axis=0),
    )

    with open(output_folder / "igpr_fit_table.md", "w") as f:
        f.write(table.to_markdown())

    with open(output_folder / "igpr_fit_table.tex", "w") as f:
        f.write(table.to_latex())

    with open(output_folder / "igpr_fit_table.csv", "w") as f:
        f.write(table.to_csv())

    aggregation_columns = ["agent", "dataset", "constant_dataset", "hps", "phase_num"]
    for combination in chain(
        combinations(aggregation_columns, 2),
        [[column] for column in aggregation_columns],
    ):
        aggregate_and_save_results(
            table,
            list(combination),
            output_folder / "igpr_fit_table",
            rounding_decimals=3,
        )


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
            final_results_pandas.groupby(
                by=["agent", "dataset", "constant_dataset", "hps"]
            )
            .apply(calculate_regret)
            .reset_index(level="phase")
        )

        table_pick_first_phase: pd.DataFrame = (
            regret_df.sort_values(["phase"], ascending=True)
            .groupby(by=regret_df.index.names)
            .nth(0)
        )  # type: ignore
        table_pick_second_phase: pd.DataFrame = (
            regret_df.sort_values(["phase"], ascending=True)
            .groupby(by=regret_df.index.names)
            .nth(1)
        )  # type: ignore

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

        aggregation_columns = ["agent", "dataset", "constant_dataset", "hps"]
        extra_combinations = [("constant_dataset", "agent")]
        for combination in chain(
            combinations(aggregation_columns, 2),
            [[column] for column in aggregation_columns],
            extra_combinations,
        ):
            aggregate_and_save_results(
                table_pick_first_phase,
                list(combination),
                output_folder / f"{regret_col}_table_first_phase",
            )
            aggregate_and_save_results(
                table_pick_second_phase,
                list(combination),
                output_folder / f"{regret_col}_table_second_phase",
            )


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
            df.groupby(by=["phase_num", "config_index"])
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
        optimum_per_phase_df = temp_df.groupby(by=["phase_num"]).apply(
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
                    df[df["phase_num"] == phase],
                    "success",
                    [f"hp.{hp_name}" for hp_name in df["hps"].iloc[0]],
                ),
                phase,
            )
            for phase in sorted(temp_df["phase_num"].unique())
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
        optimum_per_phase_df["phase_num"] = optimum_per_phase.keys()
        optimum_per_phase_df = optimum_per_phase_df.set_index("phase_num")
        optimum_per_phase_diff_df = optimum_per_phase_df.diff()
        optimum_per_phase_diff_df["optimum_shift"] = optimum_per_phase_diff_df.apply(
            lambda row: np.sqrt(row["hp0_opt"] ** 2 + row["hp1_opt"] ** 2), axis=1
        )
        return optimum_per_phase_diff_df

    full_table: pd.DataFrame = final_results_pandas.groupby(
        by=["agent", "dataset", "constant_dataset", "hps"]
    ).apply(calculate_optimum_shift)  # type: ignore

    with open(out / "optimum_shift_table_full.md", "w") as f:
        f.write(full_table.to_markdown())

    with open(out / "optimum_shift_table_full.tex", "w") as f:
        f.write(full_table.to_latex())

    with open(out / "optimum_shift_table_full.csv", "w") as f:
        f.write(full_table.to_csv())

    base_path = out / "optimum_shift_table"
    aggregate_and_save_results(full_table, ["agent", "constant_dataset"], base_path)
    aggregate_and_save_results(full_table, ["dataset"], base_path)
    aggregate_and_save_results(full_table, ["agent", "dataset"], base_path)
    aggregate_and_save_results(full_table, ["agent", "phase_num"], base_path)
    aggregate_and_save_results(full_table, ["dataset", "phase_num"], base_path)
    aggregate_and_save_results(full_table, ["agent"], base_path)
    aggregate_and_save_results(full_table, ["phase_num"], base_path)


def create_importance_divergence_table(
    results_pandas: pd.DataFrame, output_folder: Path
):
    """Create a table showing divergence per dataset-setting and algorithm.
    Uses cosine similarity for measuring the difference.

    Args:
        results_pandas: pandas dataframe containing all results (all phases)
        output_folder: folder to save tables in
    """

    def compute_importance_divergence(df: pd.DataFrame) -> pd.Series:
        # Create probability arrays
        df_prob_arrays = (
            df.reset_index(level=["hp", "trainingprogress"])
            .groupby(by="trainingprogress")
            .apply(lambda df: np.array(df.sort_values("hp")["mean"]))
            .sort_index()
        )

        cosine_df = pd.Series(
            {
                f"{df_prob_arrays.index[i - 1]}->{df_prob_arrays.index[i]}": distance.cosine(
                    df_prob_arrays.iloc[i - 1], df_prob_arrays.iloc[i]
                )
                for i in range(1, len(df_prob_arrays))
            }
        )
        cosine_df["->".join(map(str, df_prob_arrays.index))] = cosine_df.loc[
            ~cosine_df.index.str.startswith("100->")
        ].sum()

        return cosine_df

    table = results_pandas.groupby(by=["setting", "agent"]).apply(
        lambda df: compute_importance_divergence(df)
    )

    with open(output_folder / "importance_divergence_table.md", "w") as f:
        f.write(table.to_markdown())

    with open(output_folder / "importance_divergence_table.tex", "w") as f:
        f.write(table.to_latex())

    with open(output_folder / "importance_divergence_table.csv", "w") as f:
        f.write(table.to_csv())


def parse_hp_importance(zip_path: Path) -> pd.DataFrame:
    """Parse the importances from the zip file

    Args:
        zip_path: Path to importance zip file

    Returns:
        pd.DataFrame with multiindex ("setting", "agent", "trainingprogress", "hp") and columns ("mean", "std")
    """

    def parse_filename(filename: str) -> None | re.Match:
        return re.fullmatch(
            r"(?P<setting>[^/]*)/(?P<agent>[a-zA-Z]+)(?P<trainingprogress>\d+)\.json",
            filename,
        )

    def parse_importances(
        filename: str, zip_file: zipfile.ZipFile
    ) -> dict[str, dict[str, float]]:
        with zip_file.open(filename) as f:
            raw_importance_data = json.loads(
                f.read().decode(encoding="utf-8").replace("'", '"')
            )

        assert len(raw_importance_data.keys()) == 1
        run_hash = list(raw_importance_data.keys())[0]
        assert len(raw_importance_data[run_hash]) == 1
        run_number = list(raw_importance_data[run_hash].keys())[0]

        return {
            key: {
                "mean": raw_importance_data[run_hash][run_number][key][0],
                "std": raw_importance_data[run_hash][run_number][key][1],
            }
            for key in raw_importance_data[run_hash][run_number].keys()
        }

    with zipfile.ZipFile(zip_path) as zip_file:
        filenames = zip_file.namelist()
        parsed_filenames: list[re.Match] = filter(
            lambda match: match is not None,
            [parse_filename(filename) for filename in filenames],
        )  # type: ignore
        data = [
            {
                "setting": match.groupdict()["setting"],
                "agent": match.groupdict()["agent"],
                "trainingprogress": int(match.groupdict()["trainingprogress"]),
                "hp": parse_importances(match.group(0), zip_file),
            }
            for match in parsed_filenames
        ]

    df_wide = pd.json_normalize(data)
    df_long = df_wide.melt(
        id_vars=["setting", "agent", "trainingprogress"],
        value_vars=[c for c in df_wide.columns if c.startswith("hp.")],
        var_name="hp",
        value_name="importance",
    )
    df_long["hp"] = df_long["hp"].str.replace("hp.", "")
    # We now have hp_name.mean and hp_name.std as values in hp-column
    # Move this into own columns
    df_long[["hp", "stat"]] = df_long["hp"].str.split(".", expand=True)
    df = df_long.pivot(
        index=["setting", "agent", "trainingprogress", "hp"],
        columns="stat",
        values="importance",
    ).dropna()
    return df


def aggregate_and_save_results(
    table: pd.DataFrame,
    grouping_keys: list[str],
    output_base_path: Path,
    rounding_decimals: int = 2,
) -> None:
    unformatted_aggregated_table = table.groupby(by=grouping_keys).agg(
        ["mean", "std", "max"]
    )

    aggregated_table = pd.DataFrame(
        {
            col: [
                f"{m:.{rounding_decimals}f} $\\pm$ {s:.{rounding_decimals}f}"
                for m, s in zip(
                    unformatted_aggregated_table[(col, "mean")],
                    unformatted_aggregated_table[(col, "std")],
                )
            ]
            for col in unformatted_aggregated_table.columns.get_level_values(0)
            if not (
                "max " in col.lower() or "max_" in col.lower() or "max-" in col.lower()
            )
        }
        | {
            col: [
                f"{m:.{rounding_decimals}f}"
                for m in unformatted_aggregated_table[(col, "max")]
            ]
            for col in unformatted_aggregated_table.columns.get_level_values(0)
            if "max " in col.lower() or "max_" in col.lower() or "max-" in col.lower()
        },
        index=unformatted_aggregated_table.index,
    )
    aggregated_table = (
        aggregated_table.reorder_levels(grouping_keys)
        if len(grouping_keys) > 1
        else aggregated_table
    )

    with open(f"{str(output_base_path)}_{'-'.join(grouping_keys)}.md", "w") as f:
        f.write(aggregated_table.to_markdown())

    with open(f"{str(output_base_path)}_{'-'.join(grouping_keys)}.tex", "w") as f:
        f.write(aggregated_table.to_latex())

    with open(f"{str(output_base_path)}_{'-'.join(grouping_keys)}.csv", "w") as f:
        f.write(aggregated_table.to_csv())

    return


def compute_merged_df(zipfiles: list[Path]) -> pd.DataFrame:
    if not args.no_multiprocessing:
        try:
            thread_count = int(os.environ["SLURM_CPUS_ON_NODE"]) // 2
        except Exception as _:
            thread_count = multiprocessing.cpu_count() // 2

        with multiprocessing.get_context("spawn").Pool(thread_count) as pool:
            results_from_zips = pool.map(read_results_from_zip, zipfiles)
    else:
        results_from_zips = [read_results_from_zip(zipfile) for zipfile in zipfiles]

    return merge_experiments(
        {
            prefix: (
                run_info,
                compute_additional_information(phase_results_to_pandas(phase_results)),
            )
            for result_from_zip in results_from_zips
            for prefix, (run_info, phase_results) in result_from_zip.items()
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zipfiles", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--hp_importance_data",
        action="store_true",
        help="Zip contains hp importance data. Only create tables based on hp importance.",
    )
    parser.add_argument("--no_multiprocessing", action="store_true")
    parser.add_argument("--output_folder", type=Path, required=True)
    args = parser.parse_args()

    # Parse results
    output_folder = args.output_folder
    output_folder.mkdir(exist_ok=True, parents=True)

    if args.hp_importance_data:
        importance_df = pd.concat(
            [parse_hp_importance(zipfile) for zipfile in args.zipfiles]
        )
        create_importance_divergence_table(importance_df, args.output_folder)
        exit(0)

    merged_results_df: pd.DataFrame = load_or_compute(args.zipfiles, compute_merged_df)  # type: ignore

    create_phased_tables(merged_results_df, output_folder)
    create_igprfit_tables(merged_results_df, output_folder)
    create_regret_table(merged_results_df, output_folder)
    create_optimum_shift_table(merged_results_df, output_folder)

    # Do all calculations once without pure explore
    output_folder = output_folder / "wo_pure_explore"
    output_folder.mkdir(exist_ok=True)
    wo_pure_explore_df: pd.DataFrame = merged_results_df[
        merged_results_df["dataset"].apply(
            lambda datasets: any(
                ["explore-v0" not in dataset for dataset in datasets.split(",")]
            )
        )
    ]  # type: ignore
    create_phased_tables(wo_pure_explore_df, output_folder)
    create_igprfit_tables(wo_pure_explore_df, output_folder)
    create_regret_table(wo_pure_explore_df, output_folder)
    create_optimum_shift_table(wo_pure_explore_df, output_folder)
