import argparse
from .util.data import (
    ResultsPerStep,
    PhaseResult,
    phase_results_to_pandas,
    read_results_from_zip,
    EvaluationResult,
)
from pathlib import Path
from .plots.triple_gp import TripleGPModel, create_contour_plot
import numpy as np
from .configurations import get_config_space
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Any
import re
from pyfolding import FTU
import seaborn as sns
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
import os
from copy import deepcopy
from itertools import product
from gcrl_landscapes.util.eval import cvar, iqr, cvar_inv

DIM_LABEL_MAPPING = {
    "actor_p_trajgoal": "$p_{trajgoal}$",
    "discount": "Discount Factor",
}

FTU_SIGNIFICANCE_THRESHOLD = 0.05

CVAR_CONFIDENCE_LEVEL = 10
DISP_CONFIDENCE_LEVELS = (10, 90)


def map_labels(label: str) -> str:
    return DIM_LABEL_MAPPING[label] if label in DIM_LABEL_MAPPING else label


def compute_additional_information(
    phase_results: list[tuple[int, pd.DataFrame]],
) -> list[tuple[int, pd.DataFrame]]:
    phase_results_copy = deepcopy(phase_results)
    for _, phase_result in phase_results_copy:
        phase_result["normalized_goal_distances"] = phase_result["eval_result"].apply(
            lambda results_per_seed: np.array(
                [
                    np.array(result["goal_end_distances"])
                    / np.array(result["goal_start_distances"])
                    for result in results_per_seed.info
                ]
            ).reshape(-1)
        )
        phase_result["normalized_goal_distance_returns"] = phase_result[
            "normalized_goal_distances"
        ].apply(lambda distances: 1 - distances)
        phase_result["mean_normalized_goal_distance"] = phase_result[
            "normalized_goal_distances"
        ].apply(np.mean)
        phase_result["mean_normalized_goal_distance_return"] = phase_result[
            "normalized_goal_distance_returns"
        ].apply(np.mean)

        phase_result["cvar_normalized_goal_distance_return"] = phase_result[
            "normalized_goal_distance_returns"
        ].apply(
            lambda distances: cvar(distances, confidence_level=CVAR_CONFIDENCE_LEVEL)
        )
        phase_result["disp_normalized_goal_distance"] = phase_result[
            "normalized_goal_distance_returns"
        ].apply(lambda returns: iqr(returns, DISP_CONFIDENCE_LEVELS))
        phase_result["disp_normalized_goal_distance_score"] = (
            1 - phase_result["disp_normalized_goal_distance"]
        )
        assert phase_result["disp_normalized_goal_distance_score"].max() <= 1
        phase_result["cvar_normalized_goal_distance_return_half_percentile"] = (
            phase_result[
                "normalized_goal_distance_returns"
            ].apply(lambda returns: cvar_inv(returns, 0.5))
        )
        phase_result["cvar_normalized_goal_distance_return_half_percentile_score"] = (
            1 - phase_result["cvar_normalized_goal_distance_return_half_percentile"]
        )

    return phase_results_copy


def plot_igpr(
    phase: int,
    phase_result: pd.DataFrame,
    y_col: str,
    hp_names: list[str],
    output_folder: Path,
    y_label: str | None,
):
    model = TripleGPModel(
        phase_result,
        np.float64,
        y_col=y_col,
        hp_names=hp_names,
        configspace=get_config_space(""),
    )
    model.fit()
    create_contour_plot(
        model,
        x_dim=0,
        y_dim=1,
        z_dim=y_label if y_label else y_col,
        bounds=[0, 1],
        filename=output_folder / f"igpr_{y_col}_{phase}.png",
        dim_label_mapping=map_labels,
    )
    create_contour_plot(
        model,
        x_dim=0,
        y_dim=1,
        z_dim=y_label if y_label else y_col,
        bounds=[None, None],
        filename=output_folder / f"igpr_{y_col}_{phase}_scaled.png",
        dim_label_mapping=map_labels,
    )

    # Create plot without using gaussian processes
    plt.figure()
    x_scaled = model.x
    y_scaled = model.y_iqm

    x0i_scaled, x1i_scaled = np.meshgrid(
        np.linspace(x_scaled[:, 0].min(), x_scaled[:, 0].max(), 1000),
        np.linspace(x_scaled[:, 1].min(), x_scaled[:, 1].max(), 1000),
    )
    yi_scaled = griddata(
        (x_scaled[:, 0], x_scaled[:, 1]),
        y_scaled,
        (x0i_scaled, x1i_scaled),
        method="nearest",
    )
    x0i = x0i_scaled * model.x_normalizing_factor[0] + model.x_normalizing_offset[0]
    x1i = x1i_scaled * model.x_normalizing_factor[1] + model.x_normalizing_offset[1]
    yi = model._unscale_y(yi_scaled).squeeze()

    c = plt.contourf(x0i, x1i, yi, cmap="rocket", vmin=0, vmax=1)
    plt.colorbar(c, label=y_label if y_label else y_col)
    plt.savefig(output_folder / f"nearest_{y_label}_{phase}.png", bbox_inches="tight")
    plt.close()


def plot(results_pandas: pd.DataFrame, output_folder: Path, run_info: dict[str, Any]):
    """Main plotting Code to generate the landscapes

    Args:
        results_pandas: pandas dataframe containing all results (all phases) for one experiment
        output_folder: folder to save plots in
        run_info: info about the run/setup
    """
    hp_full_list = results_pandas.columns[results_pandas.columns.str.startswith("hp.")]
    hp_list = [f"hp.{hp_name}" for hp_name in run_info["arguments"]["hyperparameters"]]
    assert all(hp in hp_full_list for hp in hp_list)

    phase_starts = sorted(results_pandas["phase_start"].unique().tolist())
    phase_results = [
        (
            phase,
            results_pandas[
                (results_pandas["eval_step"] == phase)
                & (results_pandas["phase_start"] == phase_start)
            ],
        )
        for phase_start, phase in zip(phase_starts, run_info["arguments"]["phases"])
    ]
    phase_results = compute_additional_information(phase_results)  # type: ignore

    for phase, phase_result in phase_results:
        phase_result_copy = phase_result.copy()
        phase_result_copy.loc[:, "run_id"], _ = pd.factorize(
            phase_result_copy["run_id"]
        )  # TripleGPModel needs continuous run-ids starting at 0

        plot_igpr(
            phase, phase_result_copy, "success", hp_list, output_folder, "Success Rate"
        )
        plot_igpr(
            phase,
            phase_result_copy,
            "mean_normalized_goal_distance_return",
            hp_list,
            output_folder,
            "Normalized Goal Distance Return",
        )
        plot_igpr(
            phase,
            phase_result_copy,
            "cvar_normalized_goal_distance_return",
            hp_list,
            output_folder,
            "CVaR of normalized goal distance return",
        )
        plot_igpr(
            phase,
            phase_result_copy,
            "disp_normalized_goal_distance_score",
            hp_list,
            output_folder,
            "Dispersion score of normalized goal distance return",
        )
        plot_igpr(
            phase,
            phase_result_copy,
            "cvar_normalized_goal_distance_return_half_percentile_score",
            hp_list,
            output_folder,
            "1 - Percentile needed to reach CVaR of 0.5",
        )

        # Modality plots
        # Create plot based on folding test of unimodality
        def eval_result_to_mean_over_tasks(
            eval_result: EvaluationResult,
        ) -> list[float]:
            all_success_trajectories = np.array(
                [task["success_all"] for task in eval_result.info]
            )  # type: ignore
            return np.mean(all_success_trajectories, axis=0)

        phase_result_copy["success_all_mean_over_tasks"] = phase_result_copy[
            "eval_result"
        ].apply(eval_result_to_mean_over_tasks)

        def aggregate_mean_task_results(series: pd.Series) -> FTU:
            eval_results = series.tolist()
            success_per_seed_task = np.array(eval_results)
            return FTU(success_per_seed_task.reshape(-1), routine="c++")  # type: ignore  # the type is correct, there seems to be an import problem

        # group by configuration to apply statistic over seeds
        ftu_object_per_configuration = phase_result_copy.groupby(["run_id"] + hp_list)[
            "normalized_goal_distances"
        ].aggregate(aggregate_mean_task_results)
        ftu_result_per_configuration = ftu_object_per_configuration.apply(
            lambda ftu: ftu.folding_statistics
            if ftu.p_value <= FTU_SIGNIFICANCE_THRESHOLD
            else 1.0
        )

        # get configurations as x
        x_unscaled = np.array(ftu_result_per_configuration.index.tolist())[:, 1:]
        x_scaled = (x_unscaled - x_unscaled.min(axis=0)) / (
            x_unscaled.max(axis=0) - x_unscaled.min(axis=0)
        )
        # Replace nan values with 1.0, which means undecided
        y = np.nan_to_num(np.array(ftu_result_per_configuration.tolist()), nan=1.0)
        y_discrete = np.copy(y)
        y_discrete[y_discrete > 1.0] = 2.0
        y_discrete[y_discrete < 1.0] = 0.0
        x0_grid, x1_grid = np.meshgrid(
            np.linspace(x_scaled[:, 0].min(), x_scaled[:, 0].max(), 1000),
            np.linspace(x_scaled[:, 1].min(), x_scaled[:, 1].max(), 1000),
        )
        y_grid = griddata(
            (x_scaled[:, 0], x_scaled[:, 1]),
            y_discrete,
            (x0_grid, x1_grid),
            method="nearest",
        )
        assert y_grid.min() >= 0.0 and y_grid.max() <= 2.0

        fig = plt.figure()
        c = plt.contourf(
            x0_grid,
            x1_grid,
            y_grid,
            cmap=sns.color_palette("vlag", as_cmap=True),
            vmin=0.0,
            vmax=2.0,
        )
        cbar = fig.colorbar(c, label="Modality")

        cbar.ax.yaxis.set_minor_locator(ticker.FixedLocator([0.0, 1.0, 2.0]))
        cbar.ax.yaxis.set_minor_formatter(ticker.FixedFormatter(["MM", "N/A", "UM"]))
        cbar.ax.set_yticks([])

        plt.savefig(output_folder / f"modality_{phase}.png", bbox_inches="tight")
        plt.close()


def grid_plot(
    imgpath_dataframe: pd.DataFrame,
    output_folder: Path,
    title: str,
    grid_columns: tuple[str, str],
) -> None:
    indexed_imgpath_series = imgpath_dataframe.set_index(
        list(grid_columns)
    ).sort_index()

    fig = plt.figure(figsize=(16, 16))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(
            len(indexed_imgpath_series.index.get_level_values(0).unique()),
            len(indexed_imgpath_series.index.get_level_values(1).unique()),
        ),
        axes_pad=0.1,
    )
    plt.axis("off")

    for ax, ((dataset, phase), imgpath) in zip(
        grid, indexed_imgpath_series["path"].items()
    ):
        ax.imshow(Image.open(imgpath))
        ax.axis("off")
        ax.set_title(f"{dataset}-{phase}")
    fig.savefig(output_folder / f"{title}.png", bbox_inches="tight")
    plt.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zipfile", type=Path, required=True)
    args = parser.parse_args()

    plots_folder = Path("plots") / os.path.basename(args.zipfile)

    results: dict[str, tuple[dict, ResultsPerStep[PhaseResult]]] = (
        read_results_from_zip(args.zipfile)
    )

    results_pandas = {
        identifier: (run_info, phase_results_to_pandas(phase_results))
        for identifier, (run_info, phase_results) in results.items()
    }

    for prefix, (run_info, results_df) in results_pandas.items():
        run_match = re.match(r"^logs[^/]*/([^/]*)/?", prefix)
        if not run_match:
            raise ValueError("Naming inside of zipfile not as expected.")
        run_name = run_match.group(1)

        folder = plots_folder / run_name
        folder.mkdir(exist_ok=True, parents=True)
        plot(results_df, folder, run_info)

    # Build igpr grid of plots (across datasets)
    imgpath_dataframe = pd.DataFrame(
        columns=["phase", "agent", "path", "dataset", "y_col"]
    )  # type: ignore
    for experiment_name in os.listdir(plots_folder):
        if not Path(plots_folder / experiment_name).is_dir():
            continue
        experiment_name_matches = re.match(
            r"^(?P<agent>[^_]*)_(?P<dataset>[^_]*)_", experiment_name
        )
        if not experiment_name_matches:
            raise ValueError("Naming inside of plots folder not as expected.")
        agent = experiment_name_matches["agent"]
        dataset = experiment_name_matches["dataset"]

        for unscaled_igpr_plotpath in [
            plots_folder / experiment_name / filename
            for filename in os.listdir(plots_folder / experiment_name)
            if re.match(r"^igpr_.*_\d+.png$", filename)
        ]:
            plot_name_matches = re.match(
                r".*/igpr_(?P<y_col>.*)_(?P<phase>\d+).png$",
                str(unscaled_igpr_plotpath),
            )
            if not plot_name_matches:
                raise ValueError("Naming inside of plots folder not as expected.")

            imgpath_dataframe.loc[len(imgpath_dataframe)] = [
                int(plot_name_matches["phase"]),
                agent,
                str(unscaled_igpr_plotpath),
                dataset,
                plot_name_matches["y_col"],
            ]

    for agent, y_col in product(
        imgpath_dataframe["agent"].unique(), imgpath_dataframe["y_col"].unique()
    ):
        grid_plot(
            imgpath_dataframe[
                (imgpath_dataframe["agent"] == agent)
                & (imgpath_dataframe["y_col"] == y_col)
            ],  # type: ignore
            plots_folder,
            f"{agent}_{y_col}",
            grid_columns=("dataset", "phase"),
        )  # type: ignore
