import argparse
from gcrl_landscapes.util.data import (
    ResultsPerStep,
    PhaseResult,
    phase_results_to_pandas,
    read_results_from_zip,
)
from pathlib import Path
from gcrl_landscapes.plots.triple_gp import TripleGPModel, create_contour_plot
import numpy as np
from gcrl_landscapes.configurations import get_config_space
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Any
import re
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
import os
from itertools import product
from .common import (
    CVAR_CONFIDENCE_LEVELS,
    map_labels,
    compute_additional_information,
)
import toml
import zipfile
import multiprocessing
from functools import partial


def plot_landscape(
    phase: int,
    phase_result: pd.DataFrame,
    y_col: str,
    hp_names: list[str],
    output_folder: Path,
    y_label: str | None,
):
    """IGPR plot of given data. Fits gaussian process itself

    Args:
        phase: which phase is this. used for plot title
        phase_result: results for this phase
        y_col: column which we want to plot landscape for
        hp_names: names of all hyperparameter columns (to create groups from dataframe)
        output_folder: where to save plot to
        y_label: How y should be labeled
    """
    result_copy = phase_result.copy()
    if "hp.lr" in hp_names:
        hp_names = ["lr-uniform"] + [
            hp_name for hp_name in hp_names if hp_name != "hp.lr"
        ]
        result_copy["lr-uniform"] = (
            np.log10(result_copy["hp.lr"]) - np.log10(result_copy["hp.lr"].min())
        ) / (
            np.log10(result_copy["hp.lr"].max()) - np.log10(result_copy["hp.lr"].min())
        )
    model = TripleGPModel(
        result_copy,
        np.float64,
        y_col=y_col,
        hp_names=hp_names,
        configspace=get_config_space(""),
    )
    model.fit()
    # One direct plot
    create_contour_plot(
        model,
        x_dim=0,
        y_dim=1,
        z_dim=y_label if y_label else y_col,
        bounds=[0, 1],
        filename=output_folder / f"igpr-{y_col}-{phase}.png",
        dim_label_mapping=map_labels,
    )
    # Scale results to [0, 1] to better see contours
    create_contour_plot(
        model,
        x_dim=0,
        y_dim=1,
        z_dim=y_label if y_label else y_col,
        bounds=[None, None],
        filename=output_folder / f"igpr-{y_col}-{phase}-scaled.png",
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
    plt.savefig(output_folder / f"nearest_{y_label}-{phase}.png", bbox_inches="tight")
    plt.close()


def plot_eval_curve(
    phase: int,
    phase_result: pd.DataFrame,
    y_col: str,
    output_folder: Path,
    output_per_config_folder: Path,
    y_label: str | None,
):
    exploded_phase_result = phase_result[[y_col, "config_index", "eval_step"]].explode(
        y_col
    )  # type: ignore
    # per configuration
    for config_index, group in exploded_phase_result.groupby("config_index"):
        _ = plt.figure()
        ax = sns.lineplot(
            data=group,  # type: ignore
            x="eval_step",
            y=y_col,
            errorbar=("ci", 95),
        )
        plt.title(f"{y_label if y_label else y_col}", fontsize=18)
        ax.set_ylim(0, 1)
        plt.savefig(
            output_per_config_folder
            / f"eval-{y_col}-config_{config_index}-{phase}.png",
            bbox_inches="tight",
        )
        plt.close()
    ## configuration marginalized
    _ = plt.figure()
    ax = sns.lineplot(
        data=exploded_phase_result,  # type: ignore
        x="eval_step",
        y=y_col,
        errorbar=("ci", 95),
    )
    plt.title(f"{y_label if y_label else y_col}", fontsize=18)
    ax.set_ylim(0, 1)
    plt.savefig(
        output_folder / f"eval-{y_col}-{phase}.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_return_distribution(
    phase: int,
    phase_result: pd.DataFrame,
    y_col: str,
    output_folder: Path,
    output_per_config_folder: Path,
    y_label: str | None,
):
    RETURN_LIMITS = (-1, 1)
    exploded_phase_result = phase_result[[y_col, "config_index"]].explode(y_col)  # type: ignore
    exploded_phase_result[f"clipped_{y_col}"] = exploded_phase_result[y_col].clip(
        lower=RETURN_LIMITS[0], upper=RETURN_LIMITS[1]
    )
    # per configuration
    for config_index, group in exploded_phase_result.groupby("config_index"):
        _ = plt.figure()
        ax = sns.histplot(
            data=group,  # type: ignore
            x=f"clipped_{y_col}",
            stat="probability",
            bins="sturges",
        )
        ax.set_xlim(*RETURN_LIMITS)
        ax.set_ylim(0, 1)
        plt.title(f"{y_label if y_label else y_col}", fontsize=18)
        plt.savefig(
            output_per_config_folder
            / f"returndistribution-{y_col}-config_{config_index}-{phase}.png",
            bbox_inches="tight",
        )
        plt.close()
    ## configuration marginalized
    _ = plt.figure()
    ax = sns.histplot(
        data=exploded_phase_result,  # type: ignore
        x=f"clipped_{y_col}",
        stat="probability",
        bins="sturges",
    )
    ax.set_xlim(*RETURN_LIMITS)
    ax.set_ylim(0, 1)
    plt.title(f"{y_label if y_label else y_col}", fontsize=18)
    plt.savefig(
        output_folder / f"returndistribution-{y_col}-{phase}.png",
        bbox_inches="tight",
    )
    plt.close()


def plot(
    results_pandas: pd.DataFrame,
    output_folder: Path,
    run_info: dict[str, Any],
    plot_return_distributions: bool = False,
    plot_eval_curves: bool = False,
):
    """Main plotting Code to generate the landscapes

    Args:
        results_pandas: pandas dataframe containing all results (all phases) for one experiment
        output_folder: folder to save plots in
        run_info: info about the run/setup
    """
    results_pandas = compute_additional_information(results_pandas)

    per_config_folder = output_folder / "per_config"
    per_config_folder.mkdir(exist_ok=True)

    hp_full_list = results_pandas.columns[results_pandas.columns.str.startswith("hp.")]
    hp_list = [f"hp.{hp_name}" for hp_name in run_info["arguments"]["hyperparameters"]]
    assert all(hp in hp_full_list for hp in hp_list)

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

    for phase, phase_result in phase_results:
        per_config_phase_folder = per_config_folder / f"phase_{phase}"
        per_config_phase_folder.mkdir(exist_ok=True)
        phase_result_copy = phase_result.copy()
        phase_result_copy.loc[:, "run_id"], _ = pd.factorize(
            phase_result_copy["run_id"]
        )  # TripleGPModel needs continuous run-ids starting at 0

        landscape_pairs = [
            ("success", "Success Rate"),
            ("mean_normalized_goal_distance_return", "Normalized Goal Distance Return"),
            (
                "disp_normalized_goal_distance_score",
                "Dispersion score of normalized goal distance return",
            ),
        ] + [  # Gather all CVaR confidence levels
            (
                f"cvar{confidence_level}_normalized_goal_distance_return",
                f"CVaR ({confidence_level}%)of normalized goal distance return",
            )
            for confidence_level in CVAR_CONFIDENCE_LEVELS
        ]
        for col, title in landscape_pairs:
            plot_landscape(phase, phase_result_copy, col, hp_list, output_folder, title)

        if plot_return_distributions:
            plot_return_distribution(
                phase,
                phase_result_copy,
                "mean_normalized_goal_distance_return",
                output_folder,
                per_config_phase_folder,
                "Normalized Goal Distance Return Distribution",
            )
        if plot_eval_curves:
            plot_eval_curve(
                phase,
                results_pandas[results_pandas["phase"] == phase],
                "mean_normalized_goal_distance_return",
                output_folder,
                per_config_phase_folder,
                "Normalized Goal Distance Return",
            )


def grid_plot(
    imgpath_dataframe: pd.DataFrame,
    output_folder: Path,
    title: str,
    grid_columns: tuple[str, str],
) -> None:
    """Create a grid plot from all experiments.

    Args:
        imgpath_dataframe: Dataframe which maps combinatons of the grid columns to a plotpath
        output_folder: folder to save plot to
        title: title of the plot
        grid_columns: columns to use for grid
    """
    indexed_imgpath_series = imgpath_dataframe.set_index(
        list(grid_columns)
    ).sort_index()

    fig = plt.figure(figsize=(16, 16))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(
            len(indexed_imgpath_series.index.get_level_values(0).unique()),
            # get index 0 of first level and count unique phases. Pandas doesn't properly support that
            len(
                indexed_imgpath_series.iloc[
                    indexed_imgpath_series.index.get_loc(
                        indexed_imgpath_series.index.levels[0][0]
                    )
                ]
                .index.get_level_values(1)
                .unique()
            ),
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
        ax.title.set_size(7)
    fig.savefig(output_folder / f"{title}.png", bbox_inches="tight")
    plt.close()
    return


def plot_parallel_wrapper(
    plots_folder: Path,
    arg: tuple[str, tuple[dict, pd.DataFrame]],
    plot_return_distributions: bool = False,
    plot_eval_curves: bool = False,
):
    prefix, (run_info, results_df) = arg
    run_match = re.match(r"^logs[^/]*/([^/]*)/?", prefix)
    if not run_match:
        raise ValueError("Naming inside of zipfile not as expected.")
    run_name = run_match.group(1)

    folder = plots_folder / run_name
    folder.mkdir(exist_ok=True, parents=True)
    print(f"Plotting '{prefix}'")
    plot(
        results_df,
        folder,
        run_info,
        plot_return_distributions=plot_return_distributions,
        plot_eval_curves=plot_eval_curves,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zipfile", type=Path, required=True)
    parser.add_argument("--plot_return_distributions", action="store_true")
    parser.add_argument("--plot_eval_curves", action="store_true")
    args = parser.parse_args()

    plt.rcParams.update(sns.plotting_context("talk") | {"figure.figsize": [4, 3]})

    # Parse results
    plots_folder = Path("plots") / os.path.basename(args.zipfile)
    results: dict[str, tuple[dict, ResultsPerStep[PhaseResult]]] = (
        read_results_from_zip(args.zipfile)
    )
    results_pandas = {
        identifier: (run_info, phase_results_to_pandas(phase_results))
        for identifier, (run_info, phase_results) in results.items()
    }

    with multiprocessing.get_context("spawn").Pool(8) as pool:
        pool.map(
            partial(
                plot_parallel_wrapper,
                plots_folder,
                plot_return_distributions=args.plot_return_distributions,
                plot_eval_curves=args.plot_eval_curves,
            ),
            results_pandas.items(),
        )

    # Build igpr grid of plots (across datasets)
    # first builds dataframe using plots inside subfolders
    imgpath_dataframe = pd.DataFrame(
        columns=[  # type: ignore
            "plot_type",
            "phase",
            "agent",
            "path",
            "dataset",
            "actor_loss",
            "y_col",
        ]
    )  # type: ignore
    for experiment_name in os.listdir(plots_folder):
        if not Path(plots_folder / experiment_name).is_dir():
            continue

        zipf = zipfile.ZipFile(args.zipfile, "r")
        zipfiles = zipf.namelist()
        matching_info = [
            filename
            for filename in zipf.namelist()
            if re.fullmatch(r"^.*/" + experiment_name + r"/info.toml$", filename)
        ]
        if not len(matching_info):
            raise ValueError("given zip seemingly contains experiment multiple times")
        setup = toml.loads(zipf.read(matching_info[0]).decode(encoding="utf-8"))[
            "arguments"
        ]
        agent = setup["agent"]
        dataset = setup["dataset"]
        if "actor_loss" in setup:
            actor_loss = setup["actor_loss"]
        else:
            actor_loss = "default"

        for unscaled_plotpath in [
            plots_folder / experiment_name / filename
            for filename in os.listdir(plots_folder / experiment_name)
            if re.match(r"^.*-.*-\d+.png$", filename)
        ]:
            plot_name_matches = re.match(
                r".*/(?P<plot_type>[^-/]*)-(?P<y_col>[^-/]*)-(?P<phase>\d+).png$",
                str(unscaled_plotpath),
            )
            if not plot_name_matches:
                raise ValueError("Naming inside of plots folder not as expected.")

            imgpath_dataframe.loc[len(imgpath_dataframe)] = [
                plot_name_matches["plot_type"],
                int(plot_name_matches["phase"]),
                agent,
                str(unscaled_plotpath),
                dataset,
                actor_loss,
                plot_name_matches["y_col"],
            ]

    # Now actually plot all grid plots
    for plot_type, agent, y_col, actor_loss in product(
        imgpath_dataframe["plot_type"].unique(),
        imgpath_dataframe["agent"].unique(),
        imgpath_dataframe["y_col"].unique(),
        imgpath_dataframe["actor_loss"].unique(),
    ):
        grid_df = imgpath_dataframe[
            (imgpath_dataframe["plot_type"] == plot_type)
            & (imgpath_dataframe["agent"] == agent)
            & (imgpath_dataframe["y_col"] == y_col)
            & (imgpath_dataframe["actor_loss"] == actor_loss)
        ]
        if len(grid_df) == 0:
            continue
        grid_plot(
            grid_df,
            plots_folder,
            f"{plot_type}-{agent}-{y_col}-{actor_loss}",
            grid_columns=("dataset", "phase"),
        )  # type: ignore
