import argparse
from gcrl_landscapes.util.data import (
    ResultsPerStep,
    PhaseResult,
    phase_results_to_pandas,
    read_results_from_zip,
)
from gcrl_landscapes.plots.triple_gp import estimate_model_fit, TripleGPModel
from pathlib import Path
from gcrl_landscapes.plots.triple_gp import create_contour_plot
from gcrl_landscapes.util.eval import fit_model
import numpy as np
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
    calculate_regret_for_experiment,
)
import toml
import zipfile
import multiprocessing
from functools import partial
from itertools import combinations
from sklearn.metrics import mean_absolute_error, max_error, mean_squared_error
from sklearn.model_selection import ShuffleSplit
from typing import Callable


# fmt: off
def plot_landscape(
    phase: int,
    model: TripleGPModel,
    y_col: str,
    hp_names: list[str],
    agent_name: str,
    output_folder: Path,
    y_label: str | None,
    y_transform: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda y_pred, y: y_pred,
    discrete_levels: np.ndarray | None = None,
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
# fmt: on
    plot_filename_base = f"({'_'.join(hp_names)})-{y_label.lower().replace(' ', '_').replace(')', '').replace('(', '') if y_label else y_col}-{phase}"
    # One direct plot
    create_contour_plot(
        model,
        x_dim=0,
        y_dim=1,
        z_dim=y_label if y_label else y_col,
        bounds=[0, 1],
        filename=output_folder / f"igpr-{plot_filename_base}.png",
        dim_label_mapping=map_labels,
        agent_name=agent_name,
        z_transform=y_transform,
        discrete_levels=discrete_levels,
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

    c = plt.contourf(
        x0i,
        x1i,
        y_transform(yi, model._unscale_y(model.y_iqm)),
        cmap="rocket",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(c, label=y_label if y_label else y_col)
    plt.savefig(
        output_folder / f"nearest-{plot_filename_base}.png",
        bbox_inches="tight",
    )
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


def plot_gp_fit(
    phase: int,
    phase_result: pd.DataFrame,
    y_col: str,
    hp_names: list[str],
    output_folder: Path,
    y_label: str | None,
    n_samples: int = 20,
    n_splits: int = 10,
):
    gp_fit_dfs = []
    # Sample n configurations from results and fit model
    for n in np.unique(
        np.rint(
            np.linspace(2, len(phase_result.groupby(hp_names)) - 1, n_samples)
        ).astype(int)
    ):
        print(n)
        model = fit_model(phase_result, y_col, hp_names)
        data = estimate_model_fit(
            X=model.x,
            y=model.y_iqm,
            splitter=ShuffleSplit(n_splits=n_splits, train_size=n),
            y_scale=model.y_normalizing_factor,
            metrics=[mean_squared_error, mean_absolute_error, max_error],
        )
        data["n"] = n
        gp_fit_dfs.append(data)
    gp_fit_df = pd.concat(gp_fit_dfs).reset_index(drop=True)

    fig, ax = plt.subplots()
    sns.lineplot(
        data=gp_fit_df, x="n", y="mean_absolute_error", ax=ax, label="Mean Error"
    )
    sns.lineplot(data=gp_fit_df, x="n", y="max_error", ax=ax, label="Max Error")
    ax.set_ylim(0.01, 1)
    ax.set_yscale("log")
    ax.set_xlabel("Number of Configurations")
    ax.set_ylabel("Error")
    fig.savefig(
        output_folder / f"gp_fit-({'_'.join(hp_names)})-{y_col}-{phase}.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_regret_curve(
    experiment_result: pd.DataFrame,
    regret_col: str,
    base_col: str,
    hp_names: list[str],
    output_folder: Path,
    regret_label: str | None,
                ) -> None:
    regret_df = calculate_regret_for_experiment(experiment_result, regret_col, base_col)
    regret_df = regret_df[regret_df.columns[regret_df.columns.str.startswith("regret_phase_")]].reset_index().rename(columns={"phase": "pick_phase"})

    regret_df_long = regret_df.melt(id_vars="pick_phase", var_name="phase", value_name="regret")
    ordered_pick_phases = np.sort(regret_df_long["pick_phase"].unique())  # type: ignore
    regret_df_long["pick_phase"] = regret_df_long["pick_phase"].map(
        dict(zip(ordered_pick_phases, range(1, len(ordered_pick_phases) + 1)))  # type: ignore
    )
    regret_df_long["phase"] = regret_df_long["phase"].str.replace("regret_phase_", "")

    fig, ax = plt.subplots()
    sns.lineplot(data=regret_df_long, x="phase", y="regret", hue="pick_phase", ax=ax)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(output_folder / f"regret_({'_'.join(hp_names)})-{regret_label.lower().replace(' ', '_').replace(')', '').replace('(', '') if regret_label else regret_col}.png")
    plt.close()


def plot(
    results_pandas: pd.DataFrame,
    output_folder: Path,
    run_info: dict[str, Any],
    plot_return_distributions: bool = False,
    plot_eval_curves: bool = False,
    plot_gp_fits: bool = False,
    plot_regret: bool = False,
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

    # Plots using full experiment results
    regret_targets = [
        ("mean_normalized_goal_distance_return_regret", "mean_normalized_goal_distance_return", "Goal Distance Regret"),
        ("mean_normalized_goal_distance_return_normalized_regret", "mean_normalized_goal_distance_return", "Normalized Goal Distance Regret"),
        ("success_regret", "success", "Success Regret"),
        ("success_normalized_regret", "success", "Normalized Success Regret"),
    ]
    if plot_regret:
        for regret_column, base_column, regret_label in regret_targets:
            plot_regret_curve(results_pandas, regret_column, base_column, hp_list, output_folder, regret_label)


    # Plots that are local to a phase
    phases = sorted(results_pandas["phase"].unique().tolist())
    phase_results = [
        (
            phase_idx + 1,
            results_pandas[
                (results_pandas["eval_step"] == phase)
                & (results_pandas["phase"] == phase)
            ],
        )
        for phase_idx, phase in enumerate(phases)
    ]

    phase_result: pd.DataFrame
    for phase, phase_result in phase_results:  # type: ignore
        per_config_phase_folder = per_config_folder / f"phase_{phase}"
        per_config_phase_folder.mkdir(exist_ok=True)

        e_optimal_bins = np.array([0.0, 0.5, 0.8, 0.9, 0.95, 1.0])
        landscape_pairs = [
            ("success", "Success Rate", {}),
            (
                "success",
                "Success Epsilon Optimality Bins",
                # Make this stepped, so that it is easier to interpret
                {
                    "y_transform": lambda y_pred, y: y_pred / np.max(y_pred),
                    "discrete_levels": e_optimal_bins,
                }
            ),
            (
                "mean_normalized_goal_distance_return",
                "Normalized Goal Distance Return",
                {}
            ),
            (
                "mean_normalized_goal_distance_return",
                "Normalized Goal Distance Return Epsilon Optimality Bins",
                {
                    "y_transform": lambda y_pred, y: y_pred / np.max(y_pred),
                    "discrete_levels": e_optimal_bins,
                }
            ),
            (
                "mean_normalized_goal_distance_return",
                "Normalized Goal Distance Return Epsilon Optimality",
                {
                    "y_transform": lambda y_pred, y: y_pred / np.max(y_pred),
                }
            ),
            (
                "disp_normalized_goal_distance_score",
                "Dispersion score of normalized goal distance return",
                {}
            ),
            (
                "mean_normalized_goal_distance_return_regret",
                "Normalized Goal Distance Return Regret",
                {}
            ),
            (
                "mean_normalized_goal_distance_return_regret_cummean",
                "Normalized Goal Distance Return Regret Cumulative",
                {}
            ),
            (
                "success_regret",
                "Normalized Goal Distance Return Regret",
                {}
            ),
            (
                "success_regret_cummean",
                "Normalized Goal Distance Return Regret Cumulative",
                {}
            ),
        ] + [  # Gather all CVaR confidence levels
            (
                f"cvar{confidence_level}_normalized_goal_distance_return",
                f"CVaR ({confidence_level}%)of normalized goal distance return",
                {}
            )
            for confidence_level in CVAR_CONFIDENCE_LEVELS
        ]
        for hp_pair in combinations(hp_list, 2):
            for col, title, kwargs in landscape_pairs:
                model = fit_model(phase_result, col, list(hp_pair))
                plot_landscape(
                    phase,
                    model,
                    col,
                    list(hp_pair),
                    phase_result["hp.agent_name"].iloc[0],
                    output_folder,
                    title,
                    **kwargs
                )

        if plot_return_distributions:
            plot_return_distribution(
                phase,
                phase_result,
                "mean_normalized_goal_distance_return",
                output_folder,
                per_config_phase_folder,
                "Normalized Goal Distance Return Distribution",
            )
        if plot_eval_curves:
            plot_eval_curve(
                phase,
                results_pandas,
                "mean_normalized_goal_distance_return",
                output_folder,
                per_config_phase_folder,
                "Normalized Goal Distance Return",
            )
        if plot_gp_fits:
            for hp_pair in combinations(hp_list, 2):
                plot_gp_fit(
                    phase,
                    phase_result,
                    "mean_normalized_goal_distance_return",
                    list(hp_pair),
                    output_folder,
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
    plot_gp_fits: bool = False,
    plot_regret: bool = False,
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
        plot_gp_fits=plot_gp_fits,
        plot_regret=plot_regret,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zipfile", type=Path, required=True)
    parser.add_argument("--plot_return_distributions", action="store_true")
    parser.add_argument("--plot_eval_curves", action="store_true")
    parser.add_argument("--plot_gp_fits", action="store_true")
    parser.add_argument("--plot_regret", action="store_true")
    parser.add_argument("--no_multiprocessing", action="store_true")
    args = parser.parse_args()

    sns.set_theme(context="talk", rc={"figure.figsize": (4, 3)})

    # Parse results
    plots_folder = Path("plots") / os.path.basename(args.zipfile)
    results: dict[str, tuple[dict, ResultsPerStep[PhaseResult]]] = (
        read_results_from_zip(args.zipfile)
    )
    results_pandas = {
        identifier: (run_info, phase_results_to_pandas(phase_results))
        for identifier, (run_info, phase_results) in results.items()
    }

    plot_results = partial(
                    plot_parallel_wrapper,
                    plots_folder,
                    plot_return_distributions=args.plot_return_distributions,
                    plot_eval_curves=args.plot_eval_curves,
                    plot_gp_fits=args.plot_gp_fits,
                    plot_regret=args.plot_regret,
                )

    if not args.no_multiprocessing:
        try:
            thread_count = int(os.environ["SLURM_CPUS_ON_NODE"]) // 3
        except Exception as _:
            thread_count = multiprocessing.cpu_count() // 3
        with multiprocessing.get_context("spawn").Pool(thread_count) as pool:
            pool.map(
                plot_results,
                results_pandas.items(),
            )
    else:
        for item in results_pandas.items():
            plot_results(item)

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
            "hps",
        ]
    )  # type: ignore
    for experiment_name in os.listdir(plots_folder):
        if experiment_name == "grid_plots":
            continue
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
        dataset = (
            ",".join(setup["datasets"]) if "datasets" in setup else setup["dataset"]
        )
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
                r".*/(?P<plot_type>[^-/]*)-(?:\((?P<hps>[^\)]*)\)-)?(?P<y_col>[^-/]*)-(?P<phase>\d+).png$",
                str(unscaled_plotpath),
            )
            if not plot_name_matches:
                raise ValueError(
                    f"Naming inside of plots folder not as expected for {unscaled_plotpath}."
                )

            imgpath_dataframe.loc[len(imgpath_dataframe)] = [
                plot_name_matches["plot_type"],
                int(plot_name_matches["phase"]),
                agent,
                str(unscaled_plotpath),
                dataset,
                actor_loss,
                plot_name_matches["y_col"],
                plot_name_matches["hps"],
            ]

    # Now actually plot all grid plots
    for plot_type, agent, y_col, actor_loss, hps in product(
        imgpath_dataframe["plot_type"].unique(),
        imgpath_dataframe["agent"].unique(),
        imgpath_dataframe["y_col"].unique(),
        imgpath_dataframe["actor_loss"].unique(),
        imgpath_dataframe["hps"].unique(),
    ):
        grid_df = imgpath_dataframe[
            (imgpath_dataframe["plot_type"] == plot_type)
            & (imgpath_dataframe["agent"] == agent)
            & (imgpath_dataframe["y_col"] == y_col)
            & ((imgpath_dataframe["hps"] == hps) | pd.isnull(imgpath_dataframe["hps"]))
            & (imgpath_dataframe["actor_loss"] == actor_loss)
        ]
        if len(grid_df) == 0:
            continue
        plots_subfolder = plots_folder / "grid_plots" / f"{plot_type}-{y_col}"
        plots_subfolder.mkdir(exist_ok=True, parents=True)
        grid_plot(
            grid_df,
            plots_subfolder,
            f"{plot_type}-{agent}-{hps}-{y_col}-{actor_loss}",
            grid_columns=("dataset", "phase"),
        )  # type: ignore
