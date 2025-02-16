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

DIM_LABEL_MAPPING = {
    "actor_p_trajgoal": "$p_{trajgoal}$",
    "discount": "Discount Factor",
}

FTU_SIGNIFICANCE_THRESHOLD = 0.05


def map_labels(label: str) -> str:
    return DIM_LABEL_MAPPING[label] if label in DIM_LABEL_MAPPING else label


def plot(results_pandas: pd.DataFrame, folder: Path, info: dict[str, Any]):
    hp_full_list = results_pandas.columns[results_pandas.columns.str.startswith("hp.")]
    hp_list = [f"hp.{hp_name}" for hp_name in info["arguments"]["hyperparameters"]]
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
        for phase_start, phase in zip(phase_starts, info["arguments"]["phases"])
    ]

    for phase, phase_result in phase_results:
        phase_result_copy = phase_result.copy()
        phase_result_copy.loc[:, "run_id"], _ = pd.factorize(
            phase_result_copy["run_id"]
        )  # TripleGPModel needs continuous run-ids starting at 0
        model = TripleGPModel(
            phase_result_copy,
            np.float64,
            y_col="success",
            hp_names=hp_list,
            configspace=get_config_space(""),
        )
        model.fit()
        create_contour_plot(
            model,
            x_dim=0,
            y_dim=1,
            z_dim="Eval Returns",
            bounds=[0, 1],
            filename=folder / f"igpr_{phase}.png",
            dim_label_mapping=map_labels,
        )
        create_contour_plot(
            model,
            x_dim=0,
            y_dim=1,
            z_dim="Eval Returns",
            bounds=[None, None],
            filename=folder / f"igpr_{phase}_scaled.png",
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
        plt.colorbar(c, label="Success")
        plt.savefig(folder / f"nearest_{phase}.png")

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
            success_per_seed_task = np.array(
                [
                    [np.mean(task["success_all"]) for task in result.info]
                    for result in eval_results
                ]
            )
            return FTU(success_per_seed_task.reshape(-1), routine="c++")  # type: ignore  # the type is correct, there seems to be an import problem

        # group by configuration to apply statistic over seeds
        ftu_object_per_configuration = phase_result_copy.groupby(["run_id"] + hp_list)[
            "eval_result"
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

        plt.savefig(folder / f"modality_{phase}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zipfile", type=Path, required=True)
    args = parser.parse_args()

    results: dict[str, tuple[dict, ResultsPerStep[PhaseResult]]] = (
        read_results_from_zip(args.zipfile)
    )

    results_pandas = {
        identifier: (run_info, phase_results_to_pandas(phase_results))
        for identifier, (run_info, phase_results) in results.items()
    }

    for prefix, (run_info, results_df) in results_pandas.items():
        folder = Path("plots") / re.match(r"^logs/([^/]*)/?", prefix).group(1)
        folder.mkdir(exist_ok=True, parents=True)
        plot(results_df, folder, run_info)
