import argparse
from .util.data import (
    ResultsPerStep,
    PhaseResult,
    phase_results_to_pandas,
    read_results_from_zip,
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
        )
        create_contour_plot(
            model,
            x_dim=0,
            y_dim=1,
            z_dim="Eval Returns",
            bounds=[None, None],
            filename=folder / f"igpr_{phase}_scaled.png",
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
