import argparse
from util.data import ResultsPerStep, PhaseResult, phase_results_to_pandas
import json
from pathlib import Path
from plots.triple_gp import TripleGPModel, create_contour_plot
import numpy as np
from configurations import get_config_space
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfolder", type=Path, required=True)
    args = parser.parse_args()

    with open(args.logfolder / "results.json", "r") as f:
        results_raw = json.load(f)

    results: ResultsPerStep[PhaseResult] = ResultsPerStep(
        {
            int(step): PhaseResult.from_dict(result)
            for step, result in results_raw.items()
        }
    )

    results_pandas = phase_results_to_pandas(results)
    hp_full_list = results_pandas.columns[results_pandas.columns.str.startswith("hp.")]
    # keep only hyperparameters that are actually changed/do not stay constant
    hp_list = hp_full_list[results_pandas[hp_full_list].nunique() > 1].tolist()

    phase_results = [
        (
            phase,
            results_pandas[
                (results_pandas["eval_step"] == phase)
                & (results_pandas["phase"] == phase)
            ],
        )
        for phase in results_pandas["phase"].unique()
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
            filename=args.logfolder / f"igpr_{phase}.png",
        )

        # Create plot without using gaussian processes
        plt.figure()
        x_scaled = model.x
        y_scaled = model.y.reshape(-1)

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
        yi = model._unscale_y(yi_scaled)

        c = plt.pcolormesh(x0i, x1i, yi, cmap="rocket", vmin=yi.min(), vmax=yi.max())
        plt.colorbar(c, label="Success")
        plt.savefig(args.logfolder / f"nearest_{phase}.png")
