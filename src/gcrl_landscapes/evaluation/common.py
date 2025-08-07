import pandas as pd
import numpy as np
from copy import deepcopy
from gcrl_landscapes.util.eval import cvar, iqr
from gcrl_landscapes.util.data import (
    PhaseResult,
    ResultsPerStep,
    phase_results_to_pandas,
)


DIM_LABEL_MAPPING = {
    "actor_p_trajgoal": "$p_{trajgoal}$",
    "discount": "Discount Factor",
    "alpha": "Alpha",
    "lr": "Learning Rate",
    "lr-uniform": "Learning Rate",
}

FTU_SIGNIFICANCE_THRESHOLD = 0.05

CVAR_CONFIDENCE_LEVELS = (10, 20, 30, 40)
DISP_CONFIDENCE_LEVELS = (10, 90)


def map_labels(label: str) -> str:
    return DIM_LABEL_MAPPING[label] if label in DIM_LABEL_MAPPING else label


def compute_additional_information(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Compute additional metrics which can be calculated given the data. E.g. CVaR, Dispersion

    Args:
        phase_results: all phase results in list

    Returns:
        new object with additional information with same layout as original one
    """
    results_copy = deepcopy(results)
    # Normalize goal end distances by start distance to get a distance in [1, inf)
    results_copy["normalized_goal_distances"] = results_copy["eval_result"].apply(
        lambda results_per_seed: np.array(
            [
                np.array(result["goal_end_distances"])
                / np.array(result["goal_start_distances"])
                for result in results_per_seed.info
            ]
        ).reshape(-1)
    )
    # To see this as a "score", subtract from 1. Yields scores in (-inf, 1], where 1 is perfect, 0 neutral and below is bad
    results_copy["normalized_goal_distance_returns"] = results_copy[
        "normalized_goal_distances"
    ].apply(lambda distances: 1 - distances)
    results_copy["mean_normalized_goal_distance"] = results_copy[
        "normalized_goal_distances"
    ].apply(np.mean)
    results_copy["mean_normalized_goal_distance_return"] = results_copy[
        "normalized_goal_distance_returns"
    ].apply(np.mean)

    # Calculate CVaR
    for confidence_level in CVAR_CONFIDENCE_LEVELS:
        results_copy[f"cvar{confidence_level}_normalized_goal_distance_return"] = (
            results_copy["normalized_goal_distance_returns"].apply(
                lambda distances: cvar(distances, confidence_level=confidence_level)
            )
        )

    # Calculate Dispersion on goal distance distribution
    results_copy["disp_normalized_goal_distance"] = results_copy[
        "normalized_goal_distance_returns"
    ].apply(lambda returns: iqr(returns, DISP_CONFIDENCE_LEVELS))
    results_copy["disp_normalized_goal_distance_score"] = (
        1 - results_copy["disp_normalized_goal_distance"]
    )
    assert results_copy["disp_normalized_goal_distance_score"].max() <= 1

    # Regret
    ## Check how many hyperparameters are varied.
    ## This only supports 2 HPs at the same time.
    ## Otherwise we'll have to also look at the different combinations and apply some kind of aggregation
    temp_df = results_copy.loc[:, results_copy.columns.str.startswith("hp.")]
    if sum(temp_df.nunique() > 1) > 2:
        raise NotImplementedError(
            "More than 2 hyperparameters are varied. This is not supported yet for regret calculations as these are done upfront."
        )
    ## Now calculate regret
    for column in (
        "mean_normalized_goal_distance_return",
        "success",
    ):
        column_max_per_phase = (
            (
                results_copy.groupby(by=["eval_step", "phase", "config_index"])[column]
                .mean()
                .reset_index()  # marginalize seed
            )
            .groupby(by=["eval_step", "phase"])[column]
            .max()
            .reset_index()
            .rename(columns={column: f"{column}_max_per_phase"})
        )  # get best configuration per phase
        results_copy = pd.merge(
            results_copy,
            column_max_per_phase,
            how="left",
            on=["eval_step", "phase"],
        )
        results_copy[f"{column}_regret"] = (
            results_copy[f"{column}_max_per_phase"] - results_copy[column]
        )
        ## Calculate cumulative regret
        cum_regret_result: pd.DataFrame = results_copy[  # type: ignore
            (results_copy["eval_step"] == results_copy["phase"])
        ].sort_values(by="phase")
        col_cummean = (
            cum_regret_result.groupby(by=["config_index", "seed"])[f"{column}_regret"]
            .expanding()
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .rename(f"{column}_regret_cummean")  # type: ignore
        )  # type: ignore
        results_copy = pd.merge(
            results_copy,
            col_cummean,
            how="outer",
            left_index=True,
            right_index=True,
        )

    return results_copy


def merge_experiments(
    results: dict[str, tuple[dict, ResultsPerStep[PhaseResult]]],
) -> pd.DataFrame:
    return pd.concat(
        [
            phase_results_to_pandas(result).assign(
                agent=run_info["arguments"]["agent"],
                dataset=",".join(run_info["arguments"]["datasets"])
                if "datasets" in run_info["arguments"]
                else run_info["arguments"]["dataset"],
                hps=lambda x: [run_info["arguments"]["hyperparameters"]] * len(x),
            )
            for run_info, result in results.values()
        ]
    )
