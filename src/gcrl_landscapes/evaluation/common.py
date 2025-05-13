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
            results_copy[
                "normalized_goal_distance_returns"
            ].apply(
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

    return results_copy


def merge_experiments(
    results: dict[str, tuple[dict, ResultsPerStep[PhaseResult]]],
) -> pd.DataFrame:
    return pd.concat(
        [
            phase_results_to_pandas(result).assign(
                agent=run_info["arguments"]["agent"],
                dataset=run_info["arguments"]["dataset"],
            )
            for run_info, result in results.values()
        ]
    )
