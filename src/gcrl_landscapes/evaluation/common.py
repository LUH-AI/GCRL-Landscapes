import pandas as pd
import numpy as np
from copy import deepcopy
from gcrl_landscapes.util.eval import cvar, iqr


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
    phase_results: list[tuple[int, pd.DataFrame]],
) -> list[tuple[int, pd.DataFrame]]:
    """Compute additional metrics which can be calculated given the data. E.g. CVaR, Dispersion

    Args:
        phase_results: all phase results in list

    Returns:
        new object with additional information with same layout as original one
    """
    phase_results_copy = deepcopy(phase_results)
    for _, phase_result in phase_results_copy:
        # Normalize goal end distances by start distance to get a distance in [1, inf)
        phase_result["normalized_goal_distances"] = phase_result["eval_result"].apply(
            lambda results_per_seed: np.array(
                [
                    np.array(result["goal_end_distances"])
                    / np.array(result["goal_start_distances"])
                    for result in results_per_seed.info
                ]
            ).reshape(-1)
        )
        # To see this as a "score", subtract from 1. Yields scores in (-inf, 1], where 1 is perfect, 0 neutral and below is bad
        phase_result["normalized_goal_distance_returns"] = phase_result[
            "normalized_goal_distances"
        ].apply(lambda distances: 1 - distances)
        phase_result["mean_normalized_goal_distance"] = phase_result[
            "normalized_goal_distances"
        ].apply(np.mean)
        phase_result["mean_normalized_goal_distance_return"] = phase_result[
            "normalized_goal_distance_returns"
        ].apply(np.mean)

        # Calculate CVaR
        for confidence_level in CVAR_CONFIDENCE_LEVELS:
            phase_result[f"cvar{confidence_level}_normalized_goal_distance_return"] = (
                phase_result[
                    "normalized_goal_distance_returns"
                ].apply(
                    lambda distances: cvar(distances, confidence_level=confidence_level)
                )
            )

        # Calculate Dispersion on goal distance distribution
        phase_result["disp_normalized_goal_distance"] = phase_result[
            "normalized_goal_distance_returns"
        ].apply(lambda returns: iqr(returns, DISP_CONFIDENCE_LEVELS))
        phase_result["disp_normalized_goal_distance_score"] = (
            1 - phase_result["disp_normalized_goal_distance"]
        )
        assert phase_result["disp_normalized_goal_distance_score"].max() <= 1

    return phase_results_copy
