import numpy as np
import pandas as pd
from gcrl_landscapes.plots.triple_gp import TripleGPModel
from gcrl_landscapes.configurations import get_config_space, get_bounds


def cvar(values: np.ndarray, confidence_level: int = 5) -> np.floating:
    """Calculate conditional value of risk

    Args:
        values: Values to calculate CVaR for
        confidence_level: tail confidence level/percentile

    Returns:
        Conditional value of risk
    """
    tail = values[values < np.percentile(values, confidence_level)]
    return np.mean(tail)


def cvar_inv(values: np.ndarray, cvar_target: float) -> float | None:
    """Calculate inverted conditional value of risk.
    Instead of calculating the mean of the tail given a percentile, calculate the percentile given the mean of the tail.

    Args:
        values: Values to calculate inverted CVaR for
        cvar_target: Targeted CVaR value for which we calculate the needed percentile

    Returns:
        Needed percentile to get given CVaR
    """
    index = 0
    sorted_values = np.sort(values)
    while index < len(values):
        tail = sorted_values[: index + 1]
        if np.mean(tail) >= cvar_target:
            return (index + 1) / len(values)
        index += 1
    return 1


def iqr(values: np.ndarray, confidence_levels: tuple[int, int]) -> np.floating:
    """Calculate interquartile range.
    Is able to do this for any percentile combination

    Args:
        values: values to calculate interquartile range for
        confidence_levels: left- and right percentile to calculate range for

    Returns:
        range between the two percentiles
    """
    return np.abs(
        np.percentile(values, confidence_levels[0])
        - np.percentile(values, confidence_levels[1])
    )


def fit_model(
    phase_result: pd.DataFrame,
    y_col: str,
    hp_names: list[str],
) -> TripleGPModel:
    result_copy = phase_result.copy()
    result_copy.loc[:, "run_id"], _ = pd.factorize(
        result_copy["run_id"]
    )  # TripleGPModel needs continuous run-ids starting at 0

    # Learning rate is scaled logarithmically
    # -> Show model uniform distribution by appropriate scaling
    for hp_name, hp_lower_bound, hp_upper_bound in [
        (hp, hp_lower_bound, hp_upper_bound)
        for hp, (hp_lower_bound, hp_upper_bound, hp_log_scaled) in zip(
            hp_names,
            [
                get_bounds(hp_name, phase_result["hp.agent_name"].iloc[0])
                for hp_name in hp_names
            ],
        )
        if hp_log_scaled
    ]:
        hp_names = [f"{hp_name}-uniform"] + [
            hp_name for hp_name in hp_names if hp_name != f"hp.{hp_name}"
        ]
        result_copy[f"{hp_name}-uniform"] = (
            np.log10(result_copy[f"hp.{hp_name}"]) - np.log10(hp_lower_bound)
        ) / (np.log10(hp_upper_bound) - np.log10(hp_lower_bound))
    model = TripleGPModel(
        result_copy,
        np.float64,
        y_col=y_col,
        hp_names=hp_names,
        configspace=get_config_space(""),
    )
    model.fit()
    return model
