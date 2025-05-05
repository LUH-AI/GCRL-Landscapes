import numpy as np


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
