import numpy as np


def cvar(values: np.ndarray, confidence_level: int = 5) -> np.floating:
    tail = values[values < np.percentile(values, confidence_level)]
    return np.mean(tail)


def cvar_inv(values: np.ndarray, cvar_target: float) -> float | None:
    index = 0
    sorted_values = np.sort(values)
    while index < len(values):
        tail = sorted_values[: index + 1]
        if np.mean(tail) >= cvar_target:
            return (index + 1) / len(values)
        index += 1
    return 1


def iqr(values: np.ndarray, confidence_levels: tuple[int, int]) -> np.floating:
    return np.abs(
        np.percentile(values, confidence_levels[0])
        - np.percentile(values, confidence_levels[1])
    )
