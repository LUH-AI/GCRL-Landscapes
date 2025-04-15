import numpy as np


def cvar(values: np.ndarray, confidence_level: int = 5) -> np.floating:
    tail = values[values < np.percentile(values, confidence_level)]
    return np.mean(tail)


def iqr(values: np.ndarray, confidence_levels: tuple[int, int]) -> np.floating:
    return np.abs(
        np.percentile(values, confidence_levels[0])
        - np.percentile(values, confidence_levels[1])
    )
