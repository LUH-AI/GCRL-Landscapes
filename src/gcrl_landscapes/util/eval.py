import numpy as np


def cvar(values: np.ndarray, confidence_level: int = 5) -> np.floating:
    tail = values[values < np.percentile(values, confidence_level)]
    return np.mean(tail)
