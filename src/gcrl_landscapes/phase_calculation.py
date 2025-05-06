from typing import Callable


def get_all_phases(
    agent: str,
    dataset: str,
    final_performance_percentage: int,
    phase_percentages: list[int] = [25, 50, 100],
    mode: str = "linear_final",
) -> list[int]:
    raise NotImplementedError()


def fit_function(agent: str, dataset: str) -> Callable[[int], float]:
    raise NotImplementedError()


def get_data(agent: str, dataset: str) -> tuple[list[int], list[float]]:
    raise NotImplementedError()
