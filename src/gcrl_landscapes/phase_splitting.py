from typing import Callable
from pathlib import Path
from .util.data import read_results_from_zip, phase_results_to_pandas
import pandas as pd


def get_all_phases(
    agent: str,
    dataset: str,
    final_performance_percentage: int,
    phase_percentages: list[int] = [25, 50, 100],
    mode: str = "linear_final",
) -> list[int]:
    return


def fit_function(agent: str, dataset: str) -> Callable[[int], float]:
    raise NotImplementedError()


def get_data(agent: str, dataset: str, zippath: Path) -> pd.DataFrame:
    """Gather data from zipfile for agent-dataset-combination.
    Expects a zipfile with "logs/" in root as created by the runners.
    Here only only *one configuration*, *one phase* and *one agent-dataset-combination* is allowed.
    There may be as many seeds as wanted.

    Args:
        agent: Name of agent. Used for getting correct experiment from zip
        dataset: Name of dataset. Used for getting correct experiment from zip
        zipf: zipfile with results

    Returns:
        x and y datapoints, mapping training steps to success rate
        [TODO: do we want success rate here?]
    """
    results = read_results_from_zip(zippath)

    # find matching prefix
    try:
        matching_prefixes = [
            prefix for prefix in results.keys() if agent in prefix and dataset in prefix
        ]
        if len(matching_prefixes) != 1:
            raise ValueError(f"{zippath} has more than one agent-dataset-combination.")
        prefix = matching_prefixes[0]
    except StopIteration:
        raise ValueError(f"Could not find prefix for {agent}-{dataset} in {zippath}")

    result = phase_results_to_pandas(results[prefix][1])
    # sanity checks
    ## only one configuration
    if len(result["config_index"].unique()) != 1:
        raise ValueError(f"{prefix} has more than one configuration.")
    ## one phase
    if len(result["phase"].unique()) != 1:
        raise ValueError(f"{prefix} has more than one phase.")

    return result
