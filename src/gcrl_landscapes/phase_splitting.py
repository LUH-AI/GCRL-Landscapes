from typing import Callable
from pathlib import Path
from .util.data import read_results_from_zip, phase_results_to_pandas
import pandas as pd
import numpy as np
from scipy.stats import trim_mean
from scipy.optimize import root_scalar, RootResults

TARGET_EVAL_STEP = 1_000_000


def get_all_phases(
    agent: str,
    dataset: str,
    final_performance_percentage: int,
    zippath: Path,
    phase_percentages: list[int],
    mode: str = "target_ratio",
    interpolation: str = "linear",
) -> list[int]:
    """Get phases to run based on experimental performance results for long-term runs.

    Args:
        agent: Name of agent
        dataset: Name of dataset
        final_performance_percentage: percentage of final_performance to hit
        zippath: zipfile with results. see `get_data` for more details.
        phase_percentages: percentages at which a phase should be run
        mode: may be "target_ratio" or "performance_ratio".
              "performance_ratio" means given phase_percentages are percentages of the performance curve
              "target_ratio" means, we only look at the steps needed to reach the target value. Basically linear function as interpolation
        interpolation: Which interpolation mode to use. see `fit_function`
    Returns:
        phases as list
    """
    data = get_data(agent, dataset, zippath)
    performance_xs, performance_ys = zip(
        *(
            [(0, 0)]
            + [
                (
                    group["eval_step"].iloc[0],
                    trim_mean(group["success"].to_numpy(), 0.25),
                )
                for _, group in data.groupby(by=["eval_step"])
            ]
        )
    )

    performance_function = fit_function(performance_xs, performance_ys, interpolation)
    final_performance = performance_function(TARGET_EVAL_STEP)
    performance_target = final_performance * (final_performance_percentage / 100)

    # Find first bracket which contains performance_target based on data
    # -> find leftmost root
    bracket_idx_right = np.min(np.where(performance_ys >= performance_target)[0])
    bracket = (performance_xs[bracket_idx_right - 1], performance_xs[bracket_idx_right])

    if mode == "target_ratio":
        root_result: RootResults = root_scalar(
            lambda x: performance_function(x) - performance_target,
            bracket=bracket,
            method="brentq",
        )
        if not root_result.converged:
            raise Exception("could not find final performance percentage given data")
        performance_target_steps = root_result.root
        return [
            int(performance_target_steps * (phase_percentage / 100))
            for phase_percentage in phase_percentages
        ]

    raise NotImplementedError(mode)


def fit_function(
    xs: list[float], ys: list[float], interpolation_mode: str
) -> Callable[[np.ndarray | int], np.ndarray | float]:
    """fit a function given mode to data.

    Args:
        data: pandas dataframe containing data
        interpolation_mode: interpolation to apply

    Returns:
        [TODO: do we want success rate here?]
        callable which maps training steps to success rate

    Raises:
        ValueError: TARGET_EVAL_STEP could not be found in data
        NotImplementedError: interpolation mode is not known
    """

    if TARGET_EVAL_STEP not in xs:
        raise ValueError(
            f"{TARGET_EVAL_STEP} not in given data. Given max was {max(ys)}."
        )

    if interpolation_mode == "linear_target":
        return lambda x: (x / TARGET_EVAL_STEP) * ys[xs.index(TARGET_EVAL_STEP)]
    elif interpolation_mode == "linear":
        return lambda x: np.interp(x, xs, ys)

    raise NotImplementedError(f"no interpolation mode {interpolation_mode}")


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
        pandas dataframe of results for given agent-dataset-combination
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
