import numpy as np
import jax
import pandas as pd
from gcrl_landscapes.plots.triple_gp import TripleGPModel
from gcrl_landscapes.configurations import (
    get_config_space,
    get_bounds,
    hp_to_sobol_codomain,
)


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

    # Scale all variables
    # Also reverts logarithmic scaling to uniform
    for hp_name in hp_names:
        result_copy[hp_name] = hp_to_sobol_codomain(
            result_copy[hp_name],
            *get_bounds(
                hp_name.removeprefix("hp."), phase_result["hp.agent_name"].iloc[0]
            ),
        )

    model = TripleGPModel(
        result_copy,
        np.float64,
        y_col=y_col,
        hp_names=hp_names,
        configspace=get_config_space(""),
    )
    model.fit()
    return model


def get_gradients(agent, batch):
    value_grad = jax.grad(lambda grad_params: agent.value_loss(batch, grad_params), has_aux=True)(agent.network.params)[0]["modules_value"]
    actor_grad = jax.grad(lambda grad_params: agent.actor_loss(batch, grad_params), has_aux=True)(agent.network.params)[0]["modules_actor"]
    return value_grad, actor_grad


def gradient_cosine_similarity(grad1, grad2):
    flat_grad1 = grad1.reshape(-1)
    flat_grad2 = grad2.reshape(-1)
    return jax.numpy.dot(flat_grad1, flat_grad2) / (jax.numpy.linalg.norm(flat_grad1) * jax.numpy.linalg.norm(flat_grad2))


def gradient_magnitude_similarity(grad1, grad2):
    norm_grad1 = jax.numpy.linalg.norm(grad1)
    norm_grad2 = jax.numpy.linalg.norm(grad2)
    return (2 * norm_grad1 * norm_grad2) / (norm_grad1 ** 2 + norm_grad2 ** 2)


def fully_flatten_tree(x):
    return jax.numpy.concatenate(jax.tree.flatten(jax.tree.map(lambda y: jax.numpy.ravel(y), x))[0])


def flatten_tree_batch(x):
    x_flattened, x_flattened_tree = jax.tree.flatten(x)
    return x_flattened

def pairwise_cosine_similarity(x, y):
    # Stack trees onto each other, retaining batch dimension
    stacked_trees = jax.tree.map(lambda a, b: jax.numpy.stack([a, b], axis=1), x, y)

    x_flattened = flatten_tree_batch(x)
    y_flattened = flatten_tree_batch(y)

    def subtree_batch_pairwise_cosine_similarity(sub_x, sub_y):
        return jax.vmap(jax.numpy.dot, in_axes=0, out_axes=0)(jax.numpy.stack([sub_x, sub_y], axis=1))

    return sum([jax.numpy.dot(sub_x, sub_y) / (jax.numpy.linalg.norm(sub_x) * jax.numpy.linalg.norm(sub_y)) for sub_x, sub_y in zip(x_flattened, y_flattened)])
