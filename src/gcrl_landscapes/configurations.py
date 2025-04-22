from ml_collections import FrozenConfigDict
from ConfigSpace import ConfigurationSpace, Float, Categorical
from scipy.stats.qmc import Sobol
from math import log
import ogbench.impls.agents.crl
import ogbench.impls.agents.gciql
import ogbench.impls.agents.gcivl
import ogbench.impls.agents.cmd
import ogbench.impls.agents.gcbc
import ogbench.impls.agents.qrl
import ogbench.impls.agents.hiql
import warnings
import numpy as np
import logging

logger = logging.getLogger(__name__)

LEARNING_RATE_LOWER = 1e-6
LEARNING_RATE_UPPER = 1e-3

DISCOUNT_FACTOR_LOWER = 0.8
DISCOUNT_FACTOR_UPPER = 0.99

SUPPORTED_HPS = set(["lr", "discount", "actor_p_trajgoal"])


def get_config_space(agent: str) -> ConfigurationSpace:
    warnings.warn("not fully implemented, returns the same for every agent")
    return ConfigurationSpace(
        {
            "lr": Float("lr", (LEARNING_RATE_LOWER, LEARNING_RATE_UPPER)),
            "discount": Float(
                "discount", (DISCOUNT_FACTOR_LOWER, DISCOUNT_FACTOR_UPPER)
            ),
            "actor_p_trajgoal": Float("actor_p_trajgoal", (0, 1)),
            "actor_p_randomgoal": Float("actor_p_randomgoal", (0, 1)),
            "actor_p_curgoal": Float("actor_p_curgoal", (0, 1)),
            "actor_geom_sample": Categorical("actor_geom_sample", (True, False)),
        }
    )


def generate_configurations(
    n: int, agent: str, hyperparameters: set[str], seed: int = 0
) -> list[FrozenConfigDict]:
    logger.info(f"Generating {n} configurations for {agent} using {hyperparameters}")
    np.random.seed(seed)
    ld_n = log(n, 2)
    if not ld_n.is_integer():
        raise ValueError("Only supports powers of 2")

    agent_config_generators = {
        "CRL": lambda n: _generate_configurations(
            ogbench.impls.agents.crl.get_config().to_dict(), n, hyperparameters
        ),
        "CMD": lambda n: _generate_configurations(
            ogbench.impls.agents.cmd.get_config().to_dict(), n, hyperparameters
        ),
        "GCBC": lambda n: _generate_configurations(
            ogbench.impls.agents.gcbc.get_config().to_dict()
            if "discount" not in hyperparameters
            else ogbench.impls.agents.gcbc.get_config().to_dict()
            | {"actor_geom_sample": True},
            n,
            hyperparameters,
        ),
        "GCIQL": lambda n: _generate_configurations(
            ogbench.impls.agents.gciql.get_config().to_dict(), n, hyperparameters
        ),
        "GCIVL": lambda n: _generate_configurations(
            ogbench.impls.agents.gcivl.get_config().to_dict(), n, hyperparameters
        ),
        "HIQL": lambda n: _generate_configurations(
            ogbench.impls.agents.hiql.get_config().to_dict(), n, hyperparameters
        ),
        "QRL": lambda n: _generate_configurations(
            ogbench.impls.agents.qrl.get_config().to_dict(), n, hyperparameters
        ),
    }
    return agent_config_generators[agent](n)


def _generate_configurations(base_config: dict, n: int, hyperparameters: set[str]):
    if not len(hyperparameters & SUPPORTED_HPS) == len(hyperparameters):
        raise NotImplementedError(
            f"hyperparameters {hyperparameters - SUPPORTED_HPS} not supported"
        )
    learning_rates = (
        _generate_learning_rates(n)
        if "lr" in hyperparameters
        else _generate_dummy_list(n, base_config["lr"])
    )
    discount_factors = (
        _generate_discount_factors(n)
        if "discount" in hyperparameters
        else _generate_dummy_list(n, base_config["discount"])
    )
    # trajectory goals for dataset sampling
    actor_p_curgoals = _generate_dummy_list(
        n, 0
    )  # always on 0 as it shouldn't be that relevant
    actor_p_trajgoals = (
        _generate_actor_p_trajgoals(n)
        if "actor_p_trajgoal" in hyperparameters
        else _generate_dummy_list(n, base_config["actor_p_trajgoal"])
    )
    actor_p_randomgoals = list(
        np.ones(len(actor_p_trajgoals)) - np.array(actor_p_trajgoals)
    )

    return [
        FrozenConfigDict(
            initial_dictionary=base_config
            | {
                "lr": lr,
                "discount": df,
                "actor_p_curgoal": actor_p_curgoal,
                "actor_p_trajgoal": actor_p_trajgoal,
                "actor_p_randomgoal": actor_p_randomgoal,
            }
        )
        for lr, df, actor_p_curgoal, actor_p_trajgoal, actor_p_randomgoal in zip(
            learning_rates,
            discount_factors,
            actor_p_curgoals,
            actor_p_trajgoals,
            actor_p_randomgoals,
        )
    ]


def _generate_learning_rates(n: int) -> list[float]:
    return list(
        Sobol(1).random_base2(round(log(n, 2))).reshape(-1)
        * (LEARNING_RATE_UPPER - LEARNING_RATE_LOWER)
        + LEARNING_RATE_LOWER
    )


def _generate_discount_factors(n: int) -> list[float]:
    return list(
        Sobol(1).random_base2(round(log(n, 2))).reshape(-1)
        * (DISCOUNT_FACTOR_UPPER - DISCOUNT_FACTOR_LOWER)
        + DISCOUNT_FACTOR_LOWER
    )


def _generate_actor_p_trajgoals(n: int) -> list[float]:
    # Already in [0, 1], no need for scaling
    return list(Sobol(1).random_base2(round(log(n, 2))).reshape(-1))


def _generate_dummy_list(n: int, val: object) -> list[object]:
    return [val] * n
