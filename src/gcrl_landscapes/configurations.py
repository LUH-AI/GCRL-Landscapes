from ml_collections import FrozenConfigDict
from ConfigSpace import ConfigurationSpace, Float
from scipy.stats.qmc import Sobol
from math import log
import ogbench.impls.agents.crl
import ogbench.impls.agents.cmd
import ogbench.impls.agents.gcbc
import ogbench.impls.agents.qrl
import ogbench.impls.agents.hiql
import warnings

LEARNING_RATE_LOWER = 1e-6
LEARNING_RATE_UPPER = 1e-3

DISCOUNT_FACTOR_LOWER = 0.8
DISCOUNT_FACTOR_UPPER = 0.99


def get_config_space(agent: str) -> ConfigurationSpace:
    warnings.warn("not fully implemented, returns the same for every agent")
    return ConfigurationSpace(
        {
            "lr": Float("lr", (LEARNING_RATE_LOWER, LEARNING_RATE_UPPER)),
            "discount": Float(
                "discount", (DISCOUNT_FACTOR_LOWER, DISCOUNT_FACTOR_UPPER)
            ),
        }
    )


def generate_configurations(n: int, agent: str) -> list[FrozenConfigDict]:
    ld_n = log(n, 2)
    if not ld_n.is_integer():
        raise ValueError("Only supports powers of 2")

    agent_config_generators = {
        "CRL": lambda n: _generate_configurations(
            ogbench.impls.agents.crl.get_config().to_dict(), n
        ),
        "CMD": lambda n: _generate_configurations(
            ogbench.impls.agents.cmd.get_config().to_dict(), n
        ),
        "GCBC": lambda n: _generate_configurations(
            ogbench.impls.agents.gcbc.get_config().to_dict(), n
        ),
        "QRL": lambda n: _generate_configurations(
            ogbench.impls.agents.qrl.get_config().to_dict(), n
        ),
        "HIQL": lambda n: _generate_configurations(
            ogbench.impls.agents.hiql.get_config().to_dict(), n
        ),
    }
    return agent_config_generators[agent](n)


def _generate_configurations(base_config: dict, n: int):
    return [
        FrozenConfigDict(initial_dictionary=base_config | {"lr": lr, "discount": df})
        for lr, df in zip(_generate_learning_rates(n), _generate_discount_factors(n))
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
