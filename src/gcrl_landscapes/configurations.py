from ml_collections import FrozenConfigDict
from scipy.stats.qmc import Sobol
from math import log
import ogbench.impls.agents.crl
import ogbench.impls.agents.cmd

# [TODO: is it okay that this is different from the paper? With higher learning rates we get overflows]
LEARNING_RATE_LOWER = 1e-6
LEARNING_RATE_UPPER = 1e-3

DISCOUNT_FACTOR_LOWER = 0.8
DISCOUNT_FACTOR_UPPER = 0.99


def generate_configurations(n: int, agent: str) -> list[FrozenConfigDict]:
    ld_n = log(n, 2)
    if not ld_n.is_integer():
        raise ValueError("Only supports powers of 2")

    agent_config_generators = {
        "CRL": _generate_configurations_crl,
        "CMD": _generate_configurations_cmd,
    }
    return agent_config_generators[agent](n)


def _generate_configurations_cmd(n: int) -> list[FrozenConfigDict]:
    sobol_generator = Sobol(2)
    sobol_sample = sobol_generator.random_base2(round(log(n, 2)))
    learning_rates = (
        sobol_sample[:, 0] * (LEARNING_RATE_UPPER - LEARNING_RATE_LOWER)
        + LEARNING_RATE_LOWER
    )
    discount_factors = (
        sobol_sample[:, 1] * (DISCOUNT_FACTOR_UPPER - DISCOUNT_FACTOR_LOWER)
        + DISCOUNT_FACTOR_LOWER
    )
    base_config = ogbench.impls.agents.cmd.get_config().to_dict()
    return [
        FrozenConfigDict(initial_dictionary=base_config | {"lr": lr, "discount": df})
        for lr, df in zip(learning_rates, discount_factors)
    ]


def _generate_configurations_crl(n: int) -> list[FrozenConfigDict]:
    sobol_generator = Sobol(2)
    sobol_sample = sobol_generator.random_base2(round(log(n, 2)))
    learning_rates = (
        sobol_sample[:, 0] * (LEARNING_RATE_UPPER - LEARNING_RATE_LOWER)
        + LEARNING_RATE_LOWER
    )
    discount_factors = (
        sobol_sample[:, 1] * (DISCOUNT_FACTOR_UPPER - DISCOUNT_FACTOR_LOWER)
        + DISCOUNT_FACTOR_LOWER
    )
    base_config = ogbench.impls.agents.crl.get_config().to_dict()
    return [
        FrozenConfigDict(initial_dictionary=base_config | {"lr": lr, "discount": df})
        for lr, df in zip(learning_rates, discount_factors)
    ]
