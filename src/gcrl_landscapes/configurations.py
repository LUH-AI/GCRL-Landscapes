from ml_collections import FrozenConfigDict
from scipy.stats.qmc import Sobol
from math import log
import ogbench.impls.agents.crl

LEARNING_RATE_LOWER = 1e-4
LEARNING_RATE_UPPER = 1e-1

DISCOUNT_FACTOR_LOWER = 0.8
DISCOUNT_FACTOR_UPPER = 0.99


def generate_configurations_crl(n: int) -> list[FrozenConfigDict]:
    ld_n = log(n, 2)
    if not ld_n.is_integer():
        raise ValueError("Only supports powers of 2")

    sobol_generator = Sobol(2)
    sobol_sample = sobol_generator.random_base2(round(ld_n))
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
