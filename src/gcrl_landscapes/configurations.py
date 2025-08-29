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
import ogbench.impls.agents.sac
import warnings
import numpy as np
import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

LEARNING_RATE_LOWER = 1e-6
LEARNING_RATE_UPPER = 1e-2

DISCOUNT_FACTOR_LOWER = 0.8
DISCOUNT_FACTOR_UPPER = 0.99

AWR_TEMPERATURE_LOWER = 0.0
AWR_TEMPERATURE_UPPER = 30.0

DDPGBC_BC_COEFF_LOWER = 0.0
DDPGBC_BC_COEFF_UPPER = 0.5

EPS_QRL_LOWER = 0.0
EPS_QRL_UPPER = 1.0

TAU_HIQL_LOWER = 1e-4
TAU_HIQL_UPPER = 1.0

SUPPORTED_HPS = set(
    [
        "lr",
        "discount",
        "actor_p_trajgoal",
        "alpha",
        "eps",
        "low_alpha",
        "high_alpha",
        "tau",
    ]
)


def hydra_to_ogbench_config(
    hydra_config: DictConfig, phase_idx: int
) -> FrozenConfigDict:
    get_adapted_default_config(
        hydra_config["agent_name"],
        hydra_config["datasets"][phase_idx],
        hydra_config["actor_loss"],
    )

    hydra_config_modified: dict = dict(hydra_config)
    if "actor_p_curgoalshare" in hydra_config:
        hydra_config_modified["actor_p_curgoal"] = (
            1 - hydra_config["actor_p_trajgoal"]
        ) * hydra_config["actor_p_curgoalshare"]
        hydra_config_modified["actor_p_randomgoal"] = (
            1 - hydra_config["actor_p_trajgoal"]
        ) * (1 - hydra_config["actor_p_curgoalshare"])
        del hydra_config_modified["actor_p_curgoalshare"]
    if "value_p_curgoalshare" in hydra_config:
        hydra_config_modified["value_p_curgoal"] = (
            1 - hydra_config["value_p_trajgoal"]
        ) * hydra_config["value_p_curgoalshare"]
        hydra_config_modified["value_p_randomgoal"] = (
            1 - hydra_config["value_p_trajgoal"]
        ) * (1 - hydra_config["value_p_curgoalshare"])
        del hydra_config_modified["value_p_curgoalshare"]

    return FrozenConfigDict(
        initial_dictionary=dict(
            get_adapted_default_config(
                hydra_config["agent_name"],
                hydra_config["datasets"][phase_idx],
                hydra_config["actor_loss"],
            )
        )
        | hydra_config_modified
    )


def get_config_space(agent: str) -> ConfigurationSpace:
    warnings.warn("not fully implemented, returns the same for every agent")
    return ConfigurationSpace(
        {
            "lr": Float("lr", (LEARNING_RATE_LOWER, LEARNING_RATE_UPPER)),
            "discount": Float(
                "discount", (DISCOUNT_FACTOR_LOWER, DISCOUNT_FACTOR_UPPER)
            ),
            "actor_loss": Categorical("actor_loss", ("original", "ddpgbc", "awr")),
            "alpha": Float(
                "alpha",
                (
                    min(AWR_TEMPERATURE_LOWER, DDPGBC_BC_COEFF_LOWER),
                    max(AWR_TEMPERATURE_UPPER, DDPGBC_BC_COEFF_UPPER),
                ),
            ),
            "low_alpha": Float(
                "low_alpha",
                (
                    min(AWR_TEMPERATURE_LOWER, DDPGBC_BC_COEFF_LOWER),
                    max(AWR_TEMPERATURE_UPPER, DDPGBC_BC_COEFF_UPPER),
                ),
            ),
            "high_alpha": Float(
                "high_alpha",
                (
                    min(AWR_TEMPERATURE_LOWER, DDPGBC_BC_COEFF_LOWER),
                    max(AWR_TEMPERATURE_UPPER, DDPGBC_BC_COEFF_UPPER),
                ),
            ),
            "actor_p_trajgoal": Float("actor_p_trajgoal", (0, 1)),
            "actor_p_randomgoal": Float("actor_p_randomgoal", (0, 1)),
            "actor_p_curgoal": Float("actor_p_curgoal", (0, 1)),
            "actor_geom_sample": Categorical("actor_geom_sample", (True, False)),
            "eps": Float("eps", (0, 1)),
            "tau": Float("tau", (TAU_HIQL_LOWER, TAU_HIQL_UPPER)),
        }
    )


def get_adapted_default_config(
    agent: str, env: str, actor_loss: str | None = None
) -> FrozenConfigDict:
    agent_config_generators = {
        "crl": lambda: _adapt_base_config(
            ogbench.impls.agents.crl.get_config().to_dict(), env, actor_loss
        ),
        "cmd": lambda: _adapt_base_config(
            ogbench.impls.agents.cmd.get_config().to_dict(), env, actor_loss
        ),
        "gcbc": lambda: _adapt_base_config(
            ogbench.impls.agents.gcbc.get_config().to_dict(), env, actor_loss
        ),
        "gciql": lambda: _adapt_base_config(
            ogbench.impls.agents.gciql.get_config().to_dict(), env, actor_loss
        ),
        "gcivl": lambda: _adapt_base_config(
            ogbench.impls.agents.gcivl.get_config().to_dict(), env, actor_loss
        ),
        "hiql": lambda: _adapt_base_config(
            ogbench.impls.agents.hiql.get_config().to_dict(), env, actor_loss
        ),
        "qrl": lambda: _adapt_base_config(
            ogbench.impls.agents.qrl.get_config().to_dict(), env, actor_loss
        ),
        "sac": lambda: _adapt_base_config(
            ogbench.impls.agents.sac.get_config().to_dict(), env, actor_loss
        ),
    }
    return FrozenConfigDict(initial_dictionary=agent_config_generators[agent.lower()]())


def generate_configurations(
    n: int,
    agent: str,
    hyperparameters: set[str],
    env: str,
    seed: int = 0,
    actor_loss: str | None = None,
) -> list[FrozenConfigDict]:
    logger.info(f"Generating {n} configurations for {agent} using {hyperparameters}")
    np.random.seed(seed)
    ld_n = log(n, 2)
    if not ld_n.is_integer():
        raise ValueError("Only supports powers of 2")

    agent_config_generators = {
        "CRL": lambda n: _generate_configurations(
            _adapt_base_config(
                ogbench.impls.agents.crl.get_config().to_dict(), env, actor_loss
            ),
            n,
            hyperparameters,
        ),
        "CMD": lambda n: _generate_configurations(
            _adapt_base_config(
                ogbench.impls.agents.cmd.get_config().to_dict(), env, actor_loss
            ),
            n,
            hyperparameters,
        ),
        "GCBC": lambda n: _generate_configurations(
            _adapt_base_config(
                ogbench.impls.agents.gcbc.get_config().to_dict(), env, actor_loss
            )
            if "discount" not in hyperparameters
            else _adapt_base_config(
                ogbench.impls.agents.gcbc.get_config().to_dict(), env, actor_loss
            )
            | {"actor_geom_sample": True},
            n,
            hyperparameters,
        ),
        "GCIQL": lambda n: _generate_configurations(
            _adapt_base_config(
                ogbench.impls.agents.gciql.get_config().to_dict(), env, actor_loss
            ),
            n,
            hyperparameters,
        ),
        "GCIVL": lambda n: _generate_configurations(
            _adapt_base_config(
                ogbench.impls.agents.gcivl.get_config().to_dict(), env, actor_loss
            ),
            n,
            hyperparameters,
        ),
        "HIQL": lambda n: _generate_configurations(
            _adapt_base_config(
                ogbench.impls.agents.hiql.get_config().to_dict(), env, actor_loss
            ),
            n,
            hyperparameters,
        ),
        "QRL": lambda n: _generate_configurations(
            _adapt_base_config(
                ogbench.impls.agents.qrl.get_config().to_dict(), env, actor_loss
            ),
            n,
            hyperparameters,
        ),
        "SAC": lambda n: _generate_configurations(
            _adapt_base_config(
                ogbench.impls.agents.sac.get_config().to_dict(), env, actor_loss
            ),
            n,
            hyperparameters,
        ),
    }
    return agent_config_generators[agent](n)


def _generate_configurations(base_config: dict, n: int, hyperparameters: set[str]):
    if not len(hyperparameters & SUPPORTED_HPS) == len(hyperparameters):
        raise NotImplementedError(
            f"hyperparameters {hyperparameters - SUPPORTED_HPS} not supported"
        )
    if not len(hyperparameters) == 2:
        raise NotImplementedError("Only supports two hyperparameters at the moment")

    # choose between awr temperature and ddpgbc bc coefficient
    # use given actor loss as information, otherwise look at size of default alpha
    if "actor_loss" in base_config:
        actor_loss = base_config["actor_loss"]
    elif "alpha" in base_config:
        actor_loss = "ddpgbc" if base_config["alpha"] < 1.0 else "awr"
    else:
        actor_loss = None

    random_values = Sobol(d=len(hyperparameters)).random_base2(round(log(n, 2)))
    random_values_per_hp = [random_values[:, i] for i in range(len(hyperparameters))]

    learning_rates = (
        _scale_to_range(
            random_values_per_hp.pop(), LEARNING_RATE_LOWER, LEARNING_RATE_UPPER, True
        )
        if "lr" in hyperparameters
        else _generate_dummy_list(n, base_config["lr"])
    )
    discount_factors = (
        _scale_to_range(
            random_values_per_hp.pop(),
            DISCOUNT_FACTOR_LOWER,
            DISCOUNT_FACTOR_UPPER,
            False,
        )
        if "discount" in hyperparameters
        else _generate_dummy_list(n, base_config["discount"])
    )
    # trajectory goals for dataset sampling
    actor_p_curgoals = _generate_dummy_list(
        n, 0
    )  # always on 0 as it shouldn't be that relevant
    actor_p_trajgoals = (
        _scale_to_range(random_values_per_hp.pop(), 0, 1, False)
        if "actor_p_trajgoal" in hyperparameters
        else _generate_dummy_list(n, base_config["actor_p_trajgoal"])
    )
    actor_p_randomgoals = list(
        np.ones(len(actor_p_trajgoals)) - np.array(actor_p_trajgoals)
    )
    # Set alpha hyperparameter (AWR temperature or ddpgbc bc coefficient)
    ## may be unused if algorithm uses ddpgbc
    awr_temperatures = (
        _scale_to_range(
            random_values_per_hp.pop(),
            AWR_TEMPERATURE_LOWER,
            AWR_TEMPERATURE_UPPER,
            False,
        )
        if "alpha" in hyperparameters and actor_loss != "ddpgbc"
        else _generate_dummy_list(
            n, base_config["alpha"] if "alpha" in base_config else None
        )
    )
    awr_temperatures_low_level_policy = (
        _scale_to_range(
            random_values_per_hp.pop(),
            AWR_TEMPERATURE_LOWER,
            AWR_TEMPERATURE_UPPER,
            False,
        )
        if "low_alpha" in hyperparameters and actor_loss != "ddpgbc"
        else _generate_dummy_list(
            n, base_config["low_alpha"] if "low_alpha" in base_config else None
        )
    )
    awr_temperatures_high_level_policy = (
        _scale_to_range(
            random_values_per_hp.pop(),
            AWR_TEMPERATURE_LOWER,
            AWR_TEMPERATURE_UPPER,
            False,
        )
        if "high_alpha" in hyperparameters and actor_loss != "ddpgbc"
        else _generate_dummy_list(
            n, base_config["high_alpha"] if "high_alpha" in base_config else None
        )
    )

    ## may be unused if algorithm uses awr
    ddpgbc_bc_coeffs = (
        _scale_to_range(
            random_values_per_hp.pop(),
            DDPGBC_BC_COEFF_LOWER,
            DDPGBC_BC_COEFF_UPPER,
            False,
        )
        if "alpha" in hyperparameters and actor_loss == "ddpgbc"
        else _generate_dummy_list(
            n, base_config["alpha"] if "alpha" in base_config else None
        )
    )

    # Set epsilon for QRL
    ## unused for other algorithms
    epss = (
        _scale_to_range(random_values_per_hp.pop(), EPS_QRL_LOWER, EPS_QRL_UPPER, True)
        if "eps" in hyperparameters
        else _generate_dummy_list(
            n, base_config["eps"] if "eps" in base_config else None
        )
    )

    # Set tau for HIQL (and maybe other non-regarded algorithms)
    taus = (
        _scale_to_range(
            random_values_per_hp.pop(), TAU_HIQL_LOWER, TAU_HIQL_UPPER, True
        )
        if "tau" in hyperparameters
        else _generate_dummy_list(
            n, base_config["tau"] if "tau" in base_config else None
        )
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
                "alpha": ddpgbc_bc_coeff if actor_loss == "ddpgbc" else awr_temperature,
                **({"eps": eps} if "eps" in hyperparameters else {}),
                **({"tau": tau} if "tau" in hyperparameters else {}),
                **({"low_alpha": low_alpha} if "low_alpha" in hyperparameters else {}),
                **(
                    {"high_alpha": high_alpha}
                    if "high_alpha" in hyperparameters
                    else {}
                ),
            }
        )
        for lr, df, actor_p_curgoal, actor_p_trajgoal, actor_p_randomgoal, awr_temperature, low_alpha, high_alpha, ddpgbc_bc_coeff, eps, tau in zip(
            learning_rates,
            discount_factors,
            actor_p_curgoals,
            actor_p_trajgoals,
            actor_p_randomgoals,
            awr_temperatures,
            awr_temperatures_low_level_policy,
            awr_temperatures_high_level_policy,
            ddpgbc_bc_coeffs,
            epss,
            taus,
        )
    ]


def _scale_to_range(
    values: np.ndarray, lower: float, upper: float, log: bool
) -> list[float]:
    if log:
        return list(
            10 ** (np.log10(lower) + (np.log10(upper) - np.log10(lower)) * values)
        )
    else:
        return list(lower + (upper - lower) * values)


def _generate_dummy_list(n: int, val: object) -> list[object]:
    return [val] * n


def _adapt_base_config(config: dict, env: str, actor_loss: str | None) -> dict:
    visual = "visual" in env or "powderworld" in env
    discrete = "powderworld" in env
    # [TODO: name GCBCs loss properly. For now it says AWR]
    awr = config["agent_name"].lower() in ["gciql", "gcivl", "hiql", "gcbc"] or (
        discrete or config["actor_loss"] == "awr" or actor_loss == "awr"
    )
    hiql_grad_propagation = "low_actor_rep_grad" in config.keys() and visual
    key_value_pairs = [
        ("encoder", "impala_small", visual),
        ("discrete", True, discrete),
        ("actor_loss", "awr", awr),
        ("alpha", 3.0, awr),
        ("low_actor_rep_grad", True, hiql_grad_propagation),
    ]
    return config | {
        key: value for key, value, condition in key_value_pairs if condition
    }
