from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from .configurations import hydra_to_ogbench_config
from .submission import train_wrapper
import signal
import sys
import os
import yaml
import csv
import re

HYPERSWEEPER_REMOVABLE_KEYS = [
    "config_id",
    "performance",
    "budget",
    "budget_used",
    "total_wallclock_time",
    "total_optimization_time",
]

RUN_CONFIG_REMOVABLE_KEYS = [
    "tasks_per_node_parallel",
]


def find_best_agent(last_run: Path) -> Path:
    """Find best agent checkpoint based on Hypersweeper/SMAC HPO run

    Args:
        last_run: Directory of hypersweeper log

    Returns:
        Path to best checkpoint from best configuration of run

    Raises:
        ValueError: If there is a mismatch between directory configs and the logged best config
    """
    with open(last_run / "incumbent.csv", "r") as f:
        *_, best_config_full = csv.DictReader(f)
        best_config = {
            key.lower(): val.lower()
            for key, val in best_config_full.items()
            if key not in HYPERSWEEPER_REMOVABLE_KEYS
        }

    # We will find all directories containing logs for the best configuration in the following steps
    run_log_directories = filter(
        lambda dir: dir.is_dir() and re.fullmatch(r"^\d+$", str(dir.name)),
        [last_run / child for child in os.listdir(last_run)],
    )

    def read_config(dir: Path) -> dict:
        with open(dir / "hydra_config.yaml", "r") as f:
            loaded_config = yaml.load(f, yaml.BaseLoader)
        return {
            key.lower(): str(val).lower()
            for key, val in loaded_config.items()
            if key not in RUN_CONFIG_REMOVABLE_KEYS
        } | {"log_dir": dir}

    configs = map(read_config, run_log_directories)
    matching_configs = list(  # These are all seeds for the best configuration/incumbent
        filter(
            lambda config: all(
                [best_config[key] == config[key] for key in best_config.keys()]
            ),
            configs,
        )
    )

    seeds = [config["seed"] for config in matching_configs]
    if not len(seeds) == len(set(seeds)):
        raise ValueError("found more than one config matching the best config")

    # Now we can look at which seed got us the best performance and find out its checkpoint-path
    def get_final_performance(config: dict) -> dict:
        with open(config["log_dir"] / "train_log" / "eval_log.csv", "r") as f:
            *_, last_log = csv.DictReader(f)
            return {
                "success": float(last_log["success"]),
                "step": int(last_log["step"]),
            }

    config_best_seed = max(
        matching_configs, key=lambda config: get_final_performance(config)["success"]
    )

    return (
        config_best_seed["log_dir"]
        / "train_log"
        / f"params_{get_final_performance(config_best_seed)['step']}.pkl"
    )


@hydra.main(config_path="../../configs", config_name="hpo_crl", version_base="1.1")
def hpo_target(hydra_config: DictConfig) -> float:
    # submitit just bypasses SIGTERM although it should end the job, overwrite that behaviour here
    def handler(signum, frame):
        print(f"Received {signal.Signals(signum).name} ({signum}), stopping!")
        sys.exit(1)

    signal.signal(signal.SIGTERM, handler)

    with open("hydra_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(hydra_config, resolve=True))

    if (
        hydra_config["phases"].index(hydra_config["phase"]) > 0
    ):  # There has been a run before
        last_phase = hydra_config["phases"][
            hydra_config["phases"].index(hydra_config["phase"]) - 1
        ]
        agent_path = find_best_agent(Path(os.getcwd()) / ".." / ".." / str(last_phase))
        already_trained_steps = last_phase
        print(
            f"Already trained {already_trained_steps} steps. Resuming training from checkpoint {agent_path}"
        )
    else:
        already_trained_steps = 0
        agent_path = None

    config = hydra_to_ogbench_config(hydra_config)
    eval_trajectory = train_wrapper(
        agent_name=config["agent_name"],  # type: ignore
        agent_path=agent_path,
        dataset_name=config["env"],  # type: ignore
        already_trained_steps=already_trained_steps,
        eval_steps=[config["phase"]],  # type: ignore
        save_steps=[config["phase"]],  # type: ignore
        eval_episodes=config["eval_episodes"],  # type: ignore
        configuration=config,
        run_log_dir=Path("./train_log"),
        tasks_per_node_parallel=config["tasks_per_node_parallel"],  # type: ignore
        seed=config["seed"],  # type: ignore
    )
    eval_results = eval_trajectory[0]
    return -eval_results[max(eval_results.keys())].success


if __name__ == "__main__":
    for key in [key for key in os.environ.keys() if "SLURM" in key]:
        del os.environ[key]
    hpo_target()
