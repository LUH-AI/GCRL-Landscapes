from pathlib import Path
import hydra
from omegaconf import DictConfig
from .configurations import hydra_to_ogbench_config
from .submission import train_wrapper


@hydra.main(config_path="../../configs", config_name="hpo_crl", version_base="1.1")
def hpo_target(hydra_config: DictConfig) -> float:
    config = hydra_to_ogbench_config(hydra_config)
    eval_trajectory = train_wrapper(
        agent_name=config["agent_name"],  # type: ignore
        agent_path=None,
        dataset_name=config["env"],  # type: ignore
        already_trained_steps=0,
        eval_steps=[config["training_steps"]],  # type: ignore
        save_steps=[],
        eval_episodes=config["eval_episodes"],  # type: ignore
        configuration=config,
        run_log_dir=Path("./run_log") / f"{str(hash(config))[:8]}",
        tasks_per_node_parallel=config["tasks_per_node_parallel"],  # type: ignore
        seed=config["seed"],  # type: ignore
    )
    eval_results = eval_trajectory[0]
    return eval_results[max(eval_results.keys())].success


if __name__ == "__main__":
    hpo_target()
