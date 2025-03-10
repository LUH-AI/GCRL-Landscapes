import tqdm
import numpy as np
from ogbench.impls.utils.datasets import GCDataset
from ogbench.impls.utils.log_utils import CsvLogger
from ogbench.impls.utils.flax_utils import save_agent
import gymnasium as gym
import random
import time
from pathlib import Path
from ml_collections import FrozenConfigDict
from typing import Callable, Any, Optional
from .util.data import (
    EvaluationResult,
    EvalTrajectory,
    PhaseResult,
    restore_agent,
    ResultsPerStep,
)
from .util.misc import retry_call
import os
from functools import reduce
from deprecate import deprecated


@deprecated(
    target=None,
    template_mgs="Deprecated in favor of manual running in parallel and collecting afterwards.",
)
def full_phased_run(
    phase_steps: list[int],
    configs: list[FrozenConfigDict],
    agent_class: Callable[[Any, gym.Env, int], Any],
    env: gym.Env,
    train_datasets: list[GCDataset],
    val_datasets: list[GCDataset],
    eval_at_steps: list[int],
    evaluate: Callable[
        [Any, gym.Env, int, FrozenConfigDict],
        tuple[list, dict[str, np.floating], list, list],
    ],
    save_at_steps: list[int] = [],
    log_interval: int = 5000,
    eval_episodes: int = 50,
    log_dir: Path = Path("./logs"),
    seed: int = 0,
) -> ResultsPerStep[PhaseResult]:
    # basically partial function application, but without the need of proper ordering
    def run_phase_configured(
        phase_step: int, already_trained_steps: int, agent_path: Optional[Path]
    ) -> tuple[tuple[FrozenConfigDict, Path], PhaseResult]:
        return run_phase(
            configs=configs,
            phase_steps=phase_step,
            already_trained_steps=already_trained_steps,
            agent_class=agent_class,
            agent_path=agent_path,
            env=env,
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            eval_at_steps=eval_at_steps,
            evaluate=evaluate,
            save_at_steps=save_at_steps,
            log_interval=log_interval,
            eval_episodes=eval_episodes,
            log_dir=log_dir,
            seed=seed,
        )

    def run_phase_and_collect(
        agentstate_and_collector: tuple[
            tuple[int, Optional[Path]], ResultsPerStep[PhaseResult]
        ],
        phase_step: int,
    ) -> tuple[tuple[int, Path], ResultsPerStep[PhaseResult]]:
        (already_trained_steps, agent_path), collector = agentstate_and_collector
        best_config, results = run_phase_configured(
            phase_step, already_trained_steps, agent_path
        )
        collector[phase_step] = results
        return (phase_step, best_config[1]), collector

    # Pass 'None' to randomly initialize agent, Type hint does not work here properly as it expects the result of the function passed to result
    return reduce(run_phase_and_collect, phase_steps, [(0, None), ResultsPerStep()])[1]


@deprecated(
    target=None,
    template_mgs="Deprecated in favor of manual running in parallel and collecting afterwards.",
)
def run_phase(
    configs: list[FrozenConfigDict],
    phase_steps: int,
    already_trained_steps: int,
    agent_class: Callable[[Any, gym.Env, int], Any],
    agent_path: Optional[Path],
    env: gym.Env,
    train_datasets: list[GCDataset],
    val_datasets: list[GCDataset],
    eval_at_steps: list[int],
    evaluate: Callable[
        [Any, gym.Env, int, FrozenConfigDict],
        tuple[list, dict[str, np.floating], list, list],
    ],
    save_at_steps: list[int] = [],
    log_interval: int = 5000,
    eval_episodes: int = 20,
    log_dir: Path = Path("./logs"),
    seed: int = 0,
) -> tuple[tuple[FrozenConfigDict, Path], PhaseResult]:
    """Run a single phase of the training. Return best configuration from that phase

    Args:
        configs: Configurations to test
        phase_steps: How many steps this phase has (for saving of model)
        agent_class: Class to create agent with
        agent_path: Checkpoint to load from (None if starting from scratch)
        env: gymnasium environment
        train_dataset: offline dataset to train on
        val_dataset: offline dataset to validate on
        eval_at_steps: list of at which steps to evaluate
        evaluate: Function to evaluate agent. Although not in type hint, has to support optional parameter `metrics`
        save_at_steps: list of steps at which to save agent
        log_interval: how often to log training metrics
        eval_episodes: how many episodes to evaluate
        log_dir: where to save logs
        seed: seed for training

    Returns:
        first entry is the best configuration with the path to its checkpoint at phase_steps, second entry is all results for all configurations
    """
    assert phase_steps in eval_at_steps and phase_steps in save_at_steps
    assert phase_steps > already_trained_steps

    results = {
        config: train(
            agent_class=agent_class,
            agent_path=agent_path,
            env=env,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            already_trained_steps=already_trained_steps,
            eval_at_steps=eval_at_steps,
            evaluate=evaluate,
            config=config,
            save_at_steps=save_at_steps,
            log_interval=log_interval,
            eval_episodes=eval_episodes,
            log_dir=log_dir,
            seed=seed,
        )
        for config, train_dataset, val_dataset in zip(
            configs, train_datasets, val_datasets
        )
    }
    # index 0 corresponds to returned metrics, then get final evaluation
    best_config = max(
        results, key=lambda config: results[config][0].get_final_result().success
    )
    return (
        best_config,
        results[best_config][1][phase_steps],
    ), PhaseResult(results)


def train(
    agent_class: Callable[[Any, gym.Env, int], Any],
    agent_path: Optional[Path],
    env: gym.Env,
    train_dataset: GCDataset,
    val_dataset: GCDataset,
    already_trained_steps: int,
    eval_at_steps: list[int],
    evaluate: Callable[
        [Any, gym.Env, int, FrozenConfigDict],
        tuple[list, dict[str, np.floating], list, list],
    ],
    config: FrozenConfigDict,
    save_at_steps: list[int] = [],
    log_interval: int = 5000,
    eval_episodes: int = 20,
    log_dir: Path = Path("./logs"),
    seed: int = 0,
) -> EvalTrajectory:
    """Train Loop for a single configuration
    This code is adapted from [ogbench](https://github.com/seohongpark/ogbench)

    Args:
        agent_class: Class to create agent with
        agent_path: Checkpoint to load from (None if starting from scratch)
        env: gymnasium environment
        train_dataset: offline dataset to train on
        val_dataset: offline dataset to validate on
        eval_at_steps: list of at which steps to evaluate
        evaluate: Function to evaluate agent. Although not in type hint, has to support optional parameter `metrics`
        save_at_steps: list of steps at which to save agent
        log_interval: how often to log training metrics
        eval_episodes: how many episodes to evaluate
        log_dir: where to save logs
        seed: seed for training

    Returns:
        list of evaluation metrics and list of corresponding agent checkpoints, corresponding to eval_at_steps and save_at_steps
    """
    assert already_trained_steps == 0 or agent_path
    # Initialize agent.
    random.seed(seed)
    np.random.seed(seed)

    example_batch = train_dataset.sample(1)
    if config["discrete"]:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch["actions"] = np.full_like(
            example_batch["actions"], env.action_space.n - 1
        )  # type: ignore

    agent = agent_class.create(
        seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
    )
    if agent_path:
        agent = restore_agent(agent, agent_path)

    metrics: ResultsPerStep[EvaluationResult] = ResultsPerStep()
    agent_paths: ResultsPerStep[Path] = ResultsPerStep()
    save_dir = log_dir
    os.makedirs(save_dir, exist_ok=True)
    train_logger = CsvLogger(save_dir / "train_log.csv")
    eval_logger = CsvLogger(save_dir / "eval_log.csv")
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(
        range(already_trained_steps + 1, max(eval_at_steps + save_at_steps) + 1),
        smoothing=0.1,
        dynamic_ncols=True,
    ):
        # Update agent.
        batch = train_dataset.sample(config["batch_size"])
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % log_interval == 0 or i == 1:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config["batch_size"])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update(
                    {f"validation/{k}": v for k, v in val_info.items()}
                )
            train_metrics["time/epoch_time"] = (time.time() - last_time) / log_interval
            train_metrics["time/total_time"] = time.time() - first_time
            last_time = time.time()
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i in eval_at_steps:
            eval_agent = agent
            eval_info, eval_metrics, trajs, renders = evaluate(
                eval_agent, env, eval_episodes, config, metrics=["success"]
            )

            if len(renders) > 0:
                pass
                # [TODO: pass to wandb]

            metrics[i] = EvaluationResult(
                success=eval_metrics["success"], metrics=eval_metrics, info=eval_info
            )
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i in save_at_steps:
            agent_paths[i] = save_dir / f"params_{i}.pkl"
            print(f"Trying to save agent to {save_dir}")
            retry_call(lambda: save_agent(agent, save_dir, i))

    train_logger.close()
    eval_logger.close()

    return EvalTrajectory((metrics, agent_paths))
