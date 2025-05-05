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
    restore_agent,
    ResultsPerStep,
)
from .util.misc import retry_call
import os


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
        # there is an optional argument "metrics" here, not easily type hintable
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
