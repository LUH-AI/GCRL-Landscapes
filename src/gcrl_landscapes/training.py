import tqdm
import numpy as np
from ogbench.impls.utils.datasets import GCDataset
from ogbench.impls.utils.log_utils import CsvLogger, setup_wandb, get_wandb_video
from ogbench.impls.utils.flax_utils import save_agent
import gymnasium as gym
import random
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from ml_collections import ConfigDict
from typing import Callable
import os


def train(agent_class, env: gym.Env, train_dataset: GCDataset, val_dataset: GCDataset, train_steps: int, eval_interval: int, evaluate: Callable, config: ConfigDict, save_at_steps: list[int] = [], log_interval: int = 5000, eval_episodes: int = 20, log_dir: Path = Path("./logs"), seed: int = 0):
    """ Train Loop for a single configuration for n train_steps
    This code is adapted from [ogbench](https://github.com/seohongpark/ogbench)

    Args:
        agent_class: ogbench agent class. Does not have a proper type hint but inherits from flax.struct.PyTreeNode
        train_dataset: The dataset to train with (offline RL)
        train_steps: for how many steps to train
        config: The configuration to train with. Should match the agent
    """
    # Initialize agent.
    random.seed(seed)
    np.random.seed(seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)  # type: ignore

    # [TODO: implement restoring trained agents when starting next phase]
    agent = agent_class.create(
        seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    save_dir = log_dir / datetime.now().strftime("%Y-%m-%dT%H:%M")
    os.makedirs(save_dir)
    train_logger = CsvLogger(save_dir / "train_log.csv")
    eval_logger = CsvLogger(save_dir / "eval_log.csv")
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % log_interval == 0 or i == 1:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % eval_interval == 0:
            eval_agent = agent
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos  # type: ignore
            num_tasks = len(task_infos)
            eval_info, eval_metrics, trajs, renders = evaluate(
                eval_agent,
                env,
                eval_episodes,
            )

            if len(renders) > 0:
                pass
                # [TODO: pass to wandb]

            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i in save_at_steps:
            save_agent(agent, save_dir, i)

    train_logger.close()
    eval_logger.close()
