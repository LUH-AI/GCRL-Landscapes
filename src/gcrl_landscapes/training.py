import tqdm
import numpy as np
import jax
import optax
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
from .util.eval import gradient_cosine_similarity, gradient_magnitude_similarity
import os
from functools import partial

CONST_VAL_BATCH_SIZE = 256

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
    if val_dataset is not None:
        held_out_val_batch = val_dataset.sample(CONST_VAL_BATCH_SIZE)

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
        if i % log_interval == 0 or i == 1 or i == max(eval_at_steps + save_at_steps):
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

            train_metrics.update(get_metrics(agent, held_out_val_batch))
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

@jax.jit
def batched_tree_to_batched_vector(tree):
    flattened_matrices = jax.tree.map(lambda leaf: leaf.reshape((leaf.shape[0], -1)), tree)
    return jax.numpy.concatenate(jax.tree.flatten(flattened_matrices)[0], axis=1)


@partial(jax.jit, static_argnames=("fun", "batch_size", "symmetric"))
def pairwise_seq_map(fun, batch_size, symmetric, x, y):
    """This maps a function pairwise (batch dimension) over two given trees, which we here flatten to use as vectors.
    Map sequentially to save on memory.

    Args:
        fun (Callable): function to map over the two vectors
        batch_size (): size of given batch in trees. Already encoded by the trees, but needed to properly compile
        symmetric (): Is fun symmetric?
        x (): tree one
        y (): tree two

    Returns:
        vector of pairwise similarities containing some duplicates. Use `remove_duplicates` to remove 
    """

    # We limit the amount of shifts to calculate here due to duplicates. After calling this function we'll filter out the last few remainining duplicate items left over due to vectorization
    cosine_similarities = jax.lax.map(lambda batch_shift: optax.losses.cosine_similarity(batched_tree_to_batched_vector(x), batched_tree_to_batched_vector(jax.tree.map(lambda tree: jax.numpy.roll(tree, batch_shift, axis=0), y))), jax.numpy.arange(1, 1 + (batch_size // 2 + 1) if symmetric else batch_size), batch_size=1)
    return jax.numpy.concatenate(jax.tree.flatten(cosine_similarities)[0])


def remove_duplicates(pairwise_similarities, batch_size, symmetric):
    """Remove duplicates generated by pairwise_seq_map

    Args:
        pairwise_similarities (): similarity as generated by pairwise_seq_map
        batch_size (): batch_size used in pairwise_seq_map
        symmetric (): was fun symmetric in pairwise_seq_map

    Returns:
        pairwise_similarities without duplicates
    """
    return pairwise_similarities.flatten()[:(batch_size ** 2 - batch_size) // 2] if symmetric else pairwise_similarities


@jax.jit
def get_grads(agent, batch):
    value_grads = jax.vmap(lambda sample: jax.grad(lambda grad_params: agent.value_loss(sample, grad_params), has_aux=True)(agent.network.params)[0]["modules_value"], in_axes=0, out_axes=0)(batch)
    actor_grads = jax.vmap(lambda sample: jax.grad(lambda grad_params: agent.actor_loss(sample, grad_params), has_aux=True)(agent.network.params)[0]["modules_actor"], in_axes=0, out_axes=0)(batch)
    return value_grads, actor_grads


@jax.jit
def calc_cossim(grads):
    return remove_duplicates(pairwise_seq_map(gradient_cosine_similarity, CONST_VAL_BATCH_SIZE, True, grads, grads), CONST_VAL_BATCH_SIZE, True)


@jax.jit
def calc_magsim(grads):
    return remove_duplicates(pairwise_seq_map(gradient_magnitude_similarity, CONST_VAL_BATCH_SIZE, True, grads, grads), CONST_VAL_BATCH_SIZE, True)

@jax.jit
def calc_scale(grads):
    return jax.numpy.linalg.norm(batched_tree_to_batched_vector(grads), axis=1)


def get_metrics(agent, batch):
    value_grads, actor_grads = get_grads(agent, batch)
    value_cosine_similarities, actor_cosine_similarities = calc_cossim(value_grads), calc_cossim(actor_grads)
    value_magnitude_similarity, actor_magnitude_similarity = calc_magsim(value_grads), calc_magsim(actor_grads)
    value_scale, actor_scale = calc_scale(value_grads), calc_scale(actor_grads)

    # We use trajectories as the notion of a task for pairwise metrics. If a sample is from the same trajectory -> filter it
    same_task = remove_duplicates(np.concatenate([batch["trajectory_final_state_idx"] == np.roll(batch["trajectory_final_state_idx"], shift) for shift in np.arange(1, 1 + (CONST_VAL_BATCH_SIZE // 2 + 1))]), CONST_VAL_BATCH_SIZE, True)
    return {
        # Cosine similarities
        "grad/value_cosine_similarity_mean": jax.numpy.mean(value_cosine_similarities[~same_task]),
        "grad/actor_cosine_similarity_mean": jax.numpy.mean(actor_cosine_similarities[~same_task]),
        "grad/value_cosine_similarity_std": jax.numpy.std(value_cosine_similarities[~same_task]),
        "grad/actor_cosine_similarity_std": jax.numpy.std(actor_cosine_similarities[~same_task]),
        # Magnitude similarities
        "grad/value_magnitude_similarity_mean": jax.numpy.mean(value_magnitude_similarity[~same_task]),
        "grad/actor_magnitude_similarity_mean": jax.numpy.mean(actor_magnitude_similarity[~same_task]),
        "grad/value_magnitude_similarity_std": jax.numpy.std(value_magnitude_similarity[~same_task]),
        "grad/actor_magnitude_similarity_std": jax.numpy.std(actor_magnitude_similarity[~same_task]),
        # Gradient scale
        "grad/value_scale_mean": jax.numpy.mean(value_scale),
        "grad/actor_scale_mean": jax.numpy.mean(actor_scale),
        "grad/value_scale_std": jax.numpy.std(value_scale),
        "grad/actor_scale_std": jax.numpy.std(actor_scale),
    }
