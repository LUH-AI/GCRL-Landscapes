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
from .util.misc import retry_call, get_feature_embedding, crl_total_loss_individual
from .util.eval import gradient_cosine_similarity, gradient_magnitude_similarity
import os
from functools import partial

CONST_VAL_BATCH_SIZE = 256
GRAD_CHUNK_SIZE = 32  # chunk size for chunked_vmap in per-sample grad/update computation

# Top-level params subtrees that belong to the policy. Everything else — value,
# critic, their target copies, and CRL's contrastive encoder pair — is "the
# learned signal" and is what `freeze_value` holds constant.
ACTOR_MODULE_PREFIXES = ("modules_actor", "modules_low_actor", "modules_high_actor")


def snapshot_frozen_params(agent) -> dict:
    """Copy every non-actor top-level params subtree of *agent*.

    Taken right after the checkpoint restore, this is the value signal that
    `freeze_value` pins for the whole run.
    """
    return {
        key: value
        for key, value in agent.network.params.items()
        if not key.startswith(ACTOR_MODULE_PREFIXES)
    }


def restore_frozen_params(agent, frozen: dict):
    """Write the snapshotted non-actor subtrees back over an updated agent.

    Called after every ``agent.update()`` so that only the actor evolves. The
    optimizer state of the frozen modules keeps accumulating, which is
    harmless: their parameters are overwritten before they are ever read again.
    """
    if not frozen:
        return agent
    from flax.core import FrozenDict

    params = agent.network.params
    # FrozenDict.copy(add_or_replace) vs plain-dict merge — the two container
    # types take different call signatures, so dispatch explicitly.
    merged = (
        params.copy(frozen) if isinstance(params, FrozenDict) else {**params, **frozen}
    )
    return agent.replace(network=agent.network.replace(params=merged))

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
    log_count: int = 10,
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
        # Get the same batch in all cases: fixed indices and fixed goal relabeling seed.
        held_out_val_batch = val_dataset.sample(
            CONST_VAL_BATCH_SIZE,
            idxs=np.random.default_rng(seed=0).integers(
                low=0, high=val_dataset.size, size=CONST_VAL_BATCH_SIZE
            ),
            seed=0,
        )

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
        dict(config),
    )
    if agent_path:
        agent = restore_agent(agent, agent_path)

    # `freeze_value` isolates extraction from value formation: the learned
    # signal is held at its restored value while actor-side hyperparameters
    # vary. Only meaningful with a checkpoint — freezing a randomly
    # initialised critic would pin noise.
    freeze_value = bool(config.get("freeze_value", False))
    if freeze_value and not agent_path:
        raise ValueError(
            "freeze_value requires a checkpoint to freeze; submit with --agent_path"
        )
    frozen_params = snapshot_frozen_params(agent) if freeze_value else {}

    metrics: ResultsPerStep[EvaluationResult] = ResultsPerStep()
    agent_paths: ResultsPerStep[Path] = ResultsPerStep()
    save_dir = log_dir
    os.makedirs(save_dir, exist_ok=True)
    train_logger = CsvLogger(save_dir / "train_log.csv", separator=";")
    eval_logger = CsvLogger(save_dir / "eval_log.csv")
    first_time = time.time()
    last_time = time.time()
    log_steps = np.linspace(already_trained_steps + 1, max(eval_at_steps + save_at_steps), log_count, dtype=int).tolist()
    for i in tqdm.tqdm(
        range(already_trained_steps + 1, max(eval_at_steps + save_at_steps) + 1),
        smoothing=0.1,
        dynamic_ncols=True,
    ):
        # Update agent.
        batch = train_dataset.sample(config["batch_size"])
        agent, update_info = agent.update(batch)
        if freeze_value:
            agent = restore_frozen_params(agent, frozen_params)

        # Log metrics.
        if i in log_steps:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config["batch_size"])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update(
                    {f"validation/{k}": v for k, v in val_info.items()}
                )
            train_metrics["time/epoch_time"] = (time.time() - last_time) / (log_steps[1] - log_steps[0])
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
    pairwise_fun_results = jax.lax.map(lambda batch_shift: fun(batched_tree_to_batched_vector(x), batched_tree_to_batched_vector(jax.tree.map(lambda tree: jax.numpy.roll(tree, batch_shift, axis=0), y))), jax.numpy.arange(1, 1 + (batch_size // 2 + 1) if symmetric else batch_size), batch_size=1)
    return jax.numpy.concatenate(jax.tree.flatten(pairwise_fun_results)[0])


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


def chunked_vmap(f, xs, chunk_size):
    """Like jax.vmap but processes xs in sequential chunks to reduce peak memory.

    chunk_size must divide the leading axis size of xs.
    """
    batch_size = jax.tree.leaves(xs)[0].shape[0]
    n_chunks = batch_size // chunk_size
    chunks = jax.tree.map(lambda x: x.reshape(n_chunks, chunk_size, *x.shape[1:]), xs)
    results = jax.lax.map(lambda chunk: jax.vmap(f)(chunk), chunks)
    return jax.tree.map(lambda x: x.reshape(batch_size, *x.shape[2:]), results)


@jax.jit
def get_grads_standard(agent, batch):
    """Get gradients for QRL/CRL with single actor."""
    if agent.config.get('agent_name', '').lower() == 'crl':
        return chunked_vmap(
            lambda index: jax.grad(lambda grad_params: crl_total_loss_individual(agent, batch, index, grad_params), has_aux=True)(agent.network.params)[0],
            jax.numpy.arange(jax.tree_util.tree_leaves(batch)[0].shape[0]),
            GRAD_CHUNK_SIZE,
        )
    return chunked_vmap(
        lambda sample: jax.grad(lambda grad_params: agent.total_loss(sample, grad_params), has_aux=True)(agent.network.params)[0],
        batch,
        GRAD_CHUNK_SIZE,
    )


@jax.jit
def hiql_combine(grads):
    # Flatten both gradient trees and concatenate
    modules_actor = { "modules_low_actor": grads["modules_low_actor"], "modules_high_actor": grads["modules_high_actor"] }

    return {"modules_actor": modules_actor, **{k: v for k, v in grads.items() 
            if k not in ("modules_low_actor", "modules_high_actor")}}


@jax.jit
def get_grads(agent, batch):
    """Get gradients for value and actor networks, handling different agent types."""
    agent_name = agent.config.get('agent_name', '').lower()

    if agent_name == 'hiql':
        return hiql_combine(get_grads_standard(agent, batch))
    else:
        return get_grads_standard(agent, batch)


@jax.jit
def get_updates_standard(agent, batch):
    if agent.config.get('agent_name', '').lower() == 'crl':
        return chunked_vmap(
            lambda index: agent.network.tx.update(jax.grad(lambda grad_params: crl_total_loss_individual(agent, batch, index, grad_params), has_aux=True)(agent.network.params)[0], agent.network.opt_state, agent.network.params)[0],
            jax.numpy.arange(jax.tree_util.tree_leaves(batch)[0].shape[0]),
            GRAD_CHUNK_SIZE,
        )
    else:
        return chunked_vmap(
            lambda sample: agent.network.tx.update(jax.grad(lambda grad_params: agent.total_loss(sample, grad_params), has_aux=True)(agent.network.params)[0], agent.network.opt_state, agent.network.params)[0],
            batch,
            GRAD_CHUNK_SIZE,
        )


@jax.jit
def get_updates(agent, batch):
    agent_name = agent.config.get('agent_name', '').lower()

    if agent_name == 'hiql':
        return hiql_combine(get_updates_standard(agent, batch))
    else:
        return get_updates_standard(agent, batch)



@jax.jit
def calc_cossim(grads):
    return remove_duplicates(pairwise_seq_map(optax.losses.cosine_similarity, CONST_VAL_BATCH_SIZE, True, grads, grads), CONST_VAL_BATCH_SIZE, True)


@jax.jit
def calc_magsim(grads):
    return remove_duplicates(pairwise_seq_map(gradient_magnitude_similarity, CONST_VAL_BATCH_SIZE, True, grads, grads), CONST_VAL_BATCH_SIZE, True)


@jax.jit
def calc_scale(grads):
    return jax.numpy.linalg.norm(batched_tree_to_batched_vector(grads), axis=1)


@jax.jit
def calc_feature_embedding_rank(agent, batch):
    """Calculate the rank of state and goal feature embedding covariance matrices.

    Args:
        agent: The agent with value network
        batch: Batch of data

    Returns:
        state_rank: The matrix rank of the state feature embedding covariance matrix
        goal_rank: The matrix rank of the goal feature embedding covariance matrix
    """
    phi = get_feature_embedding(agent, batch)

    def get_embedding_rank(phi):
        # Calculate covariance matrices using JAX's built-in function
        # rowvar=False because features are in columns (batch_size x latent_dim)
        cov_matrix = jax.numpy.cov(phi, rowvar=False)

        # Calculate matrix ranks using SVD
        embedding_rank = jax.numpy.linalg.matrix_rank(cov_matrix)

        return embedding_rank

    if (len(phi.shape) > 2 and phi.shape[0] == 2):
        # Ensemble learning -> two embeddings
        return (get_embedding_rank(phi[0]) + get_embedding_rank(phi[1])) / 2

    return get_embedding_rank(phi)


def _qv_advantage(agent, batch):
    """adv = min(Q1, Q2) - V  — used by GCIQL and CRL-AWR."""
    v = agent.network.select('value')(batch['observations'], batch['actor_goals'])
    q1, q2 = agent.network.select('critic')(batch['observations'], batch['actor_goals'], batch['actions'])
    q = jax.numpy.minimum(q1, q2)
    return q - v, v, q


def _ensemble_td_advantage(agent, obs, next_obs, goals):
    """adv = mean(nV1,nV2) - mean(V1,V2)  — used by GCIVL and HIQL."""
    v1, v2 = agent.network.select('value')(obs, goals)
    nv1, nv2 = agent.network.select('value')(next_obs, goals)
    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2
    return nv - v, v, nv


def _mqe_advantage(agent, batch):
    """adv = q - v  — used by MQE.

    Mirrors MQEAgent.actor_loss's AWR branch (ogbench/impls/agents/mqe.py):
    v = min over ensemble of -distance(psi(s), psi(g)),
    q = min over ensemble of -distance(phi(s,a), psi(g)).
    """
    psi_s = agent.network.select('psi')(batch['observations'])
    psi_g = agent.network.select('psi')(batch['actor_goals'])
    phi = agent.network.select('phi')(batch['observations'], batch['actions'])
    v1, v2 = -agent.distance(psi_s, psi_g)
    v = jax.numpy.minimum(v1, v2)
    q1, q2 = -agent.distance(phi, psi_g)
    q = jax.numpy.minimum(q1, q2)
    return q - v, v, q


def get_advantage_metrics(agent, batch) -> dict:
    """Compute per-sample advantage values on a fixed batch for logging."""
    if agent.config.get('actor_loss') != 'awr':
        return {}

    agent_name = agent.config.get('agent_name', '').lower()
    result = {}

    if agent_name in ('gciql', 'crl'):
        adv, v, q = _qv_advantage(agent, batch)
        result['advantage/actor'] = adv.tolist()
        result['value/actor_v'] = v.tolist()
        result['value/actor_q'] = q.tolist()

    elif agent_name == 'gcivl':
        adv, v, nv = _ensemble_td_advantage(
            agent, batch['observations'], batch['next_observations'], batch['actor_goals']
        )
        result['advantage/actor'] = adv.tolist()
        result['value/actor_v'] = v.tolist()
        result['value/actor_nv'] = nv.tolist()

    elif agent_name == 'qrl':
        v = -agent.network.select('value')(batch['observations'], batch['actor_goals'])
        nv = -agent.network.select('value')(batch['next_observations'], batch['actor_goals'])
        result['advantage/actor'] = (nv - v).tolist()
        result['value/actor_v'] = v.tolist()
        result['value/actor_nv'] = nv.tolist()

    elif agent_name == 'hiql':
        adv_low, v_low, nv_low = _ensemble_td_advantage(
            agent, batch['observations'], batch['next_observations'], batch['low_actor_goals']
        )
        result['advantage/low_actor'] = adv_low.tolist()
        result['value/low_actor_v'] = v_low.tolist()
        result['value/low_actor_nv'] = nv_low.tolist()

        adv_high, v_high, nv_high = _ensemble_td_advantage(
            agent, batch['observations'], batch['high_actor_targets'], batch['high_actor_goals']
        )
        result['advantage/high_actor'] = adv_high.tolist()
        result['value/high_actor_v'] = v_high.tolist()
        result['value/high_actor_nv'] = nv_high.tolist()

    elif agent_name == 'mqe':
        adv, v, q = _mqe_advantage(agent, batch)
        result['advantage/actor'] = adv.tolist()
        result['value/actor_v'] = v.tolist()
        result['value/actor_q'] = q.tolist()

    # GCBC / SAC / CMD / CRL-ddpgbc: no advantage → empty dict
    return result


def get_metrics(agent, batch):
    agent_name = agent.config.get('agent_name', '').lower()

    target_quantiles = np.arange(100) / 100

    def get_quantile_dict(values, value_name):
        return {f"{value_name}_quant{q}": jax.numpy.quantile(values, q) for q in target_quantiles}

    # We use trajectories as the notion of a task for pairwise metrics. If a sample is from the same trajectory -> filter it
    same_task = remove_duplicates(np.concatenate([batch["trajectory_final_state_idx"] == np.roll(batch["trajectory_final_state_idx"], shift) for shift in np.arange(1, 1 + (CONST_VAL_BATCH_SIZE // 2 + 1))]), CONST_VAL_BATCH_SIZE, True)
    filtered = ~same_task

    def grad_group_metrics(prefix, grads):
        cossim = calc_cossim(grads)
        magsim = calc_magsim(grads)
        scale = calc_scale(grads)
        return {
            f"grad/{prefix}_cosine_similarity_mean": jax.numpy.mean(cossim[filtered]),
            f"grad/{prefix}_cosine_similarity_std": jax.numpy.std(cossim[filtered]),
            **get_quantile_dict(cossim[filtered], f"grad/{prefix}_cosine_similarity"),
            f"grad/{prefix}_cosine_similarity_cvar0.25": jax.numpy.mean(cossim[filtered][cossim[filtered] < jax.numpy.quantile(cossim[filtered], 0.25)]),
            f"grad/{prefix}_magnitude_similarity_mean": jax.numpy.mean(magsim[filtered]),
            f"grad/{prefix}_magnitude_similarity_std": jax.numpy.std(magsim[filtered]),
            **get_quantile_dict(magsim[filtered], f"grad/{prefix}_magnitude_similarity"),
            f"grad/{prefix}_scale_mean": jax.numpy.mean(scale),
            f"grad/{prefix}_scale_std": jax.numpy.std(scale),
        }

    def update_group_metrics(prefix, updates):
        u_cossim = calc_cossim(updates)
        u_magsim = calc_magsim(updates)
        u_scale = calc_scale(updates)
        return {
            f"update/{prefix}_cosine_similarity_mean": jax.numpy.mean(u_cossim[filtered]),
            f"update/{prefix}_cosine_similarity_std": jax.numpy.std(u_cossim[filtered]),
            **get_quantile_dict(u_cossim[filtered], f"update/{prefix}_cosine_similarity"),
            f"update/{prefix}_magnitude_similarity_mean": jax.numpy.mean(u_magsim[filtered]),
            f"update/{prefix}_magnitude_similarity_std": jax.numpy.std(u_magsim[filtered]),
            **get_quantile_dict(u_magsim[filtered], f"update/{prefix}_magnitude_similarity"),
            f"update/{prefix}_scale_mean": jax.numpy.mean(u_scale),
            f"update/{prefix}_scale_std": jax.numpy.std(u_scale),
        }

    MODULE_GROUPS = [
        ("modules_value", "value"),
        ("modules_actor", "actor"),
        ("modules_critic", "critic"),
    ]

    result: dict = {}

    # --- Phase 1 & 2: gradient and update metrics ---
    # MQE's loss is batch-level (samples a stochastic mask over the full batch), so
    # per-sample gradient computation via chunked_vmap is incompatible. Skip for MQE.
    if agent_name != 'mqe':
        # Compute raw_grads, extract all grad metrics as scalars, then free the device buffer
        # before allocating the equally-sized update array (both ~256 MB; holding both = OOM).
        raw_grads = get_grads_standard(agent, batch)
        total_grads = hiql_combine(raw_grads) if agent_name == 'hiql' else raw_grads
        available = set(total_grads.keys())

        for module_key, name in MODULE_GROUPS:
            if module_key in available:
                result.update(grad_group_metrics(name, total_grads[module_key]))
        if "modules_critic" in available and "modules_value" in available:
            result.update(grad_group_metrics("critic_value", {"critic": total_grads["modules_critic"], "value": total_grads["modules_value"]}))

        del raw_grads, total_grads  # reclaim ~256 MB before allocating updates

        # get_updates_standard re-derives per-sample grads internally (32-sample chunks),
        # so no full grad array needs to live alongside the update array.
        raw_updates = get_updates_standard(agent, batch)
        total_updates = hiql_combine(raw_updates) if agent_name == 'hiql' else raw_updates

        for module_key, name in MODULE_GROUPS:
            if module_key in available:
                result.update(update_group_metrics(name, total_updates[module_key]))
        if "modules_critic" in available and "modules_value" in available:
            result.update(update_group_metrics("critic_value", {"critic": total_updates["modules_critic"], "value": total_updates["modules_value"]}))

        del raw_updates, total_updates

    # --- Other metrics ---
    agent_modules = agent.network.model_def.modules.keys()
    embedding_rank = calc_feature_embedding_rank(agent, batch) if "value" in agent_modules else None

    if "target_value" in agent_modules:
        held_out_val_batch_values = agent.network.select("target_value")(batch["observations"], batch["value_goals"], params=agent.network.params)
        if len(held_out_val_batch_values.shape) == 3 and held_out_val_batch_values.shape[0] == 2:
            held_out_val_batch_values = held_out_val_batch_values.mean(axis=0).reshape(-1)
    else:
        held_out_val_batch_values = None

    if embedding_rank is not None:
        result["feature/embedding_rank"] = embedding_rank
    if held_out_val_batch_values is not None:
        result["target/held_out_val_batch_values"] = held_out_val_batch_values.tolist()

    result.update(get_advantage_metrics(agent, batch))

    return result
