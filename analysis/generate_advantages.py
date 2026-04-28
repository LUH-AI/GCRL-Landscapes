#!/usr/bin/env python3
"""Generate cross-goal advantage values from trained checkpoints.

For each checkpoint in the catalog, loads the agent and a fixed validation batch,
then computes advantages for every (obs_i, action_i, goal_j) combination (NxN matrix).

Usage
-----
    python analysis/generate_advantages.py --catalog checkpoints.csv --output advantages.parquet
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from ml_collections import FrozenConfigDict

from gcrl_landscapes.util.data import restore_agent
from gcrl_landscapes.util.datasets import AGENT_CLASSES, create_env_and_dataset

BATCH_SIZE = 256

# Agents that use AWR actor loss and for which we can compute advantages
_AWR_AGENTS = {"GCIQL", "CRL", "GCIVL", "QRL", "HIQL"}


def _sample_fixed_batch(dataset: Any, seed: int = 0) -> dict:
    """Sample a deterministic batch from *dataset* using a fixed RNG seed."""
    rng = np.random.default_rng(seed=seed)
    idxs = rng.integers(low=0, high=dataset.size, size=BATCH_SIZE)
    return dataset.sample(BATCH_SIZE, idxs=idxs, seed=seed)


def _load_agent(row: pd.Series) -> tuple:
    """Load agent, env, and datasets from a catalog row."""
    config = FrozenConfigDict(json.loads(Path(row["config_path"]).read_text()))
    env, train_ds, val_ds = create_env_and_dataset(row["dataset"], row["agent"], config)
    agent_cls = AGENT_CLASSES[row["agent"]]
    example = train_ds.sample(1)
    if config.get("discrete", False):
        example["actions"] = np.full_like(example["actions"], env.action_space.n - 1)
    agent = agent_cls.create(0, example["observations"], example["actions"], config)
    agent = restore_agent(agent, Path(row["checkpoint_path"]))
    return agent, val_ds, config


def _param_shape_key(params: Any) -> str:
    """Hashable string key encoding the PyTree structure and leaf shapes."""
    return str(jax.tree.map(lambda x: x.shape, params))


def _batch_compute_gciql_crl(
    ref_agent: Any,
    batched_params: Any,
    obs_rep: jnp.ndarray,
    act_rep: jnp.ndarray,
    goals_rep: jnp.ndarray,
) -> np.ndarray:
    """Returns (B, N, N) advantages for a batch of B param sets."""
    def single(params: Any) -> jnp.ndarray:
        q1, q2 = ref_agent.network.select("critic")(obs_rep, goals_rep, act_rep, params=params)
        q = jnp.minimum(q1, q2)
        v = ref_agent.network.select("value")(obs_rep, goals_rep, params=params)
        return (q - v).reshape(BATCH_SIZE, BATCH_SIZE)

    return np.array(jax.jit(jax.vmap(single))(batched_params))


def _batch_compute_gcivl(
    ref_agent: Any,
    batched_params: Any,
    obs_rep: jnp.ndarray,
    nobs_rep: jnp.ndarray,
    goals_rep: jnp.ndarray,
) -> np.ndarray:
    """Returns (B, N, N) advantages for a batch of B param sets."""
    def single(params: Any) -> jnp.ndarray:
        v1, v2 = ref_agent.network.select("value")(obs_rep, goals_rep, params=params)
        nv1, nv2 = ref_agent.network.select("value")(nobs_rep, goals_rep, params=params)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        return (nv - v).reshape(BATCH_SIZE, BATCH_SIZE)

    return np.array(jax.jit(jax.vmap(single))(batched_params))


def _batch_compute_qrl(
    ref_agent: Any,
    batched_params: Any,
    obs_rep: jnp.ndarray,
    nobs_rep: jnp.ndarray,
    goals_rep: jnp.ndarray,
) -> np.ndarray:
    """Returns (B, N, N) advantages for a batch of B param sets."""
    def single(params: Any) -> jnp.ndarray:
        v = -ref_agent.network.select("value")(obs_rep, goals_rep, params=params)
        nv = -ref_agent.network.select("value")(nobs_rep, goals_rep, params=params)
        return (nv - v).reshape(BATCH_SIZE, BATCH_SIZE)

    return np.array(jax.jit(jax.vmap(single))(batched_params))


def _batch_compute_hiql_low(
    ref_agent: Any,
    batched_params: Any,
    obs_rep: jnp.ndarray,
    nobs_rep: jnp.ndarray,
    goals_rep: jnp.ndarray,
) -> np.ndarray:
    """Returns (B, N, N) low-actor advantages for a batch of B param sets."""
    def single(params: Any) -> jnp.ndarray:
        v1, v2 = ref_agent.network.select("value")(obs_rep, goals_rep, params=params)
        nv1, nv2 = ref_agent.network.select("value")(nobs_rep, goals_rep, params=params)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        return (nv - v).reshape(BATCH_SIZE, BATCH_SIZE)

    return np.array(jax.jit(jax.vmap(single))(batched_params))


def _batch_compute_hiql_high(
    ref_agent: Any,
    batched_params: Any,
    obs_rep: jnp.ndarray,
    targets_rep: jnp.ndarray,
    goals_rep: jnp.ndarray,
) -> np.ndarray:
    """Returns (B, N, N) high-actor advantages for a batch of B param sets."""
    def single(params: Any) -> jnp.ndarray:
        v1, v2 = ref_agent.network.select("value")(obs_rep, goals_rep, params=params)
        nv1, nv2 = ref_agent.network.select("value")(targets_rep, goals_rep, params=params)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        return (nv - v).reshape(BATCH_SIZE, BATCH_SIZE)

    return np.array(jax.jit(jax.vmap(single))(batched_params))


def _build_rows(
    row: pd.Series,
    actor_name: str,
    adv_matrix: np.ndarray,
) -> list[dict]:
    """Flatten NxN advantage matrix into per-(obs_idx, goal_idx) rows."""
    N = BATCH_SIZE
    obs_idxs = np.repeat(np.arange(N), N)
    goal_idxs = np.tile(np.arange(N), N)
    is_positive = obs_idxs == goal_idxs
    advantages = adv_matrix.ravel()

    records = []
    for i in range(N * N):
        records.append(
            {
                "checkpoint_path": row["checkpoint_path"],
                "agent": row["agent"],
                "dataset": row["dataset"],
                "phase": row["phase"],
                "seed": row["seed"],
                "configuration": row["configuration"],
                "actor": actor_name,
                "obs_idx": int(obs_idxs[i]),
                "goal_idx": int(goal_idxs[i]),
                "is_positive": bool(is_positive[i]),
                "advantage": float(advantages[i]),
            }
        )
    return records


def generate_advantages(
    catalog_df: pd.DataFrame,
    chunk_size: int = 32,
    progress: bool = True,
) -> pd.DataFrame:
    """Process all rows in *catalog_df*; return combined advantages DataFrame.

    Checkpoints are grouped by (agent_name, dataset, param_shape_key) and processed
    in batches using jax.vmap over the checkpoint dimension, amortizing JIT overhead.
    """
    start = time.time()

    # --- Phase 1: load all agents sequentially ---
    loaded: list[tuple] = []
    load_iter = tqdm(
        catalog_df.iterrows(),
        total=len(catalog_df),
        desc="Loading checkpoints",
        disable=not progress,
    )
    for _, row in load_iter:
        if row["agent"] not in _AWR_AGENTS:
            continue
        try:
            agent, val_ds, config = _load_agent(row)
        except Exception as exc:
            warnings.warn(f"Failed to load agent for {row['checkpoint_path']}: {exc}")
            continue
        if config.get("actor_loss", "awr") != "awr":
            continue
        shape_key = _param_shape_key(agent.network.params)
        loaded.append((row, agent, val_ds, shape_key))

    if progress:
        print(f"Loaded {len(loaded)} checkpoints in {time.time() - start:.1f}s")

    # --- Phase 2: group by (agent_name, dataset, param_shape_key) ---
    groups: dict[tuple, list] = defaultdict(list)
    for row, agent, val_ds, shape_key in loaded:
        key = (row["agent"], row["dataset"], shape_key)
        groups[key].append((row, agent, val_ds))

    # --- Phase 3: process each group in chunks with vmap ---
    frames: list[pd.DataFrame] = []
    group_iter = tqdm(
        groups.items(),
        total=len(groups),
        desc="Computing advantages",
        disable=not progress,
    )
    for (agent_name, dataset, _shape_key), group_items in group_iter:
        ref_agent = group_items[0][1]
        val_ds = group_items[0][2]
        batch = _sample_fixed_batch(val_ds, seed=0)

        # Pre-compute shared tiled inputs (identical for all checkpoints in group)
        N = BATCH_SIZE
        obs_rep = jnp.repeat(jnp.array(batch["observations"]), N, axis=0)
        goals_rep = jnp.tile(jnp.array(batch["value_goals"]), (N, 1))

        if agent_name in ("GCIQL", "CRL"):
            act_rep = jnp.repeat(jnp.array(batch["actions"]), N, axis=0)
        elif agent_name in ("GCIVL", "QRL", "HIQL"):
            nobs_rep = jnp.repeat(jnp.array(batch["next_observations"]), N, axis=0)

        if agent_name == "HIQL":
            has_low = "low_actor_goals" in batch
            has_high = "high_actor_goals" in batch and "high_actor_targets" in batch
            if has_low:
                low_goals_rep = jnp.tile(jnp.array(batch["low_actor_goals"]), (N, 1))
            if has_high:
                targets_rep = jnp.repeat(jnp.array(batch["high_actor_targets"]), N, axis=0)
                high_goals_rep = jnp.tile(jnp.array(batch["high_actor_goals"]), (N, 1))

        for chunk_start in range(0, len(group_items), chunk_size):
            chunk = group_items[chunk_start : chunk_start + chunk_size]
            rows_chunk = [c[0] for c in chunk]
            batched_params = jax.tree.map(
                lambda *xs: np.stack(xs),
                *[c[1].network.params for c in chunk],
            )

            try:
                if agent_name in ("GCIQL", "CRL"):
                    adv_batch = _batch_compute_gciql_crl(
                        ref_agent, batched_params, obs_rep, act_rep, goals_rep
                    )
                    for i, row in enumerate(rows_chunk):
                        frames.append(pd.DataFrame(_build_rows(row, "actor", adv_batch[i])))

                elif agent_name == "GCIVL":
                    adv_batch = _batch_compute_gcivl(
                        ref_agent, batched_params, obs_rep, nobs_rep, goals_rep
                    )
                    for i, row in enumerate(rows_chunk):
                        frames.append(pd.DataFrame(_build_rows(row, "actor", adv_batch[i])))

                elif agent_name == "QRL":
                    adv_batch = _batch_compute_qrl(
                        ref_agent, batched_params, obs_rep, nobs_rep, goals_rep
                    )
                    for i, row in enumerate(rows_chunk):
                        frames.append(pd.DataFrame(_build_rows(row, "actor", adv_batch[i])))

                elif agent_name == "HIQL":
                    if has_low:
                        adv_low_batch = _batch_compute_hiql_low(
                            ref_agent, batched_params, obs_rep, nobs_rep, low_goals_rep
                        )
                        for i, row in enumerate(rows_chunk):
                            frames.append(
                                pd.DataFrame(_build_rows(row, "low_actor", adv_low_batch[i]))
                            )
                    if has_high:
                        adv_high_batch = _batch_compute_hiql_high(
                            ref_agent, batched_params, obs_rep, targets_rep, high_goals_rep
                        )
                        for i, row in enumerate(rows_chunk):
                            frames.append(
                                pd.DataFrame(_build_rows(row, "high_actor", adv_high_batch[i]))
                            )

            except Exception as exc:
                for row in rows_chunk:
                    warnings.warn(
                        f"Failed advantage computation for {row['checkpoint_path']}: {exc}"
                    )

    elapsed = time.time() - start
    if progress:
        print(f"\nProcessed {len(frames)} checkpoint results in {elapsed:.1f}s total")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_advantages(df: pd.DataFrame, output_path: Path) -> None:
    """Write *df* to parquet at *output_path*."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cross-goal advantage values from trained checkpoints."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="CSV file produced by catalog_checkpoints.py (must include config_path column).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parquet file path.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help=(
            "Checkpoints per vmap batch. Memory scales as B × N² × hidden_dim × layers × 4 bytes"
            " in activations. Default 32 is safe for 20 GB VRAM; try 64 if not OOM."
        ),
    )
    args = parser.parse_args()

    catalog_df = pd.read_csv(args.catalog)
    if "config_path" not in catalog_df.columns:
        raise ValueError(
            "catalog CSV missing 'config_path' column — re-run catalog_checkpoints.py"
        )

    result = generate_advantages(catalog_df, chunk_size=args.chunk_size)
    if result.empty:
        print("No advantages generated.")
    else:
        print(
            f"Generated {len(result):,} rows ({result['checkpoint_path'].nunique()} checkpoints)"
        )
        save_advantages(result, args.output)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
