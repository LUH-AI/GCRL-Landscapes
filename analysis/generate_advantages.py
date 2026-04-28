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
import os
import pickle
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from tqdm import tqdm

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from ml_collections import FrozenConfigDict

from gcrl_landscapes.util.datasets import AGENT_CLASSES, create_env_and_dataset

BATCH_SIZE = 256

# Agents that use AWR actor loss and for which we can compute advantages
_AWR_AGENTS = {"GCIQL", "CRL", "GCIVL", "QRL", "HIQL"}


def _sample_fixed_batch(dataset: Any, seed: int = 0) -> dict:
    """Sample a deterministic batch from *dataset* using a fixed RNG seed."""
    rng = np.random.default_rng(seed=seed)
    idxs = rng.integers(low=0, high=dataset.size, size=BATCH_SIZE)
    return dataset.sample(BATCH_SIZE, idxs=idxs, seed=seed)


def _load_raw_checkpoint(checkpoint_path: Path) -> dict:
    """Load raw pickle dict from disk — I/O only, thread-safe."""
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def _construct_agent(row: pd.Series, raw_ckpt: dict, dataset_cache: dict) -> tuple:
    """Construct agent from cached dataset + raw checkpoint dict. Main thread only."""
    config = FrozenConfigDict(json.loads(Path(row["config_path"]).read_text()))
    key = (row["dataset"], row["agent"])
    if key not in dataset_cache:
        env, train_ds, val_ds = create_env_and_dataset(row["dataset"], row["agent"], config)
        dataset_cache[key] = (env, train_ds, val_ds)
    env, train_ds, val_ds = dataset_cache[key]

    agent_cls = AGENT_CLASSES[row["agent"]]
    example = train_ds.sample(1)
    if config.get("discrete", False):
        example["actions"] = np.full_like(example["actions"], env.action_space.n - 1)
    agent = agent_cls.create(0, example["observations"], example["actions"], config)
    agent = flax.serialization.from_state_dict(agent, raw_ckpt["agent"])
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


def _process_chunk_frames(
    agent_name: str,
    chunk: list[tuple],  # [(row, agent, val_ds), ...]
    frames: list[pd.DataFrame],
) -> None:
    """Compute advantages for one chunk and append DataFrames to *frames*.

    The caller should clear *chunk* after this returns to release agent params from VRAM.
    """
    ref_agent = chunk[0][1]
    val_ds = chunk[0][2]
    batch = _sample_fixed_batch(val_ds, seed=0)
    rows_chunk = [c[0] for c in chunk]

    N = BATCH_SIZE
    obs_rep = jnp.repeat(jnp.array(batch["observations"]), N, axis=0)
    goals_rep = jnp.tile(jnp.array(batch["value_goals"]), (N, 1))

    batched_params = jax.tree.map(
        lambda *xs: np.stack(xs),
        *[c[1].network.params for c in chunk],
    )

    try:
        if agent_name in ("GCIQL", "CRL"):
            act_rep = jnp.repeat(jnp.array(batch["actions"]), N, axis=0)
            adv_batch = _batch_compute_gciql_crl(
                ref_agent, batched_params, obs_rep, act_rep, goals_rep
            )
            for i, row in enumerate(rows_chunk):
                frames.append(pd.DataFrame(_build_rows(row, "actor", adv_batch[i])))

        elif agent_name == "GCIVL":
            nobs_rep = jnp.repeat(jnp.array(batch["next_observations"]), N, axis=0)
            adv_batch = _batch_compute_gcivl(
                ref_agent, batched_params, obs_rep, nobs_rep, goals_rep
            )
            for i, row in enumerate(rows_chunk):
                frames.append(pd.DataFrame(_build_rows(row, "actor", adv_batch[i])))

        elif agent_name == "QRL":
            nobs_rep = jnp.repeat(jnp.array(batch["next_observations"]), N, axis=0)
            adv_batch = _batch_compute_qrl(
                ref_agent, batched_params, obs_rep, nobs_rep, goals_rep
            )
            for i, row in enumerate(rows_chunk):
                frames.append(pd.DataFrame(_build_rows(row, "actor", adv_batch[i])))

        elif agent_name == "HIQL":
            nobs_rep = jnp.repeat(jnp.array(batch["next_observations"]), N, axis=0)
            has_low = "low_actor_goals" in batch
            has_high = "high_actor_goals" in batch and "high_actor_targets" in batch
            if has_low:
                low_goals_rep = jnp.tile(jnp.array(batch["low_actor_goals"]), (N, 1))
                adv_low_batch = _batch_compute_hiql_low(
                    ref_agent, batched_params, obs_rep, nobs_rep, low_goals_rep
                )
                for i, row in enumerate(rows_chunk):
                    frames.append(pd.DataFrame(_build_rows(row, "low_actor", adv_low_batch[i])))
            if has_high:
                targets_rep = jnp.repeat(jnp.array(batch["high_actor_targets"]), N, axis=0)
                high_goals_rep = jnp.tile(jnp.array(batch["high_actor_goals"]), (N, 1))
                adv_high_batch = _batch_compute_hiql_high(
                    ref_agent, batched_params, obs_rep, targets_rep, high_goals_rep
                )
                for i, row in enumerate(rows_chunk):
                    frames.append(
                        pd.DataFrame(_build_rows(row, "high_actor", adv_high_batch[i]))
                    )

    except Exception as exc:
        for row in rows_chunk:
            warnings.warn(f"Failed advantage computation for {row['checkpoint_path']}: {exc}")


def generate_advantages(
    catalog_df: pd.DataFrame,
    chunk_size: int = 32,
    num_workers: int = 8,
    progress: bool = True,
) -> pd.DataFrame:
    """Process all rows in *catalog_df*; return combined advantages DataFrame.

    Checkpoints are pre-grouped by (agent_name, dataset) from the DataFrame (no I/O),
    then processed in chunks of *chunk_size*. Per chunk: parallel I/O → construct →
    vmap → free. Peak CPU RAM and VRAM scale with chunk_size, not total checkpoints.
    """
    start = time.time()

    eligible = [
        (idx, row)
        for idx, row in catalog_df.iterrows()
        if row["agent"] in _AWR_AGENTS
    ]

    # Pre-group by (agent_name, dataset) — O(N) scan, zero I/O
    by_agent_dataset: dict[tuple, list] = defaultdict(list)
    for idx, row in eligible:
        by_agent_dataset[(row["agent"], row["dataset"])].append((idx, row))

    dataset_cache: dict = {}
    frames: list[pd.DataFrame] = []
    n_constructed = 0

    outer_iter = tqdm(
        by_agent_dataset.items(),
        total=len(by_agent_dataset),
        desc="Groups",
        disable=not progress,
    )
    for (agent_name, dataset_name), group_rows in outer_iter:
        ref_shape: str | None = None

        for chunk_start in range(0, len(group_rows), chunk_size):
            chunk_rows = group_rows[chunk_start : chunk_start + chunk_size]

            # Parallel I/O — only chunk_size files in flight at once
            raw_ckpts: dict[int, dict] = {}
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_idx = {
                    executor.submit(_load_raw_checkpoint, Path(str(row["checkpoint_path"]))): idx
                    for idx, row in chunk_rows
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        raw_ckpts[idx] = future.result()
                    except Exception as exc:
                        warnings.warn(f"Failed to read checkpoint for row {idx}: {exc}")

            # Sequential construction (JAX, main thread); free raw pkl immediately
            chunk: list[tuple] = []
            for idx, row in chunk_rows:
                if idx not in raw_ckpts:
                    continue
                try:
                    agent, val_ds, config = _construct_agent(row, raw_ckpts[idx], dataset_cache)
                    del raw_ckpts[idx]
                except Exception as exc:
                    warnings.warn(
                        f"Failed to construct agent for {row['checkpoint_path']}: {exc}"
                    )
                    continue
                if config.get("actor_loss", "awr") != "awr":
                    continue
                chunk.append((row, agent, val_ds))
                n_constructed += 1

            if not chunk:
                continue

            # Shape check: vmap requires identical param shapes within a chunk.
            # Same (agent, dataset) group almost always shares one shape; warn + skip outliers.
            if ref_shape is None:
                ref_shape = _param_shape_key(chunk[0][1].network.params)

            good = [c for c in chunk if _param_shape_key(c[1].network.params) == ref_shape]
            skipped = len(chunk) - len(good)
            if skipped:
                warnings.warn(
                    f"{skipped} checkpoint(s) in {agent_name}/{dataset_name} have mismatched "
                    f"param shapes; skipping."
                )

            if good:
                _process_chunk_frames(agent_name, good, frames)
            # good goes out of scope here → agent params freed before next chunk

    elapsed = time.time() - start
    if progress:
        print(f"\nProcessed {n_constructed} checkpoints in {elapsed:.1f}s total")
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, (os.cpu_count() or 4)),
        help="Worker threads for parallel checkpoint I/O. Does not affect VRAM.",
    )
    args = parser.parse_args()

    catalog_df = pd.read_csv(args.catalog)
    if "config_path" not in catalog_df.columns:
        raise ValueError(
            "catalog CSV missing 'config_path' column — re-run catalog_checkpoints.py"
        )

    result = generate_advantages(catalog_df, chunk_size=args.chunk_size, num_workers=args.num_workers)
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
