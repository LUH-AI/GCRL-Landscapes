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
from pathlib import Path
from typing import Any
from tqdm import tqdm

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


def _compute_advantages_gciql_crl(agent: Any, batch: dict) -> np.ndarray:
    """advantage[i,j] = min(Q1,Q2)(s_i,a_i,g_j) - V(s_i,g_j)."""
    N = BATCH_SIZE
    obs = jnp.array(batch["observations"])  # (N, obs_dim)
    actions = jnp.array(batch["actions"])  # (N, act_dim)
    goals = jnp.array(batch["value_goals"])  # (N, goal_dim)

    obs_rep = jnp.repeat(obs, N, axis=0)  # (N*N, obs_dim)
    act_rep = jnp.repeat(actions, N, axis=0)  # (N*N, act_dim)
    goals_rep = jnp.tile(goals, (N, 1))  # (N*N, goal_dim)

    q1, q2 = agent.network.select("critic")(obs_rep, goals_rep, act_rep)
    q = jnp.minimum(q1, q2)
    v = agent.network.select("value")(obs_rep, goals_rep)

    adv = (q - v).reshape(N, N)
    return np.array(adv)


def _compute_advantages_gcivl(agent: Any, batch: dict) -> np.ndarray:
    """advantage[i,j] = mean(V(s'_i,g_j)) - mean(V(s_i,g_j)) (ensemble mean)."""
    N = BATCH_SIZE
    obs = jnp.array(batch["observations"])
    next_obs = jnp.array(batch["next_observations"])
    goals = jnp.array(batch["value_goals"])

    obs_rep = jnp.repeat(obs, N, axis=0)
    nobs_rep = jnp.repeat(next_obs, N, axis=0)
    goals_rep = jnp.tile(goals, (N, 1))

    v1, v2 = agent.network.select("value")(obs_rep, goals_rep)
    nv1, nv2 = agent.network.select("value")(nobs_rep, goals_rep)

    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2
    adv = (nv - v).reshape(N, N)
    return np.array(adv)


def _compute_advantages_qrl(agent: Any, batch: dict) -> np.ndarray:
    """advantage[i,j] = V(s_i,g_j) - V(s'_i,g_j)  (negated quasimetric distances)."""
    N = BATCH_SIZE
    obs = jnp.array(batch["observations"])
    next_obs = jnp.array(batch["next_observations"])
    goals = jnp.array(batch["value_goals"])

    obs_rep = jnp.repeat(obs, N, axis=0)
    nobs_rep = jnp.repeat(next_obs, N, axis=0)
    goals_rep = jnp.tile(goals, (N, 1))

    v = -agent.network.select("value")(obs_rep, goals_rep)  # negate distance
    nv = -agent.network.select("value")(nobs_rep, goals_rep)

    adv = (nv - v).reshape(N, N)
    return np.array(adv)


def _compute_advantages_hiql_low(agent: Any, batch: dict) -> np.ndarray:
    """advantage[i,j] for low actor: V(s'_i,g_j) - V(s_i,g_j) (ensemble mean)."""
    N = BATCH_SIZE
    obs = jnp.array(batch["observations"])
    next_obs = jnp.array(batch["next_observations"])
    goals = jnp.array(batch["low_actor_goals"])

    obs_rep = jnp.repeat(obs, N, axis=0)
    nobs_rep = jnp.repeat(next_obs, N, axis=0)
    goals_rep = jnp.tile(goals, (N, 1))

    v1, v2 = agent.network.select("value")(obs_rep, goals_rep)
    nv1, nv2 = agent.network.select("value")(nobs_rep, goals_rep)

    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2
    adv = (nv - v).reshape(N, N)
    return np.array(adv)


def _compute_advantages_hiql_high(agent: Any, batch: dict) -> np.ndarray:
    """advantage[i,j] for high actor: V(target_i,g_j) - V(s_i,g_j) (ensemble mean)."""
    N = BATCH_SIZE
    obs = jnp.array(batch["observations"])
    targets = jnp.array(batch["high_actor_targets"])
    goals = jnp.array(batch["high_actor_goals"])

    obs_rep = jnp.repeat(obs, N, axis=0)
    targets_rep = jnp.repeat(targets, N, axis=0)
    goals_rep = jnp.tile(goals, (N, 1))

    v1, v2 = agent.network.select("value")(obs_rep, goals_rep)
    nv1, nv2 = agent.network.select("value")(targets_rep, goals_rep)

    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2
    adv = (nv - v).reshape(N, N)
    return np.array(adv)


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


def _process_row(row: pd.Series) -> pd.DataFrame:
    """Compute cross-goal advantages for a single catalog row.

    Returns empty DataFrame for non-AWR agents or on error.
    """
    agent_name = row["agent"]
    if agent_name not in _AWR_AGENTS:
        return pd.DataFrame()

    try:
        agent, val_ds, config = _load_agent(row)
    except Exception as exc:
        warnings.warn(f"Failed to load agent for {row['checkpoint_path']}: {exc}")
        return pd.DataFrame()

    if config.get("actor_loss", "awr") != "awr":
        return pd.DataFrame()

    batch = _sample_fixed_batch(val_ds, seed=0)

    records: list[dict] = []
    try:
        if agent_name in ("GCIQL", "CRL"):
            adv = _compute_advantages_gciql_crl(agent, batch)
            records.extend(_build_rows(row, "actor", adv))
        elif agent_name == "GCIVL":
            adv = _compute_advantages_gcivl(agent, batch)
            records.extend(_build_rows(row, "actor", adv))
        elif agent_name == "QRL":
            adv = _compute_advantages_qrl(agent, batch)
            records.extend(_build_rows(row, "actor", adv))
        elif agent_name == "HIQL":
            if "low_actor_goals" in batch:
                adv_low = _compute_advantages_hiql_low(agent, batch)
                records.extend(_build_rows(row, "low_actor", adv_low))
            if "high_actor_goals" in batch and "high_actor_targets" in batch:
                adv_high = _compute_advantages_hiql_high(agent, batch)
                records.extend(_build_rows(row, "high_actor", adv_high))
    except Exception as exc:
        warnings.warn(
            f"Failed advantage computation for {row['checkpoint_path']}: {exc}"
        )
        return pd.DataFrame()

    return pd.DataFrame(records)


def generate_advantages(
    catalog_df: pd.DataFrame, progress: bool = True
) -> pd.DataFrame:
    """Process all rows in *catalog_df*; return combined advantages DataFrame."""
    frames = []
    start = time.time()
    iterator = tqdm(
        catalog_df.iterrows(),
        total=len(catalog_df),
        desc="Generating advantages",
        disable=not progress,
    )
    for idx, row in iterator:
        df = _process_row(row)
        if not df.empty:
            frames.append(df)
    elapsed = time.time() - start
    if progress:
        print(f"\nProcessed {len(frames)} of {len(catalog_df)} rows in {elapsed:.1f}s")
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
    args = parser.parse_args()

    catalog_df = pd.read_csv(args.catalog)
    if "config_path" not in catalog_df.columns:
        raise ValueError(
            "catalog CSV missing 'config_path' column — re-run catalog_checkpoints.py"
        )

    result = generate_advantages(catalog_df)
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
