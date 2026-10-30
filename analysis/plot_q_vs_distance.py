#!/usr/bin/env python3
"""Plot Q-value (or value proxy) vs. goal distance for a single agent checkpoint.

Produces a single-panel hexbin figure saved as both PDF and PNG.

Usage
-----
    python -m analysis.plot_q_vs_distance \
        --checkpoint path/to/agent.pkl AGENT_NAME \
        --config-path path/to/config.json \
        --dataset cube-single-play-v0 \
        --output analysis/plots/q_vs_distance_cube_GCIQL.pdf
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import flax.serialization
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from ml_collections import FrozenConfigDict
from scipy.spatial.distance import cdist

from gcrl_landscapes.util.datasets import AGENT_CLASSES, create_env_and_dataset
from analysis.generate_advantages import _sample_fixed_batch

BATCH_SIZE = 256

# Position slice per dataset — used to compute meaningful L2 distance.
_POS_SLICES: dict[str, slice] = {
    "cube-single-play-v0": slice(19, 22),
    "antmaze-medium-navigate-v0": slice(0, 2),
    "antmaze-large-navigate-v0": slice(0, 2),
    "scene-play-v0": slice(0, 2),
}
_DIST_LABELS: dict[str, str] = {
    "cube-single-play-v0": r"$\|$state − goal$\|_2$ (scaled)",
    "antmaze-medium-navigate-v0": r"$\|$state − goal$\|_2$",
    "antmaze-large-navigate-v0": r"$\|$state − goal$\|_2$",
    "scene-play-v0": r"$\|$state − goal$\|_2$",
}


def _load_agent(checkpoint_path: Path, config_path: Path, agent_name: str, example_batch: dict) -> Any:
    """Load agent from checkpoint and config."""
    with open(checkpoint_path, "rb") as f:
        raw_ckpt = pickle.load(f)

    config = FrozenConfigDict(json.loads(config_path.read_text()))
    agent_cls = AGENT_CLASSES[agent_name]
    agent = agent_cls.create(
        0,
        example_batch["observations"][:1],
        example_batch["actions"][:1],
        config,
    )
    agent = flax.serialization.from_state_dict(agent, raw_ckpt["agent"])
    return agent


def _compute_q_matrix(agent: Any, agent_name: str, batch: dict) -> np.ndarray:
    """Compute NxN Q-value (or value-proxy) matrix for a fixed batch.

    Returns array of shape (N, N) where entry [i, j] is the value for
    (obs_i, action_i, goal_j).
    """
    N = BATCH_SIZE
    obs_rep  = jnp.repeat(jnp.array(batch["observations"]),      N, axis=0)  # (N², D)
    act_rep  = jnp.repeat(jnp.array(batch["actions"]),           N, axis=0)  # (N², A)
    nobs_rep = jnp.repeat(jnp.array(batch["next_observations"]), N, axis=0)  # (N², D)
    goal_rep = jnp.tile(jnp.array(batch["actor_goals"]),    (N, 1))           # (N², D)

    if agent_name in ("GCIQL", "CRL"):
        q1, q2 = agent.network.select("critic")(obs_rep, goal_rep, act_rep)
        q_flat = jnp.minimum(q1, q2)

    elif agent_name == "GCIVL":
        v1, v2 = agent.network.select("value")(nobs_rep, goal_rep)
        q_flat = (v1 + v2) / 2.0

    elif agent_name == "QRL":
        q_flat = -agent.network.select("value")(obs_rep, goal_rep)

    else:
        raise ValueError(f"Unsupported agent: {agent_name}")

    return np.array(q_flat).reshape(N, N)


def _compute_distance_matrix(batch: dict, pos_slice: slice) -> np.ndarray:
    """Compute NxN L2-distance matrix between agent/object positions in obs and goals."""
    pos_obs  = np.array(batch["observations"])[:, pos_slice]
    pos_goal = np.array(batch["actor_goals"])[:, pos_slice]
    return cdist(pos_obs, pos_goal)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Q-value vs. goal distance for a single agent checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        nargs=2,
        required=True,
        metavar=("PATH", "AGENT_NAME"),
        help="Checkpoint path and agent name, e.g. --checkpoint path/to/agent.pkl GCIQL",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        metavar="CONFIG",
        help="Config JSON path for the checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cube-single-play-v0",
        help="OGBench dataset name (default: cube-single-play-v0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/plots/q_vs_distance.pdf"),
        help="Output figure path (PDF). A PNG is also saved at the same path.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint[0])
    agent_name = args.checkpoint[1]
    config_path = args.config_path
    dataset_name = args.dataset

    pos_slice = _POS_SLICES.get(dataset_name, slice(0, 2))
    dist_label = _DIST_LABELS.get(dataset_name, r"$\|$obs − goal$\|_2$")

    config = FrozenConfigDict(json.loads(config_path.read_text()))
    print(f"Loading dataset {dataset_name!r} ...")
    _, _, val_ds = create_env_and_dataset(dataset_name, agent_name, config)
    batch = _sample_fixed_batch(val_ds, seed=0)
    print(f"Batch shapes: obs={np.array(batch['observations']).shape}, "
          f"goals={np.array(batch['actor_goals']).shape}")

    dist_mat = _compute_distance_matrix(batch, pos_slice)
    print(f"Distance matrix: min={dist_mat.min():.3f}, max={dist_mat.max():.3f}")

    print(f"Loading {agent_name} from {ckpt_path} ...")
    agent = _load_agent(ckpt_path, config_path, agent_name, batch)
    q_mat = _compute_q_matrix(agent, agent_name, batch)
    print(f"  Q matrix: min={q_mat.min():.3f}, max={q_mat.max():.3f}")

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    distances = dist_mat.ravel()
    q_vals = q_mat.ravel()

    hb = ax.hexbin(distances, q_vals, gridsize=60, cmap="viridis", mincnt=1)
    counts = np.asarray(hb.get_array())
    density = counts / counts.sum()
    hb.set_array(density)
    hb.set_clim(density.min(), density.max())
    fig.colorbar(hb, ax=ax, label="density")

    n_bins = 60
    bin_edges = np.linspace(distances.min(), distances.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = np.array([
        q_vals[(distances >= bin_edges[k]) & (distances < bin_edges[k + 1])].mean()
        if np.any((distances >= bin_edges[k]) & (distances < bin_edges[k + 1]))
        else np.nan
        for k in range(n_bins)
    ])
    valid = ~np.isnan(bin_means)
    ax.plot(bin_centers[valid], bin_means[valid], color="tomato", lw=1.8, label="bin mean")

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(dist_label, fontsize=9)
    y_label = {"GCIQL": "Q-value", "CRL": "Q-value", "GCIVL": "V-value", "QRL": "Distance-value"}.get(agent_name, "Value")
    ax.set_ylabel(y_label, fontsize=9)
    ax.legend(fontsize=8)

    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    png_path = args.output.with_suffix(".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output} and {png_path}")


if __name__ == "__main__":
    main()
