#!/usr/bin/env python3
"""Plot Q-value (or value proxy) vs. goal distance for multiple agents.

For each supplied checkpoint, loads the agent, computes Q-values on a fixed 256-sample
validation batch, and plots Q vs. L2 distance between the current agent/object position
and the goal position in a 2×2 figure.

Usage
-----
    python -m analysis.plot_q_vs_distance \
        --checkpoints path/to/gciql.pkl GCIQL path/to/crl.pkl CRL \
                      path/to/gcivl.pkl GCIVL path/to/qrl.pkl QRL \
        --config-paths gciql_cfg.json crl_cfg.json gcivl_cfg.json qrl_cfg.json \
        --dataset cube-single-play-v0 \
        --output analysis/q_vs_distance_cube.pdf
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
# Cube XYZ starts at index 19 in the 28D cube-single observation:
#   [0:6]   joint_pos, [6:12] joint_vel, [12:19] effector state
#   [19:22] cube_pos (scaled)
# Antmaze observation = concat(qpos, qvel); qpos[:2] = agent XY.
_POS_SLICES: dict[str, slice] = {
    "cube-single-play-v0": slice(19, 22),
    "antmaze-medium-navigate-v0": slice(0, 2),
    "antmaze-large-navigate-v0": slice(0, 2),
    "scene-play-v0": slice(0, 2),  # placeholder; update if needed
}
_DIST_LABELS: dict[str, str] = {
    "cube-single-play-v0": r"$\|$cube$_\mathrm{obs}$ − cube$_\mathrm{goal}\|_2$ (scaled)",
    "antmaze-medium-navigate-v0": r"$\|$agent$_\mathrm{obs}$ − agent$_\mathrm{goal}\|_2$ (XY)",
    "antmaze-large-navigate-v0": r"$\|$agent$_\mathrm{obs}$ − agent$_\mathrm{goal}\|_2$ (XY)",
    "scene-play-v0": r"$\|$obs − goal$\|_2$",
}
_TITLES: dict[str, str] = {
    "cube-single-play-v0": "Q-function behaviour: value vs. goal distance (cube-single)",
    "antmaze-medium-navigate-v0": "Q-function behaviour: value vs. goal distance (antmaze-medium)",
    "antmaze-large-navigate-v0": "Q-function behaviour: value vs. goal distance (antmaze-large)",
    "scene-play-v0": "Q-function behaviour: value vs. goal distance (scene)",
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
        # Ensemble of two value heads; use next_obs as in advantage computation
        v1, v2 = agent.network.select("value")(nobs_rep, goal_rep)
        q_flat = (v1 + v2) / 2.0

    elif agent_name == "QRL":
        # Negated quasimetric distance: larger = closer to goal
        q_flat = -agent.network.select("value")(obs_rep, goal_rep)

    else:
        raise ValueError(f"Unsupported agent: {agent_name}")

    return np.array(q_flat).reshape(N, N)


def _compute_distance_matrix(batch: dict, pos_slice: slice) -> np.ndarray:
    """Compute NxN L2-distance matrix between agent/object positions in obs and goals."""
    pos_obs  = np.array(batch["observations"])[:, pos_slice]   # (N, k)
    pos_goal = np.array(batch["actor_goals"])[:, pos_slice]    # (N, k)
    return cdist(pos_obs, pos_goal)  # (N, N)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Q-value vs. goal distance for multiple agents."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        metavar="PATH_OR_AGENT",
        help="Alternating checkpoint paths and agent names: path AGENT path AGENT ...",
    )
    parser.add_argument(
        "--config-paths",
        nargs="+",
        required=True,
        metavar="CONFIG",
        help="Config JSON paths, one per checkpoint (in same order).",
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
        default=Path("analysis/q_vs_distance.pdf"),
        help="Output figure path (PDF/PNG/SVG).",
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    pos_slice = _POS_SLICES.get(dataset_name, slice(0, 2))
    dist_label = _DIST_LABELS.get(dataset_name, r"$\|$obs − goal$\|_2$")
    fig_title = _TITLES.get(dataset_name, f"Q-function behaviour: value vs. goal distance ({dataset_name})")

    # Parse --checkpoints: alternating path, agent_name pairs
    if len(args.checkpoints) % 2 != 0:
        parser.error("--checkpoints must be pairs of <path> <AGENT_NAME>")
    ckpt_pairs = [
        (Path(args.checkpoints[i]), args.checkpoints[i + 1])
        for i in range(0, len(args.checkpoints), 2)
    ]
    config_paths = [Path(p) for p in args.config_paths]
    if len(config_paths) != len(ckpt_pairs):
        parser.error("Number of --config-paths must match number of checkpoint pairs.")

    # Load dataset once
    first_agent_name = ckpt_pairs[0][1]
    first_config = FrozenConfigDict(json.loads(config_paths[0].read_text()))
    print(f"Loading dataset {dataset_name!r} ...")
    _, _, val_ds = create_env_and_dataset(dataset_name, first_agent_name, first_config)
    batch = _sample_fixed_batch(val_ds, seed=0)
    print(f"Batch shapes: obs={np.array(batch['observations']).shape}, "
          f"goals={np.array(batch['actor_goals']).shape}")

    dist_mat = _compute_distance_matrix(batch, pos_slice)
    print(f"Distance matrix: min={dist_mat.min():.3f}, max={dist_mat.max():.3f}")

    # Compute Q matrices for all checkpoints
    results: dict[str, np.ndarray] = {}
    for (ckpt_path, agent_name), cfg_path in zip(ckpt_pairs, config_paths):
        print(f"Loading {agent_name} from {ckpt_path} ...")
        agent = _load_agent(ckpt_path, cfg_path, agent_name, batch)
        q_mat = _compute_q_matrix(agent, agent_name, batch)
        key = agent_name
        idx = 1
        while key in results:
            idx += 1
            key = f"{agent_name}_{idx}"
        results[key] = q_mat
        print(f"  Q matrix: min={q_mat.min():.3f}, max={q_mat.max():.3f}")

    n_agents = len(results)
    ncols = 2
    nrows = (n_agents + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 4 * nrows), sharex=True, sharey=False)
    axes_flat = np.array(axes).ravel()

    for ax, (agent_name, q_mat) in zip(axes_flat, results.items()):
        distances = dist_mat.ravel()
        q_vals = q_mat.ravel()

        hb = ax.hexbin(distances, q_vals, gridsize=60, cmap="viridis", mincnt=1)
        counts = hb.get_array()
        density = counts / counts.sum()
        hb.set_array(density)
        hb.set_clim(density.min(), density.max())
        fig.colorbar(hb, ax=ax, label="density")

        # Binned-mean trend line
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
        ax.set_title(agent_name, fontsize=12)
        ax.set_xlabel(dist_label, fontsize=9)
        y_label = "Q-value" if agent_name.split("_")[0] in ("GCIQL", "CRL") else "Value proxy"
        ax.set_ylabel(y_label, fontsize=9)
        ax.legend(fontsize=8)

    for ax in axes_flat[n_agents:]:
        ax.set_visible(False)

    fig.suptitle(fig_title, fontsize=13)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
