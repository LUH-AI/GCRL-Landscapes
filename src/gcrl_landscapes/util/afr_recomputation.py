"""AFR (Future-Random Advantage Separation) recomputation from trained checkpoints.

Computes AFR metrics by loading trained model weights, reconstructing the OG-Bench
dataset, and computing per-sample advantages for both active goals and cross-
trajectory goals on-the-fly using the agent's forward pass methods.

Usage:
    from gcrl_landscapes.util.afr_recomputation import run_afr_for_phase
    df = run_afr_for_phase(phase_dir, alpha=0.5, n_batches=5)
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import flax.serialization
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ml_collections import FrozenConfigDict

from gcrl_landscapes.util.datasets import DATASET_CLASSES


class AfrRecomputationError(RuntimeError):
    """Raised when AFR recomputation fails."""


class AgentWrapper:
    """Wraps a trained Flax agent with JAX-compatible advantage function dispatch."""

    def __init__(self, agent: Any, config: FrozenConfigDict) -> None:
        """
        Args:
            agent: A trained Flax agent with network.select() method
            config: Training configuration dict
        """
        self.agent = agent
        self.config = config

    @property
    def params(self) -> Any:
        """Access the agent's network parameters."""
        return self.agent.network.params

    # ===========================================================================
    # Public API
    # ===========================================================================

    def get_advantage_for_goals(
        self,
        batch: Dict[str, Any],
        goals: Any,
    ) -> np.ndarray:
        """Compute A(s_i, a_i, goal_j) for each row i and all goal rows j.

        Args:
            batch: Dict with keys 'observations', 'actions', 'next_observations'
            goals: Goals array of shape (K, D_goal) or (B, K, D_goal)

        Returns:
            Array of shape (B,) per-sample advantages, averaged over K goals if
            goals has shape (B, K, D_goal).
        """
        g = jnp.asarray(goals)
        agent_name = str(self.config.get("agent_name", "crl")).lower()

        obs = jnp.asarray(batch["observations"])
        act = jnp.asarray(batch["actions"])
        next_obs = jnp.asarray(batch.get("next_observations", obs))

        if agent_name == "crl":
            # A(s, a, g) = critic(s, a, g) - value(s, g)
            q1, q2 = self.agent.network.select("critic")(
                obs, act, g, params=self.params
            )
            q = jnp.minimum(q1, q2)
            v = self.agent.network.select("value")(obs, g, params=self.params)
            result = q - v

        elif agent_name in ("gciql",):
            # A(s, a, g) = min(Q1, Q2)(s, a, g) - V(s, g)
            q1, q2 = self.agent.network.select("critic")(
                obs, act, g, params=self.params
            )
            q = jnp.minimum(q1, q2)
            v = self.agent.network.select("value")(obs, g, params=self.params)
            result = q - v

        elif agent_name in ("gcivl", "hiql"):
            # A(s, a, g) = mean(nV1, nV2)(s_next, g) - mean(V1, V2)(s, g)
            v = self.agent.network.select("value")(obs, g, params=self.params)
            nv = self.agent.network.select("value")(next_obs, g, params=self.params)
            result = nv - v

        elif agent_name in ("qrl",):
            # A(s, a, g) = d(s, g) - d(s_next, g) (quasimetric distance reduction)
            d = self.agent.network.select("qmetric")(obs, g, params=self.params)
            d_next = self.agent.network.select("qmetric")(
                next_obs, g, params=self.params
            )
            result = d - d_next

        else:
            raise AfrRecomputationError(f"Unsupported agent for AFR: {agent_name}")

        result = result.astype(jnp.float32)
        return np.asarray(result)

    def get_cross_trajs_gaps(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute advantage gaps between active goals and cross-trajectory goals.

        For each row i with active goal g_i, computes:
        - A(s_i, a_i, g_i) — advantage for own goal (active)
        - A(s_i, a_i, g_j) for all j where traj_j != traj_i (cross-trajectory)
        - gap_{i,j} = A(s_i, a_i, g_j) - A(s_i, a_i, g_i)

        Uses the same same-task / same-traj filtering logic as `pairwise_seq_map`
        and `remove_duplicates` in training.py.

        Args:
            batch: Dict from dataset.sample() with observations, actions, actor_goals,
                   and trajectory membership info.

        Returns:
            active_adv: [B] advantages for active goals
            gap_rows: [M] sparse row indices of valid pairs
            gap_cols: [M] sparse column indices of valid pairs
            gap_values: [M] advantage(row_i, goal_j) - advantage(row_i, goal_i)
            trajectory_idx: [B] trajectory membership for reference
        """
        observations = np.asarray(batch["observations"], dtype=np.float32)
        actions = np.asarray(batch["actions"], dtype=np.float32)
        actor_goals = np.asarray(
            batch.get("actor_goals", batch.get("goals", observations)),
            dtype=np.float32,
        )
        next_observations = np.asarray(
            batch.get("next_observations", observations), dtype=np.float32
        )
        trajectory_idx = np.asarray(
            batch.get(
                "trajectory_idx",
                batch.get(
                    "trajectory_final_state_idx", np.arange(observations.shape[0])
                ),
            )
        )

        batch_size = observations.shape[0]

        # Active advantages (row_i vs own goal)
        active_batch = {
            "observations": observations,
            "actions": actions,
            "next_observations": next_observations,
        }
        active_adv = self.get_advantage_for_goals(
            active_batch,
            actor_goals,  # shape (B, D_goal)
        )

        # --- Build trajectory mask: True where different trajectory ---
        traj_arr = jnp.asarray(trajectory_idx)
        # traj_mask[i, j] == True means trajectory_idx[i] != trajectory_idx[j]
        traj_mask = traj_arr[:, None] != traj_arr[None, :]  # (B, B)

        # --- Vmap-over-chunk approach: memory-efficient cross-trajectory ---
        obs_flat = jnp.reshape(observations, (batch_size, -1))
        act_flat = jnp.reshape(actions, (batch_size, -1))
        next_obs_flat = jnp.reshape(next_observations, (batch_size, -1))
        goals_flat = jnp.reshape(actor_goals, (batch_size, -1))

        flat_batch = {
            "observations": obs_flat,
            "actions": act_flat,
            "next_observations": next_obs_flat,
        }

        # active_adv_flat is [B] — advantage for own goal
        active_adv_flat = active_adv

        gap_rows_list: List[np.ndarray] = []  # type: ignore[var-annotated]
        gap_cols_list: List[np.ndarray] = []  # type: ignore[var-annotated]
        gap_values_list: List[np.ndarray] = []  # type: ignore[var-annotated]

        # Process goals in chunks to keep memory bounded
        for j_start in range(0, batch_size, 64):
            j_end = min(j_start + 64, batch_size)
            chunk_goals = goals_flat[j_start:j_end]  # (chunk, D_flat)

            # [chunk, B] — advantage(row_i vs goals chunk) for all (i, j) in chunk
            chunk_adv = self.get_advantage_for_goals(flat_batch, chunk_goals)

            # Keep only cross-trajectory (different-traj) entries
            traj_chunk = (
                trajectory_idx[j_start:j_end][:, None] != trajectory_idx[None, :]
            )
            r_rows, i_cols = jnp.where(traj_chunk)

            if len(r_rows) == 0:
                continue

            # gap = A(cross_goal) - A(own_goal)
            gap_vals = chunk_adv[r_rows, i_cols] - active_adv_flat[i_cols]

            gap_rows_list.append(i_cols)
            gap_cols_list.append(j_start + r_rows)
            gap_values_list.append(gap_vals.astype(np.float32))

        gap_rows = (
            np.concatenate(gap_rows_list)
            if gap_rows_list
            else np.array([], dtype=np.int64)
        )
        gap_cols = (
            np.concatenate(gap_cols_list)
            if gap_cols_list
            else np.array([], dtype=np.int64)
        )
        gap_values = (
            np.concatenate(gap_values_list)
            if gap_values_list
            else np.array([], dtype=np.float32)
        )

        return active_adv, gap_rows, gap_cols, gap_values, trajectory_idx


# ===========================================================================
# Low-level: save/restore agents
# ===========================================================================


def _load_agent_from_checkpoint(
    checkpoint_path: Path,
    agent_class: Any,
    seed: int,
    config: FrozenConfigDict,
) -> AgentWrapper:
    """Load a trained Flax agent from a pickle checkpoint file.

    Args:
        checkpoint_path: Path to params_*.pkl
        agent_class: The agent class (must have a .create() method)
        seed: Random seed
        config: Training configuration dict

    Returns:
        AgentWrapper ready for AFR computation
    """
    with open(checkpoint_path, "rb") as f:
        load_dict = pickle.load(f)

    # Create fresh agent with example data
    # Use the agent config to determine observation/action dims
    obs_dim = int(config.get("obs_dim", 29))  # antmaze default
    act_dim = int(config.get("act_dim", 1))

    random_obs = np.zeros((1, obs_dim), dtype=np.float32)
    random_act = np.zeros((1, act_dim), dtype=np.float32)

    agent = agent_class.create(seed, random_obs, random_act, config)

    # Restore trained params from state dict
    agent_network = load_dict.get("agent", load_dict)
    if isinstance(agent_network, dict):
        agent.network.params = flax.serialization.from_state_dict(
            agent.network.params, agent_network
        )
    else:
        agent.network.params = flax.serialization.from_state_dict(
            agent.network.params, agent_network
        )

    return AgentWrapper(agent, config)


def _load_checkpoint_params(checkpoint_path: Path) -> Dict[str, Any]:
    """Load raw params and config from a pickle checkpoint."""
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def _extract_agent_class(agent_name: str) -> Any:
    """Import and return the agent class for the given agent name."""
    from gcrl_landscapes.util.datasets import AGENT_CLASSES

    if agent_name not in AGENT_CLASSES:
        raise AfrRecomputationError(
            f"Unknown agent name: {agent_name}. Available: {list(AGENT_CLASSES.keys())}"
        )
    return AGENT_CLASSES[agent_name]


# ===========================================================================
# Dataset reconstruction
# ===========================================================================


def _load_dataset_from_config(
    agent_name: str,
    config: FrozenConfigDict,
) -> Any:
    """Reconstruct the OG-Bench dataset for training.

    Args:
        agent_name: Agent name (e.g., 'CRL', 'GCIQL')
        config: Full training configuration dict

    Returns:
        A GCRLLandscapes dataset object with .sample() method
    """
    # The dataset name is in the config
    dataset_name = config.get(
        "dataset_name", config.get("dataset", "antmaze-medium-navigate-v0")
    )

    try:
        import gymnasium as gym
        from ogbench.impls.utils.datasets import (
            create_env_and_datasets as make_env_and_datasets,
        )
    except ImportError:
        raise AfrRecomputationError(
            "OG-Bench is not installed. Install it with: pip install og-bench"
        )

    # Recreate the dataset using OG-Bench's loader
    env, train_dataset, val_dataset = make_env_and_datasets(dataset_name)
    train_dataset = DATASET_CLASSES[agent_name.upper()](train_dataset, config)

    return train_dataset


def _detect_checkpoint_format(checkpoint_path: Path) -> Tuple[str, FrozenConfigDict]:
    """Detect which agent/class format the checkpoint uses.

    Returns:
        Tuple of (agent_name, config)
    """
    with open(checkpoint_path, "rb") as f:
        load_dict = pickle.load(f)

    # Try to detect agent name from the loaded params
    # Common keys: 'agent' dict, or direct params
    top_keys = list(load_dict.keys()) if isinstance(load_dict, dict) else []

    # Check if there's a 'config' or 'agent_config' key
    for key in ["agent_config", "config", "configuration"]:
        if key in load_dict:
            cfg = load_dict[key]
            if isinstance(cfg, dict):
                if "agent_name" in cfg:
                    return cfg["agent_name"].lower(), FrozenConfigDict(cfg)
                # Inferred from structure
                return "crl", FrozenConfigDict(cfg)

    # If agent_name not found, try to infer from module structure in params
    agent = load_dict.get("agent", load_dict)
    if hasattr(agent, "network"):
        network = agent.network
        if hasattr(network, "model_def"):
            model_def = network.model_def
            modules = list(model_def.modules.keys())
            # CRL uses "critic" + "value" modules
            # QRL uses "qmetric" module
            if "qmetric" in modules:
                return "qrl", FrozenConfigDict({})
            return "crl", FrozenConfigDict({})
        if hasattr(network, "select"):
            # Has select method, infer from available modules
            return "crl", FrozenConfigDict({})

    raise AfrRecomputationError(
        f"Cannot detect agent format from checkpoint {checkpoint_path}. "
        "The checkpoint should contain 'agent_config' with 'agent_name'."
    )


# ===========================================================================
# Configuration discovery
# ===========================================================================


def _find_config_for_phase(
    phase_dir: Path,
) -> FrozenConfigDict:
    """Find the configuration JSON for a given phase directory.

    Phase directory structure:
        phase_dir/seed_X/params_*.pkl
        phase_dir/.../configuration_X.json (in parent run_logs dir)

    Returns:
        FrozenConfigDict with full training configuration
    """
    # Configuration is stored in run_logs/configuration_*.json
    parent = phase_dir
    for _ in range(5):  # traverse up to 5 levels
        config_files = sorted(parent.glob("configuration_*.json"))
        if config_files:
            config_path = config_files[0]
            with open(config_path, "r") as f:
                config = json.loads(f.read())
            return FrozenConfigDict(config)
        parent = parent.parent
        if parent is None:
            break

    raise FileNotFoundError(
        f"No configuration file found traversing up from {phase_dir}. "
        "Expected configuration_0.json in run_logs/ or parent dirs."
    )


# ===========================================================================
# AFR Metrics computation (mirrors afr_signal_ogbench_diagnostic.md)
# ===========================================================================


def compute_afr_metrics_from_gaps(
    gap_values: np.ndarray,
    active_adv: np.ndarray,
    gap_rows: np.ndarray,
    gap_cols: np.ndarray,
    trajectory_idx: np.ndarray,
    alpha: Optional[float] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute AFR metrics from gap arrays.

    Args:
        gap_values: advantage(row_i, goal_j) - advantage(row_i, goal_i) for valid (i,j)
        active_adv: advantage(row_i, own_goal_i), shape [B]
        gap_rows: sparse row indices of valid pairs
        gap_cols: sparse column indices of valid pairs
        trajectory_idx: [B] trajectory membership
        alpha: AWR temperature (optional)
        eps: numerical stabilizer

    Returns:
        Dict with FR-AUC, EI, log_weight_ratio, etc.
    """
    gap_flat = jnp.asarray(gap_values, dtype=jnp.float64)

    # FR-AUC: Pr(A+ > A-). Since gap = A(g_j) - A(g_i), gap > 0 means g_j has higher advantage
    # For AFR: A+ = future_goal advantage, A- = random_goal advantage
    # Here, when gap > 0, the cross-trajectory goal (g_j) has higher advantage than
    # the own-goal (g_i), which means the opposite of what we expect.
    # Wait — we need to be careful. The AFR paper expects:
    # A+ = advantage for GOAL FROM SAME TRAJECTORY (reachable in dataset future)
    # A- = advantage for GOAL FROM DIFFERENT TRAJECTORY (random)
    # gap = A(g_same_traj) - A(g_diff_traj)
    # Our computed gap = A(g_cross_traj) - A(g_own_goal)
    # For AFR-AUC we want Pr(A(g_future) > A(g_random))
    # The cross-trajectory goals ARE the random ones (g-)
    # The own-goal (or same-trajectory goals) ARE the future ones (g+)
    # So fr_auc = Pr(A(g_own) < A(g_cross)) = Pr(gap < 0) for "future > random"
    # BUT — this depends on how the advantage is defined!
    #
    # Actually, looking at the AFR spec: for CRL, advantage is a reachability-lift.
    # Higher advantage = more reachable. So for CRL, A+ (same-traj, more reachable) > A- (random).
    # Our gap = A(cross) - A(active). So gap < 0 means active (same-traj) > cross → A+ > A-
    # fr_auc = Pr(gap < 0) = 1 - Pr(gap > 0)
    #
    # For the general case (algorithm-agnostic):
    # We define fr_auc = Pr(A(g_future_goals) > A(g_random_goals))
    # In our sparse representation:
    # - The "active goal" for each row comes from the SAME trajectory (it's the forward trajectory)
    # - The "cross-trajectory goals" are the random ones
    # - So we want: active_adv (g+) vs cross_goals (g-)
    # - fr_auc = Pr(active_adv > cross_adv) = Pr(gap < 0) where gap = cross - active
    #
    # BUT — the standard AFR metric definition (from the spec) is:
    # fr_auc = Pr(A+ > A-) where A+ is advantage for SAME-TRAJ goals, A- is advantage for CROSS
    # Our gap = A(cross) - A(active), so:
    # A+ > A- ⟺ A(active) > A(cross) ⟺ gap < 0

    # Compute FR-AUC
    fr_auc = float(jnp.mean(gap_flat < 0.0))  # gap < 0 → A(active) > A(cross) → A+ > A-
    gap_mean = float(jnp.mean(gap_flat))
    gap_std = float(jnp.std(gap_flat) + eps)
    ei = float(gap_mean / gap_std)

    metrics: Dict[str, float] = {
        "fr_auc": fr_auc,
        "gap_mean": -gap_mean,  # Flip sign: gap = A(active) - A(cross) for A+ > A-
        "gap_std": gap_std,
        "extractability_index": -ei,  # Flip for consistency
    }

    if alpha is not None:
        # log_weight_ratio = alpha * (A+ - A-) = alpha * (-gap) in our representation
        log_wr = -alpha * gap_flat
        metrics.update(
            {
                "log_weight_ratio_mean": float(jnp.mean(log_wr)),
                "log_weight_ratio_std": float(jnp.std(log_wr)),
                "frac_weight_ratio_gt_10": float(jnp.mean(log_wr > jnp.log(10.0))),
                "frac_weight_ratio_gt_100": float(jnp.mean(log_wr > jnp.log(100.0))),
            }
        )

    return metrics


def run_afr_for_phase(
    phase_dir: Path,
    alpha: Optional[float] = None,
    n_batches: int = 5,
    batch_size: int = 256,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[int, FrozenConfigDict]]:
    """Run AFR recomputation for all seeds in a phase directory.

    For each seed in phase_dir:
    1. Load trained agent weights (params_*.pkl)
    2. Determine agent type from checkpoint/config
    3. Recreate OG-Bench dataset
    4. Sample transitions and compute cross-trajectory advantage gaps
    5. Compute FR-AUC, EI, log_weight_ratio metrics

    Args:
        phase_dir: Directory containing seed_*/ subdirectories with params_*.pkl files.
                   Expected structure: phase_dir/seed_X/params_*.pkl
        alpha: AWR temperature parameter. If None, log_weight_ratio metrics are skipped.
        n_batches: Number of dataset batches to sample per seed.
                   More batches = more precise metrics (but slower).
        batch_size: Batch size for advantage computation (default 256, matches CONST_VAL_BATCH_SIZE).
        seed: Fixed random seed for reproducibility (optional).

    Returns:
        df: DataFrame with columns (phase, seed, fr_auc, ei, metrics...).
        configs: Dict mapping seed → FrozenConfigDict for reference.
    """
    if seed is not None:
        np.random.seed(seed)

    # Discover seeds in phase_dir
    seed_dirs = sorted([d for d in phase_dir.iterdir() if d.name.startswith("seed_")])
    if not seed_dirs:
        raise AfrRecomputationError(f"No seed directories found in {phase_dir}")

    # Load configuration from parent directory
    config: FrozenConfigDict = _find_config_for_phase(phase_dir)

    # The config should contain agent_name and dataset info
    agent_name = str(config.get("agent_name", config.get("agent", "crl"))).lower()
    dataset_name = config.get(
        "dataset_name", config.get("dataset", "antmaze-medium-navigate-v0")
    )

    metrics_rows: List[Dict[str, Any]] = []
    configs: Dict[int, FrozenConfigDict] = {}

    for seed_dir in seed_dirs:
        seed_num = int(seed_dir.name.split("_")[1])
        print(f"[AFR] Processing seed {seed_num} from {seed_dir}")

        # Find checkpoint
        ckpt_files = sorted(seed_dir.glob("params_*.pkl"))
        if not ckpt_files:
            print(f"[AFR] No checkpoint found in {seed_dir}, skipping")
            continue

        checkpoint_path = ckpt_files[-1]
        configs[seed_num] = config

        # Load agent
        agent_class = _extract_agent_class(agent_name)
        agent = _load_agent_from_checkpoint(
            checkpoint_path, agent_class, seed_num, config
        )
        wrapped_agent = AgentWrapper(
            agent.agent, config
        )  # agent from checkpoint has same structure

        # Recreate dataset
        dataset = _load_dataset_from_config(agent_name.upper(), config)

        # Sample batches and compute AFR
        fr_aucs: List[float] = []
        eis: List[float] = []
        lw_means: List[float] = []
        current_gap_means: List[float] = []
        lw_stds: List[float] = []

        for b in range(n_batches):
            # Sample batch from dataset (with trajectory membership)
            batch = _sample_batch_with_trajectory(dataset, batch_size)

            # Compute cross-trajectory advantages
            active_adv, gap_rows, gap_cols, gap_values, trajectory_idx = (
                wrapped_agent.get_cross_trajs_gaps(batch)
            )

            if len(gap_values) == 0:
                print(
                    f"[AFR] Seed {seed_num}, batch {b}: no cross-traj pairs found, skipping"
                )
                continue

            # Compute FR-AUC, EI, log_weight_ratio etc.
            metrics = compute_afr_metrics_from_gaps(
                gap_values, active_adv, gap_rows, gap_cols, trajectory_idx, alpha=alpha
            )

            fr_aucs.append(metrics["fr_auc"])
            eis.append(metrics["extractability_index"])
            if "log_weight_ratio_mean" in metrics:
                lw_means.append(metrics["log_weight_ratio_mean"])
                lw_stds.append(metrics["log_weight_ratio_std"])

            # Collect gap mean for this batch
            current_gap_means.append(float(jnp.mean(gap_values)))

        # Aggregate across batches
        metrics_dict: Dict[str, Any] = {
            "phase": int(re.search(r"phase_(\d+)", str(phase_dir)).group(1)),
            "seed": seed_num,
            "n_batches_sampled": len(fr_aucs),
            "fr_auc": float(np.mean(fr_aucs)) if fr_aucs else float("nan"),
            "fr_auc_std": float(np.std(fr_aucs)) if fr_aucs else 0.0,
            "ei": float(np.mean(eis)) if eis else float("nan"),
            "ei_std": float(np.std(eis)) if eis else 0.0,
            "gap_mean": float(np.mean(current_gap_means))
            if current_gap_means
            else float("nan"),
        }

        if alpha is not None and lw_means:
            metrics_dict["log_weight_ratio_mean"] = float(np.mean(lw_means))
            metrics_dict["log_weight_ratio_std"] = float(np.std(lw_stds))

        metrics_rows.append(metrics_dict)
        print(
            f"[AFR] Seed {seed_num}: FR-AUC={metrics_dict['fr_auc']:.3f}, EI={metrics_dict['ei']:.3f}"
        )

    df = pd.DataFrame(metrics_rows)
    return df, configs


# ===========================================================================
# Dataset sampling helpers
# ===========================================================================


def _sample_batch_with_trajectory(
    dataset: Any,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """Sample a batch from the OG-Bench dataset with trajectory membership info.

    Returns a dict with keys:
    - 'observations': [B, D_obs]
    - 'actions': [B, D_act]
    - 'next_observations': [B, D_obs]
    - 'actor_goals': [B, D_goal]
    - 'goal_obs': [B, D_goal]
    - 'next_goal_obs': [B, D_goal]
    - 'trajectory_final_state_idx': [B] trajectory membership
    """
    batch = dataset.sample(batch_size)

    # Add trajectory_idx from observations
    trajectory_idx = np.asarray(
        batch.get("trajectory_final_state_idx", np.arange(batch_size))
    )

    # actor_goals may not always be present, fallback to goals or observations
    actor_goals = batch.get(
        "actor_goals",
        batch.get("goals", batch.get("goal_obs", batch["observations"][:batch_size])),
    )
    # If actor_goals are the same shape as observations, they might be position encodings
    # In that case, we need to distinguish between observation and goal
    if actor_goals.shape[1] == batch["observations"].shape[1]:
        # They might be obs — use goal-specific keys
        actor_goals = batch.get("goal_obs", batch.get("goals", actor_goals))

    return {
        "observations": batch["observations"],
        "actions": batch["actions"],
        "next_observations": batch.get("next_observations", batch["observations"]),
        "goals": batch.get("goals", batch.get("actor_goals", actor_goals)),
        "actor_goals": actor_goals,
        "trajectory_final_state_idx": trajectory_idx,
    }


# ===========================================================================
# Quick entry point for testing
# ===========================================================================


def create_agent_wrapper_from_checkpoint(
    checkpoint_path: Path,
) -> Tuple[AgentWrapper, FrozenConfigDict]:
    """Convenience function to create an AgentWrapper from a checkpoint.

    Args:
        checkpoint_path: Path to params_*.pkl

    Returns:
        Tuple of (AgentWrapper, config)
    """
    agent_name, config = _detect_checkpoint_format(checkpoint_path)
    agent_class = _extract_agent_class(agent_name)

    # Extract seed and config index from path
    seed = int(checkpoint_path.parent.name.split("_")[1])
    ckpt_name = str(checkpoint_path.name)
    step_match = re.search(r"params_(\d+)", ckpt_name)
    step = int(step_match.group(1)) if step_match else 0

    agent = _load_agent_from_checkpoint(checkpoint_path, agent_class, seed, config)
    return AgentWrapper(agent.agent, config), config
