from ogbench import make_env_and_datasets
from ogbench.impls.utils.datasets import HGCDataset, GCDataset, Dataset
from ogbench.impls.agents import (
    CRLAgent,
    CMDAgent,
    GCBCAgent,
    MQEAgent,
    QRLAgent,
    HIQLAgent,
    SACAgent,
)
from ml_collections import FrozenConfigDict
import numpy as np
from math import ceil, floor
import warnings
import re

from ..agents import FQLAgent, NStepGCIQLAgent, NStepGCIVLAgent

AGENT_CLASSES = {
    "CRL": CRLAgent,
    "CMD": CMDAgent,
    "FQL": FQLAgent,
    "GCBC": GCBCAgent,
    # n-step subclasses behave identically to the base agents unless the batch
    # carries per-sample 'discounts' (only produced when n_step > 1).
    "GCIQL": NStepGCIQLAgent,
    "GCIVL": NStepGCIVLAgent,
    "MQE": MQEAgent,
    "QRL": QRLAgent,
    "HIQL": HIQLAgent,
    "SAC": SACAgent,
}
DATASET_CLASSES = {
    "CRL": GCDataset,
    "CMD": GCDataset,
    "FQL": GCDataset,
    "GCBC": GCDataset,
    "GCIQL": GCDataset,
    "GCIVL": GCDataset,
    "MQE": GCDataset,
    "QRL": GCDataset,
    "HIQL": HGCDataset,
    "SAC": GCDataset,
}


class NStepGCDataset(GCDataset):
    """GCDataset variant producing n-step TD targets.

    For n = config['n_step'] > 1, each sampled transition at index t gets:
    - next_observations: the state n_used steps ahead, where
      n_used = min(n, steps to the trajectory's final state),
    - rewards: the discounted n-step reward sum under the relabeled sparse
      goal reward, treating the goal state as absorbing,
    - masks: 0 if the sampled value goal is reached within the window,
    - discounts: gamma^{n_used}, the per-sample bootstrap factor.

    Goal sampling (value/actor goals) is unchanged. Only state-based datasets
    are supported (frame_stack must be None).
    """

    def sample(self, batch_size, idxs=None, evaluation=False, seed=None):
        n = int(self.config["n_step"])
        if n <= 1:
            return super().sample(
                batch_size, idxs=idxs, evaluation=evaluation, seed=seed
            )
        assert self.config["frame_stack"] is None, (
            "n_step > 1 does not support frame stacking"
        )

        rng = np.random.default_rng(seed) if seed is not None else None

        if idxs is None:
            if rng is not None:
                if "valids" in self.dataset._dict:
                    idxs = self.dataset.valid_idxs[
                        rng.integers(len(self.dataset.valid_idxs), size=batch_size)
                    ]
                else:
                    idxs = rng.integers(self.dataset.size, size=batch_size)
            else:
                idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config["value_p_curgoal"],
            self.config["value_p_trajgoal"],
            self.config["value_p_randomgoal"],
            self.config["value_geom_sample"],
            rng=rng,
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config["actor_p_curgoal"],
            self.config["actor_p_trajgoal"],
            self.config["actor_p_randomgoal"],
            self.config["actor_geom_sample"],
            rng=rng,
        )
        batch["value_goals"] = self.get_observations(value_goal_idxs)
        batch["actor_goals"] = self.get_observations(actor_goal_idxs)

        final = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        gamma = self.config["discount"]

        # Steps until the goal state is the current state (success at step k
        # means index t+k equals the goal index, within the same trajectory).
        offset = value_goal_idxs - idxs
        success_in_window = (
            (offset >= 0) & (offset <= n - 1) & (value_goal_idxs <= final)
        )

        n_traj = np.maximum(final - idxs, 1)  # never step past the final state
        n_used = np.minimum(n, n_traj)
        k = np.where(success_in_window, offset, n_used).astype(np.int64)

        if self.config["gc_negative"]:
            # r_j = -1 before the goal, 0 at the goal (absorbing).
            rewards = -(1.0 - gamma**k) / (1.0 - gamma)
        else:
            # r_j = 1 at the goal, 0 otherwise.
            rewards = np.where(success_in_window, gamma ** k.astype(np.float64), 0.0)

        batch["masks"] = 1.0 - success_in_window.astype(np.float64)
        batch["rewards"] = rewards.astype(np.float64)
        batch["discounts"] = (gamma**n_used).astype(np.float64)
        next_idxs = np.minimum(idxs + n_used, final)
        batch["next_observations"] = self.get_observations(next_idxs)
        batch["trajectory_final_state_idx"] = final

        if self.config["p_aug"] is not None and not evaluation:
            rand_val = rng.random() if rng is not None else np.random.rand()
            if rand_val < self.config["p_aug"]:
                self.augment(
                    batch,
                    ["observations", "next_observations", "value_goals", "actor_goals"],
                )

        return batch


class MixedDataset(GCDataset):
    def __init__(self, dataset1, dataset2, second_share: int, freeze=True):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.second_share_factor = second_share / 100

        self.size = self.dataset1.size + self.dataset2.size
        self.terminal_locs = np.concatenate(
            (self.dataset1.terminal_locs, self.dataset2.terminal_locs)
        )
        self.initial_locs = np.concatenate(
            (self.dataset1.initial_locs, self.dataset2.initial_locs)
        )

    def _split_indices(self, idxs):
        idxs1 = idxs[idxs < self.dataset1.size] if idxs is not None else None
        idxs2 = (
            idxs[idxs >= self.dataset1.size] - self.dataset1.size
            if idxs is not None
            else None
        )

        return idxs1, idxs2

    def sample(self, batch_size, idxs=None, evaluation=False, seed=None):
        if idxs is not None:
            warnings.warn(
                "setting idxs for sampling from MixedDataset disregards set ratio between datasets"
            )
        idxs1, idxs2 = self._split_indices(idxs)
        batch1_size = ceil((1 - self.second_share_factor) * batch_size)
        batch2_size = floor(self.second_share_factor * batch_size)

        batch1 = self.dataset1.sample(
            batch1_size, idxs=idxs1, evaluation=evaluation, seed=seed
        )
        batch2 = self.dataset2.sample(
            batch2_size, idxs=idxs2, evaluation=evaluation, seed=seed
        )

        return {
            key: np.concatenate((batch1[key], batch2[key])) for key in batch1.keys()
        }

    def sample_goals(
        self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, rng=None
    ):
        idxs1, idxs2 = self._split_indices(idxs)
        goals1 = self.dataset1.sample_goals(
            idxs1, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, rng=rng
        )
        goals2 = self.dataset2.sample_goals(
            idxs2, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, rng=rng
        )

        return np.concatenate((goals1, goals2))

    def get_observations(self, idxs):
        idxs1, idxs2 = self._split_indices(idxs)
        obs1 = self.dataset1.get_observations(idxs1)
        obs2 = self.dataset1.get_observations(idxs2)

        return np.concatenate((obs1, obs2))

    def get_stacked_observations(self, idxs):
        idxs1, idxs2 = self._split_indices(idxs)
        obs1 = self.dataset1.get_observations(idxs1)
        obs2 = self.dataset1.get_observations(idxs2)

        return np.concatenate((obs1, obs2))


def dataset_constructor(dataset_raw, agent_name: str, configuration: FrozenConfigDict):
    dataset_class = DATASET_CLASSES[agent_name.upper()]
    if int(configuration.get("n_step", 1)) > 1:
        assert agent_name.upper() in ("GCIQL", "GCIVL"), (
            f"n_step > 1 is only supported for GCIQL and GCIVL, got {agent_name}"
        )
        dataset_class = NStepGCDataset
    return dataset_class(Dataset.create(**dataset_raw), configuration)


def create_env_and_dataset(
    dataset_name: str, agent_name: str, configuration: FrozenConfigDict
) -> tuple:
    explore_mix_match = re.fullmatch(
        r"^.*explore(?P<explore_share>\d+)(?P<secondtype>[^-]*).*$", dataset_name
    )
    if explore_mix_match:
        base_dataset = re.sub(r"explore\d+", "", dataset_name)
        explore_dataset = re.sub(r"explore\d+[^-]*", "explore", dataset_name)
        explore_share = int(explore_mix_match.groupdict()["explore_share"])
        print(
            f"Found mixed dataset, mixing {explore_dataset} into {base_dataset} ({explore_share}%)"
        )
        env, train_dataset1_raw, val_dataset1_raw = make_env_and_datasets(base_dataset)  # type: ignore
        _, train_dataset2_raw, val_dataset2_raw = make_env_and_datasets(explore_dataset)  # type: ignore
        train_dataset = MixedDataset(
            dataset_constructor(train_dataset1_raw, agent_name, configuration),
            dataset_constructor(train_dataset2_raw, agent_name, configuration),
            explore_share,
        )
        val_dataset = MixedDataset(
            dataset_constructor(val_dataset1_raw, agent_name, configuration),
            dataset_constructor(val_dataset2_raw, agent_name, configuration),
            explore_share,
        )
    else:
        env, train_dataset_raw, val_dataset_raw = make_env_and_datasets(dataset_name)  # type: ignore
        train_dataset = dataset_constructor(
            train_dataset_raw, agent_name, configuration
        )
        val_dataset = dataset_constructor(val_dataset_raw, agent_name, configuration)

    return env, train_dataset, val_dataset
