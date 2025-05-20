from ogbench.impls.utils.datasets import GCDataset
import numpy as np
from math import ceil, floor
import warnings


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
        idxs1 = idxs[idxs < self.dataset1.size] if idxs else None
        idxs2 = idxs[idxs >= self.dataset1.size] - self.dataset1.size if idxs else None

        return idxs1, idxs2

    def sample(self, batch_size, idxs=None, evaluation=False):
        if idxs is not None:
            warnings.warn(
                "setting idxs for sampling from MixedDataset disregards set ratio between datasets"
            )
        idxs1, idxs2 = self._split_indices(idxs)
        batch1_size = ceil((1 - self.second_share_factor) * batch_size)
        batch2_size = floor(self.second_share_factor * batch_size)

        batch1 = self.dataset1.sample(batch1_size, idxs=idxs1, evaluation=evaluation)
        batch2 = self.dataset2.sample(batch2_size, idxs=idxs2, evaluation=evaluation)

        return {
            key: np.concatenate((batch1[key], batch2[key])) for key in batch1.keys()
        }

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        idxs1, idxs2 = self._split_indices(idxs)
        goals1 = self.dataset1.sample_goals(
            idxs1, p_curgoal, p_trajgoal, p_randomgoal, geom_sample
        )
        goals2 = self.dataset2.sample_goals(
            idxs2, p_curgoal, p_trajgoal, p_randomgoal, geom_sample
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
