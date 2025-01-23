# +----------------------------------------------------------------------------------------------------------------------------------------------+
# | The following code is taken from https://colab.research.google.com/drive/1VjWng8KeGiW1RnsU6FYriaxAOxkBooXc?usp=sharing#scrollTo=81Mam8kV7ksN |
# | and was created by Aditya Mohan (https://amsks.github.io/)                                                                                   |
# +----------------------------------------------------------------------------------------------------------------------------------------------+

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import gpflow
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import trim_mean
from ConfigSpace import ConfigurationSpace, Float, Categorical
from arlbench.core.algorithms import DQN, PPO, SAC
from autorl_landscape.visualize import (
    LEGEND_FSIZE,
    TITLE_FSIZE,
)

from matplotlib.gridspec import GridSpecFromSubplotSpec
from itertools import zip_longest
from pandas import DataFrame
from autorl_landscape.analyze.visualization import Visualization

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from pandas import DataFrame
from sklearn.base import BaseEstimator


import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator

from autorl_landscape.analyze.visualization import Visualization
from autorl_landscape.run.compare import iqm

def iqm(x, axis=None):
    """
    Helper to comput the IQM. Wraps Scipys trim_mean function
    """
    iqms = trim_mean(x, proportiontocut=0.25, axis=axis)
    means = np.mean(x, axis=axis)
    iqms[np.isnan(means)] = np.nan
    return iqms

class TripleGPModel(BaseEstimator):
    """
    Implements the IGPR approach by Mohan et al. 2023 to computes the landscapes on lower-, middle- and upper-confidence
    intervals. Although Gaussian Processes (GPs) can inherently model uncertainty which could be used to fully model
    lower and upper quantiles as well as the mean of normal distributions, this instead uses just the mean of the GP to
    model the surfaces independently. This uses an RBF kernel and optimizes parameters and length scales with scipy’s
    L-BFGS-B optimizer.
    """

    def __init__(
        self,
        data,
        dtype: type,
        y_col: str = "performance",
        y_bounds: tuple[float, float] | None = None,
        best_conf = None,
        hp_names = None,
        configspace: ConfigurationSpace = None,
        ci: float = 0.95,
    ) -> None:
        super().__init__()
        gpflow.config.set_default_float(dtype)
        self.data = data
        self.y_col = y_col
        self.y_bounds = y_bounds
        self.ci = ci
        self.best_conf = best_conf
        self.model_layer_names = ["upper", "middle", "lower"]
        self.hp_names = hp_names
        self.dtype = dtype
        self.y_info = "yelp"

        # group runs with the same configuration:
        self.dim_info = hp_names
        conf_groups = data.groupby(["run_id"] + hp_names)
        # all groups (configurations):
        self.x = np.array(list(conf_groups.groups.keys()))[:, 1:]
        """(num_confs, num_ls_dims). LS dimensions are sorted by name"""
        # all evaluations (y values) for a group (configuration):

        y = np.concatenate(conf_groups[y_col].apply(list))
        y = np.array([np.array(ys) for ys in y])

        # handle crashed runs by assigning 0 return:
        self.crashed = np.isnan(y)
        y[np.isnan(y)] = 0

        # scale ls dims into [0, 1] interval:
        for i in range(len(hp_names)):
            if configspace[hp_names[i].split(".")[-1]].__class__ == Float:
                self.x[:, i] = (self.x[:, i] - configspace[hp_names[i]].lower) / (configspace[hp_names[i]].upper - configspace[hp_names[i]].lower)
        # scale y into [0, 1] interval:
        self.y = (y - y.min()) / (y.max() - y.min())
        """(num_confs, samples_per_conf)"""

        # just all the single evaluation values, not grouped (but still scaled to [0, 1] interval):
        self.x_samples = np.repeat(self.x, self.y.shape[1], axis=0)
        """(num_confs * samples_per_conf, num_ls_dims)"""
        self.y_samples = self.y.reshape(-1, 1)
        """(num_confs * samples_per_conf, 1)"""

        upper_quantile = 1 - ((1 - ci) / 2)
        lower_quantile = 0 + ((1 - ci) / 2)

        # statistical information about each configuration:
        self.y_iqm = iqm(self.y, axis=1).reshape(-1, 1)
        """(num_confs, 1)"""
        # first, select ci quantile:
        self.y_ci_upper = np.quantile(self.y, upper_quantile, method="median_unbiased", axis=1, keepdims=True)
        """(num_confs, 1)"""
        self.y_ci_lower = np.quantile(self.y, lower_quantile, method="median_unbiased", axis=1, keepdims=True)
        """(num_confs, 1)"""

        self._viz_infos: list[Visualization] = [
            Visualization(
                "Raw Return Samples",
                "scatter",
                "graphs",
                self.build_df(self.x_samples, self.y_samples, "performance"),
                {},
                # {"color": "red"},
            )
        ]

    def fit(self):
        """Fit the three GPs to IQM, upper and lower CI."""
        self.iqm_model = gpflow.models.GPR((self.x, self.y_iqm), kernel=gpflow.kernels.SquaredExponential())
        self.upper_model = gpflow.models.GPR((self.x, self.y_ci_upper), kernel=gpflow.kernels.SquaredExponential())
        self.lower_model = gpflow.models.GPR((self.x, self.y_ci_lower), kernel=gpflow.kernels.SquaredExponential())

        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.iqm_model.training_loss, self.iqm_model.trainable_variables)
        opt.minimize(self.upper_model.training_loss, self.upper_model.trainable_variables)
        opt.minimize(self.lower_model.training_loss, self.lower_model.trainable_variables)

    def estimate_iqm_fit(self):
        print("-"*50)
        print("Estimate IQM surface fit")
        data = estimate_model_fit(X=self.x, y=self.y_iqm, k=5)
        for c in data.columns:
            if c != "fold":
                print(c, data[c].mean(), data[c].std())
        return data

    def get_upper(self, x, assimilate_factor: float = 1.0):
        """Return the upper CI estimate of y at the position(s) x."""
        f_mean, _ = self.upper_model.predict_f(x)
        return self._ci_scale(x, f_mean.numpy(), assimilate_factor)

    def get_middle(self, x):
        """Return the IQM estimate of y at the position(s) x."""
        f_mean, _ = self.iqm_model.predict_f(x)
        return f_mean.numpy()

    def get_lower(self, x, assimilate_factor: float = 1.0):
        """Return the lower CI estimate of y at the position(s) x."""
        f_mean, _ = self.lower_model.predict_f(x)
        return self._ci_scale(x, f_mean.numpy(), assimilate_factor)

    @staticmethod
    def get_model_name() -> str:
        """Return name of this model, for naming files and the like."""
        return "igpr_"

    def add_viz_info(self, viz_info) -> None:
        """Add a visualization to this model."""
        self._viz_infos.append(viz_info)

    def get_viz_infos(self):
        """Return visualization info(s) for data points used for training the model."""
        return self._viz_infos

    def build_df(self, x, y, y_axis_label: str):
        """Helper to construct a `DataFrame` given x points and y readings and a label for y.

        Labels for x are taken from the model.
        """
        assert x.shape[1] == len(self.dim_info)
        return DataFrame(np.concatenate([x, y], axis=1), columns=self.get_ls_dim_names() + [y_axis_label])

    def _ci_scale(self, x, y, assimilate_factor: float = 1.0):
        """Assimilate passed y values (assumed to come from `get_upper` or `get_lower`) towards the middle values."""
        if assimilate_factor == 1.0:
            return y

        y_middle = self.get_middle(x)
        return assimilate_factor * y + (1 - assimilate_factor) * y_middle

    def get_ls_dim_names(self) -> list[str]:
        """Get the list of hyperparameter landscape dimension names."""
        return self.hp_names

    def get_dim_info(self, name: str):
        """Return matching `DimInfo` to a passed name (can be y_info of any dim_info)."""
        return None

    def plot_fit(self):
        """Plot the fit of the IQM model with confidence intervals."""
        # Create a grid of x values to predict over
        x_grid = np.linspace(0, 1, 100).reshape(-1, self.x.shape[1])

        # Get predictions for mean, upper, and lower confidence intervals
        y_middle = self.get_middle(x_grid)
        y_upper = self.get_upper(x_grid)
        y_lower = self.get_lower(x_grid)

        # Actual y values (scaled back to [0, 1] interval) and configurations
        plt.figure(figsize=(10, 6))
        plt.scatter(self.x_samples[:, 0], self.y_samples, color='blue', label='Actual Data', alpha=0.5)

        # Plot the model's mean predictions
        plt.plot(x_grid[:, 0], y_middle, color='green', label='IQM (Mean Prediction)', linewidth=2)

        # Plot the confidence intervals
        plt.fill_between(x_grid[:, 0], y_lower.flatten(), y_upper.flatten(), color='gray', alpha=0.3, label='Confidence Interval')

        # Customize plot
        plt.title('Triple GP Model Fit with Confidence Intervals')
        plt.xlabel('Hyperparameter Dimension')
        plt.ylabel(self.y_col)
        plt.legend()
        plt.show()
