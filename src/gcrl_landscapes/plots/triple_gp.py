# +-----------------------------------------------------------------------------------------------------------------------------------------------------+
# | The following code is adapted from https://colab.research.google.com/drive/1VjWng8KeGiW1RnsU6FYriaxAOxkBooXc?usp=sharing#scrollTo=81Mam8kV7ksN |
# | and was created by Aditya Mohan (https://amsks.github.io/)                                                                                          |
# +-----------------------------------------------------------------------------------------------------------------------------------------------------+

import numpy as np
import matplotlib.pyplot as plt
import gpflow
from sklearn.base import BaseEstimator
from scipy.stats import trim_mean
from ConfigSpace import ConfigurationSpace, Float

from pandas import DataFrame
from autorl_landscape.analyze.visualization import Visualization

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, max_error, mean_squared_error
from sklearn.model_selection import KFold

from autorl_landscape.analyze.visualization import Visualization
from autorl_landscape.run.compare import iqm
from typing import Callable, Any

from gcrl_landscapes.configurations import get_bounds, sobol_codomain_to_hp


def estimate_model_fit(X, y, y_scale, splitter: Any = KFold(n_splits=5, shuffle=True), metrics: list[Callable] | None = None) -> DataFrame:
    if metrics is None:
        metrics = [mean_squared_error, mean_absolute_error]

    data = []
    for i, (train_index, test_index) in enumerate(splitter.split(X=X, y=y)):
        X_i = X[train_index]
        Y_i = y[train_index]
        model = gpflow.models.GPR((X_i, Y_i), kernel=gpflow.kernels.SquaredExponential())
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)
        f_mean, _ = model.predict_f(X[test_index])
        y_pred = f_mean.numpy()
        results = {}
        results["fold"] = i
        for metric in metrics:
            results[metric.__name__] = metric(y[test_index] * y_scale, y_pred * y_scale)
        data.append(results)
    data = pd.DataFrame(data)

    return data


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
        conf_groups = data.groupby(hp_names)
        # all groups (configurations):
        self.x_unscaled = np.array(list(conf_groups.groups.keys()))[:, :]
        """(num_confs, num_ls_dims). LS dimensions are sorted by name"""
        # all evaluations (y values) for a group (configuration):
        self.x_normalizing_offset = self.x_unscaled.min(axis=0)
        self.x_normalizing_factor = self.x_unscaled.max(axis=0) - self.x_unscaled.min(axis=0)
        # max == min -> don't divide by zero
        self.x_normalizing_factor[np.where(self.x_normalizing_factor == 0.0)] = 1.0

        self.x = self._scale_x(self.x_unscaled)

        y = np.array(conf_groups[y_col].apply(list).tolist())

        # handle crashed runs by assigning 0 return:
        self.crashed = np.isnan(y)
        y[np.isnan(y)] = 0

        # scale ls dims into [0, 1] interval:
        for i in range(len(hp_names)):
            if "-uniform" not in hp_names[i] and configspace[hp_names[i].split(".")[-1]].__class__ == Float:
                self.x[:, i] = (self.x[:, i] - configspace[hp_names[i]].lower) / (configspace[hp_names[i]].upper - configspace[hp_names[i]].lower)
        # scale y into [0, 1] interval:
        self.y_normalizing_offset = y.min()
        self.y_normalizing_factor = y.max() - y.min() if y.max() != y.min() else 1.0
        self.y_unscaled = y
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

    def estimate_iqm_fit(self, print: bool = False, metrics: list[Callable] = [mean_squared_error, mean_absolute_error, max_error]) -> DataFrame:
        # unscale estimates to get interpretable results
        data = estimate_model_fit(X=self.x, y=self.y_iqm, y_scale=self.y_normalizing_factor, metrics=metrics)
        if not print:
            return data

        print("-"*50)
        print("Estimate IQM surface fit")
        for c in data.columns:
            if c != "fold":
                print(c, data[c].mean(), data[c].std())
        return data

    def _scale_x(self, x):
        return (x - self.x_normalizing_offset) / self.x_normalizing_factor

    def _unscale_x(self, x):
        return x * self.x_normalizing_factor + self.x_normalizing_offset

    def _unscale_y(self, y):
        return (y * self.y_normalizing_factor) + self.y_normalizing_offset

    def get_upper(self, x_unscaled, assimilate_factor: float = 1.0):
        """Return the upper CI estimate of y at the position(s) x."""
        x = self._scale_x(x_unscaled)
        f_mean, _ = self.upper_model.predict_f(x)
        return self._ci_scale(x, f_mean.numpy(), assimilate_factor)

    def get_middle(self, x_unscaled):
        """Return the IQM estimate of y at the position(s) x."""
        x = self._scale_x(x_unscaled)
        f_mean, _ = self.iqm_model.predict_f(x)
        return self._unscale_y(f_mean.numpy())

    def get_lower(self, x_unscaled, assimilate_factor: float = 1.0):
        """Return the lower CI estimate of y at the position(s) x."""
        x = self._scale_x(x_unscaled)
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

def create_contour_plot(model, x_dim, y_dim, z_dim, bounds, filename, dim_label_mapping: Callable[[str], str],  agent_name:str, z_transform: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda z_pred, z: z_pred, discrete_levels: np.ndarray | None = None, last_phase_best_config: pd.DataFrame | None = None, grid_length=100):
    x_lower, x_upper, x_log = get_bounds(model.hp_names[x_dim].removeprefix("hp."), agent_name)
    y_lower, y_upper, y_log = get_bounds(model.hp_names[y_dim].removeprefix("hp."), agent_name)
    # Generate a finer grid for contour plot
    x = np.linspace(0, 1, grid_length)
    y = np.linspace(0, 1, grid_length)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    points = np.vstack([X.ravel(), Y.ravel()]).transpose()
    Z_untransformed = model.get_middle(points)
    Z = z_transform(Z_untransformed.reshape(-1), model._unscale_y(model.y_iqm)).reshape(X.shape)
    if bounds[0] != None and bounds[1] != None:
        Z = np.clip(Z, bounds[0], bounds[1])

    # Create contour plot
    fig, ax = plt.subplots(figsize=[4, 3])

    X = sobol_codomain_to_hp(X, x_lower, x_upper, x_log)
    Y = sobol_codomain_to_hp(Y, y_lower, y_upper, y_log)
    if x_log:
        ax.set_xscale('log', base=10)

    if y_log:
        ax.set_xscale('log', base=10)


    if discrete_levels is None:
        if bounds[0] != None and bounds[1] != None:
            levels = np.linspace(bounds[0], bounds[1], 21)
            cmap = plt.get_cmap("rocket", len(levels) - 1)
            norm = mcolors.BoundaryNorm(levels, cmap.N, clip=True)

            cbarnorm = mcolors.Normalize(vmin=bounds[0], vmax=bounds[1])
            cbarmappable = cm.ScalarMappable(norm=cbarnorm, cmap="rocket")
            cbar = plt.colorbar(cbarmappable, ax=ax)
            contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
        else:

            levels = np.linspace(Z.min(), Z.max(), 21)
            cmap = plt.get_cmap("rocket", len(levels) - 1)
            norm = mcolors.BoundaryNorm(levels, cmap.N, clip=True)

            contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
            cbar = plt.colorbar(contour, ax=ax)

    else:
        levels = discrete_levels

        labels = [f">={level:.2f}" for level in levels[:-1]]

        cmap = plt.get_cmap("viridis", len(levels) - 1)
        norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        handles = [
            Patch(color=cmap(i), label=labels[i]) for i in range(len(labels))
        ]
        ax.legend(handles=handles, title="Value Range")
        contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)

    # Mark best previous configuration with asterisk
    if last_phase_best_config is not None:
        x = last_phase_best_config[model.hp_names[x_dim]].iloc[0]
        y = last_phase_best_config[model.hp_names[y_dim]].iloc[0]
        ax.scatter([x], [y], color='white', marker='*', s=100, edgecolor='black')  # Peaks

    plt.xlabel(dim_label_mapping(model.hp_names[x_dim].split('.')[-1]), fontsize=18)
    plt.ylabel(dim_label_mapping(model.hp_names[y_dim].split('.')[-1]), fontsize=18)
    # plt.title(f'{z_dim}', fontsize=18)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
