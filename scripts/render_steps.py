"""Render the four pipeline steps for one method and phase, one PNG per step.

Usage: python render_steps.py <grids.pkl> <returns.pkl> <output_dir> <method> <phase>
"""
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

grids = pickle.load(open(sys.argv[1], "rb"))
returns = pickle.load(open(sys.argv[2], "rb"))
out_dir, method, phase = Path(sys.argv[3]), sys.argv[4], int(sys.argv[5])
out_dir.mkdir(parents=True, exist_ok=True)

INK = "#1d1d1f"
VMIN = 0.35
OFF = "#a9a9b3"
RAMP = LinearSegmentedColormap.from_list("ramp", ["#eaf5f4", "#21918c", "#0d3f3d"])
# Same colour as this phase in the landscape GIF, so step 4 reads as one of its panels.
REGION = sns.color_palette("viridis", n_colors=4)[phase - 1]

sns.set_theme(style="white", context="paper")
grid = grids[(method, phase)]
obs = returns[(method, phase)]
# One row per (configuration, seed): aggregate to one dot per configuration, by IQM.
per_config = pd.DataFrame(obs, columns=["lr", "alpha", "score"]).assign(
    score=lambda d: d["score"].clip(0, 1)
).groupby(["lr", "alpha"])["score"].apply(lambda v: np.sort(v)[1:-1].mean()).reset_index()
assert per_config["score"].notna().all() and obs.shape[0] == 5 * len(per_config)
lr, alpha, score = per_config["lr"].to_numpy(), per_config["alpha"].to_numpy(), per_config["score"].to_numpy()
surface = grid["Z"].reshape(100, 100)
vmax = max(float(score.max()), float(surface.max()))
# Most of the plane sits well above zero; a floor spends the colour range where the shape is.
vmin = VMIN
print("surface quantiles 10/25/50/75/90:", np.round(np.quantile(surface, [.1, .25, .5, .75, .9]), 3))
(bx, by) = grid["bounds"]
print(f"{method} phase {phase}: n={len(per_config)} raw max {score.max():.3f} "
      f"GP max {surface.max():.3f} colour max {vmax:.3f}")


def axes():
    fig, ax = plt.subplots(figsize=(3.9, 3.2))
    ax.set_xscale("log")
    ax.set_xlim(bx[0], bx[1])
    ax.set_ylim(by[0], by[1])
    ax.set_xlabel("Learning rate", fontsize=10)
    ax.set_ylabel("AWR temperature", fontsize=10)
    ax.tick_params(labelsize=9)
    sns.despine(ax=ax)
    return fig, ax


def save(fig, name):
    fig.tight_layout()
    path = out_dir / f"step{name}.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(path, path.stat().st_size // 1024, "KB")


fig, ax = axes()
ax.grid(True, alpha=0.15)
ax.scatter(lr, alpha, s=34, facecolors="white", edgecolors=OFF, linewidths=1.2, zorder=3)
save(fig, 1)

fig, ax = axes()
ax.grid(True, alpha=0.15)
ax.scatter(lr, alpha, s=42, c=score, cmap=RAMP, vmin=vmin, vmax=vmax, edgecolors="white",
           linewidths=0.6, zorder=3)
save(fig, 2)

fig, ax = axes()
X = grid["px"].reshape(100, 100)
Y = grid["py"].reshape(100, 100)
ax.pcolormesh(X, Y, surface, cmap=RAMP, vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
ax.contour(X, Y, surface / surface.max(), levels=[0.8], colors="white", linewidths=1.8, zorder=2)
ax.scatter(lr, alpha, s=9, c="white", edgecolors=INK, linewidths=0.4, alpha=0.75, zorder=3)
save(fig, 3)

fig, ax = axes()
ax.grid(True, alpha=0.15)
keep = (grid["Z"] / grid["Z"].max()) > 0.8
region = pd.DataFrame({"x": grid["px"][keep], "y": grid["py"][keep]})
kde = dict(data=region, x="x", y="y", log_scale=(bx[2], by[2]), thresh=0.05, bw_adjust=0.7, ax=ax)
sns.kdeplot(fill=True, levels=4, alpha=0.28, color=REGION, linewidth=0, **kde)
sns.kdeplot(fill=False, levels=[0.7], color=REGION, linewidths=2.8, **kde)
ax.set_xlim(bx[0], bx[1])
ax.set_ylim(by[0], by[1])
ax.set_xlabel("Learning rate", fontsize=10)
ax.set_ylabel("AWR temperature", fontsize=10)
save(fig, 4)
