"""Landscape-level analyses: mobility plots and optimum-overlap metrics.

These analyses require multiple experiments from one zip (different datasets / phases)
and are heavier than per-experiment plots (GP fitting per phase/quality level).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

from gcrl_landscapes.configurations import get_bounds, sobol_codomain_to_hp
from gcrl_landscapes.evaluation.common import map_labels
from gcrl_landscapes.util.eval import fit_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def datasets_to_exploration_schedule(dataset_str: str) -> str:
    """Parse a (possibly multi-env) dataset string into exploration percentages.

    Returns a comma-separated string of integer percentages, one per env.
    """

    def _pct(dataset: str) -> int:
        m = re.search(r"explore(\d+)\w+", dataset)
        if m:
            return int(m.group(1))
        if re.match(r".*explore-.*", dataset):
            return 100
        if re.match(r".*navigate-.*", dataset):
            return 0
        return 0  # non-antmaze datasets treated as fully expert

    return ",".join(str(_pct(d)) for d in dataset_str.split(","))


# ---------------------------------------------------------------------------
# Mobility plot (per experiment, per phase or per quality level)
# ---------------------------------------------------------------------------


def mobility_plot(
    df: pd.DataFrame,
    agent_name: str,
    hp_list: list[str],
    output_path: Path,
    by_col: str = "phase_num",
    performance_threshold: float = 0.95,
    grid_length: int = 100,
) -> None:
    """KDE visualisation of optimal HP regions across phases or data-quality levels.

    Args:
        df: Results DataFrame for one (agent, dataset) combination. Must contain
            ``mean_normalized_goal_distance_return``, ``hp.*`` columns and ``by_col``.
        agent_name: Agent name string (used for ``get_bounds`` lookup).
        hp_list: Two HP column names, e.g. ``["hp.lr", "hp.alpha"]``.
        output_path: Full path where the PNG will be saved (parent is created).
        by_col: Column whose distinct values become KDE layers.
        performance_threshold: Minimum *normalised* performance for a GP-predicted
            point to be included in the KDE.
        grid_length: Side length of the prediction grid (``grid_length**2`` points).
    """
    clipped_df = df.copy()
    clipped_df["mean_normalized_goal_distance_return"] = clipped_df[
        "mean_normalized_goal_distance_return"
    ].clip(0, 1)

    # Resolve HP names and axis bounds from a sample model
    _sample = fit_model(clipped_df, "mean_normalized_goal_distance_return", hp_list)
    _sample.fit()
    hp_name_x = _sample.hp_names[0].removeprefix("hp.")
    hp_name_y = _sample.hp_names[1].removeprefix("hp.")
    x_lower, x_upper, x_log = get_bounds(hp_name_x, agent_name)
    y_lower, y_upper, y_log = get_bounds(hp_name_y, agent_name)

    # Build Sobol-space prediction grid
    x_grid = np.linspace(0, 1, grid_length)
    y_grid = np.linspace(0, 1, grid_length)
    X, Y = np.meshgrid(x_grid, y_grid)
    points = np.vstack([X.ravel(), Y.ravel()]).T

    point_dfs: list[pd.DataFrame] = []
    for name, group in clipped_df.groupby(by_col):
        model = fit_model(
            group.copy().reset_index(),
            "mean_normalized_goal_distance_return",
            hp_list,
        )
        model.fit()
        Z = model.get_middle(points).clip(0, 1)
        Z_norm = (Z / Z.max()).squeeze()
        pts_x = sobol_codomain_to_hp(points[:, 0], x_lower, x_upper, x_log)
        pts_y = sobol_codomain_to_hp(points[:, 1], y_lower, y_upper, y_log)
        pdf = pd.DataFrame(
            {
                hp_list[0]: pts_x,
                hp_list[1]: pts_y,
                "mean_normalized_goal_distance_return": Z_norm,
            }
        )
        pdf[by_col] = name
        point_dfs.append(pdf)

    point_df = pd.concat(point_dfs)
    df_filtered = point_df[
        point_df["mean_normalized_goal_distance_return"] > performance_threshold
    ]

    if df_filtered.empty:
        print(
            f"mobility_plot: no points above threshold {performance_threshold} — skipping {output_path}"
        )
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    palette = sns.color_palette("viridis", n_colors=df_filtered[by_col].nunique())
    sns.set_context(context="paper", font_scale=1.75)

    texts: list = []
    centroids: list[tuple] = []
    for i, phase in enumerate(sorted(df_filtered[by_col].unique())):
        phase_df = df_filtered[df_filtered[by_col] == phase]
        color = palette[i]

        # Filled density
        sns.kdeplot(
            data=phase_df,
            x=hp_list[0],
            y=hp_list[1],
            log_scale=(x_log, y_log),
            fill=True,
            levels=4,
            thresh=0.05,
            bw_adjust=0.7,
            alpha=0.12,
            color=color,
            linewidth=0,
            ax=ax,
        )
        # Contour line
        sns.kdeplot(
            data=phase_df,
            x=hp_list[0],
            y=hp_list[1],
            log_scale=(x_log, y_log),
            fill=False,
            levels=[0.7],
            thresh=0.05,
            bw_adjust=0.7,
            alpha=0.9,
            color=color,
            linewidths=3.5,
            ax=ax,
            label=str(phase),
        )

        cx = phase_df[hp_list[0]].median()
        cy = phase_df[hp_list[1]].median()
        ax.scatter(
            cx,
            cy,
            s=750 if by_col == "phase_num" else 1500,
            c=[color],
            edgecolors="white",
            linewidths=1,
            zorder=100,
            marker="o",
            alpha=0.75,
        )
        texts.append(
            ax.annotate(
                str(phase),
                xy=(cx, cy),
                xytext=(cx, cy),
                fontsize=20,
                fontweight="bold",
                ha="center",
                va="center",
                color="white",
                zorder=101,
                alpha=1,
            )
        )
        centroids.append((cx, cy))

    ax.set_xlabel(map_labels(hp_name_x))
    ax.set_ylabel(map_labels(hp_name_y))
    ax.grid(True, alpha=0.15)
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)
    if x_log:
        ax.set_xscale("log", base=10)
    if y_log:
        ax.set_yscale("log", base=10)

    adjusted = adjust_text(
        texts,
        avoid_self=False,
        pull_threshold=0.000001,
        pull_force=0.1,
        force_static=0.001,
        force_explode=0.5,
        arrowprops=dict(arrowstyle="-", color="white"),
    )
    # adjust_text returns (texts, patches) or just texts depending on version
    if isinstance(adjusted, tuple):
        adj_texts, adj_patches = adjusted
    else:
        adj_texts, adj_patches = adjusted, []
    for item in adj_texts:
        item.set_zorder(102)
    for item in adj_patches:
        item.set_zorder(101)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=1200)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()


# ---------------------------------------------------------------------------
# Optimum overlap (Jaccard index across phase / quality transitions)
# ---------------------------------------------------------------------------


def optimum_share_table(
    df: pd.DataFrame,
    agent_name: str,
    hp_list: list[str],
    by_col: str = "phase_num",
    performance_threshold: float = 0.95,
    sorter: Callable[[list], list] = sorted,
    grid_length: int = 100,
) -> pd.Series:
    """Jaccard index of optimal HP regions between consecutive transitions.

    For each pair of consecutive values in ``by_col`` (e.g. phase 1→2, 2→3),
    fits a GP, selects grid points whose normalised prediction exceeds
    ``performance_threshold``, and returns intersection/union.

    Returns:
        Series keyed ``"1->2"``, ``"2->3"``, … with float Jaccard values.
    """
    clipped_df = df.copy()
    clipped_df["mean_normalized_goal_distance_return"] = clipped_df[
        "mean_normalized_goal_distance_return"
    ].clip(0, 1)

    # Sample model to establish HP axis names
    sample_model = fit_model(
        clipped_df, "mean_normalized_goal_distance_return", hp_list
    )
    hp_name_x = sample_model.hp_names[0].removeprefix("hp.")
    hp_name_y = sample_model.hp_names[1].removeprefix("hp.")
    sample_model.fit()

    # Prediction grid
    X, Y = np.meshgrid(np.linspace(0, 1, grid_length), np.linspace(0, 1, grid_length))
    points = np.vstack([X.ravel(), Y.ravel()]).T

    def _select(group_df: pd.DataFrame) -> np.ndarray:
        m = fit_model(
            group_df.copy().reset_index(),
            "mean_normalized_goal_distance_return",
            hp_list,
        )
        m.fit()
        assert (
            m.hp_names[0].removeprefix("hp.") == hp_name_x
            and m.hp_names[1].removeprefix("hp.") == hp_name_y
        ), "HP names changed across groups — cannot compare grids"
        Z = m.get_middle(points).clip(0, 1)
        return ((Z / Z.max()) > performance_threshold).squeeze()

    optimal = clipped_df.groupby(by_col).apply(_select)
    col_sorted = sorter(optimal.index.tolist())

    transitions: dict[str, float] = {}
    for c1, c2 in zip(col_sorted[:-1], col_sorted[1:]):
        union = int(np.sum(optimal.loc[c1] | optimal.loc[c2]))
        intersect = int(np.sum(optimal.loc[c1] & optimal.loc[c2]))
        transitions[f"{c1}->{c2}"] = intersect / union if union > 0 else float("nan")
    return pd.Series(transitions, name="transition")


def plot_optimum_overlap(
    results_df: pd.DataFrame,
    hp_list: list[str],
    experiment_folder: Path,
    grid_plots_folder: Path,
    performance_threshold: float = 0.90,
) -> None:
    """Generate optimum-overlap line plot (grid_plots) and LaTeX table (tables/).

    Args:
        results_df: Combined results DataFrame for the full zip (all agents/datasets).
        hp_list: Two HP column names, e.g. ``["hp.lr", "hp.alpha"]``.
        experiment_folder: Per-experiment output folder (used to derive tables path).
        grid_plots_folder: Root ``grid_plots/`` folder.
        performance_threshold: Threshold for optimality mask passed to
            ``optimum_share_table``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = results_df.groupby(["hp.agent_name", "dataset"]).apply(
            lambda g: optimum_share_table(
                g,
                g["hp.agent_name"].iloc[0],
                hp_list,
                by_col="phase_num",
                performance_threshold=performance_threshold,
            )
        )
    if isinstance(raw, pd.Series):
        raw = raw.unstack()

    long_df = raw.reset_index().melt(
        id_vars=["hp.agent_name", "dataset"],
        var_name="transition",
        value_name="Jaccard Index of Optimum",
    )
    long_df["hp.agent_name"] = long_df["hp.agent_name"].str.upper().astype("category")

    # Line plot → grid_plots/optimum-overlap/
    out_dir = grid_plots_folder / "optimum-overlap"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    sns.lineplot(
        data=long_df,
        x="transition",
        y="Jaccard Index of Optimum",
        hue="hp.agent_name",
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Phase Transition")
    ax.set_ylabel("Overlap of Optimum")
    ax.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig(out_dir / "optimum-overlap-phases.png", dpi=1200)
    plt.close()

    # LaTeX table → tables/{experiment_name}/
    tables_dir = experiment_folder.parent / "tables" / experiment_folder.name
    tables_dir.mkdir(parents=True, exist_ok=True)
    latex_df = (
        raw.swaplevel(0, 1)
        .rename_axis(index={"hp.agent_name": "Algorithm", "dataset": "Setting"})
        .sort_index()
    )
    with open(tables_dir / "optimum-overlap-phases.tex", "w") as fh:
        fh.write(
            latex_df.to_latex(
                index=True,
                caption="Optimum Overlap across Phase Transitions",
                label="tab:optimum-overlap-phases",
                float_format="%.3f",
                bold_rows=False,
                longtable=False,
            )
        )


# ---------------------------------------------------------------------------
# Per-experiment orchestrator
# ---------------------------------------------------------------------------


def plot_mobility_for_experiment(
    results_df: pd.DataFrame,
    hp_list: list[str],
    experiment_folder: Path,
) -> None:
    """Phase-based mobility plots for one experiment (one agent + dataset combination).

    Saves plots to ``experiment_folder/mobility/{threshold_pct}/``.
    Cross-quality mobility (antmaze explore/navigate sweep) is handled separately
    at the combined level via ``plot_optimum_overlap``.
    """
    mobility_folder = experiment_folder / "mobility"
    mobility_folder.mkdir(exist_ok=True, parents=True)

    results_df = results_df.copy()
    results_df["dataset_condensed"] = results_df["dataset"].apply(
        lambda x: x.split(",")[0] if len(set(x.split(","))) == 1 else x
    )

    for (agent_name, dataset), group in results_df.groupby(
        ["hp.agent_name", "dataset_condensed"]
    ):
        for threshold in [0.95, 0.90, 0.85, 0.80]:
            threshold_pct = int(threshold * 100)
            # Filename: agent has no hyphens, so first hyphen after "mobility-" is safe
            output_path = (
                mobility_folder
                / str(threshold_pct)
                / f"mobility-{agent_name}-{dataset}.png"
            )
            try:
                mobility_plot(
                    group,
                    str(agent_name),
                    hp_list,
                    output_path,
                    by_col="phase_num",
                    performance_threshold=threshold,
                )
            except Exception as exc:
                print(
                    f"Skipping mobility plot {agent_name}/{dataset} "
                    f"(threshold={threshold_pct}%): {exc}"
                )


# ---------------------------------------------------------------------------
# Grid overview for mobility plots
# ---------------------------------------------------------------------------


def grid_mobility_plot(plots_folder: Path) -> None:
    """Assemble per-experiment mobility PNGs into grid overviews.

    Scans ``plots_folder/*/mobility/{threshold}/mobility-*.png`` and creates
    one grid per (threshold, agent) saved to ``plots_folder/grid_plots/mobility/``.
    """
    grid_dir = plots_folder / "grid_plots" / "mobility"

    rows: list[dict] = []
    for exp_dir in sorted(plots_folder.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == "grid_plots":
            continue
        mob_dir = exp_dir / "mobility"
        if not mob_dir.exists():
            continue
        for threshold_dir in sorted(mob_dir.iterdir()):
            if not threshold_dir.is_dir() or not threshold_dir.name.isdigit():
                continue
            threshold = threshold_dir.name
            for png in sorted(threshold_dir.glob("mobility-*.png")):
                # mobility-{agent}-{dataset...}.png
                # Agent names are single lowercase words (no hyphens)
                rest = png.stem[len("mobility-") :]  # e.g. "hiql-antmaze-medium-..."
                hyphen_idx = rest.index("-")
                agent = rest[:hyphen_idx]
                dataset = rest[hyphen_idx + 1 :]
                rows.append(
                    {
                        "threshold": threshold,
                        "agent": agent,
                        "dataset": dataset,
                        "path": str(png),
                    }
                )

    if not rows:
        return

    grid_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    for (threshold, agent), group in df.groupby(["threshold", "agent"]):
        group = group.sort_values("dataset").reset_index(drop=True)
        n = len(group)
        if n < 2:
            continue
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
        grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.15)
        for ax, (_, row) in zip(grid, group.iterrows()):
            ax.imshow(Image.open(row["path"]))
            ax.axis("off")
            ax.set_title(row["dataset"], fontsize=6)
        # Hide unused axes
        for ax in list(grid)[n:]:
            ax.axis("off")
        plt.suptitle(f"{agent.upper()} — mobility (≥{threshold}% optimal)", fontsize=10)
        plt.savefig(
            grid_dir / f"mobility-grid-{agent}-{threshold}.png", bbox_inches="tight"
        )
        plt.close()
