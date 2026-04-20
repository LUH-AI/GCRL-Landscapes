import argparse
import ast
from gcrl_landscapes.util.data import (
    ResultsPerStep,
    PhaseResult,
    phase_results_to_pandas,
    training_logs_to_pandas,
    read_results_from_zip,
)
from gcrl_landscapes.plots.triple_gp import estimate_model_fit, TripleGPModel
from pathlib import Path
from gcrl_landscapes.plots.triple_gp import create_contour_plot
from gcrl_landscapes.util.eval import fit_model
from gcrl_landscapes.util.data import get_best_config_row
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Any
import re
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
import os
from itertools import product
from .common import (
    CVAR_CONFIDENCE_LEVELS,
    map_labels,
    compute_additional_information,
    resolve_alpha_sync,
    calculate_regret_for_experiment,
)
import toml
import traceback
import zipfile
import multiprocessing
from functools import partial
from itertools import combinations
from sklearn.metrics import mean_absolute_error, max_error, mean_squared_error
from sklearn.model_selection import ShuffleSplit
from typing import Callable


# fmt: off
def plot_landscape(
    phase: int,
    model: TripleGPModel,
    y_col: str,
    hp_names: list[str],
    agent_name: str,
    output_folder: Path,
    y_label: str | None,
    y_transform: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda y_pred, y: y_pred,
    discrete_levels: np.ndarray | None = None,
    last_phase_best_config: pd.DataFrame | None = None,
    actor_loss: str | None = None,
):
    """IGPR plot of given data. Fits gaussian process itself

    Args:
        phase: which phase is this. used for plot title
        phase_result: results for this phase
        y_col: column which we want to plot landscape for
        hp_names: names of all hyperparameter columns (to create groups from dataframe)
        output_folder: where to save plot to
        y_label: How y should be labeled
    """
# fmt: on
    plot_filename_base = f"({'_'.join(hp_names)})-{y_label.lower().replace(' ', '_').replace(')', '').replace('(', '') if y_label else y_col}-{phase}"
    # One direct plot
    create_contour_plot(
        model,
        x_dim=0,
        y_dim=1,
        z_dim=y_label if y_label else y_col,
        bounds=[0, 1],
        filename=output_folder / f"igpr-{plot_filename_base}.pdf",
        dim_label_mapping=map_labels,
        agent_name=agent_name,
        z_transform=y_transform,
        discrete_levels=discrete_levels,
        last_phase_best_config=last_phase_best_config,
        actor_loss=actor_loss,
    )

    # Create plot without using gaussian processes
    plt.figure()
    x_scaled = model.x
    y_scaled = model.y_iqm

    x0i_scaled, x1i_scaled = np.meshgrid(
        np.linspace(x_scaled[:, 0].min(), x_scaled[:, 0].max(), 1000),
        np.linspace(x_scaled[:, 1].min(), x_scaled[:, 1].max(), 1000),
    )
    yi_scaled = griddata(
        (x_scaled[:, 0], x_scaled[:, 1]),
        y_scaled,
        (x0i_scaled, x1i_scaled),
        method="nearest",
    )
    x0i = x0i_scaled * model.x_normalizing_factor[0] + model.x_normalizing_offset[0]
    x1i = x1i_scaled * model.x_normalizing_factor[1] + model.x_normalizing_offset[1]
    yi = model._unscale_y(yi_scaled).squeeze()

    c = plt.contourf(
        x0i,
        x1i,
        y_transform(yi, model._unscale_y(model.y_iqm)),
        cmap="rocket",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(c, label=y_label if y_label else y_col)
    plt.savefig(
        output_folder / f"nearest-{plot_filename_base}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_eval_curve(
    phase_result: pd.DataFrame,
    y_col: str,
    output_folder: Path,
    y_label: str | None,
):
    exploded_phase_result = phase_result[[y_col, "config_index", "eval_step"]].explode(
        y_col
    )  # type: ignore

    fig, ax = plt.subplots(figsize=[4, 3])
    ax = sns.lineplot(
        data=exploded_phase_result,  # type: ignore
        x="eval_step",
        y=y_col,
        errorbar=("ci", 95),
    )
    # plt.title(f"{y_label if y_label else y_col}", fontsize=18)
    plt.ylabel(y_label if y_label else y_col)
    plt.xlabel("Training Step")
    ax.set_ylim(0, 1)
    plt.savefig(
        output_folder / f"eval-{y_col}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_return_distribution(
    phase: int,
    phase_result: pd.DataFrame,
    y_col: str,
    output_folder: Path,
    output_per_config_folder: Path,
    y_label: str | None,
):
    RETURN_LIMITS = (-1, 1)
    exploded_phase_result = phase_result[[y_col, "config_index"]].explode(y_col)  # type: ignore
    exploded_phase_result[f"clipped_{y_col}"] = exploded_phase_result[y_col].clip(
        lower=RETURN_LIMITS[0], upper=RETURN_LIMITS[1]
    )
    # per configuration
    for config_index, group in exploded_phase_result.groupby("config_index"):
        _ = plt.figure()
        ax = sns.histplot(
            data=group,  # type: ignore
            x=f"clipped_{y_col}",
            stat="probability",
            bins="sturges",
        )
        ax.set_xlim(*RETURN_LIMITS)
        ax.set_ylim(0, 1)
        # plt.title(f"{y_label if y_label else y_col}", fontsize=18)
        plt.savefig(
            output_per_config_folder
            / f"returndistribution-{y_col}-config_{config_index}-{phase}.pdf",
            bbox_inches="tight",
        )
        plt.close()
    ## configuration marginalized
    _ = plt.figure()
    ax = sns.histplot(
        data=exploded_phase_result,  # type: ignore
        x=f"clipped_{y_col}",
        stat="probability",
        bins="sturges",
    )
    ax.set_xlim(*RETURN_LIMITS)
    ax.set_ylim(0, 1)
    # plt.title(f"{y_label if y_label else y_col}", fontsize=18)
    plt.savefig(
        output_folder / f"returndistribution-{y_col}-{phase}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_gp_fit(
    phase: int,
    phase_result: pd.DataFrame,
    y_col: str,
    hp_names: list[str],
    output_folder: Path,
    y_label: str | None,
    n_samples: int = 20,
    n_splits: int = 10,
):
    gp_fit_dfs = []
    # Sample n configurations from results and fit model
    for n in np.unique(
        np.rint(
            np.linspace(2, len(phase_result.groupby(hp_names)) - 1, n_samples)
        ).astype(int)
    ):
        model = fit_model(phase_result, y_col, hp_names)
        data = estimate_model_fit(
            X=model.x,
            y=model.y_iqm,
            splitter=ShuffleSplit(n_splits=n_splits, train_size=n),
            y_scale=model.y_normalizing_factor,
            metrics=[mean_squared_error, mean_absolute_error, max_error],
        )
        data["n"] = n
        gp_fit_dfs.append(data)
    gp_fit_df = pd.concat(gp_fit_dfs).reset_index(drop=True)

    fig, ax = plt.subplots()
    sns.lineplot(
        data=gp_fit_df, x="n", y="mean_absolute_error", ax=ax, label="Mean Error"
    )
    sns.lineplot(data=gp_fit_df, x="n", y="max_error", ax=ax, label="Max Error")
    ax.set_ylim(0.01, 1)
    ax.set_yscale("log")
    ax.set_xlabel("Number of Configurations")
    ax.set_ylabel("Error")
    fig.savefig(
        output_folder / f"gp_fit-({'_'.join(hp_names)})-{y_col}-{phase}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_regret_curve(
    experiment_result: pd.DataFrame,
    regret_col: str,
    base_col: str,
    hp_names: list[str],
    output_folder: Path,
    regret_label: str | None,
                ) -> None:
    regret_df = calculate_regret_for_experiment(experiment_result, regret_col, base_col)
    regret_df = regret_df[regret_df.columns[regret_df.columns.str.startswith("regret_phase_")]].reset_index().rename(columns={"phase": "pick_phase"})

    regret_df_long = regret_df.melt(id_vars="pick_phase", var_name="phase", value_name="regret")
    ordered_pick_phases = np.sort(regret_df_long["pick_phase"].unique())  # type: ignore
    regret_df_long["pick_phase"] = regret_df_long["pick_phase"].map(
        dict(zip(ordered_pick_phases, range(1, len(ordered_pick_phases) + 1)))  # type: ignore
    )
    regret_df_long["phase"] = regret_df_long["phase"].str.replace("regret_phase_", "")

    fig, ax = plt.subplots()
    sns.lineplot(data=regret_df_long, x="phase", y="regret", hue="pick_phase", ax=ax)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(output_folder / f"regret_({'_'.join(hp_names)})-{regret_label.lower().replace(' ', '_').replace(')', '').replace('(', '') if regret_label else regret_col}.pdf")
    plt.close()


def _build_adv_long_df(
    training_df: pd.DataFrame,
    adv_cols: list[str],
    end_only: bool = True,
) -> pd.DataFrame:
    """Build long-format DataFrame: one row per (config, seed, batch sample).

    Parses stringified list columns in ``adv_cols`` to numpy arrays, optionally
    filters to the final ``eval_step`` per (agent, dataset), then explodes into
    one row per sample with ``advantage`` and ``weight = exp(alpha * advantage)``.
    """
    df = training_df.copy()
    for col in adv_cols:
        df[col] = df[col].apply(
            lambda s: np.array(ast.literal_eval(s), dtype=np.float32)
            if isinstance(s, str)
            else (s if isinstance(s, np.ndarray) else None)
        )

    if end_only:
        df = df[
            df["eval_step"]
            == df.groupby(["hp.agent_name", "dataset"])["eval_step"].transform("max")
        ].copy()

    rows = []
    for _, row in df.iterrows():
        for adv_col in adv_cols:
            adv_arr = row.get(adv_col)
            if adv_arr is None or not isinstance(adv_arr, np.ndarray):
                continue
            alpha = float(row.get("hp.alpha", np.nan))
            df_tmp = pd.DataFrame(
                {
                    "hp.agent_name": row["hp.agent_name"],
                    "dataset": row["dataset"],
                    "config_index": row["config_index"],
                    "seed": row["seed"],
                    "eval_step": row["eval_step"],
                    "actor": adv_col,
                    "advantage": adv_arr,
                    "alpha": alpha,
                }
            )
            df_tmp["weight"] = np.exp(alpha * adv_arr.astype(np.float64))
            rows.append(df_tmp)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_advantage_distributions(
    training_df: pd.DataFrame,
    output_folder: Path,
    results_df: pd.DataFrame | None = None,
) -> None:
    """KDE plots of advantage distributions per (agent, actor).

    Saves end-of-training plots to ``output_folder/advantages/``.
    If ``results_df`` is provided (and has a ``phase_num`` column), also saves
    per-phase plots as ``adv_dist_{agent}_{actor}_phase{n}.png``.

    Args:
        training_df: Training-log DataFrame; must contain ``advantage/*`` columns
            as stringified arrays, ``hp.agent_name``, ``dataset``, ``eval_step``.
        output_folder: Per-experiment output folder.
        results_df: Optional processed results DataFrame used to map eval_step →
            phase_num for per-phase plots.
    """
    adv_cols = [c for c in training_df.columns if c.startswith("advantage/")]
    if not adv_cols:
        return

    adv_folder = output_folder / "advantages"
    adv_folder.mkdir(exist_ok=True, parents=True)

    # Build end-of-training long-format df
    adv_long = _build_adv_long_df(training_df, adv_cols, end_only=True)
    if adv_long.empty:
        return

    # Global x-limits (1st / 99th percentile)
    adv_vals = adv_long["advantage"].dropna()
    xlim = (float(adv_vals.quantile(0.01)), float(adv_vals.quantile(0.99)))

    for (agent_name, actor), grp in adv_long.groupby(["hp.agent_name", "actor"]):
        fig, ax = plt.subplots(figsize=(5, 3))
        for _, cfg_grp in grp.groupby("config_index"):
            sns.kdeplot(
                bw_adjust=0.5, data=cfg_grp, x="advantage", ax=ax,
                alpha=0.3, linewidth=0.8, color="steelblue",
            )
        sns.kdeplot(
            bw_adjust=0.5, data=grp, x="advantage", ax=ax,
            color="black", linewidth=2, label="overall",
        )
        ax.axvline(0, color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_xlim(xlim)
        ax.set_title(f"{agent_name.upper()} — advantage distribution")
        ax.set_xlabel("Advantage")
        ax.set_ylabel("Density")
        plt.tight_layout()
        fname = adv_folder / f"adv_dist_{agent_name}_{actor.replace('/', '_')}.png"
        plt.savefig(fname, dpi=300)
        plt.close()

    # Per-phase plots (if results_df provides phase mapping)
    if results_df is not None and "phase_num" in results_df.columns:
        phase_map = (
            results_df[
                ["hp.agent_name", "dataset", "config_index", "eval_step", "phase_num"]
            ]
            .drop_duplicates()
        )
        adv_all = _build_adv_long_df(training_df, adv_cols, end_only=False)
        if adv_all.empty:
            return
        adv_all = adv_all.merge(
            phase_map,
            on=["hp.agent_name", "dataset", "config_index", "eval_step"],
            how="left",
        )
        for (agent_name, actor, phase_num), grp in adv_all.dropna(
            subset=["phase_num"]
        ).groupby(["hp.agent_name", "actor", "phase_num"]):
            fig, ax = plt.subplots(figsize=(5, 3))
            for _, cfg_grp in grp.groupby("config_index"):
                sns.kdeplot(
                    bw_adjust=0.5, data=cfg_grp, x="advantage", ax=ax,
                    alpha=0.3, linewidth=0.8, color="steelblue",
                )
            sns.kdeplot(
                bw_adjust=0.5, data=grp, x="advantage", ax=ax,
                color="black", linewidth=2, label="overall",
            )
            ax.axvline(0, color="red", linestyle="--", alpha=0.6, linewidth=1)
            ax.set_xlim(xlim)
            ax.set_title(f"{agent_name.upper()} — phase {int(phase_num)} advantage")
            ax.set_xlabel("Advantage")
            ax.set_ylabel("Density")
            plt.tight_layout()
            fname = (
                adv_folder
                / f"adv_dist_{agent_name}_{actor.replace('/', '_')}_phase{int(phase_num)}.png"
            )
            plt.savefig(fname, dpi=300)
            plt.close()


def plot_awr_weights(
    training_df: pd.DataFrame,
    output_folder: Path,
) -> None:
    """Log-scale KDE plots of AWR weight distributions per (agent, actor).

    Saves to ``output_folder/advantages/weight_dist_{agent}_{actor}.png`` and a
    clipped variant at 100.
    """
    adv_cols = [c for c in training_df.columns if c.startswith("advantage/")]
    if not adv_cols:
        return

    adv_folder = output_folder / "advantages"
    adv_folder.mkdir(exist_ok=True, parents=True)

    adv_long = _build_adv_long_df(training_df, adv_cols, end_only=True)
    if adv_long.empty:
        return

    # Global log-scale x-limits
    w_vals = adv_long["weight"].apply(
        lambda x: x if np.isfinite(x) and x > 0 else np.nan
    ).dropna()
    if w_vals.empty:
        return
    w_xlim = (
        max(float(w_vals.quantile(0.01)), 1e-10),
        float(w_vals.quantile(0.99)) if np.isfinite(w_vals.quantile(0.99)) else 1e30,
    )

    for (agent_name, actor), grp in adv_long.groupby(["hp.agent_name", "actor"]):
        finite_mask = grp["weight"].apply(np.isfinite)
        grp_valid = grp[finite_mask]
        if grp_valid.empty:
            continue

        alpha_median = float(grp["alpha"].median())

        # Log-scale weight distribution
        fig, ax = plt.subplots(figsize=(5, 3))
        for _, cfg_grp in grp_valid.groupby("config_index"):
            cfg_pos = cfg_grp[cfg_grp["weight"] > 0]
            if cfg_pos.empty:
                continue
            sns.kdeplot(
                bw_adjust=0.5, data=cfg_pos, x="weight", ax=ax,
                alpha=0.3, linewidth=0.8, color="darkorange", log_scale=True,
            )
        overall_pos = grp_valid[grp_valid["weight"] > 0]
        sns.kdeplot(
            bw_adjust=0.5, data=overall_pos, x="weight", ax=ax,
            color="black", linewidth=2, label="overall", log_scale=True,
        )
        ax.set_title(
            f"{agent_name.upper()} — AWR weight distribution (median α={alpha_median:.2f})"
        )
        ax.set_xlabel("AWR Weight  exp(α · adv)")
        ax.set_ylabel("Density")
        ax.set_xlim(w_xlim)
        plt.tight_layout()
        fname = adv_folder / f"weight_dist_{agent_name}_{actor.replace('/', '_')}.png"
        plt.savefig(fname, dpi=300)
        plt.close()

        # Clipped at 100
        grp_clipped = grp_valid.copy()
        grp_clipped["weight_clipped"] = grp_clipped["weight"].clip(upper=100)
        fig, ax = plt.subplots(figsize=(5, 3))
        for _, cfg_grp in grp_clipped.groupby("config_index"):
            sns.kdeplot(
                bw_adjust=0.5, data=cfg_grp, x="weight_clipped", ax=ax,
                alpha=0.3, linewidth=0.8, color="darkorange", clip=(0, 100),
            )
        sns.kdeplot(
            bw_adjust=0.5, data=grp_clipped, x="weight_clipped", ax=ax,
            color="black", linewidth=2, label="overall", clip=(0, 100),
        )
        ax.set_xlim(0, 100)
        ax.set_ylim(top=1)
        ax.axvline(100, color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_title(
            f"{agent_name.upper()} — AWR weights clipped at 100 (median α={alpha_median:.2f})"
        )
        ax.set_xlabel("AWR Weight  exp(α · adv), clipped at 100")
        ax.set_ylabel("Density")
        plt.tight_layout()
        fname = (
            adv_folder
            / f"weight_dist_clipped_{agent_name}_{actor.replace('/', '_')}.png"
        )
        plt.savefig(fname, dpi=300)
        plt.close()


def grid_advantage_plot(plots_folder: Path) -> None:
    """Assemble per-experiment advantage KDE plots into grid overviews.

    Scans ``plots_folder/*/advantages/adv_dist_*.png`` and creates one grid per
    actor, saved to ``plots_folder/grid_plots/advantages/``.
    End-of-training and per-phase plots are assembled separately.
    """
    grid_dir = plots_folder / "grid_plots" / "advantages"

    rows: list[dict] = []
    for exp_dir in sorted(plots_folder.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == "grid_plots":
            continue
        adv_dir = exp_dir / "advantages"
        if not adv_dir.exists():
            continue
        for png in sorted(adv_dir.glob("adv_dist_*.png")):
            # adv_dist_{agent}_{actor_sanitized}[_phase{n}].png
            stem = png.stem[len("adv_dist_"):]
            # Last token: phase{n} or actor part
            phase_match = re.search(r"_phase(\d+)$", stem)
            phase = int(phase_match.group(1)) if phase_match else None
            stem_no_phase = stem[: phase_match.start()] if phase_match else stem
            # Agent name is first token (no underscores)
            parts = stem_no_phase.split("_", 1)
            if len(parts) < 2:
                continue
            agent, actor_sanitized = parts
            rows.append(
                {
                    "agent": agent,
                    "actor": actor_sanitized,
                    "phase": phase,
                    "dataset": exp_dir.name,
                    "path": str(png),
                }
            )

    if not rows:
        return

    grid_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    for (actor, phase), group in df.groupby(
        ["actor", "phase"], dropna=False
    ):
        group = group.sort_values(["agent", "dataset"]).reset_index(drop=True)
        n = len(group)
        if n < 2:
            continue
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig = plt.figure(figsize=(6 * ncols, 4 * nrows))
        grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.15)
        for ax, (_, row) in zip(grid, group.iterrows()):
            ax.imshow(Image.open(row["path"]))
            ax.axis("off")
            ax.set_title(f"{row['agent']} / {row['dataset']}", fontsize=6)
        for ax in list(grid)[n:]:
            ax.axis("off")
        phase_label = f"phase {int(phase)}" if phase is not None else "end-of-training"
        plt.suptitle(f"Advantage distributions — {actor} ({phase_label})", fontsize=9)
        safe_actor = actor.replace("/", "_")
        phase_str = f"_phase{int(phase)}" if phase is not None else ""
        plt.savefig(
            grid_dir / f"advantage-grid-{safe_actor}{phase_str}.png",
            bbox_inches="tight",
        )
        plt.close()


def plot_eval_results(
    results_pandas: pd.DataFrame,
    output_folder: Path,
    run_info: dict[str, Any],
    plot_return_distributions: bool = False,
    plot_eval_curves: bool = False,
    plot_gp_fits: bool = False,
    plot_regret: bool = False,
    plot_landscapes: bool = True,
    plot_mobility: bool = True,
    plot_advantages: bool = False,
    training_df: pd.DataFrame | None = None,
):
    """Main plotting Code to generate the landscapes

    Args:
        results_pandas: pandas dataframe containing all results (all phases) for one experiment
        output_folder: folder to save plots in
        run_info: info about the run/setup
    """
    if plot_eval_curves:
        plot_eval_curve(
            results_pandas,
            "success",
            output_folder,
            "Success Rate",
        )

    results_pandas = resolve_alpha_sync(results_pandas, run_info["arguments"]["hyperparameters"])
    results_pandas = compute_additional_information(results_pandas)

    per_config_folder = output_folder / "per_config"
    per_config_folder.mkdir(exist_ok=True)

    hp_full_list = results_pandas.columns[results_pandas.columns.str.startswith("hp.")]
    hp_list = [f"hp.{hp_name}" for hp_name in run_info["arguments"]["hyperparameters"]]
    assert all(hp in hp_full_list for hp in hp_list)
    if not len(hp_list) == 2:
        raise NotImplementedError("Currently plotting landscapes is only implemented for runs with 2 hyperparameters")


    # Plots using full experiment results
    regret_targets = [
        ("mean_normalized_goal_distance_return_regret", "mean_normalized_goal_distance_return", "Goal Distance Regret"),
        ("mean_normalized_goal_distance_return_normalized_regret", "mean_normalized_goal_distance_return", "Normalized Goal Distance Regret"),
        ("success_regret", "success", "Success Regret"),
        ("success_normalized_regret", "success", "Normalized Success Regret"),
    ]
    if plot_regret:
        for regret_column, base_column, regret_label in regret_targets:
            plot_regret_curve(results_pandas, regret_column, base_column, hp_list, output_folder, regret_label)


    # Plots that are local to a phase
    phases = sorted(results_pandas["phase"].unique().tolist())
    phase_results = [
        (
            phase_idx + 1,
            results_pandas[
                (results_pandas["eval_step"] == phase)
                & (results_pandas["phase"] == phase)
            ],
        )
        for phase_idx, phase in enumerate(phases)
    ]

    best_config_per_phase: dict[int, pd.DataFrame] = {}
    phase_result: pd.DataFrame
    for phase, phase_result in phase_results:  # type: ignore
        per_config_phase_folder = per_config_folder / f"phase_{phase}"
        per_config_phase_folder.mkdir(exist_ok=True)

        if plot_landscapes:
            e_optimal_bins = np.array([0.0, 0.5, 0.8, 0.9, 0.95, 1.0])
            landscape_pairs = [
                ("success", "Success Rate", {}),
                (
                    "success",
                    "Success Epsilon Optimality Bins",
                    # Make this stepped, so that it is easier to interpret
                    {
                        "y_transform": lambda y_pred, y: y_pred / np.max(y_pred),
                        "discrete_levels": e_optimal_bins,
                    }
                ),
                (
                    "mean_normalized_goal_distance_return",
                    "Normalized Goal Distance Return",
                    {}
                ),
                (
                    "mean_normalized_goal_distance_return",
                    "Normalized Goal Distance Return Epsilon Optimality Bins",
                    {
                        "y_transform": lambda y_pred, y: y_pred / np.max(y_pred),
                        "discrete_levels": e_optimal_bins,
                    }
                ),
                (
                    "mean_normalized_goal_distance_return",
                    "Normalized Goal Distance Return Epsilon Optimality",
                    {
                        "y_transform": lambda y_pred, y: y_pred / np.max(y_pred),
                    }
                ),
                (
                    "disp_normalized_goal_distance_score",
                    "Dispersion score of normalized goal distance return",
                    {}
                ),
                (
                    "mean_normalized_goal_distance_return_regret",
                    "Normalized Goal Distance Return Regret",
                    {}
                ),
                (
                    "mean_normalized_goal_distance_return_regret_cummean",
                    "Normalized Goal Distance Return Regret Cumulative",
                    {}
                ),
                (
                    "success_regret",
                    "Normalized Goal Distance Return Regret",
                    {}
                ),
                (
                    "success_regret_cummean",
                    "Normalized Goal Distance Return Regret Cumulative",
                    {}
                ),
            ] + [  # Gather all CVaR confidence levels
                (
                    f"cvar{confidence_level}_normalized_goal_distance_return",
                    f"CVaR ({confidence_level}%)of normalized goal distance return",
                    {}
                )
                for confidence_level in CVAR_CONFIDENCE_LEVELS
            ]
            for col, title, kwargs in landscape_pairs:
                last_phase_best_config = best_config_per_phase[phase-1] if phase > 1 else None
                model = fit_model(phase_result, col, hp_list)
                best_config_per_phase[phase] = get_best_config_row(phase_result)
                plot_landscape(
                    phase,
                    model,
                    col,
                    hp_list,
                    phase_result["hp.agent_name"].iloc[0],
                    output_folder,
                    title,
                    last_phase_best_config=last_phase_best_config,
                    actor_loss=phase_result["hp.actor_loss"].iloc[0] if "hp.actor_loss" in phase_result.columns else None,
                    **kwargs
                )

        if plot_return_distributions:
            plot_return_distribution(
                phase,
                phase_result,
                "mean_normalized_goal_distance_return",
                output_folder,
                per_config_phase_folder,
                "Normalized Goal Distance Return Distribution",
            )
        if plot_gp_fits:
            for hp_pair in combinations(hp_list, 2):
                plot_gp_fit(
                    phase,
                    phase_result,
                    "mean_normalized_goal_distance_return",
                    list(hp_pair),
                    output_folder,
                    "Normalized Goal Distance Return",
                )

    # Mobility plots (phase-based KDE of optimal HP regions)
    if plot_mobility:
        from gcrl_landscapes.evaluation.landscapes import plot_mobility_for_experiment
        try:
            plot_mobility_for_experiment(results_pandas, hp_list, output_folder)
        except Exception:
            print(
                f"ERROR in mobility plots for {output_folder.name}:\n{traceback.format_exc()}",
                flush=True,
            )

    # Advantage distribution and AWR weight plots
    if plot_advantages and training_df is not None:
        try:
            plot_advantage_distributions(training_df, output_folder, results_pandas)
            plot_awr_weights(training_df, output_folder)
        except Exception:
            print(
                f"ERROR in advantage plots for {output_folder.name}:\n{traceback.format_exc()}",
                flush=True,
            )


def plot_train_results(
    results_pandas: pd.DataFrame,
    output_folder: Path,
    run_info: dict[str, Any],
    plot_return_distributions: bool = False,
    plot_eval_curves: bool = False,
    plot_gp_fits: bool = False,
    plot_regret: bool = False,
):
    hp_full_list = results_pandas.columns[results_pandas.columns.str.startswith("hp.")]
    hp_list = [f"hp.{hp_name}" for hp_name in run_info["arguments"]["hyperparameters"]]
    assert all(hp in hp_full_list for hp in hp_list)

    phases = sorted(results_pandas["phase"].unique().tolist())
    phase_results = [
        (
            phase,
            results_pandas[
                (results_pandas["eval_step"] == phase)
                & (results_pandas["phase"] == phase)
            ],
        )
        for phase in phases
    ]
    for hp_pair in combinations(hp_list, 2):
        for phase, phase_result in phase_results:
            model = fit_model(phase_result, "grad/value_cosine_similarity_mean", list(hp_pair))
            plot_landscape(
                phase,
                model,
                "grad/value_cosine_similarity_mean",
                list(hp_pair),
                phase_result["hp.agent_name"].iloc[0],
                output_folder,
                "Gradient-Value Cosine Similarity",
            )




def grid_plot(
    imgpath_dataframe: pd.DataFrame,
    output_folder: Path,
    title: str,
    grid_columns: tuple[str, str],
) -> None:
    """Create a grid plot from all experiments.

    Args:
        imgpath_dataframe: Dataframe which maps combinatons of the grid columns to a plotpath
        output_folder: folder to save plot to
        title: title of the plot
        grid_columns: columns to use for grid
    """
    indexed_imgpath_series = imgpath_dataframe.set_index(
        list(grid_columns)
    ).sort_index()

    fig = plt.figure(figsize=(16, 16))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(
            len(indexed_imgpath_series.index.get_level_values(0).unique()),
            # get index 0 of first level and count unique phases. Pandas doesn't properly support that
            len(
                indexed_imgpath_series.iloc[
                    indexed_imgpath_series.index.get_loc(
                        indexed_imgpath_series.index.levels[0][0]
                    )
                ]
                .index.get_level_values(1)
                .unique()
            ),
        ),
        axes_pad=0.1,
    )
    plt.axis("off")

    for ax, ((dataset, phase), imgpath) in zip(
        grid, indexed_imgpath_series["path"].items()
    ):
        ax.imshow(Image.open(imgpath))
        ax.axis("off")
        ax.set_title(f"{dataset}-{phase}")
        ax.title.set_size(7)
    fig.savefig(output_folder / f"{title}.pdf", bbox_inches="tight")
    plt.close()
    return


def plot_parallel_wrapper(
    plots_folder: Path,
    arg: tuple[str, tuple[dict, pd.DataFrame, pd.DataFrame]],
    plot_return_distributions: bool = False,
    plot_eval_curves: bool = False,
    plot_gp_fits: bool = False,
    plot_regret: bool = False,
    plot_landscapes: bool = True,
    plot_mobility: bool = True,
    plot_advantages: bool = False,
):
    prefix, (run_info, results_df, train_log_df) = arg
    run_match = re.match(r"^logs[^/]*/([^/]*)/?", prefix)
    if not run_match:
        raise ValueError("Naming inside of zipfile not as expected.")
    run_name = run_match.group(1)

    folder = plots_folder / run_name
    folder.mkdir(exist_ok=True, parents=True)
    print(f"Plotting '{prefix}'")

    # Ensure training_df has dataset and hp.agent_name columns for advantage plotting
    training_df: pd.DataFrame | None = train_log_df if plot_advantages else None
    if training_df is not None and not training_df.empty:
        dataset = ",".join(
            run_info["arguments"].get(
                "datasets", [run_info["arguments"].get("dataset", "")]
            )
        )
        if "dataset" not in training_df.columns:
            training_df = training_df.assign(dataset=dataset)
        if "hp.agent_name" not in training_df.columns:
            training_df = training_df.assign(
                **{"hp.agent_name": run_info["arguments"].get("agent", "")}
            )

    try:
        plot_eval_results(
            results_df,
            folder,
            run_info,
            plot_return_distributions=plot_return_distributions,
            plot_eval_curves=plot_eval_curves,
            plot_gp_fits=plot_gp_fits,
            plot_regret=plot_regret,
            plot_landscapes=plot_landscapes,
            plot_mobility=plot_mobility,
            plot_advantages=plot_advantages,
            training_df=training_df,
        )
    except Exception:
        print(f"ERROR while plotting '{prefix}':\n{traceback.format_exc()}", flush=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zipfile", type=Path, required=True)
    parser.add_argument("--plot_return_distributions", action="store_true")
    parser.add_argument("--plot_eval_curves", action="store_true")
    parser.add_argument("--plot_gp_fits", action="store_true")
    parser.add_argument("--plot_regret", action="store_true")
    parser.add_argument("--no_plot_landscapes", action="store_true")
    parser.add_argument("--no_plot_mobility", action="store_true")
    parser.add_argument("--plot_advantages", action="store_true")
    parser.add_argument("--no_multiprocessing", action="store_true")
    args = parser.parse_args()

    sns.set_theme(context="talk", rc={"figure.figsize": (4, 3)})

    # Parse results — load training logs only when needed for advantage plots
    plots_folder = Path("plots") / os.path.basename(args.zipfile)
    results: dict[str, tuple[dict, ResultsPerStep[PhaseResult], ResultsPerStep[PhaseResult]]] = (
        read_results_from_zip(args.zipfile, load_training_logs=args.plot_advantages)
    )
    results_pandas = {
        identifier: (run_info, phase_results_to_pandas(phase_results), training_logs_to_pandas(training_results))
        for identifier, (run_info, phase_results, training_results) in results.items()
    }

    plot_results = partial(
                    plot_parallel_wrapper,
                    plots_folder,
                    plot_return_distributions=args.plot_return_distributions,
                    plot_eval_curves=args.plot_eval_curves,
                    plot_gp_fits=args.plot_gp_fits,
                    plot_regret=args.plot_regret,
                    plot_landscapes=not args.no_plot_landscapes,
                    plot_mobility=not args.no_plot_mobility,
                    plot_advantages=args.plot_advantages,
                )

    if not args.no_multiprocessing:
        try:
            thread_count = int(os.environ["SLURM_CPUS_ON_NODE"]) // 4
        except Exception as _:
            thread_count = multiprocessing.cpu_count() // 4
        with multiprocessing.get_context("spawn").Pool(thread_count) as pool:
            pool.map(
                plot_results,
                results_pandas.items(),
            )
    else:
        for item in results_pandas.items():
            plot_results(item)

    # Build igpr grid of plots (across datasets)
    # first builds dataframe using plots inside subfolders
    imgpath_dataframe = pd.DataFrame(
        columns=[  # type: ignore
            "plot_type",
            "phase",
            "agent",
            "path",
            "dataset",
            "actor_loss",
            "y_col",
            "hps",
        ]
    )  # type: ignore
    for experiment_name in os.listdir(plots_folder):
        if experiment_name == "grid_plots":
            continue
        if not Path(plots_folder / experiment_name).is_dir():
            continue

        zipf = zipfile.ZipFile(args.zipfile, "r")
        zipfiles = zipf.namelist()
        matching_info = [
            filename
            for filename in zipf.namelist()
            if re.fullmatch(r"^.*/" + re.escape(experiment_name) + r"/info.toml$", filename)
        ]
        if not len(matching_info):
            continue  # not an experiment dir (e.g. 'advantages', 'cdf_swapped')
        if len(matching_info) > 1:
            raise ValueError("given zip seemingly contains experiment multiple times")
        setup = toml.loads(zipf.read(matching_info[0]).decode(encoding="utf-8"))[
            "arguments"
        ]
        agent = setup["agent"]
        dataset = (
            ",".join(setup["datasets"]) if "datasets" in setup else setup["dataset"]
        )
        if "actor_loss" in setup:
            actor_loss = setup["actor_loss"]
        else:
            actor_loss = "default"

        for unscaled_plotpath in [
            plots_folder / experiment_name / filename
            for filename in os.listdir(plots_folder / experiment_name)
            if re.match(r"^.*-.*-\d+.pdf$", filename)
        ]:
            plot_name_matches = re.match(
                r".*/(?P<plot_type>[^-/]*)-(?:\((?P<hps>[^\)]*)\)-)?(?P<y_col>[^-/]*)-(?P<phase>\d+).pdf$",
                str(unscaled_plotpath),
            )
            if not plot_name_matches:
                raise ValueError(
                    f"Naming inside of plots folder not as expected for {unscaled_plotpath}."
                )

            imgpath_dataframe.loc[len(imgpath_dataframe)] = [
                plot_name_matches["plot_type"],
                int(plot_name_matches["phase"]),
                agent,
                str(unscaled_plotpath),
                dataset,
                actor_loss,
                plot_name_matches["y_col"],
                plot_name_matches["hps"],
            ]

    # Now actually plot all grid plots
    for plot_type, agent, y_col, actor_loss, hps in product(
        imgpath_dataframe["plot_type"].unique(),
        imgpath_dataframe["agent"].unique(),
        imgpath_dataframe["y_col"].unique(),
        imgpath_dataframe["actor_loss"].unique(),
        imgpath_dataframe["hps"].unique(),
    ):
        grid_df = imgpath_dataframe[
            (imgpath_dataframe["plot_type"] == plot_type)
            & (imgpath_dataframe["agent"] == agent)
            & (imgpath_dataframe["y_col"] == y_col)
            & ((imgpath_dataframe["hps"] == hps) | pd.isnull(imgpath_dataframe["hps"]))
            & (imgpath_dataframe["actor_loss"] == actor_loss)
        ]
        if len(grid_df) == 0:
            continue
        plots_subfolder = plots_folder / "grid_plots" / f"{plot_type}-{y_col}"
        plots_subfolder.mkdir(exist_ok=True, parents=True)
        grid_plot(
            grid_df,
            plots_subfolder,
            f"{plot_type}-{agent}-{hps}-{y_col}-{actor_loss}",
            grid_columns=("dataset", "phase"),
        )  # type: ignore

    # Combined post-processing: mobility grid, advantage grid, optimum overlap
    from gcrl_landscapes.evaluation.landscapes import (
        plot_optimum_overlap,
        grid_mobility_plot,
    )

    if not args.no_plot_mobility:
        # Collect and process per-experiment results for combined analyses
        combined_parts: list[pd.DataFrame] = []
        combined_hp_list: list[str] = []
        for prefix, (run_info, results_df, _) in results_pandas.items():
            try:
                dataset_str = ",".join(
                    run_info["arguments"].get(
                        "datasets", [run_info["arguments"].get("dataset", "")]
                    )
                )
                proc = results_df.copy().assign(dataset=dataset_str)
                proc = resolve_alpha_sync(proc, run_info["arguments"]["hyperparameters"])
                proc = compute_additional_information(proc)
                hp_list_local = [
                    f"hp.{hp}" for hp in run_info["arguments"]["hyperparameters"]
                ]
                combined_parts.append(proc)
                if not combined_hp_list:
                    combined_hp_list = hp_list_local
            except Exception:
                print(
                    f"WARNING: skipping {prefix} in combined analysis:\n"
                    f"{traceback.format_exc()}",
                    flush=True,
                )

        if combined_parts and len(combined_hp_list) == 2:
            combined_df = pd.concat(combined_parts, ignore_index=True)
            # Optimum overlap (needs multiple datasets / agents to be meaningful)
            n_agents = combined_df["hp.agent_name"].nunique() if "hp.agent_name" in combined_df.columns else 0
            n_datasets = combined_df["dataset"].nunique()
            if n_agents >= 1 and n_datasets >= 1:
                first_exp_folder = plots_folder / list(
                    re.match(r"^logs[^/]*/([^/]*)/?", k).group(1)  # type: ignore[union-attr]
                    for k in results_pandas
                ).__next__()
                try:
                    plot_optimum_overlap(
                        combined_df,
                        combined_hp_list,
                        first_exp_folder,
                        plots_folder / "grid_plots",
                    )
                except Exception:
                    print(
                        f"ERROR in optimum overlap:\n{traceback.format_exc()}",
                        flush=True,
                    )

        grid_mobility_plot(plots_folder)

    if args.plot_advantages:
        grid_advantage_plot(plots_folder)
