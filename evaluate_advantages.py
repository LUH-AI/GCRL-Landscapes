# %%
# ruff: noqa
# %%
# %% tags=["parameters"]
from pathlib import Path
import argparse

_p = argparse.ArgumentParser()
_p.add_argument("--zipfiles", nargs="+", type=Path)
_args, _ = _p.parse_known_args()
zipfiles = _args.zipfiles or [
    Path(
        "../gcrl_results/2026-04-19-logs-antmaze.zip"
    )
]  # <- edit this
plot_dir = Path("plots") / zipfiles[0].name / "advantages"

# %%
import os

import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs(plot_dir, exist_ok=True)

# %%
import types
import gcrl_landscapes.evaluation.tabular as _tabular

_tabular.args = types.SimpleNamespace(no_multiprocessing=True)

from gcrl_landscapes.util.data import load_or_compute
from gcrl_landscapes.evaluation.tabular import compute_merged_df

merged_results_df, merged_training_df = load_or_compute(zipfiles, compute_merged_df)

# Memory optimization
to_category_columns = ["dataset", "hps", "agent"] + merged_training_df.columns[
    merged_training_df.columns.str.startswith("hp.")
].tolist()
for col in to_category_columns:
    merged_training_df[col] = merged_training_df[col].astype("category")
float_cols = merged_training_df.select_dtypes(include="float64").columns
for col in float_cols:
    merged_training_df[col] = merged_training_df[col].astype("float16")
merged_training_df = merged_training_df.drop(
    columns=["target/held_out_val_batch_values"], errors="ignore"
)

# %%
import ast
import numpy as np
import pandas as pd


def literal_lists_to_numpy(s):
    try:
        return np.array(ast.literal_eval(s), dtype=np.float32)
    except:
        return None


ADV_COLS = [c for c in merged_training_df.columns if c.startswith("advantage/")]
print(f"Advantage columns found: {ADV_COLS}")

for col in ADV_COLS:
    merged_training_df[col] = merged_training_df[col].apply(literal_lists_to_numpy)
# %% [markdown]
#
#

# %% [markdown]
# # Advantage Distribution Across Configurations

# %%
adv_long_df.groupby(["hp.agent_name", "dataset", "actor"])["advantage"].describe()

# %%
sns.set_theme(context="paper", style="whitegrid")

if ADV_COLS:
    # Compute global x-limits for advantage plots (1st/99th percentile to avoid extreme outliers)
    _adv_vals = adv_long_df["advantage"].dropna()
    _adv_xlim = (float(_adv_vals.quantile(0.25)), float(_adv_vals.quantile(0.75)))

    for (agent_name, dataset, actor), grp in adv_long_df.groupby(
        ["hp.agent_name", "dataset", "actor"]
    ):
        fig, ax = plt.subplots(figsize=(5, 3))
        for config_id, cfg_grp in grp.groupby("config_index"):
            sns.kdeplot(
                bw_adjust=0.5,
                data=cfg_grp,
                x="advantage",
                ax=ax,
                alpha=0.3,
                linewidth=0.8,
                color="steelblue",
            )
        sns.kdeplot(
            bw_adjust=0.5,
            data=grp,
            x="advantage",
            ax=ax,
            color="black",
            linewidth=2,
            label="overall",
        )
        ax.set_ylim(0, 1)
        ax.axvline(0, color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_xlim(-10, 10)
        ax.set_title(f"{agent_name.upper()} — advantage distribution")
        ax.set_xlabel("Advantage")
        ax.set_ylabel("Density")
        plt.tight_layout()
        fname = plot_dir / f"adv_dist_{agent_name}_{dataset}_{actor.replace('/', '_')}.png"
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()

# %% [markdown]
# # AWR Weight Distribution Across Configurations

# %%
if ADV_COLS:
    # Compute global log-scale x-limits for weight plots
    _w_vals = adv_long_df["weight"].apply(lambda x: x if np.isfinite(x) and x > 0 else np.nan).dropna()
    _w_xlim = (float(_w_vals.quantile(0.01)), float(_w_vals.quantile(0.99)))
    if not (np.isfinite(_w_xlim[0]) and _w_xlim[0] > 0):
        _w_xlim = (1e-10, _w_xlim[1])
    if not np.isfinite(_w_xlim[1]):
        _w_xlim = (_w_xlim[0], 1e30)

    for (agent_name, dataset, actor), grp in adv_long_df.groupby(
        ["hp.agent_name", "dataset", "actor"]
    ):
        # Drop non-finite weights (can occur if alpha * adv overflows float32)
        finite_mask: pd.Series = grp["weight"].apply(np.isfinite)  # type: ignore[arg-type]
        grp_valid: pd.DataFrame = grp[finite_mask]
        if len(grp_valid) == 0:
            print(f"Skipping {agent_name}/{actor}: no finite weights")
            continue

        alpha_median = grp["alpha"].median()
        fig, ax = plt.subplots(figsize=(5, 3))
        for config_id, cfg_grp in grp_valid.groupby("config_index"):
            cfg_valid: pd.DataFrame = cfg_grp[cfg_grp["weight"] > 0]
            if len(cfg_valid) == 0:
                continue
            sns.kdeplot(
                bw_adjust=0.5,
                data=cfg_valid,
                x="weight",
                ax=ax,
                alpha=0.3,
                linewidth=0.8,
                color="darkorange",
                log_scale=True,
            )
        overall_valid = grp_valid[grp_valid["weight"] > 0]
        sns.kdeplot(
            bw_adjust=0.5,
            data=overall_valid,
            x="weight",
            ax=ax,
            color="black",
            linewidth=2,
            label="overall",
            log_scale=True,
        )
        ax.set_title(
            f"{agent_name.upper()} — AWR weight distribution (median α={alpha_median:.2f})"
        )
        ax.set_xlabel("AWR Weight  exp(α · adv)")
        ax.set_ylabel("Density")
        ax.set_xlim(_w_xlim)
        plt.tight_layout()
        fname = (
            plot_dir
            / f"weight_dist_{agent_name}_{dataset}_{actor.replace('/', '_')}.pdf"
        )
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()

        # Same plot with weights clipped to 100
        grp_clipped = grp_valid.copy()
        grp_clipped["weight_clipped"] = grp_clipped["weight"].clip(upper=100)
        fig, ax = plt.subplots(figsize=(5, 3))
        for config_id, cfg_grp in grp_clipped.groupby("config_index"):
            sns.kdeplot(
                bw_adjust=0.5,
                data=cfg_grp,
                x="weight_clipped",
                ax=ax,
                alpha=0.3,
                linewidth=0.8,
                color="darkorange",
                clip=(0, 100),
            )
        sns.kdeplot(
            bw_adjust=0.5,
            data=grp_clipped,
            x="weight_clipped",
            ax=ax,
            color="black",
            linewidth=2,
            label="overall",
            clip=(0, 100),
        )
        ax.set_xlim(0, 100)
        ax.set_ylim(top=1)
        ax.axvline(100, color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_title(
            f"{agent_name.upper()} — AWR weight distribution clipped at 100 (median α={alpha_median:.2f})"
        )
        ax.set_xlabel("AWR Weight  exp(α · adv), clipped at 100")
        ax.set_ylabel("Density")
        plt.tight_layout()
        fname = (
            plot_dir
            / f"weight_dist_clipped_{agent_name}_{dataset}_{actor.replace('/', '_')}.pdf"
        )
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()

# %%
# Ratio of weights > 100 per config, then statistics across configs
if ADV_COLS:
    high_ratio = (
        adv_long_df.groupby(
            ["hp.agent_name", "dataset", "actor", "config_index", "seed"]
        )["weight"]
        .apply(lambda w: (w > 100).sum() / len(w))  # type: ignore[arg-type]
        .reset_index(name="high_ratio")
    )
    print("\nFraction of weights > 100 per config — statistics across configs:")
    print(high_ratio.groupby(["hp.agent_name", "actor"])["high_ratio"].describe())

# %%
# Correlation between high-weight ratio and config performance (end of training)
if ADV_COLS and merged_results_df is not None:
    from scipy.stats import pearsonr as _pearsonr

    end_perf = merged_results_df[
        merged_results_df["eval_step"]
        == merged_results_df.groupby(["hp.agent_name", "dataset"])[
            "eval_step"
        ].transform("max")
    ][
        [
            "hp.agent_name",
            "dataset",
            "config_index",
            "seed",
            "mean_normalized_goal_distance_return",
        ]
    ].copy()

    corr_df = high_ratio.merge(
        end_perf, on=["hp.agent_name", "dataset", "config_index", "seed"], how="inner"
    )

    def _pearson_r(g: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if len(g) < 3:
            return pd.Series({"pearson_r": float("nan"), "p_value": float("nan")})
        r, p = _pearsonr(g["high_ratio"], g["mean_normalized_goal_distance_return"])
        return pd.Series({"pearson_r": float(r), "p_value": float(p)})

    print(
        "\nCorrelation between fraction of weights > 100 and end-of-training mean_normalized_goal_distance_return:"
    )
    print(
        corr_df.groupby(["hp.agent_name", "actor"])
        .apply(_pearson_r)
        .reset_index()
        .to_string(index=False)
    )

# %%
# Build long-format dataframe over ALL phases (not just end of training)
# NOTE: merged_results_df only contains return data at phase-boundary eval_steps
# (one per phase, e.g. 205069, 410138, 615208, 820277), even though merged_training_df
# logs advantages at ~10x more intermediate steps. Any join with return data is therefore
# limited to those phase-boundary steps — this is a structural property of the zip format,
# not a filtering bug.
if ADV_COLS and merged_results_df is not None:
    # 1. Create the phase map as before
    phase_map = merged_results_df[
        ["hp.agent_name", "dataset", "config_index", "eval_step", "phase_num"]
    ].drop_duplicates()

    # 2. Select only necessary columns to save memory
    cols_to_keep = ["hp.agent_name", "dataset", "config_index", "seed", "eval_step", "hp.alpha"] + ADV_COLS
    
    # 3. Melt ADV_COLS into 'actor' and 'advantage' columns
    adv_all_df = merged_training_df[cols_to_keep].melt(
        id_vars=["hp.agent_name", "dataset", "config_index", "seed", "eval_step", "hp.alpha"],
        value_vars=ADV_COLS,
        var_name="actor",
        value_name="advantage"
    )

    # 4. Remove rows where advantage is None or not an array (mimicking your if-check)
    adv_all_df = adv_all_df[adv_all_df["advantage"].apply(lambda x: isinstance(x, np.ndarray))]

    # 5. Explode the 'advantage' column (turns arrays into individual rows)
    adv_all_df = adv_all_df.explode("advantage")

    # 6. Ensure numeric types and calculate weights vectorially
    adv_all_df["advantage"] = adv_all_df["advantage"].astype(np.float64)
    adv_all_df["weight_unclipped"] = np.exp(adv_all_df["hp.alpha"].astype(np.float64) * adv_all_df["advantage"])
    adv_all_df["weight"] = adv_all_df["weight_unclipped"].clip(upper=100)

    # 7. Final Merge
    adv_all_df = adv_all_df.merge(
        phase_map,
        on=["hp.agent_name", "dataset", "config_index", "eval_step"],
        how="left",
    )

    print(
        f"All-phases long-format dataframe: {len(adv_all_df)} rows, "
        f"phases: {sorted(adv_all_df['phase_num'].dropna().unique().tolist())}"
    )

# %%
# High-weight ratio and correlation with performance, per phase
if ADV_COLS and merged_results_df is not None:
    high_ratio_phased = (
        adv_all_df.groupby(
            ["hp.agent_name", "dataset", "actor", "config_index", "seed", "phase_num"]
        )["weight"]
        .apply(lambda w: (w > 100).sum() / len(w))  # type: ignore[arg-type]
        .reset_index(name="high_ratio")
    )

    perf_per_phase = merged_results_df[
        [
            "hp.agent_name",
            "dataset",
            "config_index",
            "seed",
            "phase_num",
            "mean_normalized_goal_distance_return",
        ]
    ].copy()
    # Normalize return to [0, 1] per (agent, phase, env) using max scaling
    grp_keys = ["hp.agent_name", "dataset", "phase_num", "dataset"]
    _max = perf_per_phase.groupby(grp_keys)[
        "mean_normalized_goal_distance_return"
    ].transform("max")
    perf_per_phase["return_normalized"] = perf_per_phase[
        "mean_normalized_goal_distance_return"
    ] / _max.replace(0, float("nan"))

    corr_phased_df = high_ratio_phased.merge(
        perf_per_phase,
        on=["hp.agent_name", "config_index", "seed", "phase_num"],
        how="inner",
    )

    def _pearson_r_phase(g: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if len(g) < 3:
            return pd.Series({"pearson_r": float("nan"), "p_value": float("nan")})
        r, p = _pearsonr(g["high_ratio"], g["return_normalized"])
        return pd.Series({"pearson_r": float(r), "p_value": float(p)})

    print(
        "\nCorrelation between fraction of weights > 100 and normalized return, per phase:"
    )
    print(
        corr_phased_df.groupby(["hp.agent_name", "actor", "phase_num"])
        .apply(_pearson_r_phase)
        .reset_index()
        .sort_values(["hp.agent_name", "actor", "phase_num"])
        .to_string(index=False)
    )


# %%
# Normalized ESS per config (end of training) + correlation with performance
# ESS = (sum(w))^2 / sum(w^2), normalized by n -> in [0, 1]; 1 = uniform weights, 0 = single sample dominates
# Weights are clipped to 100 before computation, consistent with the clipped weight distribution plots.
def _ess(w: pd.Series) -> float:
    w_arr = w.clip(upper=100).values.astype(np.float64)
    w_finite = w_arr[np.isfinite(w_arr)]
    if len(w_finite) == 0:
        return float("nan")
    # Normalize by max before squaring — ESS is scale-invariant, avoids float64 overflow
    if w_arr.max() == 0.0:
        return float(len(w))
    w_scaled = w_finite / w_finite.max()
    return float(w_scaled.sum() ** 2 / (len(w_scaled) * (w_scaled**2).sum()))


if ADV_COLS:
    ess_df = (
        adv_long_df.groupby(
            ["hp.agent_name", "dataset", "actor", "config_index", "seed"]
        )["weight"]
        .apply(_ess)  # type: ignore[arg-type]
        .reset_index(name="ess")
    )
    print("\nNormalized ESS per config (end of training) — statistics across configs:")
    print(ess_df.groupby(["hp.agent_name", "actor"])["ess"].describe())

if ADV_COLS and merged_results_df is not None:
    corr_ess_df = ess_df.merge(
        end_perf, on=["hp.agent_name", "dataset", "config_index", "seed"], how="inner"
    )

    def _pearson_r_ess(g: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if len(g) < 3:
            return pd.Series({"pearson_r": float("nan"), "p_value": float("nan")})
        r, p = _pearsonr(g["ess"], g["mean_normalized_goal_distance_return"])
        return pd.Series({"pearson_r": float(r), "p_value": float(p)})

    print(
        "\nCorrelation between ESS and end-of-training mean_normalized_goal_distance_return:"
    )
    print(
        corr_ess_df.groupby(["hp.agent_name", "actor"])
        .apply(_pearson_r_ess)
        .reset_index()
        .to_string(index=False)
    )

# %%
# ESS per phase + correlation with normalized performance
if ADV_COLS and merged_results_df is not None:
    ess_phased = (
        adv_all_df.groupby(
            ["hp.agent_name", "dataset", "actor", "config_index", "seed", "phase_num"]
        )["weight"]
        .apply(_ess)  # type: ignore[arg-type]
        .reset_index(name="ess")
    )

    corr_ess_phased_df = ess_phased.merge(
        perf_per_phase,
        on=["hp.agent_name", "dataset", "config_index", "seed", "phase_num"],
        how="inner",
    )

    def _pearson_r_ess_phase(g: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if len(g) < 3:
            return pd.Series({"pearson_r": float("nan"), "p_value": float("nan")})
        r, p = _pearsonr(g["ess"], g["return_normalized"])
        return pd.Series({"pearson_r": float(r), "p_value": float(p)})

    print("\nCorrelation between ESS and normalized return, per phase:")
    print(
        corr_ess_phased_df.groupby(["hp.agent_name", "dataset", "actor"])
        .apply(_pearson_r_ess_phase)
        .reset_index()
        .sort_values(["hp.agent_name", "dataset", "actor"])
        .to_string(index=False)
    )
