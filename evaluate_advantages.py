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
        "/home/mtoepperwien/Documents/gcrl/log_zips/2026-04-07-logs_antmaze-medium-all-algs.zip"
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

# %%
# Filter to end of training only
end_of_training_df = merged_training_df[
    merged_training_df["eval_step"]
    == merged_training_df.groupby(["hp.agent_name", "dataset"])["eval_step"].transform(
        "max"
    )
].copy()

# %%
# Build long-format dataframe: one row per (config, seed, batch sample)
adv_long_df = pd.DataFrame()
if ADV_COLS:
    rows = []
    for _, row in end_of_training_df.iterrows():
        for adv_col in ADV_COLS:
            adv_arr = row.get(adv_col)
            if adv_arr is None or not isinstance(adv_arr, np.ndarray):
                continue
            alpha = row.get("hp.alpha", np.nan)
            df_tmp = pd.DataFrame(
                {
                    "hp.agent_name": row["hp.agent_name"],
                    "dataset": row["dataset"],
                    "config_index": row["config_index"],
                    "seed": row["seed"],
                    "actor": adv_col,
                    "advantage": adv_arr,
                    "alpha": alpha,
                }
            )
            df_tmp["weight"] = np.exp(alpha * adv_arr.astype(np.float64))
            rows.append(df_tmp)

    adv_long_df = pd.concat(rows, ignore_index=True)
    print(f"Long-format dataframe: {len(adv_long_df)} rows")
    print(adv_long_df.groupby(["hp.agent_name", "actor"])[["advantage", "weight"]].describe())

# %% [markdown]
# # Advantage Distribution Across Configurations

# %%
sns.set_theme(context="paper", style="whitegrid")

if ADV_COLS:
    for (agent_name, actor), grp in adv_long_df.groupby(["hp.agent_name", "actor"]):
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
        ax.axvline(0, color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_title(f"{agent_name.upper()} — advantage distribution")
        ax.set_xlabel("Advantage")
        ax.set_ylabel("Density")
        plt.tight_layout()
        fname = plot_dir / f"adv_dist_{agent_name}_{actor.replace('/', '_')}.png"
        plt.savefig(fname, dpi=300)
        print(f"Saved {fname}")
        plt.close()

# %% [markdown]
# # AWR Weight Distribution Across Configurations

# %%
if ADV_COLS:
    for (agent_name, actor), grp in adv_long_df.groupby(["hp.agent_name", "actor"]):
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
        plt.tight_layout()
        fname = plot_dir / f"weight_dist_{agent_name}_{actor.replace('/', '_')}.png"
        plt.savefig(fname, dpi=300)
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
        ax.set_ylim(top=1)
        ax.axvline(100, color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_title(
            f"{agent_name.upper()} — AWR weight distribution clipped at 100 (median α={alpha_median:.2f})"
        )
        ax.set_xlabel("AWR Weight  exp(α · adv), clipped at 100")
        ax.set_ylabel("Density")
        plt.tight_layout()
        fname = plot_dir / f"weight_dist_clipped_{agent_name}_{actor.replace('/', '_')}.png"
        plt.savefig(fname, dpi=300)
        print(f"Saved {fname}")
        plt.close()

# %%
# Ratio of weights > 100 per config, then statistics across configs
if ADV_COLS:
    high_ratio = (
        adv_long_df.groupby(["hp.agent_name", "actor", "config_index", "seed"])["weight"]
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
        == merged_results_df.groupby(["hp.agent_name", "dataset"])["eval_step"].transform("max")
    ][["hp.agent_name", "dataset", "config_index", "seed", "mean_normalized_goal_distance_return"]].copy()

    corr_df = high_ratio.merge(end_perf, on=["hp.agent_name", "config_index", "seed"], how="inner")

    def _pearson_r(g: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if len(g) < 3:
            return pd.Series({"pearson_r": float("nan"), "p_value": float("nan")})
        r, p = _pearsonr(g["high_ratio"], g["mean_normalized_goal_distance_return"])
        return pd.Series({"pearson_r": float(r), "p_value": float(p)})

    print("\nCorrelation between fraction of weights > 100 and end-of-training mean_normalized_goal_distance_return:")
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
    phase_map = merged_results_df[
        ["hp.agent_name", "dataset", "config_index", "eval_step", "phase_num"]
    ].drop_duplicates()

    rows_all = []
    for _, row in merged_training_df.iterrows():
        for adv_col in ADV_COLS:
            adv_arr = row.get(adv_col)
            if adv_arr is None or not isinstance(adv_arr, np.ndarray):
                continue
            alpha = row.get("hp.alpha", np.nan)
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
            rows_all.append(df_tmp)

    adv_all_df = pd.concat(rows_all, ignore_index=True).merge(
        phase_map, on=["hp.agent_name", "dataset", "config_index", "eval_step"], how="left"
    )
    print(
        f"All-phases long-format dataframe: {len(adv_all_df)} rows, "
        f"phases: {sorted(adv_all_df['phase_num'].dropna().unique().tolist())}"
    )

# %%
# High-weight ratio and correlation with performance, per phase
if ADV_COLS and merged_results_df is not None:
    high_ratio_phased = (
        adv_all_df.groupby(["hp.agent_name", "actor", "config_index", "seed", "phase_num"])["weight"]
        .apply(lambda w: (w > 100).sum() / len(w))  # type: ignore[arg-type]
        .reset_index(name="high_ratio")
    )

    perf_per_phase = merged_results_df[
        ["hp.agent_name", "dataset", "config_index", "seed", "phase_num", "mean_normalized_goal_distance_return"]
    ].copy()
    # Normalize return to [0, 1] per (agent, phase, env) using max scaling
    grp_keys = ["hp.agent_name", "phase_num", "dataset"]
    _max = perf_per_phase.groupby(grp_keys)["mean_normalized_goal_distance_return"].transform("max")
    perf_per_phase["return_normalized"] = perf_per_phase["mean_normalized_goal_distance_return"] / _max.replace(0, float("nan"))

    corr_phased_df = high_ratio_phased.merge(
        perf_per_phase, on=["hp.agent_name", "config_index", "seed", "phase_num"], how="inner"
    )

    def _pearson_r_phase(g: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if len(g) < 3:
            return pd.Series({"pearson_r": float("nan"), "p_value": float("nan")})
        r, p = _pearsonr(g["high_ratio"], g["return_normalized"])
        return pd.Series({"pearson_r": float(r), "p_value": float(p)})

    print("\nCorrelation between fraction of weights > 100 and normalized return, per phase:")
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
def _ess(w: pd.Series) -> float:
    w_arr = w.values.astype(np.float64)
    w_finite = w_arr[np.isfinite(w_arr)]
    if len(w_finite) == 0:
        return float("nan")
    # Normalize by max before squaring — ESS is scale-invariant, avoids float64 overflow
    w_scaled = w_finite / w_finite.max()
    return float(w_scaled.sum() ** 2 / (len(w_scaled) * (w_scaled ** 2).sum()))

if ADV_COLS:
    ess_df = (
        adv_long_df.groupby(["hp.agent_name", "actor", "config_index", "seed"])["weight"]
        .apply(_ess)  # type: ignore[arg-type]
        .reset_index(name="ess")
    )
    print("\nNormalized ESS per config (end of training) — statistics across configs:")
    print(ess_df.groupby(["hp.agent_name", "actor"])["ess"].describe())

if ADV_COLS and merged_results_df is not None:
    corr_ess_df = ess_df.merge(end_perf, on=["hp.agent_name", "config_index", "seed"], how="inner")

    def _pearson_r_ess(g: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if len(g) < 3:
            return pd.Series({"pearson_r": float("nan"), "p_value": float("nan")})
        r, p = _pearsonr(g["ess"], g["mean_normalized_goal_distance_return"])
        return pd.Series({"pearson_r": float(r), "p_value": float(p)})

    print("\nCorrelation between ESS and end-of-training mean_normalized_goal_distance_return:")
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
        adv_all_df.groupby(["hp.agent_name", "actor", "config_index", "seed", "phase_num"])["weight"]
        .apply(_ess)  # type: ignore[arg-type]
        .reset_index(name="ess")
    )

    corr_ess_phased_df = ess_phased.merge(
        perf_per_phase, on=["hp.agent_name", "config_index", "seed", "phase_num"], how="inner"
    )

    def _pearson_r_ess_phase(g: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if len(g) < 3:
            return pd.Series({"pearson_r": float("nan"), "p_value": float("nan")})
        r, p = _pearsonr(g["ess"], g["return_normalized"])
        return pd.Series({"pearson_r": float(r), "p_value": float(p)})

    print("\nCorrelation between ESS and normalized return, per phase:")
    print(
        corr_ess_phased_df.groupby(["hp.agent_name", "actor", "phase_num"])
        .apply(_pearson_r_ess_phase)
        .reset_index()
        .sort_values(["hp.agent_name", "actor", "phase_num"])
        .to_string(index=False)
    )
