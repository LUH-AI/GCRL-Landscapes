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
        "/home/mtoepperwien/Documents/gcrl/gcrl_results/2026-04-19-logs-antmaze.zip"
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
adv_long_end_of_training_df = pd.DataFrame()
if ADV_COLS:
    rows = []
    for _, row in end_of_training_df.iterrows():
        for adv_col in ADV_COLS:
            adv_arr = row.get(adv_col)
            if adv_arr is None or not isinstance(adv_arr, np.ndarray):
                continue
            alpha = row.get("hp.alpha", np.nan)
            adv_f64 = adv_arr.astype(np.float64)
            _mu, _sigma = adv_f64.mean(), adv_f64.std()
            adv_norm = (adv_f64 - _mu) / _sigma if _sigma > 0 else np.zeros_like(adv_f64)
            adv_norm_spread = (adv_f64) / _sigma if _sigma > 0 else np.zeros_like(adv_f64)
            df_tmp = pd.DataFrame(
                {
                    "hp.agent_name": row["hp.agent_name"],
                    "dataset": row["dataset"],
                    "config_index": row["config_index"],
                    "seed": row["seed"],
                    "eval_step": row["eval_step"],
                    "actor": adv_col,
                    "advantage": adv_arr,
                    "advantage_norm": adv_norm,
                    "advantage_norm_spread": adv_norm_spread,
                    "alpha": alpha,
                }
            )
            df_tmp["weight"] = np.exp(alpha * adv_f64)
            rows.append(df_tmp)

    adv_long_end_of_training_df = pd.concat(rows, ignore_index=True)
    print(f"Long-format dataframe: {len(adv_long_end_of_training_df)} rows")
    print(
        adv_long_end_of_training_df.groupby(["hp.agent_name", "dataset", "actor"])[
            ["advantage", "weight"]
        ].describe()
    )

# %% [markdown]
# # Advantage Distribution Across Configurations

# %%
sns.set_theme(context="paper", style="whitegrid")

if ADV_COLS:
    # Compute global x-limits for advantage plots (1st/99th percentile to avoid extreme outliers)
    _adv_vals = adv_long_end_of_training_df["advantage"].dropna()
    _adv_xlim = (float(_adv_vals.quantile(0.01)), float(_adv_vals.quantile(0.99)))
    _adv_xlim = (-10, 10)

    for (agent_name, dataset, actor), grp in adv_long_end_of_training_df.groupby(
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
        ax.axvline(0, color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_xlim(_adv_xlim)
        ax.set_title(f"{agent_name.upper()} — advantage distribution")
        ax.set_xlabel("Advantage")
        ax.set_ylabel("Density")
        plt.tight_layout()
        fname = plot_dir / f"adv_dist_{agent_name}_{dataset}_{actor.replace('/', '_')}.png"
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()

# %%
# Normalized advantage distribution (z-scored per list / per config-seed row)
if ADV_COLS:
    _norm_vals = adv_long_end_of_training_df["advantage_norm"].dropna()
    _norm_xlim = (float(_norm_vals.quantile(0.01)), float(_norm_vals.quantile(0.99)))
    _norm_xlim = (-10, 10)

    for (agent_name, dataset, actor), grp in adv_long_end_of_training_df.groupby(
        ["hp.agent_name", "dataset", "actor"]
    ):
        fig, ax = plt.subplots(figsize=(5, 3))
        for config_id, cfg_grp in grp.groupby("config_index"):
            sns.kdeplot(
                bw_adjust=0.5,
                data=cfg_grp,
                x="advantage_norm",
                ax=ax,
                alpha=0.3,
                linewidth=0.8,
                color="steelblue",
            )
        sns.kdeplot(
            bw_adjust=0.5,
            data=grp,
            x="advantage_norm",
            ax=ax,
            color="black",
            linewidth=2,
            label="overall",
        )
        ax.axvline(0, color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_xlim(_norm_xlim)
        ax.set_title(f"{agent_name.upper()} — normalized advantage distribution")
        ax.set_xlabel("Normalized Advantage  (adv − μ) / σ  per config-seed")
        ax.set_ylabel("Density")
        plt.tight_layout()
        fname = plot_dir / f"adv_dist_norm_{agent_name}_{dataset}_{actor.replace('/', '_')}.png"
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()

# %%
# describe comparison — raw vs normalised advantages
if ADV_COLS:
      for (agent, actor), grp in adv_long_end_of_training_df.groupby(["hp.agent_name", "actor"]):
          print(f"\n=== {agent} | {actor} ===")
          desc = grp[["advantage", "advantage_norm", "advantage_norm_spread"]].describe()
          desc.columns = ["advantage (raw)", "advantage (z-scored)", "advantage spread normalized"]
          print(desc.to_string())

# %%
# describe comparison — raw vs normalised advantage means per batch
if ADV_COLS:
    print("Statistics over mean of advantages in batch")
    for (agent, actor), grp in adv_long_end_of_training_df.groupby(["hp.agent_name", "actor"]):
        print(f"\n=== {agent} | {actor} ===")
        temp_df = grp.groupby(["dataset", "config_index", "seed", "eval_step"])["advantage"].mean()
        desc = temp_df.describe()
        print(desc.to_string())

# %%
# describe comparison — raw vs normalised advantages, cut out outliers per batch
# Outliers defined as outside [Q5, Q95] computed per (config_index, seed, actor, eval_step) batch
if ADV_COLS:
    q25 = adv_long_end_of_training_df.groupby(["hp.agent_name", "dataset", "config_index", "seed", "actor", "eval_step"])["advantage"].transform("quantile", 0.05)
    q75 = adv_long_end_of_training_df.groupby(["hp.agent_name", "dataset", "config_index", "seed", "actor", "eval_step"])["advantage"].transform("quantile", 0.95)

    adv_trimmed_df = adv_long_end_of_training_df[(adv_long_end_of_training_df["advantage"] >= q25) & (adv_long_end_of_training_df["advantage"] <= q75)].copy()
    print(f"Trimmed dataframe: {len(adv_trimmed_df)} rows ({len(adv_long_end_of_training_df) - len(adv_trimmed_df)} outliers removed, {100 * (1 - len(adv_trimmed_df)/len(adv_long_end_of_training_df)):.1f}%)")
    for (agent, actor), grp in adv_trimmed_df.groupby(["hp.agent_name", "actor"]):
        print(f"\n=== {agent} | {actor} ===")
        desc = grp[["advantage", "advantage_norm", "weight"]].describe()
        desc.columns = ["advantage (raw)", "advantage (z-scored)", "weight"]
        print(desc.to_string())


# %%
# describe comparison — raw vs normalised advantages
if ADV_COLS:
      for (agent, actor), grp in adv_long_end_of_training_df.groupby(["hp.agent_name", "actor"]):
          print(f"\n=== {agent} | {actor} ===")
          desc = grp[["weight"]].clip(upper=100).describe()
          desc.columns = ["advantage"]
          print(desc.to_string())


# %%
# describe comparison — raw vs normalised advantages (positive only)
if ADV_COLS:
    for (agent, actor), grp in adv_long_end_of_training_df.groupby(["hp.agent_name", "actor"]):
        pos = grp[grp["advantage"] > 0]
        if pos.empty:
            continue
        print(f"\n=== {agent} | {actor} ===")
        desc = pos[["advantage", "advantage_norm"]].describe()
        desc.columns = ["advantage (raw)", "advantage (z-scored)"]
        print(desc.to_string())

# %%
# Fraction of negative advantages per algorithm-dataset combination
if ADV_COLS:
    neg_frac = (
        adv_long_end_of_training_df.groupby(["hp.agent_name", "actor"])["advantage"]
        .apply(lambda x: (x < 0).sum() / len(x))
        .reset_index(name="frac_negative")
    )
    print("\nFraction of negative advantages per agent × dataset × actor:")
    print(neg_frac.to_string(index=False))

# %% [markdown]
# # AWR Weight Distribution Across Configurations

# %%
if ADV_COLS:
    # Compute global log-scale x-limits for weight plots
    _w_vals = adv_long_end_of_training_df["weight"].apply(lambda x: x if np.isfinite(x) and x > 0 else np.nan).dropna()
    _w_xlim = (float(_w_vals.quantile(0.01)), float(_w_vals.quantile(0.99)))
    if not (np.isfinite(_w_xlim[0]) and _w_xlim[0] > 0):
        _w_xlim = (1e-10, _w_xlim[1])
    if not np.isfinite(_w_xlim[1]):
        _w_xlim = (_w_xlim[0], 1e30)

    for (agent_name, dataset, actor), grp in adv_long_end_of_training_df.groupby(
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
        adv_long_end_of_training_df.groupby(
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
            adv_f64 = adv_arr.astype(np.float64)
            _mu, _sigma = adv_f64.mean(), adv_f64.std()
            adv_norm = (adv_f64 - _mu) / _sigma if _sigma > 0 else np.zeros_like(adv_f64)
            adv_norm_spread = (adv_f64) / _sigma if _sigma > 0 else np.zeros_like(adv_f64)

            df_tmp = pd.DataFrame(
                {
                    "hp.agent_name": row["hp.agent_name"],
                    "dataset": row["dataset"],
                    "config_index": row["config_index"],
                    "seed": row["seed"],
                    "eval_step": row["eval_step"],
                    "actor": adv_col,
                    "advantage": adv_arr,
                    "advantage_norm": adv_norm,
                    "advantage_norm_spread": adv_norm_spread,
                    "alpha": alpha,
                }
            )
            df_tmp["weight"] = np.exp(alpha * adv_arr.astype(np.float64))
            rows_all.append(df_tmp)

    adv_all_df = pd.concat(rows_all, ignore_index=True).merge(
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
# describe comparison — raw vs normalised advantages
if ADV_COLS:
      for (agent, actor), grp in adv_all_df.groupby(["hp.agent_name", "actor"]):
          print(f"\n=== {agent} | {actor} ===")
          desc = grp[["advantage", "advantage_norm", "advantage_norm_spread"]].describe()
          desc.columns = ["advantage (raw)", "advantage (z-scored)", "advantage spread normalized"]
          print(desc.to_string())

# %%
# describe comparison — raw vs normalised advantage means per batch
if ADV_COLS:
    print("Statistics over mean of advantages in batch")
    for (agent, actor), grp in adv_all_df.groupby(["hp.agent_name", "actor"]):
        print(f"\n=== {agent} | {actor} ===")
        temp_df = grp.groupby(["dataset", "config_index", "seed", "eval_step"])["advantage"].mean()
        desc = temp_df.describe()
        print(desc.to_string())

# %%
# describe comparison — raw vs normalised advantages, cut out outliers per batch
# Outliers defined as outside [Q5, Q95] computed per (config_index, seed, actor, eval_step) batch
if ADV_COLS:
    q25 = adv_all_df.groupby(["hp.agent_name", "dataset", "config_index", "seed", "actor", "eval_step"])["advantage"].transform("quantile", 0.05)
    q75 = adv_all_df.groupby(["hp.agent_name", "dataset", "config_index", "seed", "actor", "eval_step"])["advantage"].transform("quantile", 0.95)

    adv_trimmed_df = adv_all_df[(adv_all_df["advantage"] >= q25) & (adv_all_df["advantage"] <= q75)].copy()
    print(f"Trimmed dataframe: {len(adv_trimmed_df)} rows ({len(adv_all_df) - len(adv_trimmed_df)} outliers removed, {100 * (1 - len(adv_trimmed_df)/len(adv_all_df)):.1f}%)")
    for (agent, actor), grp in adv_trimmed_df.groupby(["hp.agent_name", "actor"]):
        print(f"\n=== {agent} | {actor} ===")
        desc = grp[["advantage", "advantage_norm", "weight"]].describe()
        desc.columns = ["advantage (raw)", "advantage (z-scored)", "weight"]
        print(desc.to_string())


# %%
# describe comparison — raw vs normalised advantages
if ADV_COLS:
      for (agent, actor), grp in adv_all_df.groupby(["hp.agent_name", "actor"]):
          print(f"\n=== {agent} | {actor} ===")
          desc = grp[["weight"]].clip(upper=100).describe()
          desc.columns = ["advantage"]
          print(desc.to_string())


# %%
# describe comparison — raw vs normalised advantages (positive only)
if ADV_COLS:
    for (agent, actor), grp in adv_all_df.groupby(["hp.agent_name", "actor"]):
        pos = grp[grp["advantage"] > 0]
        if pos.empty:
            continue
        print(f"\n=== {agent} | {actor} ===")
        desc = pos[["advantage", "advantage_norm"]].describe()
        desc.columns = ["advantage (raw)", "advantage (z-scored)"]
        print(desc.to_string())

# %%
# Fraction of negative advantages per algorithm-dataset combination
if ADV_COLS:
    neg_frac = (
        adv_all_df.groupby(["hp.agent_name", "actor"])["advantage"]
        .apply(lambda x: (x < 0).sum() / len(x))
        .reset_index(name="frac_negative")
    )
    print("\nFraction of negative advantages per agent × dataset × actor:")
    print(neg_frac.to_string(index=False))

# %%
import warnings
# Normalized ESS per config (end of training) + correlation with performance
# ESS = (sum(w))^2 / sum(w^2), normalized by n -> in [0, 1]; 1 = uniform weights, 0 = single sample dominates
def _ess(w: pd.Series) -> float:
    w_arr = w.values.astype(np.float64)
    w_finite = w_arr.clip(min=10e-10, max=100)
    if len(w_finite) == 0:
        return float("nan")
    # Normalize by max before squaring — ESS is scale-invariant, avoids float64 overflow
    w_scaled = w_finite / w_finite.max()
    return float(w_scaled.sum() ** 2 / (len(w_scaled) * (w_scaled**2).sum()))


if ADV_COLS:
    ess_df = (
        adv_long_end_of_training_df.groupby(
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
            print("here")
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

# %%
# Correlation between fraction of positive advantages and end-of-training return
if ADV_COLS and merged_results_df is not None:
    pos_frac = (
        adv_long_end_of_training_df.groupby(
            ["hp.agent_name", "dataset", "actor", "config_index", "seed"]
        )["advantage"]
        .apply(lambda x: (x > 0).sum() / len(x))
        .reset_index(name="frac_positive")
    )

    corr_pos_frac_df = pos_frac.merge(
        end_perf, on=["hp.agent_name", "dataset", "config_index", "seed"], how="inner"
    )

    def _pearson_r_pos_frac(g: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if len(g) < 3:
            return pd.Series({"pearson_r": float("nan"), "p_value": float("nan")})
        r, p = _pearsonr(g["frac_positive"], g["mean_normalized_goal_distance_return"])
        return pd.Series({"pearson_r": float(r), "p_value": float(p)})

    print(
        "\nCorrelation between fraction of positive advantages and end-of-training mean_normalized_goal_distance_return:"
    )
    print(
        corr_pos_frac_df.groupby(["hp.agent_name", "actor"])
        .apply(_pearson_r_pos_frac)
        .reset_index()
        .to_string(index=False)
    )

    # Per-phase version
    pos_frac_phased = (
        adv_all_df.groupby(
            ["hp.agent_name", "dataset", "actor", "config_index", "seed", "phase_num"]
        )["advantage"]
        .apply(lambda x: (x > 0).sum() / len(x))
        .reset_index(name="frac_positive")
    )

    def _pearson_r_pos_frac_phase(g: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if len(g) < 3:
            return pd.Series({"pearson_r": float("nan"), "p_value": float("nan")})
        r, p = _pearsonr(g["frac_positive"], g["return_normalized"])
        return pd.Series({"pearson_r": float(r), "p_value": float(p)})

    corr_pos_frac_phased_df = pos_frac_phased.merge(
        perf_per_phase,
        on=["hp.agent_name", "dataset", "config_index", "seed", "phase_num"],
        how="inner",
    )

    print(
        "\nCorrelation between fraction of positive advantages and normalized return, per phase:"
    )
    print(
        corr_pos_frac_phased_df.groupby(["hp.agent_name", "actor", "phase_num"])
        .apply(_pearson_r_pos_frac_phase)
        .reset_index()
        .sort_values(["hp.agent_name", "actor", "phase_num"])
        .to_string(index=False)
    )

# %%
# Pairwise Spearman rank correlation of advantages across configs/seeds per phase
# Measures how consistent the relative ordering of batch samples is across different configs/seeds
if ADV_COLS and merged_results_df is not None:
    from scipy.stats import spearmanr as _spearmanr, kendalltau as _kendalltau

    # Assign a phase_num to every eval_step in merged_training_df via forward merge_asof:
    # each eval_step maps to the phase whose end-boundary is >= that step.
    # This covers intermediate (non-boundary) eval_steps, not just phase-boundary ones.
    _phase_boundaries = (
        phase_map.groupby(["hp.agent_name", "dataset", "eval_step"])["phase_num"]
        .first()
        .reset_index()
        .sort_values(["hp.agent_name", "dataset", "eval_step"])
    )
    _all_train_steps = (
        merged_training_df[["hp.agent_name", "dataset", "eval_step"]]
        .drop_duplicates()
        .sort_values(["hp.agent_name", "dataset", "eval_step"])
    )
    _parts = []
    for (_agent, _dataset), _grp in _all_train_steps.groupby(["hp.agent_name", "dataset"]):
        _bounds = _phase_boundaries[
            (_phase_boundaries["hp.agent_name"] == _agent)
            & (_phase_boundaries["dataset"] == _dataset)
        ].sort_values("eval_step")
        if _bounds.empty:
            continue
        _parts.append(
            pd.merge_asof(
                _grp.sort_values("eval_step"),
                _bounds[["eval_step", "phase_num"]],
                on="eval_step",
                direction="forward",
            )
        )
    phase_map_simple = (
        pd.concat(_parts, ignore_index=True).dropna(subset=["phase_num"])
        if _parts
        else pd.DataFrame(columns=["hp.agent_name", "dataset", "eval_step", "phase_num"])
    )

    spearman_records = []

    for adv_col in ADV_COLS:
        valid_mask = merged_training_df[adv_col].apply(
            lambda x: isinstance(x, np.ndarray) and len(x) > 0
        )
        adv_df = merged_training_df[valid_mask][
            ["hp.agent_name", "dataset", "config_index", "seed", "eval_step", adv_col]
        ].merge(
            phase_map_simple,
            on=["hp.agent_name", "dataset", "eval_step"],
            how="inner",
        )

        for (agent_name, dataset, seed, eval_step, phase_num), grp in adv_df.groupby(
            ["hp.agent_name", "dataset", "seed", "eval_step", "phase_num"]
        ):
            arrays = [
                row[adv_col].astype(np.float64)
                for _, row in grp.iterrows()
                if isinstance(row[adv_col], np.ndarray)
            ]
            if len(arrays) < 2:
                continue

            min_len = min(len(a) for a in arrays)
            mat = np.stack([a[:min_len] for a in arrays])
            n = mat.shape[0]

            corr_result = _spearmanr(mat.T)  # cols of mat.T = different configs, same seed
            if n == 2:
                corrs = [float(corr_result.statistic)]
                kt_corrs = [float(_kendalltau(mat[0], mat[1]).statistic)]
            else:
                triu_idx = np.triu_indices(n, k=1)
                corrs = corr_result.statistic[triu_idx].tolist()
                kt_corrs = [float(_kendalltau(mat[i], mat[j]).statistic) for i, j in zip(triu_idx[0], triu_idx[1])]

            spearman_records.append(
                {
                    "hp.agent_name": agent_name,
                    "dataset": dataset,
                    "actor": adv_col,
                    "seed": seed,
                    "phase_num": phase_num,
                    "mean_spearman_r": float(np.mean(corrs)),
                    "median_spearman_r": float(np.median(corrs)),
                    "mean_kendall_tau": float(np.mean(kt_corrs)),
                    "n_pairs": len(corrs),
                }
            )

    if spearman_records:
        spearman_df = pd.DataFrame(spearman_records)
        print("\nPairwise Spearman rank correlation of advantages across configs (same seed):")
        agg = (
            spearman_df.groupby(["hp.agent_name", "actor", "phase_num"])
            .agg(
                mean_spearman_r=("mean_spearman_r", "mean"),
                median_spearman_r=("median_spearman_r", "median"),
                n_eval_steps=("phase_num", "count"),
            )
            .reset_index()
            .sort_values(["hp.agent_name", "actor", "phase_num"])
        )
        print(agg.to_string(index=False))
    else:
        print("\nNo Spearman records computed (no phase-boundary eval_steps with multiple configs/seeds found).")

# %%
# Pairwise Spearman rank correlation — positive advantages only
# Masks out non-positive entries per config/seed before comparing ranks
if ADV_COLS and merged_results_df is not None:
    spearman_pos_records = []

    for adv_col in ADV_COLS:
        valid_mask = merged_training_df[adv_col].apply(
            lambda x: isinstance(x, np.ndarray) and (x > 0).any()
        )
        adv_df = merged_training_df[valid_mask][
            ["hp.agent_name", "dataset", "config_index", "seed", "eval_step", adv_col]
        ].merge(
            phase_map_simple,
            on=["hp.agent_name", "dataset", "eval_step"],
            how="inner",
        )

        for (agent_name, dataset, seed, eval_step, phase_num), grp in adv_df.groupby(
            ["hp.agent_name", "dataset", "seed", "eval_step", "phase_num"]
        ):
            # For each config, keep only the indices where ALL configs are positive (same seed)
            arrays = [
                row[adv_col].astype(np.float64)
                for _, row in grp.iterrows()
                if isinstance(row[adv_col], np.ndarray)
            ]
            if len(arrays) < 2:
                continue

            min_len = min(len(a) for a in arrays)
            arrays = [a[:min_len] for a in arrays]

            # Mask to positions that are positive in ALL arrays
            pos_mask = np.ones(min_len, dtype=bool)
            for a in arrays:
                pos_mask &= a > 0

            if pos_mask.sum() < 2:
                continue

            mat = np.stack([a[pos_mask] for a in arrays])
            n = mat.shape[0]

            corr_result = _spearmanr(mat.T)
            if n == 2:
                corrs = [float(corr_result.statistic)]
            else:
                triu_idx = np.triu_indices(n, k=1)
                corrs = corr_result.statistic[triu_idx].tolist()

            spearman_pos_records.append(
                {
                    "hp.agent_name": agent_name,
                    "dataset": dataset,
                    "actor": adv_col,
                    "seed": seed,
                    "phase_num": phase_num,
                    "mean_spearman_r": float(np.mean(corrs)),
                    "median_spearman_r": float(np.median(corrs)),
                    "n_pairs": len(corrs),
                    "n_pos_samples": int(pos_mask.sum()),
                }
            )

    if spearman_pos_records:
        spearman_pos_df = pd.DataFrame(spearman_pos_records)
        print("\nPairwise Spearman rank correlation of advantages (positive only) across configs (same seed):")
        agg_pos = (
            spearman_pos_df.groupby(["hp.agent_name", "actor", "phase_num"])
            .agg(
                mean_spearman_r=("mean_spearman_r", "mean"),
                median_spearman_r=("median_spearman_r", "median"),
                mean_n_pos_samples=("n_pos_samples", "mean"),
                n_eval_steps=("phase_num", "count"),
            )
            .reset_index()
            .sort_values(["hp.agent_name", "actor", "phase_num"])
        )
        print(agg_pos.to_string(index=False))
    else:
        print("\nNo positive-only Spearman records computed.")

# %%
# Pairwise Spearman rank correlation across phases within the same config/seed
# Measures how stable the relative ordering of batch samples is over training
if ADV_COLS and merged_results_df is not None:
    spearman_cross_phase_records = []

    for adv_col in ADV_COLS:
        valid_mask = merged_training_df[adv_col].apply(
            lambda x: isinstance(x, np.ndarray) and len(x) > 0
        )
        adv_df = merged_training_df[valid_mask][
            ["hp.agent_name", "dataset", "config_index", "seed", "eval_step", adv_col]
        ].merge(
            phase_map_simple,
            on=["hp.agent_name", "dataset", "eval_step"],
            how="inner",
        )

        for (agent_name, dataset, config_index, seed), grp in adv_df.groupby(
            ["hp.agent_name", "dataset", "config_index", "seed"]
        ):
            grp = grp.sort_values("phase_num")
            phases = grp["phase_num"].tolist()
            arrays = [
                row[adv_col].astype(np.float64)
                for _, row in grp.iterrows()
                if isinstance(row[adv_col], np.ndarray)
            ]
            if len(arrays) < 2:
                continue

            min_len = min(len(a) for a in arrays)
            mat = np.stack([a[:min_len] for a in arrays])
            n = mat.shape[0]

            corr_result = _spearmanr(mat.T)
            if n == 2:
                pairs = [(phases[0], phases[1], float(corr_result.statistic))]
            else:
                triu_idx = np.triu_indices(n, k=1)
                pairs = [
                    (phases[i], phases[j], float(corr_result.statistic[i, j]))
                    for i, j in zip(triu_idx[0], triu_idx[1])
                ]

            for phase_a, phase_b, r in pairs:
                spearman_cross_phase_records.append(
                    {
                        "hp.agent_name": agent_name,
                        "dataset": dataset,
                        "actor": adv_col,
                        "config_index": config_index,
                        "seed": seed,
                        "phase_a": phase_a,
                        "phase_b": phase_b,
                        "spearman_r": r,
                    }
                )

    if spearman_cross_phase_records:
        spearman_cross_phase_df = pd.DataFrame(spearman_cross_phase_records)
        print("\nPairwise Spearman rank correlation across phases (within same config/seed):")
        agg_cross_phase = (
            spearman_cross_phase_df.groupby(["hp.agent_name", "actor", "phase_a", "phase_b"])
            .agg(
                mean_spearman_r=("spearman_r", "mean"),
                median_spearman_r=("spearman_r", "median"),
                n_configs_seeds=("spearman_r", "count"),
            )
            .reset_index()
            .sort_values(["hp.agent_name", "actor", "phase_a", "phase_b"])
        )
        print(agg_cross_phase.to_string(index=False))
    else:
        print("\nNo cross-phase Spearman records computed.")

# %%
# Pairwise Spearman rank correlation across configs/seeds per phase — top 50% only
# Keeps only positions that are above each array's own median, then intersects masks.
# Measures ranking agreement among samples that are high-advantage for all configs/seeds.
if ADV_COLS and merged_results_df is not None:
    spearman_top50_records = []

    for adv_col in ADV_COLS:
        valid_mask = merged_training_df[adv_col].apply(
            lambda x: isinstance(x, np.ndarray) and len(x) > 0
        )
        adv_df = merged_training_df[valid_mask][
            ["hp.agent_name", "dataset", "config_index", "seed", "eval_step", adv_col]
        ].merge(
            phase_map_simple,
            on=["hp.agent_name", "dataset", "eval_step"],
            how="inner",
        )

        for (agent_name, dataset, seed, eval_step, phase_num), grp in adv_df.groupby(
            ["hp.agent_name", "dataset", "seed", "eval_step", "phase_num"]
        ):
            arrays = [
                row[adv_col].astype(np.float64)
                for _, row in grp.iterrows()
                if isinstance(row[adv_col], np.ndarray)
            ]
            if len(arrays) < 2:
                continue

            min_len = min(len(a) for a in arrays)
            arrays = [a[:min_len] for a in arrays]

            # Intersect per-array top-50% masks (above each array's own median)
            shared_mask = np.ones(min_len, dtype=bool)
            for a in arrays:
                shared_mask &= a > np.median(a)

            if shared_mask.sum() < 2:
                continue

            mat = np.stack([a[shared_mask] for a in arrays])
            n = mat.shape[0]

            corr_result = _spearmanr(mat.T)
            if n == 2:
                corrs = [float(corr_result.statistic)]
            else:
                triu_idx = np.triu_indices(n, k=1)
                corrs = corr_result.statistic[triu_idx].tolist()

            spearman_top50_records.append(
                {
                    "hp.agent_name": agent_name,
                    "dataset": dataset,
                    "actor": adv_col,
                    "seed": seed,
                    "phase_num": phase_num,
                    "mean_spearman_r": float(np.mean(corrs)),
                    "median_spearman_r": float(np.median(corrs)),
                    "n_pairs": len(corrs),
                    "n_shared_samples": int(shared_mask.sum()),
                }
            )

    if spearman_top50_records:
        spearman_top50_df = pd.DataFrame(spearman_top50_records)
        print("\nPairwise Spearman rank correlation (top 50% per array, intersected) across configs (same seed):")
        agg_top50 = (
            spearman_top50_df.groupby(["hp.agent_name", "actor", "phase_num"])
            .agg(
                mean_spearman_r=("mean_spearman_r", "mean"),
                median_spearman_r=("median_spearman_r", "median"),
                mean_n_shared_samples=("n_shared_samples", "mean"),
                n_eval_steps=("phase_num", "count"),
            )
            .reset_index()
            .sort_values(["hp.agent_name", "actor", "phase_num"])
        )
        print(agg_top50.to_string(index=False))
    else:
        print("\nNo top-50% Spearman records computed.")

# %%
# Compute oracle (maze-distance) advantages post-hoc by reproducing the held_out_val_batch.
# The `dataset` column stores comma-separated per-phase dataset names, so each phase may use
# a different dataset.
# oracle[i] = dist(obs_xy[i], goal_xy[i]) - dist(next_obs_xy[i], goal_xy[i])
# Positive = moved closer to goal = genuinely good transition.
# Only valid for locomaze environments (AntEnv, HumanoidEnv, PointEnv).
#
# actor_p_trajgoal is a SWEPT hyperparameter (range [0,1]) that controls goal relabeling
# in GCDataset.sample(). The oracle must use the same actor_p_trajgoal as each training
# config so that actor_goals match. Oracle is therefore keyed by
# (single_phase_dataset, actor_p_trajgoal).
#
# Raw datasets are cached per single_phase_dataset to avoid re-loading HDF5 files.
# GCDataset is re-wrapped with the per-config actor_p_trajgoal on top of the cached raw data.
#
# GCDataset.sample() does not yet support a seed kwarg for goal relabeling;
# training.py calls np.random.seed(seed) immediately before sampling, so we replicate
# that here with seed=0 (matching seed=0 training runs only).
from ogbench import make_env_and_datasets as _make_env_and_datasets
from ogbench.locomaze.ant import AntEnv
from ogbench.locomaze.humanoid import HumanoidEnv
from ogbench.locomaze.point import PointEnv
from ogbench.impls.agents.crl import get_config as _crl_get_config
from ogbench.impls.utils.datasets import GCDataset as _GCDataset, Dataset as _OGBDataset
from ml_collections import ConfigDict as _MLConfigDict
_CONST_VAL_BATCH_SIZE = 256
ORACLE_ADV_KEY = "advantage/oracle"

# Collect unique (full_dataset, phase_num, actor_p_trajgoal, seed) from all seeds.
# Oracle batch varies per seed because goal relabeling in GCDataset.sample() uses
# np.random state, which training.py seeds to the training seed before sampling.
_oracle_all_base = merged_training_df.merge(
    phase_map_simple, on=["hp.agent_name", "dataset", "eval_step"], how="inner"
)
_phase_oracle_inputs = (
    _oracle_all_base[["dataset", "phase_num", "hp.actor_p_trajgoal", "seed"]]
    .drop_duplicates()
)

# Cache: single_dataset_name -> (env, val_raw_dict).  Env kept open for distance queries.
_oracle_env_cache: dict[str, tuple] = {}
# Cache: (single_dataset, actor_p_trajgoal) -> GCDataset.  Avoids rebuilding per seed.
_oracle_gc_cache: dict[tuple, object] = {}
# Single datasets that failed to load — skip all seeds for these.
_oracle_failed_datasets: set[str] = set()
# Results keyed by (single_dataset, actor_p_trajgoal, seed).
oracle_advantages: dict[tuple[str, float, int], np.ndarray | None] = {}

for _, _phase_row in _phase_oracle_inputs.iterrows():
    _full_dataset = str(_phase_row["dataset"])
    _phase_idx = int(_phase_row["phase_num"]) - 1  # phase_num is 1-indexed
    _actor_p_trajgoal = float(_phase_row.get("hp.actor_p_trajgoal", 1.0))
    _seed_int = int(_phase_row["seed"])
    _phase_datasets = [s.strip() for s in _full_dataset.split(",")]
    _single_dataset = _phase_datasets[min(_phase_idx, len(_phase_datasets) - 1)]
    _key = (_single_dataset, round(_actor_p_trajgoal, 8), _seed_int)

    if _key in oracle_advantages:
        continue

    if _single_dataset in _oracle_failed_datasets:
        oracle_advantages[_key] = None
        continue

    # Load raw dataset once per single_dataset name.
    if _single_dataset not in _oracle_env_cache:
        try:
            _env_tmp, _, _val_raw = _make_env_and_datasets(_single_dataset)
        except Exception as _e:
            print(f"Could not load dataset {_single_dataset}: {_e}")
            _oracle_failed_datasets.add(_single_dataset)
            oracle_advantages[_key] = None
            continue

        if not isinstance(_env_tmp.unwrapped, (AntEnv, HumanoidEnv, PointEnv)):
            print(f"Skipping oracle for {_single_dataset} (not a locomaze env)")
            _oracle_failed_datasets.add(_single_dataset)
            oracle_advantages[_key] = None
            _env_tmp.close()
            continue

        _oracle_env_cache[_single_dataset] = (_env_tmp, _val_raw)

    _env, _val_raw = _oracle_env_cache[_single_dataset]

    # Build GCDataset once per (single_dataset, actor_p_trajgoal); reuse across seeds.
    _gc_key = (_single_dataset, round(_actor_p_trajgoal, 8))
    if _gc_key not in _oracle_gc_cache:
        _config_dict = _crl_get_config().to_dict()
        _config_dict["actor_p_trajgoal"] = _actor_p_trajgoal
        _config_dict["actor_p_randomgoal"] = 1.0 - _actor_p_trajgoal
        _config_dict["actor_p_curgoal"] = 0.0
        _oracle_gc_cache[_gc_key] = _GCDataset(_OGBDataset.create(**_val_raw), _MLConfigDict(_config_dict))
    _val_dataset = _oracle_gc_cache[_gc_key]

    # Reproduce held_out_val_batch for this training seed.
    # training.py seeds np.random to the training seed before sampling, so goal
    # relabeling in GCDataset.sample() uses that seed — actor_goals differ per seed.
    # Indices are fixed via rng(seed=0), matching training.py exactly.
    np.random.seed(_seed_int)
    _idxs = np.random.default_rng(seed=0).integers(
        low=0, high=_val_dataset.size, size=_CONST_VAL_BATCH_SIZE
    )
    _batch = _val_dataset.sample(_CONST_VAL_BATCH_SIZE, idxs=_idxs)

    _obs_xy  = np.asarray(_batch["observations"])[:, :2]
    _next_xy = np.asarray(_batch["next_observations"])[:, :2]
    _goal_xy = np.asarray(_batch["actor_goals"])[:, :2]

    _oracle = (
        np.linalg.norm(_obs_xy - _goal_xy, axis=1)
        - np.linalg.norm(_next_xy - _goal_xy, axis=1)
    ).astype(np.float64)

    oracle_advantages[_key] = _oracle
    _inf_count = np.sum(np.isinf(_oracle))
    _nan_count = np.sum(np.isnan(_oracle))
    print(
        f"Oracle computed for {_single_dataset} (actor_p_trajgoal={_actor_p_trajgoal:.3f}, seed={_seed_int}): "
        f"mean={np.nanmean(_oracle):.3f}, std={np.nanstd(_oracle):.3f}, "
        f"frac_positive={((_oracle[np.isfinite(_oracle)] > 0).mean()):.2f}, "
        f"unique_vals={len(np.unique(_oracle[np.isfinite(_oracle)]))}, "
        f"inf={_inf_count}, nan={_nan_count}"
    )

# Close cached envs now that all oracle batches are computed.
for _env, _ in _oracle_env_cache.values():
    _env.close()
_oracle_env_cache.clear()

# %%
# Correlation between oracle advantage and predicted advantages per phase.
# Oracle is looked up per-phase using the phase-specific dataset name.
_predicted_cols = [c for c in ADV_COLS if c != ORACLE_ADV_KEY]

if any(v is not None for v in oracle_advantages.values()) and _predicted_cols:
    from scipy.stats import spearmanr as _spearmanr_oracle, pearsonr as _pearsonr_oracle, kendalltau as _kendalltau_oracle

    corr_oracle_records = []
    corr_oracle_top50_records = []
    _diag_printed = False  # print one detailed diagnostic row

    # Build lookup: (agent, dataset, config_index, seed, eval_step, actor) -> advantage_norm array.
    # advantage_norm is already z-scored per batch in adv_all_df.
    _norm_lookup: dict = (
        adv_all_df
        .groupby(["hp.agent_name", "dataset", "config_index", "seed", "eval_step", "actor"])["advantage_norm"]
        .apply(np.array)
        .to_dict()
    )
    _raw_lookup: dict = (
        adv_all_df
        .groupby(["hp.agent_name", "dataset", "config_index", "seed", "eval_step", "actor"])["advantage"]
        .apply(np.array)
        .to_dict()
    )

    # Need actor_p_trajgoal for oracle key lookup — pull from merged_training_df (all seeds).
    _oracle_merged_df = merged_training_df.merge(
        phase_map_simple,
        on=["hp.agent_name", "dataset", "eval_step"],
        how="inner",
    )[["hp.agent_name", "dataset", "config_index", "seed", "eval_step", "phase_num", "hp.actor_p_trajgoal"]].drop_duplicates()

    for _, _row in _oracle_merged_df.iterrows():
        _phase_idx = int(_row["phase_num"]) - 1
        _phase_datasets = [s.strip() for s in str(_row["dataset"]).split(",")]
        _single_dataset = _phase_datasets[min(_phase_idx, len(_phase_datasets) - 1)]
        _actor_p_trajgoal = float(_row.get("hp.actor_p_trajgoal", 1.0))
        _oracle_raw = oracle_advantages.get((_single_dataset, round(_actor_p_trajgoal, 8), int(_row["seed"])))
        if _oracle_raw is None:
            continue
        # Normalize oracle the same way (z-score).
        _o_f = _oracle_raw.astype(np.float64)
        _o_std = _o_f.std()
        _oracle_norm = (_o_f - _o_f.mean()) / _o_std if _o_std > 0 else np.zeros_like(_o_f)

        for _pred_col in _predicted_cols:
            _lookup_key = (
                _row["hp.agent_name"], _row["dataset"],
                _row["config_index"], _row["seed"],
                _row["eval_step"], _pred_col,
            )
            _p_norm = _norm_lookup.get(_lookup_key)
            _p_raw = _raw_lookup.get(_lookup_key)
            if _p_norm is None or len(_p_norm) == 0:
                continue
            if not _diag_printed:
                _diag_printed = True
                print(
                    f"\n[DIAG] First correlation row: config={_row.get('config_index')}, "
                    f"seed={_row.get('seed')}, eval_step={_row.get('eval_step')}, "
                    f"dataset={_single_dataset}, actor_p_trajgoal={_actor_p_trajgoal:.3f}, col={_pred_col}"
                )
                print(
                    f"  oracle_norm: min={np.nanmin(_oracle_norm):.4f}, max={np.nanmax(_oracle_norm):.4f}, "
                    f"mean={np.nanmean(_oracle_norm):.4f}, std={np.nanstd(_oracle_norm):.4f}"
                )
                print(
                    f"  pred_norm:   min={np.nanmin(_p_norm):.4f}, max={np.nanmax(_p_norm):.4f}, "
                    f"mean={np.nanmean(_p_norm):.4f}, std={np.nanstd(_p_norm):.4f}"
                )
            _min_len = min(len(_oracle_norm), len(_p_norm))
            _o = _oracle_norm[:_min_len]
            _p = _p_norm[:_min_len]
            # Filter out inf/nan in oracle (e.g. unreachable maze cells).
            _finite_mask = np.isfinite(_o) & np.isfinite(_p)
            _o = _o[_finite_mask]
            _p = _p[_finite_mask]
            if len(_o) < 10:
                continue
            _sp_r, _ = _spearmanr_oracle(_o, _p)
            _pe_r, _ = _pearsonr_oracle(_o, _p)
            _kt, _ = _kendalltau_oracle(_o, _p)
            _mae = float(np.mean(np.abs(_o - _p)))
            _mse = float(np.mean((_o - _p) ** 2))
            _bias = float(np.mean(_p - _o))
            # Raw (non-normalized) MAE, MSE and bias
            _oracle_raw_clipped = _o_f[:_min_len][_finite_mask]
            _p_raw_arr = _p_raw[:_min_len][_finite_mask] if _p_raw is not None else None
            if _p_raw_arr is not None and len(_p_raw_arr) >= 10:
                _raw_finite = np.isfinite(_oracle_raw_clipped) & np.isfinite(_p_raw_arr)
                _diff_raw = _p_raw_arr[_raw_finite] - _oracle_raw_clipped[_raw_finite]
                _mae_raw = float(np.mean(np.abs(_diff_raw)))
                _mse_raw = float(np.mean(_diff_raw ** 2))
                _bias_raw = float(np.mean(_diff_raw))
            else:
                _mae_raw = float("nan")
                _mse_raw = float("nan")
                _bias_raw = float("nan")
            corr_oracle_records.append(
                {
                    "hp.agent_name": _row["hp.agent_name"],
                    "dataset": _row["dataset"],
                    "config_index": _row["config_index"],
                    "seed": _row["seed"],
                    "phase_num": _row["phase_num"],
                    "predicted": _pred_col,
                    "spearman_r": float(_sp_r),
                    "pearson_r": float(_pe_r),
                    "kendall_tau": float(_kt),
                    "mae": _mae,
                    "mse": _mse,
                    "bias": _bias,
                    "mae_raw": _mae_raw,
                    "mse_raw": _mse_raw,
                    "bias_raw": _bias_raw,
                }
            )
            # Top-50% oracle: restrict to transitions where oracle advantage >= median.
            _top50_mask = _o >= np.median(_o)
            _o50 = _o[_top50_mask]
            _p50 = _p[_top50_mask]
            if len(_o50) >= 10:
                _sp_r50, _ = _spearmanr_oracle(_o50, _p50)
                _pe_r50, _ = _pearsonr_oracle(_o50, _p50)
                _kt50, _ = _kendalltau_oracle(_o50, _p50)
                _mae50 = float(np.mean(np.abs(_o50 - _p50)))
                _mse50 = float(np.mean((_o50 - _p50) ** 2))
                _bias50 = float(np.mean(_p50 - _o50))
                _oraw50 = _oracle_raw_clipped[_top50_mask]
                _praw50 = _p_raw_arr[_top50_mask] if _p_raw_arr is not None else None
                if _praw50 is not None and len(_praw50) >= 10:
                    _rfinite50 = np.isfinite(_oraw50) & np.isfinite(_praw50)
                    _diff50 = _praw50[_rfinite50] - _oraw50[_rfinite50]
                    _mae_raw50 = float(np.mean(np.abs(_diff50)))
                    _mse_raw50 = float(np.mean(_diff50 ** 2))
                    _bias_raw50 = float(np.mean(_diff50))
                else:
                    _mae_raw50 = float("nan")
                    _mse_raw50 = float("nan")
                    _bias_raw50 = float("nan")
                corr_oracle_top50_records.append(
                    {
                        "hp.agent_name": _row["hp.agent_name"],
                        "dataset": _row["dataset"],
                        "config_index": _row["config_index"],
                        "seed": _row["seed"],
                        "phase_num": _row["phase_num"],
                        "predicted": _pred_col,
                        "spearman_r": float(_sp_r50),
                        "pearson_r": float(_pe_r50),
                        "kendall_tau": float(_kt50),
                        "mae": _mae50,
                        "mse": _mse50,
                        "bias": _bias50,
                        "mae_raw": _mae_raw50,
                        "mse_raw": _mse_raw50,
                        "bias_raw": _bias_raw50,
                    }
                )

    if corr_oracle_records:
        corr_oracle_df = pd.DataFrame(corr_oracle_records)
        print("\nCorrelation between oracle advantage and predicted advantages per phase:")
        agg_oracle = (
            corr_oracle_df.groupby(["hp.agent_name", "predicted", "phase_num"])
            .agg(
                mean_spearman_r=("spearman_r", "mean"),
                median_spearman_r=("spearman_r", "median"),
                mean_pearson_r=("pearson_r", "mean"),
                mean_kendall_tau=("kendall_tau", "mean"),
                mean_mae=("mae", "mean"),
                mean_mse=("mse", "mean"),
                mean_bias=("bias", "mean"),
                mean_mae_raw=("mae_raw", "mean"),
                mean_mse_raw=("mse_raw", "mean"),
                mean_bias_raw=("bias_raw", "mean"),
                n=("spearman_r", "count"),
            )
            .reset_index()
            .sort_values(["hp.agent_name", "predicted", "phase_num"])
        )
        print(agg_oracle.to_string(index=False))

    if corr_oracle_top50_records:
        corr_oracle_top50_df = pd.DataFrame(corr_oracle_top50_records)
        print("\nCorrelation between oracle advantage and predicted advantages per phase (top-50% oracle):")
        agg_oracle_top50 = (
            corr_oracle_top50_df.groupby(["hp.agent_name", "predicted", "phase_num"])
            .agg(
                mean_spearman_r=("spearman_r", "mean"),
                median_spearman_r=("spearman_r", "median"),
                mean_pearson_r=("pearson_r", "mean"),
                mean_kendall_tau=("kendall_tau", "mean"),
                mean_mae=("mae", "mean"),
                mean_mse=("mse", "mean"),
                mean_bias=("bias", "mean"),
                mean_mae_raw=("mae_raw", "mean"),
                mean_mse_raw=("mse_raw", "mean"),
                mean_bias_raw=("bias_raw", "mean"),
                n=("spearman_r", "count"),
            )
            .reset_index()
            .sort_values(["hp.agent_name", "predicted", "phase_num"])
        )
        print(agg_oracle_top50.to_string(index=False))
    else:
        print("\nNo oracle top-50% correlation records computed.")
