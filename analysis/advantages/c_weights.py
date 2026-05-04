# ruff: noqa
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gcrl_landscapes.util.data import load_or_compute
from gcrl_landscapes.evaluation.tabular import compute_merged_df
from _common import parse_args, build_adv_df, filter_adv_data, get_top_k_configs

zipfiles, plot_dir, top_k = parse_args()
os.makedirs(plot_dir, exist_ok=True)

merged_results_df, merged_training_df = load_or_compute(zipfiles, compute_merged_df)
adv_data = load_or_compute(zipfiles, build_adv_df)
if top_k is not None:
    top_configs = get_top_k_configs(merged_results_df, k=top_k)
    adv_data = filter_adv_data(adv_data, top_configs)
    print(f"Filtered to top-{top_k} configs per agent")
adv_all_df = adv_data["adv_all_df"]
adv_last_phase_df = adv_data["adv_last_phase_df"]
ADV_COLS = adv_data["ADV_COLS"]

# --- C: AWR Weight Distribution (last phase) ---

if ADV_COLS:
    _w_vals = adv_last_phase_df["weight"].apply(lambda x: x if np.isfinite(x) and x > 0 else np.nan).dropna()
    _w_xlim = (float(_w_vals.quantile(0.01)), float(_w_vals.quantile(0.99)))
    if not (np.isfinite(_w_xlim[0]) and _w_xlim[0] > 0):
        _w_xlim = (1e-10, _w_xlim[1])
    if not np.isfinite(_w_xlim[1]):
        _w_xlim = (_w_xlim[0], 1e30)

    for (agent_name, dataset, actor), grp in adv_last_phase_df.groupby(
        ["hp.agent_name", "dataset", "actor"]
    ):
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

# Ratio of weights > 100 per config — statistics across configs
if ADV_COLS:
    high_ratio_last = (
        adv_last_phase_df.groupby(
            ["hp.agent_name", "dataset", "actor", "config_index", "seed"]
        )["weight"]
        .apply(lambda w: (w > 100).sum() / len(w))  # type: ignore[arg-type]
        .reset_index(name="high_ratio")
    )
    print("\nFraction of weights > 100 per config — statistics across configs (last phase):")
    print(high_ratio_last.groupby(["hp.agent_name", "actor"])["high_ratio"].describe())
