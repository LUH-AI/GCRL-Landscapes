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
from _common import parse_args, build_adv_df

zipfiles, plot_dir, top_k = parse_args()
os.makedirs(plot_dir, exist_ok=True)

merged_results_df, merged_training_df = load_or_compute(zipfiles, compute_merged_df)
adv_data = load_or_compute(zipfiles, lambda: build_adv_df(zipfiles, top_k=top_k))
adv_all_df = adv_data["adv_all_df"]
adv_last_phase_df = adv_data["adv_last_phase_df"]
ADV_COLS = adv_data["ADV_COLS"]

# --- A: Advantage Distribution Across Configurations (last phase) ---

sns.set_theme(context="paper", style="whitegrid")

if ADV_COLS and merged_results_df is not None:
    _adv_xlim = (-10, 10)

    for (agent_name, dataset, actor), grp in adv_last_phase_df.groupby(
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
        fname = (
            plot_dir / f"adv_dist_{agent_name}_{dataset}_{actor.replace('/', '_')}.png"
        )
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()

    # Normalized advantage distribution (z-scored per list / per config-seed row)
    _norm_xlim = (-10, 10)

    for (agent_name, dataset, actor), grp in adv_last_phase_df.groupby(
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
        fname = (
            plot_dir
            / f"adv_dist_norm_{agent_name}_{dataset}_{actor.replace('/', '_')}.png"
        )
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
