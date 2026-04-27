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

zipfiles, plot_dir = parse_args()
os.makedirs(plot_dir, exist_ok=True)

merged_results_df, merged_training_df = load_or_compute(zipfiles, compute_merged_df)
adv_data = load_or_compute(zipfiles, build_adv_df)
adv_all_df = adv_data["adv_all_df"]
ADV_COLS = adv_data["ADV_COLS"]

# --- B: Descriptive Statistics (all phases) ---

if ADV_COLS:
    for (agent, actor), grp in adv_all_df.groupby(["hp.agent_name", "actor"]):
        print(f"\n=== {agent} | {actor} ===")
        desc = grp[["advantage", "advantage_norm", "advantage_norm_spread"]].describe()
        desc.columns = ["advantage (raw)", "advantage (z-scored)", "advantage spread normalized"]
        print(desc.to_string())

if ADV_COLS:
    print("Statistics over mean of advantages in batch")
    for (agent, actor), grp in adv_all_df.groupby(["hp.agent_name", "actor"]):
        print(f"\n=== {agent} | {actor} ===")
        temp_df = grp.groupby(["dataset", "config_index", "seed", "eval_step"])["advantage"].mean()
        print(temp_df.describe().to_string())

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

if ADV_COLS:
    for (agent, actor), grp in adv_all_df.groupby(["hp.agent_name", "actor"]):
        print(f"\n=== {agent} | {actor} ===")
        desc = grp[["weight"]].clip(upper=100).describe()
        desc.columns = ["advantage"]
        print(desc.to_string())

# Fraction of negative advantages per algorithm-dataset combination
if ADV_COLS:
    neg_frac = (
        adv_all_df.groupby(["hp.agent_name", "actor"])["advantage"]
        .apply(lambda x: (x < 0).sum() / len(x))
        .reset_index(name="frac_negative")
    )
    print("\nFraction of negative advantages per agent × dataset × actor:")
    print(neg_frac.to_string(index=False))
