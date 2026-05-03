# ruff: noqa
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr as _spearmanr, kendalltau as _kendalltau
from gcrl_landscapes.util.data import load_or_compute
from gcrl_landscapes.evaluation.tabular import compute_merged_df
from _common import parse_args, build_adv_df, literal_lists_to_numpy

zipfiles, plot_dir, top_k = parse_args()
os.makedirs(plot_dir, exist_ok=True)

merged_results_df, merged_training_df = load_or_compute(zipfiles, compute_merged_df)
adv_data = load_or_compute(zipfiles, lambda: build_adv_df(zipfiles, top_k=top_k))
ADV_COLS = adv_data["ADV_COLS"]
phase_map_simple = adv_data["phase_map_simple"]

# Re-apply in-memory transforms (fast, cache stores raw)
merged_training_df.drop(columns=["target/held_out_val_batch_values"], errors="ignore", inplace=True)
for col in ADV_COLS:
    merged_training_df[col] = merged_training_df[col].apply(literal_lists_to_numpy)

# --- G: Spearman Rank Correlations ---

if ADV_COLS and merged_results_df is not None:
    # Variant 1: Pairwise Spearman across seeds per config
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

        for (agent_name, dataset, config_index, eval_step, phase_num), grp in adv_df.groupby(
            ["hp.agent_name", "dataset", "config_index", "eval_step", "phase_num"]
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

            corr_result = _spearmanr(mat.T)
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
                    "config_index": config_index,
                    "phase_num": phase_num,
                    "mean_spearman_r": float(np.mean(corrs)),
                    "median_spearman_r": float(np.median(corrs)),
                    "mean_kendall_tau": float(np.mean(kt_corrs)),
                    "n_pairs": len(corrs),
                }
            )

    if spearman_records:
        spearman_df = pd.DataFrame(spearman_records)
        print("\nPairwise Spearman rank correlation of advantages across seeds (same config):")
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
        print("\nNo Spearman records computed (no phase-boundary eval_steps with multiple seeds found).")

    # Variant 2: Positive advantages only, across seeds per config
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

        for (agent_name, dataset, config_index, eval_step, phase_num), grp in adv_df.groupby(
            ["hp.agent_name", "dataset", "config_index", "eval_step", "phase_num"]
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
                    "config_index": config_index,
                    "phase_num": phase_num,
                    "mean_spearman_r": float(np.mean(corrs)),
                    "median_spearman_r": float(np.median(corrs)),
                    "n_pairs": len(corrs),
                    "n_pos_samples": int(pos_mask.sum()),
                }
            )

    if spearman_pos_records:
        spearman_pos_df = pd.DataFrame(spearman_pos_records)
        print("\nPairwise Spearman rank correlation of advantages (positive only) across seeds (same config):")
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

    # Variant 3: Across phases within the same config/seed
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

    # Variant 4: Top 50% only, across seeds per config
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

        for (agent_name, dataset, config_index, eval_step, phase_num), grp in adv_df.groupby(
            ["hp.agent_name", "dataset", "config_index", "eval_step", "phase_num"]
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
                    "config_index": config_index,
                    "phase_num": phase_num,
                    "mean_spearman_r": float(np.mean(corrs)),
                    "median_spearman_r": float(np.median(corrs)),
                    "n_pairs": len(corrs),
                    "n_shared_samples": int(shared_mask.sum()),
                }
            )

    if spearman_top50_records:
        spearman_top50_df = pd.DataFrame(spearman_top50_records)
        print("\nPairwise Spearman rank correlation (top 50% per array, intersected) across seeds (same config):")
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
