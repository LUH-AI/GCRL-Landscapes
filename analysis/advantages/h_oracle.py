# ruff: noqa
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.stats import (
    spearmanr as _spearmanr_oracle,
    pearsonr as _pearsonr_oracle,
    kendalltau as _kendalltau_oracle,
)
from gcrl_landscapes.util.data import load_or_compute
from gcrl_landscapes.evaluation.tabular import compute_merged_df
from _common import parse_args, build_adv_df, filter_adv_data, get_top_k_configs, literal_lists_to_numpy


def _save_tex(filename: Path, caption: str, label: str, df: pd.DataFrame) -> None:
    """Write a DataFrame to a booktabs-style .tex table."""
    col_format = "l" + "r" * len(df.columns)
    header = " & ".join(df.columns.tolist()) + r" \midrule\n"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        rows.append(" & ".join(vals) + r" \ ")
    body = "\n".join(rows)

    tex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\begin{{tabular}}{{{col_format}}}\n"
        "\\toprule\n" + header + body + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}"
    )
    with open(filename, "w") as f:
        f.write(tex)
    print(f"Saved {filename}")


zipfiles, plot_dir, top_k = parse_args()
os.makedirs(plot_dir, exist_ok=True)

merged_results_df, merged_training_df = load_or_compute(zipfiles, compute_merged_df)
adv_data = load_or_compute(zipfiles, build_adv_df)
if top_k is not None:
    top_configs = get_top_k_configs(merged_results_df, k=top_k)
    adv_data = filter_adv_data(adv_data, top_configs)
    print(f"Filtered to top-{top_k} configs per agent")
ADV_COLS = adv_data["ADV_COLS"]
phase_map_simple = adv_data["phase_map_simple"]
adv_all_df = adv_data["adv_all_df"]

# Re-apply in-memory transforms (fast, cache stores raw)
merged_training_df.drop(
    columns=["target/held_out_val_batch_values"], errors="ignore", inplace=True
)
for col in ADV_COLS:
    merged_training_df[col] = merged_training_df[col].apply(literal_lists_to_numpy)

# --- H: Oracle (maze-distance) Advantage Comparison ---

from ogbench import make_env_and_datasets as _make_env_and_datasets
from ogbench.locomaze.ant import AntEnv
from ogbench.locomaze.humanoid import HumanoidEnv
from ogbench.locomaze.point import PointEnv
from ogbench.impls.agents.crl import get_config as _crl_get_config
from ogbench.impls.utils.datasets import GCDataset as _GCDataset, Dataset as _OGBDataset
from ml_collections import ConfigDict as _MLConfigDict

_CONST_VAL_BATCH_SIZE = 256
ORACLE_ADV_KEY = "advantage/oracle"

_oracle_all_base = merged_training_df.merge(
    phase_map_simple, on=["hp.agent_name", "dataset", "eval_step"], how="inner"
)
_phase_oracle_inputs = _oracle_all_base[
    ["dataset", "phase_num", "hp.actor_p_trajgoal", "seed"]
].drop_duplicates()

_oracle_env_cache: dict[str, tuple] = {}
_oracle_gc_cache: dict[tuple, object] = {}
_oracle_failed_datasets: set[str] = set()
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

    if _single_dataset not in _oracle_env_cache:
        try:
            _env_tmp, _, _val_raw = _make_env_and_datasets(_single_dataset)
        except Exception as _e:
            print(f"Could not load dataset {_single_dataset}: {_e}")
            _oracle_failed_datasets.add(_single_dataset)
            oracle_advantages[_key] = None
            continue

        if not isinstance(_env_tmp.unwrapped, (AntEnv, HumanoidEnv, PointEnv)):
            # Cube / other envs: still compute Euclidean oracle
            _oracle_env_cache[_single_dataset] = (_env_tmp, _val_raw)
        else:
            _oracle_env_cache[_single_dataset] = (_env_tmp, _val_raw)

    _env, _val_raw = _oracle_env_cache[_single_dataset]

    _gc_key = (_single_dataset, round(_actor_p_trajgoal, 8))
    if _gc_key not in _oracle_gc_cache:
        _config_dict = _crl_get_config().to_dict()
        _config_dict["actor_p_trajgoal"] = _actor_p_trajgoal
        _config_dict["actor_p_randomgoal"] = 1.0 - _actor_p_trajgoal
        _config_dict["actor_p_curgoal"] = 0.0
        _oracle_gc_cache[_gc_key] = _GCDataset(
            _OGBDataset.create(**_val_raw), _MLConfigDict(_config_dict)
        )
    _val_dataset = _oracle_gc_cache[_gc_key]

    np.random.seed(_seed_int)
    _idxs = np.random.default_rng(seed=0).integers(
        low=0, high=_val_dataset.size, size=_CONST_VAL_BATCH_SIZE
    )
    _batch = _val_dataset.sample(_CONST_VAL_BATCH_SIZE, idxs=_idxs)

    _obs_xy = np.asarray(_batch["observations"])[:, :2]
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

for _env, _ in _oracle_env_cache.values():
    _env.close()
_oracle_env_cache.clear()

# Correlation between oracle advantage and predicted advantages per phase
_predicted_cols = [c for c in ADV_COLS if c != ORACLE_ADV_KEY]

if any(v is not None for v in oracle_advantages.values()) and _predicted_cols:
    corr_oracle_records = []
    corr_oracle_top50_records = []
    _diag_printed = False

    _norm_lookup: dict = (
        adv_all_df.groupby(
            ["hp.agent_name", "dataset", "config_index", "seed", "eval_step", "actor"]
        )["advantage_norm"]
        .apply(np.array)
        .to_dict()
    )
    _raw_lookup: dict = (
        adv_all_df.groupby(
            ["hp.agent_name", "dataset", "config_index", "seed", "eval_step", "actor"]
        )["advantage"]
        .apply(np.array)
        .to_dict()
    )

    _oracle_merged_df = merged_training_df.merge(
        phase_map_simple,
        on=["hp.agent_name", "dataset", "eval_step"],
        how="inner",
    )[
        [
            "hp.agent_name",
            "dataset",
            "config_index",
            "seed",
            "eval_step",
            "phase_num",
            "hp.actor_p_trajgoal",
        ]
    ].drop_duplicates()

    for _, _row in _oracle_merged_df.iterrows():
        _phase_idx = int(_row["phase_num"]) - 1
        _phase_datasets = [s.strip() for s in str(_row["dataset"]).split(",")]
        _single_dataset = _phase_datasets[min(_phase_idx, len(_phase_datasets) - 1)]
        _actor_p_trajgoal = float(_row.get("hp.actor_p_trajgoal", 1.0))
        _oracle_raw = oracle_advantages.get(
            (_single_dataset, round(_actor_p_trajgoal, 8), int(_row["seed"]))
        )
        if _oracle_raw is None:
            continue
        _o_f = _oracle_raw.astype(np.float64)
        _o_std = _o_f.std()
        _oracle_norm = (
            (_o_f - _o_f.mean()) / _o_std if _o_std > 0 else np.zeros_like(_o_f)
        )

        for _pred_col in _predicted_cols:
            _lookup_key = (
                _row["hp.agent_name"],
                _row["dataset"],
                _row["config_index"],
                _row["seed"],
                _row["eval_step"],
                _pred_col,
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
            _oracle_raw_clipped = _o_f[:_min_len][_finite_mask]
            _p_raw_arr = _p_raw[:_min_len][_finite_mask] if _p_raw is not None else None
            if _p_raw_arr is not None and len(_p_raw_arr) >= 10:
                _raw_finite = np.isfinite(_oracle_raw_clipped) & np.isfinite(_p_raw_arr)
                _diff_raw = _p_raw_arr[_raw_finite] - _oracle_raw_clipped[_raw_finite]
                _mae_raw = float(np.mean(np.abs(_diff_raw)))
                _mse_raw = float(np.mean(_diff_raw**2))
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
                    _mse_raw50 = float(np.mean(_diff50**2))
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
        print(
            "\nCorrelation between oracle advantage and predicted advantages per phase:"
        )
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

        # --- .tex output: oracle Spearman r per agent x phase ---
        oracle_tex = (
            agg_oracle[["hp.agent_name", "predicted", "phase_num", "mean_spearman_r"]]
            .copy()
            .rename(
                columns={
                    "hp.agent_name": "Agent",
                    "predicted": "Actor",
                    "phase_num": "Phase",
                    "mean_spearman_r": "Oracle r",
                }
            )
        )
        oracle_tex["Actor"] = oracle_tex["Actor"].str.strip()
        oracle_tex["Phase"] = oracle_tex["Phase"].astype(int).astype(str)

        # Keep only the last phase per agent
        last_phase = int(oracle_tex["Phase"].max())
        oracle_tex = oracle_tex[oracle_tex["Phase"] == str(last_phase)].copy()
        oracle_tex = oracle_tex[["Agent", "Oracle r"]].sort_values("Agent")

        _save_tex(
            plot_dir / "oracle_corr.tex",
            "Oracle (maze-distance) Spearman correlation per agent, actor, and training phase.",
            "tab:oracle_corr",
            oracle_tex,
        )

    if corr_oracle_top50_records:
        corr_oracle_top50_df = pd.DataFrame(corr_oracle_top50_records)
        print(
            "\nCorrelation between oracle advantage and predicted advantages per phase (top-50% oracle):"
        )
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
