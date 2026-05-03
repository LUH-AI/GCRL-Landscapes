# ruff: noqa
import ast
import types
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr as _pearsonr
import gcrl_landscapes.evaluation.tabular as _tabular

_tabular.args = types.SimpleNamespace(no_multiprocessing=True)

from gcrl_landscapes.util.data import load_or_compute
from gcrl_landscapes.evaluation.tabular import compute_merged_df


def parse_args():
    import argparse

    _p = argparse.ArgumentParser()
    _p.add_argument("--zipfiles", nargs="+", type=Path)
    _p.add_argument("--top-k", type=int, default=None)
    _args, _ = _p.parse_known_args()
    zipfiles = _args.zipfiles or [
        Path(
            "/home/mtoepperwien/Documents/gcrl/gcrl_results/2026-04-19-logs-antmaze.zip"
        )
    ]
    plot_dir = Path("plots") / zipfiles[0].name / "advantages"
    return zipfiles, plot_dir, _args.top_k


def _iqm(series: pd.Series | np.ndarray) -> float:
    """Interquartile mean: trim 25%% each tail, compute mean of remainder."""
    arr = np.sort(np.asarray(series).flatten())
    trim = len(arr) // 4
    if trim == 0:
        return float(arr.mean())
    return float(arr[trim:-trim].mean())


def get_top_k_configs(
    merged_results_df: pd.DataFrame, k: int = 5
) -> dict[str, list[int]]:
    """
    Identify top-k configs per agent by final phase success (IQM across seeds).
    Returns dict[agent -> sorted list of top-k config_indices].
    Raises ValueError if fewer than k configs exist for any agent.
    """
    # Get last phase for each agent
    last_phase = (
        merged_results_df.groupby("hp.agent_name")["phase_num"]
        .max()
        .reset_index()
        .rename(columns={"phase_num": "last_phase_num"})
    )

    # Filter to last phase only
    last_phase_df = merged_results_df.merge(last_phase, on="hp.agent_name")
    last_phase_df = last_phase_df[
        last_phase_df["phase_num"] == last_phase_df["last_phase_num"]
    ]

    # IQM across seeds per config
    iqm_success = (
        last_phase_df.groupby(["hp.agent_name", "config_index"])["success"]
        .apply(_iqm)
        .reset_index()
        .rename(columns={"success": "iqm_success"})
    )

    # Top-k per agent
    top_configs: dict[str, list[int]] = {}
    for agent, grp in iqm_success.groupby("hp.agent_name"):
        grp_sorted = grp.sort_values("iqm_success", ascending=False)
        if len(grp_sorted) < k:
            raise ValueError(
                f"Agent {agent} has only {len(grp_sorted)} configs, "
                f"fewer than requested k={k}"
            )
        top_configs[agent] = grp_sorted.head(k)["config_index"].tolist()

    return top_configs


def literal_lists_to_numpy(s):
    try:
        return np.array(ast.literal_eval(s), dtype=np.float32)
    except:
        return None


def _pearson_r(g: pd.DataFrame, x: str, y: str) -> pd.Series:
    if len(g) < 3:
        return pd.Series({"pearson_r": float("nan"), "p_value": float("nan")})
    r, p = _pearsonr(g[x], g[y])
    return pd.Series({"pearson_r": float(r), "p_value": float(p)})


def _ess(w: pd.Series) -> float:
    """Normalized ESS in [0,1]: 1 = uniform weights, 0 = single sample dominates."""
    w_arr = w.values.astype(np.float64)
    w_finite = w_arr.clip(min=10e-10, max=100)
    if len(w_finite) == 0:
        return float("nan")
    w_scaled = w_finite / w_finite.max()
    return float(w_scaled.sum() ** 2 / (len(w_scaled) * (w_scaled**2).sum()))


def build_adv_df(filepaths, top_k: int | None = None):
    """Cached via load_or_compute. Builds adv_all_df + phase_map_simple."""
    merged_results_df, merged_training_df = load_or_compute(
        filepaths, compute_merged_df
    )

    # Memory optimization
    to_category_columns = ["dataset", "hps", "agent"] + merged_training_df.columns[
        merged_training_df.columns.str.startswith("hp.")
    ].tolist()
    for col in to_category_columns:
        if col in merged_training_df.columns:
            merged_training_df[col] = merged_training_df[col].astype("category")
    float_cols = merged_training_df.select_dtypes(include="float64").columns
    for col in float_cols:
        merged_training_df[col] = merged_training_df[col].astype("float16")
    merged_training_df.drop(
        columns=["target/held_out_val_batch_values"], errors="ignore", inplace=True
    )

    ADV_COLS = [c for c in merged_training_df.columns if c.startswith("advantage/")]
    print(f"Advantage columns found: {ADV_COLS}")

    for col in ADV_COLS:
        merged_training_df[col] = merged_training_df[col].apply(literal_lists_to_numpy)

    # Build long-format dataframe over ALL phases
    adv_all_df = pd.DataFrame()
    adv_last_phase_df = pd.DataFrame()
    phase_map_simple = pd.DataFrame(
        columns=["hp.agent_name", "dataset", "eval_step", "phase_num"]
    )

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
                adv_norm = (
                    (adv_f64 - _mu) / _sigma if _sigma > 0 else np.zeros_like(adv_f64)
                )
                adv_norm_spread = (
                    (adv_f64) / _sigma if _sigma > 0 else np.zeros_like(adv_f64)
                )

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

        _last_phase = adv_all_df["phase_num"].max()
        adv_last_phase_df = adv_all_df[adv_all_df["phase_num"] == _last_phase].copy()
        print(
            f"Last-phase dataframe (phase {_last_phase}): {len(adv_last_phase_df)} rows"
        )

        # Build phase_map_simple: assigns phase_num to every eval_step via forward merge_asof
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
        for (_agent, _dataset), _grp in _all_train_steps.groupby(
            ["hp.agent_name", "dataset"]
        ):
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
            else pd.DataFrame(
                columns=["hp.agent_name", "dataset", "eval_step", "phase_num"]
            )
        )

    result = {
        "adv_all_df": adv_all_df,
        "adv_last_phase_df": adv_last_phase_df,
        "phase_map_simple": phase_map_simple,
        "ADV_COLS": ADV_COLS,
    }

    # Filter to top-k configs per agent
    if top_k is not None:
        top_configs = get_top_k_configs(merged_results_df, k=top_k)
        for agent, configs in top_configs.items():
            agent_mask = result["adv_all_df"]["hp.agent_name"] == agent
            config_mask = result["adv_all_df"]["config_index"].isin(configs)
            result["adv_all_df"] = result["adv_all_df"][~(agent_mask & ~config_mask)]
            result["adv_last_phase_df"] = result["adv_last_phase_df"][
                ~(agent_mask & ~config_mask)
            ]
        print(f"Filtered to top-{top_k} configs per agent")

    return result
