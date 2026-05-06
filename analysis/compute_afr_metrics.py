#!/usr/bin/env python3
"""Compute AFR (Future-Random Advantage) metrics from advantages.parquet.

For each (checkpoint, actor) matrix the parquet stores A(obs_i, goal_j) for
every (i, j) pair from a 256-sample validation batch.  Diagonal entries
(obs_idx == goal_idx, is_positive=True) are the future/geometric goal pairs
(A+).  Off-diagonal entries are random-goal pairs (A-).

Metrics computed per checkpoint:
  - fr_auc:               Pr(A+ > A-)  across all off-diagonal (i,j) pairs
  - gap_mean:             E[A+ - A-]
  - gap_std:              Std[A+ - A-]
  - extractability_index: gap_mean / (gap_std + eps)
  - adv_plus_mean:        mean of diagonal advantages
  - adv_minus_mean:       mean of off-diagonal advantages

Reference: afr_signal_ogbench_diagnostic.md
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


BATCH_SIZE = 256
EPS = 1e-8

_GROUP_COLS = [
    "checkpoint_path",
    "agent",
    "dataset",
    "phase",
    "seed",
    "configuration",
    "actor",
]

_META_COLS = [
    "checkpoint_path",
    "agent",
    "dataset",
    "phase",
    "seed",
    "configuration",
    "actor",
    "batch_idx",
]


CLUSTER_MOUNT = Path("/tmp/cluster-mount")
CLUSTER_PREFIX = "/bigwork/USERNAME/gcrl/code/"

# Compiled regex to extract agent dir and config index from checkpoint_path
# e.g. /bigwork/.../CRL_antmaze-large-navigate-v0_64c_lr-alpha/run_logs/configuration_0/phase_926500/seed_0/params_926500.pkl
_CKPT_RE = re.compile(
    r"(?P<agent_dir>CRL_.*?|GCIQL_.*?|GCIVL_.*?|QRL_.*?|HIQL_.*?)"
    r"/run_logs/configuration_(?P<config_idx>\d+)"
)


def _get_alpha(checkpoint_path: str, config_idx: str) -> float:
    """Derive config JSON path from checkpoint_path and extract 'alpha'.

    checkpoint_path points to a .pkl inside a run_logs/configuration_N/ subdir.
    The corresponding config file lives at:
      cluster-mount/{experiment_root}/configurations/configuration_N.json

    Falls back to 0.5 if the file cannot be read.
    """
    m = _CKPT_RE.search(checkpoint_path)
    if m:
        agent_dir = m.group("agent_dir")
        idx = m.group("config_idx")
        # Locate the agent dir start in the string to get the experiment root
        prefix_end = checkpoint_path.index(agent_dir) + len(agent_dir)
        experiment_root = checkpoint_path[:prefix_end]
        # Strip cluster prefix and route through cluster-mount
        local_root = CLUSTER_MOUNT / Path(experiment_root).relative_to(CLUSTER_PREFIX)
        config_path = local_root / "configurations" / f"configuration_{idx}.json"
        try:
            config = json.loads(config_path.read_text())
            for key, val in config.items():
                if key.lower() == "alpha":
                    return float(val)
        except Exception:
            pass
    return 0.5


def afr_metrics(
    adv_matrix: np.ndarray, alpha: float, w_max: float = 100.0, eps: float = EPS
) -> dict:
    """Compute AFR metrics from NxN advantage matrix.

    adv_matrix[i, j] = A_alg(obs_i, goal_j)
    Diagonal (i == j): A+ — future/geometric goal paired with obs_i
    Off-diagonal (i != j): A- — random goal for obs_i

    For every pair (i, j) with i != j:
        gap = A+[i] - A-[i, j]

    Returns scalar metrics dict.
    """
    N = adv_matrix.shape[0]
    adv_plus = np.diag(adv_matrix)  # [N]

    # Boolean mask: True where i != j
    off_diag = ~np.eye(N, dtype=bool)  # [N, N]

    # gaps[i, j] = A+[i] - adv_matrix[i, j]  for j != i
    # Shape after masking: [N*(N-1)]
    gaps = (adv_plus[:, None] - adv_matrix)[off_diag]

    gap_mean = float(np.mean(gaps))
    gap_std = float(np.std(gaps))
    fr_auc = float(np.mean(gaps > 0.0))
    ei = gap_mean / (gap_std + eps)

    # Cross-goal ranking: rank all 256 goals by descending advantage,
    # find the rank of the positive goal (g_i^+) for each anchor row i.
    # r_i = 1 means highest advantage.
    reciprocal_ranks = np.empty(N, dtype=np.float32)
    for i in range(N):
        ranks = np.argsort(adv_matrix[i])[::-1]  # goal indices sorted by descending A
        rank_of_positive = int(np.where(ranks == i)[0][0]) + 1  # 1-indexed
        reciprocal_ranks[i] = 1.0 / rank_of_positive
    mrr = float(np.mean(reciprocal_ranks))

    # AWR concentration: compute exp(alpha * A+) weights with w_max clipping
    with np.errstate(over="ignore"):
        raw_weights = np.exp(alpha * adv_plus)
    clipped = raw_weights > w_max
    weights = np.minimum(raw_weights, w_max)

    top_k = int(np.ceil(N * 0.05))  # 13
    sorted_w = np.sort(weights)[::-1]
    top_5_pct_mass = float(np.sum(sorted_w[:top_k]) / (np.sum(weights) + eps))
    saturation_mass = float(np.sum(weights[clipped]) / (np.sum(weights) + eps))

    return {
        "fr_auc": fr_auc,
        "gap_mean": gap_mean,
        "gap_std": gap_std,
        "extractability_index": ei,
        "adv_plus_mean": float(np.mean(adv_plus)),
        "adv_minus_mean": float(np.mean(adv_matrix[off_diag])),
        "adv_total_mean": float(np.mean(adv_matrix)),
        "adv_total_std": float(np.std(adv_matrix)),
        "mrr": mrr,
        "top_5_pct_mass": top_5_pct_mass,
        "saturation_mass": saturation_mass,
    }


def _dict_col_scalar(col: pa.ChunkedArray) -> str:
    """Extract the single string value from a constant dictionary-encoded column."""
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    if pa.types.is_dictionary(arr.type):
        arr = arr.dictionary_decode()
    return str(arr[0].as_py())


def process_batch(batch: pa.RecordBatch) -> Optional[dict]:
    """One row group (one checkpoint×actor NxN matrix) → metrics row dict."""
    if batch.num_rows != BATCH_SIZE * BATCH_SIZE:
        # Incomplete checkpoint — skip
        return None

    meta = {col: _dict_col_scalar(batch.column(col)) for col in _META_COLS}
    checkpoint_path = meta["checkpoint_path"]
    config_idx = meta["configuration"]
    alpha = _get_alpha(checkpoint_path, config_idx)

    obs_np = batch.column("obs_idx").to_numpy().astype(np.int32)
    goal_np = batch.column("goal_idx").to_numpy().astype(np.int32)
    adv_np = batch.column("advantage").to_numpy(zero_copy_only=False).astype(np.float32)

    matrix = np.empty((BATCH_SIZE, BATCH_SIZE), dtype=np.float32)
    matrix[obs_np, goal_np] = adv_np

    metrics = afr_metrics(matrix, alpha)
    metrics["alpha"] = alpha
    return {**meta, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute AFR metrics from advantages.parquet.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("advantages.parquet"),
        help="Path to advantages.parquet (default: advantages.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Analysis/afr_metrics.csv"),
        help="Output CSV path (default: Analysis/afr_metrics.csv)",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip aggregation across batches (output one row per batch instead of mean±std).",
    )
    parser.add_argument(
        "--batches",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated list of batch indices to include (e.g. 0,1,2 or 0-5). "
        "If unset, all batches are used.",
    )
    parser.add_argument(
        "--config-ids",
        type=Path,
        default=None,
        help="Path to JSON file with {agent: [config_ids]} to filter to.",
    )
    args = parser.parse_args()

    # Parse batch filter (batch_idx is stored as string in the parquet)
    batch_filter: set[str] | None = None
    if args.batches is not None:
        batch_filter = set()
        for part in args.batches.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                batch_filter.update(str(i) for i in range(int(lo), int(hi) + 1))
            else:
                batch_filter.add(part)
        print(f"Batch filter: {sorted(batch_filter, key=int)}")

    pf = pq.ParquetFile(args.input)
    n_groups = pf.metadata.num_row_groups
    print(f"Processing {n_groups} row groups from {args.input}")

    results = []
    skipped = 0
    for i, batch in enumerate(pf.iter_batches()):
        row = process_batch(batch)
        if row is not None:
            results.append(row)
        else:
            skipped += 1
        if (i + 1) % 100 == 0 or (i + 1) == n_groups:
            print(f"  {i + 1}/{n_groups}  ok={len(results)}  skipped={skipped}")

    result_df = pd.DataFrame(results)

    # Filter by batch index if requested
    if batch_filter is not None:
        before = len(result_df)
        result_df = result_df[result_df["batch_idx"].isin(batch_filter)]
        print(f"Filtered {before} → {len(result_df)} rows (batch filter)")

    # Aggregate across batches: one row per (checkpoint, agent, dataset, phase, seed,
    # configuration, actor) with mean ± std of each metric.
    agg_cols = [
        "fr_auc",
        "gap_mean",
        "gap_std",
        "extractability_index",
        "adv_plus_mean",
        "adv_minus_mean",
        "adv_total_mean",
        "adv_total_std",
        "mrr",
        "top_5_pct_mass",
        "saturation_mass",
    ]
    agg_df = result_df.groupby(_GROUP_COLS)[agg_cols].agg(["mean", "std"]).round(6)
    # Flatten: "fr_auc_mean", "fr_auc_std", ...
    agg_df.columns = [f"{col}_{stat}" for col, stat in agg_df.columns]
    agg_df = agg_df.reset_index()

    n_batches = result_df["batch_idx"].nunique()
    print(f"\nAggregated {n_batches} batch(es) into {len(agg_df)} rows")

    if args.no_aggregate:
        out_df = result_df[_GROUP_COLS + ["batch_idx"] + agg_cols]
    else:
        out_df = agg_df

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Filter by config IDs if provided (must be after aggregation)
    if args.config_ids is not None:
        config_filter: dict[str, list[str]] = json.loads(args.config_ids.read_text())
        # Normalize: uppercase agent keys, stringify config IDs
        config_filter = {
            agent.upper(): [str(c) for c in ids] for agent, ids in config_filter.items()
        }
        before = len(out_df)
        mask = out_df.apply(
            lambda row: (
                str(row["configuration"]) in config_filter.get(row["agent"], [])
            ),
            axis=1,
        )
        out_df = out_df[mask]
        print(f"Filtered to config IDs: {before} → {len(out_df)} rows")

    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(out_df)} rows → {args.output}")

    summary = out_df.groupby("agent")[
        [
            c
            for c in out_df.columns
            if c.startswith("fr_auc")
            or c.startswith("extractability")
            or c.startswith("gap_mean")
            or c.startswith("adv_total")
            or c.startswith("mrr")
            or c.startswith("top_5_pct")
        ]
    ].agg(["mean", "std"])
    print("\nSummary by agent:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
