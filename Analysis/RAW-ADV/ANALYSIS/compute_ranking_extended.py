#!/usr/bin/env python3
"""Extended goal-ranking diagnostics from the 256×256 cross-goal advantage matrix.

For each batch (one 256×256 matrix) computes per-row ranks of the true goal and
derives richer statistics than prec@1/prec@5/MRR already in advantage_matrix_metrics.parquet:

    prec_at_10          fraction of rows where true goal is rank ≤ 10  (top ~4%)
    prec_at_25          fraction of rows where true goal is rank ≤ 25  (top ~10%)
    rank_median         median rank of true goal across 256 rows
    rank_worst_half     fraction of rows where true goal is in the bottom 128 (rank > 128)
    dominance_mean      mean of (A+ - max_j≠i A(s_i, g_j)) — margin over best random competitor
    dominance_pos_frac  fraction of rows where A+ strictly beats the best random goal
                        (= prec_at_1 exactly; included for cross-check)

dominance_mean is the key new metric: it asks not "does the true goal beat a random one?"
(FR-AUC) but "does the true goal beat the BEST random one?" A positive dominance_mean
means the true goal is not just above average — it dominates the hardest competition.

Outputs
-------
    ENVS/<env>/advantage_ranking_extended.parquet  — one row per batch
    ANALYSIS/metric_summaries/ranking_extended_<env>.csv  — agent×env pivot

Usage
-----
    python ANALYSIS/compute_ranking_extended.py --env antmaze-medium
    python ANALYSIS/compute_ranking_extended.py --env antmaze-large
    python ANALYSIS/compute_ranking_extended.py --env cube
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

N = 256


def ranking_metrics(adv_matrix: np.ndarray) -> dict:
    A_plus = np.diag(adv_matrix)  # [N]

    # Rank of the true goal in each row (1 = highest advantage, N = lowest)
    # rank_i = number of goals with advantage >= A+[i]   (includes self → min rank is 1)
    ranks = (adv_matrix >= A_plus[:, None]).sum(axis=1).astype(np.int32)

    # Per-row max of off-diagonal advantages
    tmp = adv_matrix.copy()
    np.fill_diagonal(tmp, -np.inf)
    max_random = tmp.max(axis=1)  # [N]

    dominance = A_plus - max_random  # [N]  positive = beats best rival

    return {
        "prec_at_10": float(np.mean(ranks <= 10)),
        "prec_at_25": float(np.mean(ranks <= 25)),
        "rank_median": float(np.median(ranks)),
        "rank_worst_half": float(np.mean(ranks > 128)),
        "dominance_mean": float(np.mean(dominance)),
        "dominance_pos_frac": float(np.mean(dominance > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    args = parser.parse_args()

    _root = Path(__file__).parent.parent
    env_dir = _root / "ENVS" / args.env
    parquet = env_dir / "advantages.parquet"

    if not parquet.exists():
        raise FileNotFoundError(parquet)

    print(f"Processing {args.env}  ({parquet.stat().st_size / 1e9:.1f} GB) …")
    pf = pq.ParquetFile(parquet)
    n_groups = pf.metadata.num_row_groups

    records = []
    for i, batch in enumerate(pf.iter_batches()):
        if batch.num_rows != N * N:
            continue
        df = batch.to_pandas()

        mat = np.empty((N, N), dtype=np.float32)
        mat[df["obs_idx"].to_numpy(), df["goal_idx"].to_numpy()] = df[
            "advantage"
        ].to_numpy(dtype=np.float32)

        rec = {
            "agent": str(df["agent"].iloc[0]),
            "config": int(df["configuration"].iloc[0]),
            "seed": int(df["seed"].iloc[0]),
            "phase": str(df["phase"].iloc[0]),
            "actor": str(df["actor"].iloc[0]),
            "batch_idx": str(df["batch_idx"].iloc[0]),
        }
        rec.update(ranking_metrics(mat))
        records.append(rec)

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_groups}")

    df_out = pd.DataFrame(records)
    out_path = env_dir / "advantage_ranking_extended.parquet"
    df_out.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}  ({len(df_out)} rows)")

    # Agent-level summary
    agents = ["CRL", "GCIQL", "GCIVL", "QRL"]
    metrics = [
        "prec_at_10",
        "prec_at_25",
        "rank_median",
        "rank_worst_half",
        "dominance_mean",
        "dominance_pos_frac",
    ]
    print(f"\n=== {args.env} ===")
    print(df_out.groupby("agent")[metrics].mean().reindex(agents).round(4).to_string())

    # Write to metric_summaries
    summary_dir = Path(__file__).parent / "metric_summaries"
    summary_dir.mkdir(exist_ok=True)
    agg = df_out.groupby("agent")[metrics].mean().reindex(agents).round(4)
    agg.to_csv(summary_dir / f"ranking_extended_{args.env.replace('-', '_')}.csv")
    print(f"Saved → metric_summaries/ranking_extended_{args.env.replace('-', '_')}.csv")


if __name__ == "__main__":
    main()
