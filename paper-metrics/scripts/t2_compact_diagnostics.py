"""Reproduce `tab:compact_diagnostics` from `PAPER/experiments.tex`.

Per (env, agent) row reports:
  - FR-AUC              = mean over off-diagonal pairs of (A_plus[i] - M[i,j] > 0)
  - Gap                 = mean over off-diagonal pairs of (A_plus[i] - M[i,j])
  - MRR                 = mean over rows of 1 / argsort-rank of the diagonal goal
  - ESS                 = Kish ESS on AWR weights w = min(exp(alpha * A_plus), 100)
  - Top-5 Mass          = sum of top-13 (=ceil(0.05*256)) weights / total weight
  - Max Success         = re-used from outputs/t1_final_landscape_ci.csv

All cross-goal metrics use the per-checkpoint × per-batch 256x256 matrices
streamed from the parquet, then averaged across (configuration, seed, batch_idx)
per (env, agent).

Inputs:
  data/parquets/<env>/advantages.parquet           (one row per matrix cell)
  data/zips/<env-fixed-batch>.zip                  (read-only — alpha lookup from configs)
  outputs/t1_final_landscape_ci.csv                (Max Success column; run t1 first)
Output:
  outputs/t2_compact_diagnostics.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from _common import AGENTS, ENVS, OUTPUTS, alpha_map_from_zip, parquet_path, save_csv

N = 256
W_MAX = 100.0
TOP_K = int(np.ceil(0.05 * N))  # = 13, matches Malte's compute_afr_metrics.py


def per_matrix_metrics(M: np.ndarray, alpha: float) -> dict:
    A_plus = np.diag(M)
    off = ~np.eye(N, dtype=bool)
    gaps = (A_plus[:, None] - M)[off]

    fr_auc = float((gaps > 0).mean())
    gap = float(gaps.mean())

    # MRR via argsort (Malte's formulation; tie-aware against diagonal goal)
    rec = np.empty(N, dtype=np.float32)
    for i in range(N):
        order = np.argsort(M[i])[::-1]
        rec[i] = 1.0 / (int(np.where(order == i)[0][0]) + 1)
    mrr = float(rec.mean())

    # AWR weights (clipped) for ESS and Top-5
    raw = np.exp(np.clip(alpha * A_plus.astype(np.float64), -500, 500))
    w = np.minimum(raw, W_MAX)
    total = w.sum()
    if total <= 0:
        return dict(
            fr_auc=fr_auc, gap=gap, mrr=mrr, ess=float("nan"), top5_mass=float("nan")
        )

    w_sc = w / w.max()
    ess = float(w_sc.sum() ** 2 / (N * (w_sc**2).sum()))
    top5_mass = float(np.partition(w, -TOP_K)[-TOP_K:].sum() / total)
    return dict(fr_auc=fr_auc, gap=gap, mrr=mrr, ess=ess, top5_mass=top5_mass)


def process_env(env: str) -> pd.DataFrame:
    print(f"  {env}: streaming parquet…")
    parquet = parquet_path(env)
    if not parquet.exists():
        raise FileNotFoundError(parquet)
    alpha_map = alpha_map_from_zip(env)
    if not alpha_map:
        raise RuntimeError(f"No alpha map for {env} — check the zip path.")

    pf = pq.ParquetFile(parquet)
    records = []
    for batch in pf.iter_batches():
        if batch.num_rows != N * N:
            continue
        df = batch.to_pandas()
        agent = str(df["agent"].iloc[0])
        config = int(df["configuration"].iloc[0])
        seed = int(df["seed"].iloc[0])
        b_idx = str(df["batch_idx"].iloc[0])
        alpha = alpha_map.get((agent, config))
        if alpha is None:
            continue
        M = np.empty((N, N), dtype=np.float32)
        M[df["obs_idx"].to_numpy(), df["goal_idx"].to_numpy()] = df[
            "advantage"
        ].to_numpy()
        rec = per_matrix_metrics(M, alpha)
        rec.update(env=env, agent=agent, config=config, seed=seed, batch_idx=b_idx)
        records.append(rec)
    return pd.DataFrame(records)


def main() -> None:
    parts = []
    for env in ENVS:
        parts.append(process_env(env))
    per_matrix = pd.concat(parts, ignore_index=True)

    agg = (
        per_matrix.groupby(["env", "agent"])[
            ["fr_auc", "gap", "mrr", "ess", "top5_mass"]
        ]
        .mean()
        .round(4)
        .reset_index()
    )

    # Pull Max Success from t1's output
    t1_path = OUTPUTS / "t1_final_landscape_ci.csv"
    if not t1_path.exists():
        raise FileNotFoundError(
            f"{t1_path} not found — run t1_final_landscape_ci.py first."
        )
    t1 = pd.read_csv(t1_path)[["env", "agent", "max_success"]]
    out = agg.merge(t1, on=["env", "agent"])
    out = out.rename(
        columns={
            "fr_auc": "FR-AUC",
            "gap": "Gap",
            "mrr": "MRR",
            "ess": "ESS",
            "top5_mass": "Top-5 Mass",
            "max_success": "Max Success",
        }
    )

    out["env_order"] = out["env"].map({e: i for i, e in enumerate(ENVS)})
    out["agent_order"] = out["agent"].map({a: i for i, a in enumerate(AGENTS)})
    out = out.sort_values(["env_order", "agent_order"]).drop(
        columns=["env_order", "agent_order"]
    )

    save_csv(out, "t2_compact_diagnostics")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
