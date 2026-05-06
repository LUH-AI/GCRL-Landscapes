#!/usr/bin/env python3
"""Run the full advantages.parquet extraction pipeline for one environment folder.

Usage
-----
    python extract_env.py --env antmaze-large
    python extract_env.py --env cube

Outputs (all written inside <env>/)
-------------------------------------
    advantages_summary.parquet       — per-(agent,config,seed) stats
    advantages_histograms.npz        — 200-bin histograms per group
    advantage_matrix_metrics.parquet — per-batch AFR metrics (fr_auc, gap, MRR, …)
    ess_per_batch.parquet            — per-batch ESS from diagonal AWR weights
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

CLIP = 100.0
N = 256  # obs/goal matrix size
N_HIST_BINS = 200
BATCH_SIZE = 2**17  # streaming batch size

_RAW_DATA = Path("/Users/adityamohan/git/GCRL/plots/new-plots/raw-data")
LOGS_BASES = [
    _RAW_DATA / "logs-antmaze",
    _RAW_DATA / "logs-cube",
    _RAW_DATA / "logs-scene",
]


# ── helpers ───────────────────────────────────────────────────────────────────


def _ess(adv: np.ndarray, alpha: float) -> float:
    w = np.exp((alpha * adv.astype(np.float64)).clip(-500, 500))
    w = w.clip(min=1e-9, max=CLIP)
    w_scaled = w / w.max()
    denom = len(w_scaled) * float((w_scaled**2).sum())
    return float(w_scaled.sum() ** 2 / denom) if denom > 0 else float("nan")


def load_alpha_map(env_tag: str) -> dict[tuple[str, int], float]:
    """Read alpha per (agent, config) from configuration JSONs in raw log dirs."""
    agents = ["CRL", "GCIQL", "GCIVL", "QRL"]
    alpha_map: dict[tuple[str, int], float] = {}
    for base in LOGS_BASES:
        for agent in agents:
            cfg_dir = base / f"{agent}_{env_tag}_64c_lr-alpha" / "configurations"
            if not cfg_dir.exists():
                continue
            for cjson in cfg_dir.glob("configuration_*.json"):
                idx = int(re.search(r"(\d+)", cjson.name).group(1))
                data = json.loads(cjson.read_text())
                alpha = data.get("alpha")
                if alpha and alpha > 0:
                    alpha_map[(agent, idx)] = float(alpha)
    print(f"  alpha map: {len(alpha_map)} entries")
    return alpha_map


def afr_metrics(adv_matrix: np.ndarray) -> dict:
    A_plus = np.diag(adv_matrix)
    off_diag = ~np.eye(N, dtype=bool)
    gaps = (A_plus[:, None] - adv_matrix)[off_diag]
    ranks = (adv_matrix >= A_plus[:, None]).sum(axis=1)
    return {
        "fr_auc": float(np.mean(gaps > 0)),
        "gap_mean": float(gaps.mean()),
        "gap_std": float(gaps.std()),
        "mrr": float(np.mean(1.0 / ranks)),
        "prec_at_1": float(np.mean(ranks == 1)),
        "prec_at_5": float(np.mean(ranks <= 5)),
        "Aplus_mean": float(A_plus.mean()),
        "Aplus_std": float(A_plus.std()),
        "Aminus_std": float(adv_matrix[off_diag].std()),
        "extractability_index": float(gaps.mean() / (gaps.std() + 1e-8)),
        "adv_minus_mean": float(adv_matrix[off_diag].mean()),
    }


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", required=True, help="subfolder name, e.g. antmaze-large or cube"
    )
    args = parser.parse_args()

    env_dir = Path(__file__).parent.parent / "ENVS" / args.env
    parquet = env_dir / "advantages.parquet"

    if not parquet.exists():
        print(f"ERROR: {parquet} not found")
        return

    print(f"\n{'=' * 60}")
    print(f"Processing {args.env}  ({parquet.stat().st_size / 1e9:.1f} GB)")
    print(f"{'=' * 60}")

    # ── alpha map (env-specific) ───────────────────────────────────────────────
    # Derive the raw-logs env tag: antmaze-large -> antmaze-large-navigate-v0
    # cube -> cube-* (try a few patterns)
    possible_tags = [
        args.env,
        f"{args.env}-navigate-v0",
        f"{args.env}-single-v0",
        f"{args.env}-single-play-v0",
        f"{args.env}-play-v0",
    ]
    alpha_map: dict = {}
    for tag in possible_tags:
        alpha_map = load_alpha_map(tag)
        if alpha_map:
            print(f"  matched env tag: {tag}")
            break
    if not alpha_map:
        print("  [warn] no alpha values found — ESS will be skipped")

    # ── 1. streaming summary + histograms ─────────────────────────────────────
    print("\n[1/3] Streaming summary + histograms …")
    dataset = ds.dataset(parquet, format="parquet")

    from collections import defaultdict

    class Accum:
        __slots__ = ("adv_chunks", "n_pos", "n_total")

        def __init__(self):
            self.adv_chunks = []
            self.n_pos = 0
            self.n_total = 0

    groups: dict[tuple, Accum] = defaultdict(Accum)
    KEY_COLS = ["agent", "actor", "phase", "seed", "configuration"]
    total = 0

    scanner = dataset.scanner(
        columns=KEY_COLS + ["advantage", "is_positive"],
        batch_size=BATCH_SIZE,
    )
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        total += len(df)
        for key, grp in df.groupby(KEY_COLS, sort=False):
            g = groups[key]
            g.adv_chunks.append(grp["advantage"].to_numpy(dtype=np.float32))
            g.n_pos += int(grp["is_positive"].sum())
            g.n_total += len(grp)
        if total % (BATCH_SIZE * 32) < BATCH_SIZE:
            print(f"  {total:>12,} rows …")

    print(f"  {total:,} rows, {len(groups)} groups")

    summary_rows, hist_data = [], {}
    for key, g in groups.items():
        adv = np.concatenate(g.adv_chunks)
        pcts = np.percentile(adv, [5, 25, 50, 75, 95]).tolist()
        row = dict(
            zip(KEY_COLS, key),
            n=g.n_total,
            n_positive=g.n_pos,
            pos_fraction=g.n_pos / g.n_total,
            adv_mean=float(adv.mean()),
            adv_std=float(adv.std()),
            adv_p5=pcts[0],
            adv_p25=pcts[1],
            adv_median=pcts[2],
            adv_p75=pcts[3],
            adv_p95=pcts[4],
        )
        summary_rows.append(row)
        counts, edges = np.histogram(adv, bins=N_HIST_BINS)
        tag = "_".join(str(k) for k in key)
        hist_data[f"{tag}_counts"] = counts.astype(np.int32)
        hist_data[f"{tag}_edges"] = edges.astype(np.float32)

    summary = pd.DataFrame(summary_rows).sort_values(KEY_COLS).reset_index(drop=True)
    summary.to_parquet(env_dir / "advantages_summary.parquet", index=False)
    np.savez_compressed(env_dir / "advantages_histograms.npz", **hist_data)
    print("  → advantages_summary.parquet, advantages_histograms.npz")

    # ── 2. AFR metrics + MRR per row group ────────────────────────────────────
    print("\n[2/3] AFR + MRR metrics per row group …")
    pf = pq.ParquetFile(parquet)
    n_groups = pf.metadata.num_row_groups
    matrix_records = []
    for i, batch in enumerate(pf.iter_batches()):
        if batch.num_rows != N * N:
            continue
        df = batch.to_pandas()
        agent = df["agent"].iloc[0]
        config = int(df["configuration"].iloc[0])
        seed = int(df["seed"].iloc[0])
        phase = df["phase"].iloc[0]
        actor = df["actor"].iloc[0]
        b_idx = df["batch_idx"].iloc[0]

        mat = np.empty((N, N), dtype=np.float32)
        mat[df["obs_idx"].to_numpy(), df["goal_idx"].to_numpy()] = df[
            "advantage"
        ].to_numpy(dtype=np.float32)

        rec = {
            "agent": agent,
            "config": config,
            "seed": seed,
            "phase": phase,
            "actor": actor,
            "batch_idx": b_idx,
        }
        rec.update(afr_metrics(mat))
        matrix_records.append(rec)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_groups}")

    df_mat = pd.DataFrame(matrix_records)
    df_mat.to_parquet(env_dir / "advantage_matrix_metrics.parquet", index=False)
    print(f"  → advantage_matrix_metrics.parquet  ({len(df_mat)} rows)")

    # ── 3. ESS from diagonal AWR weights ──────────────────────────────────────
    if alpha_map:
        print("\n[3/3] ESS from diagonal advantages …")
        group_diag: dict[tuple, list] = defaultdict(list)
        for batch in pf.iter_batches():
            if batch.num_rows != N * N:
                continue
            df = batch.to_pandas()
            agent = df["agent"].iloc[0]
            config = int(df["configuration"].iloc[0])
            seed = int(df["seed"].iloc[0])
            diag = df.loc[df["is_positive"], "advantage"].to_numpy(dtype=np.float32)
            group_diag[(agent, config, seed)].append(diag)

        ess_records = []
        for batch in pf.iter_batches():
            if batch.num_rows != N * N:
                continue
            df = batch.to_pandas()
            agent = df["agent"].iloc[0]
            config = int(df["configuration"].iloc[0])
            seed = int(df["seed"].iloc[0])
            phase = df["phase"].iloc[0]
            actor = df["actor"].iloc[0]
            b_idx = df["batch_idx"].iloc[0]
            alpha = alpha_map.get((agent, config))
            if not alpha:
                continue
            diag = df.loc[df["is_positive"], "advantage"].to_numpy(dtype=np.float32)
            ess_records.append(
                {
                    "agent": agent,
                    "config": config,
                    "seed": seed,
                    "phase": phase,
                    "actor": actor,
                    "batch_idx": b_idx,
                    "ess": _ess(diag, alpha),
                }
            )

        df_ess = pd.DataFrame(ess_records)
        df_ess.to_parquet(env_dir / "ess_per_batch.parquet", index=False)
        print(f"  → ess_per_batch.parquet  ({len(df_ess)} rows)")
        print("\nESS by agent (mean ± std across all configs/seeds/batches):")
        print(
            df_ess.groupby("agent")["ess"]
            .agg(["mean", "std", "count"])
            .round(3)
            .to_string()
        )
    else:
        print("\n[3/3] ESS skipped (no alpha map)")

    print(f"\nDone. All outputs in {env_dir}/")


if __name__ == "__main__":
    main()
