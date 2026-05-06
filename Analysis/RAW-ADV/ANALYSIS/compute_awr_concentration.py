#!/usr/bin/env python3
"""Compute AWR weight-concentration diagnostics from diagonal advantages.

For every (agent, config, seed, phase, actor, batch) in advantages.parquet,
extracts the diagonal (matched-goal) advantages A_i = A(s_i, a_i, g_i) and
computes, at the configured training alpha AND across a log-spaced alpha sweep:

    ess               — normalized effective sample size
    saturation_mass   — weight fraction held by clipped (w_max=100) samples
    fraction_clipped  — sample fraction hitting the clip
    top5_mass         — weight fraction of the top-5% samples
    entropy           — weight entropy  -sum p_i log p_i

Outputs (written to ENVS/<env>/):
    awr_concentration.csv  — per-(agent,config,seed,phase,actor), mean±std over batches
    awr_alpha_sweep.csv    — per-(agent,config,seed,actor) × 60 alpha values
    awr_vs_return.csv      — (antmaze-medium only) concentration-return correlations

Usage
-----
    python ANALYSIS/compute_awr_concentration.py --env antmaze-medium
    python ANALYSIS/compute_awr_concentration.py --env antmaze-large
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr

CLIP = 100.0
ALPHA_GRID = np.logspace(-2, np.log10(20), 60)  # 0.01 → 20

_RAW_DATA = Path("/Users/adityamohan/git/GCRL/plots/new-plots/raw-data")
LOGS_BASES = [
    _RAW_DATA / "logs-antmaze",
    _RAW_DATA / "logs-cube",
    _RAW_DATA / "logs-scene",
]
STATS_CSV = Path(
    "/Users/adityamohan/git/GCRL/plots/new-plots/"
    "advantage-dist-phases/basin-analysis/additional_stats_raw.csv"
)


# ── helpers ───────────────────────────────────────────────────────────────────


def load_alpha_map(env_tag: str) -> dict[tuple[str, int], float]:
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
                if alpha and float(alpha) > 0:
                    alpha_map[(agent, idx)] = float(alpha)
    return alpha_map


def concentration_metrics(adv: np.ndarray, alpha: float) -> dict:
    """AWR concentration metrics for a vector of advantages at one alpha value."""
    raw = np.exp(np.clip(alpha * adv.astype(np.float64), -500, 500))
    clipped = np.minimum(raw, CLIP)
    total_w = clipped.sum()

    if total_w <= 0:
        nan = float("nan")
        return dict(
            ess=nan,
            saturation_mass=nan,
            fraction_clipped=nan,
            top5_mass=nan,
            entropy=nan,
        )

    # ESS: max-normalise to prevent scale sensitivity (same as extract_env.py)
    w_sc = clipped / clipped.max()
    ess = float(w_sc.sum() ** 2 / (len(w_sc) * (w_sc**2).sum()))

    # Saturation: weight mass AND sample fraction at the clip ceiling
    sat_mask = raw >= CLIP
    saturation_mass = float(clipped[sat_mask].sum() / total_w)
    fraction_clipped = float(sat_mask.mean())

    # Top-5% weight mass
    k = max(1, int(0.05 * len(clipped)))
    top5_mass = float(np.partition(clipped, -k)[-k:].sum() / total_w)

    # Weight entropy
    p = clipped / total_w
    entropy = float(-np.sum(p * np.log(p + 1e-300)))

    return dict(
        ess=ess,
        saturation_mass=saturation_mass,
        fraction_clipped=fraction_clipped,
        top5_mass=top5_mass,
        entropy=entropy,
    )


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute AWR weight-concentration diagnostics."
    )
    parser.add_argument(
        "--env", required=True, help="Environment subfolder name, e.g. antmaze-medium"
    )
    args = parser.parse_args()

    _root = Path(__file__).parent.parent
    env_dir = _root / "ENVS" / args.env
    parquet_p = env_dir / "advantages.parquet"

    if not parquet_p.exists():
        raise FileNotFoundError(parquet_p)

    # ── alpha map ────────────────────────────────────────────────────────────
    possible_tags = [
        args.env,
        f"{args.env}-navigate-v0",
        f"{args.env}-single-v0",
        f"{args.env}-single-play-v0",
        f"{args.env}-play-v0",
    ]
    alpha_map: dict[tuple[str, int], float] = {}
    for tag in possible_tags:
        alpha_map = load_alpha_map(tag)
        if alpha_map:
            print(f"Alpha map: {len(alpha_map)} entries  (tag: {tag})")
            break
    if not alpha_map:
        print(
            "WARNING: no alpha map found — concentration will use alpha=1.0 as fallback"
        )

    # ── stream parquet, collect diagonal rows per batch ───────────────────────
    print(f"\nStreaming {parquet_p.name}  ({parquet_p.stat().st_size / 1e9:.1f} GB) …")
    pf = pq.ParquetFile(parquet_p)
    n_groups = pf.metadata.num_row_groups

    batch_records = []
    sweep_records = []
    n_batches_seen = 0

    for i, batch in enumerate(pf.iter_batches()):
        if batch.num_rows != 256 * 256:
            continue

        df = batch.to_pandas()
        diag = df.loc[df["is_positive"].astype(bool), "advantage"].to_numpy(
            dtype=np.float64
        )
        if len(diag) == 0:
            continue

        agent = str(df["agent"].iloc[0])
        config = int(df["configuration"].iloc[0])
        seed = int(df["seed"].iloc[0])
        phase = str(df["phase"].iloc[0])
        actor = str(df["actor"].iloc[0])
        b_idx = str(df["batch_idx"].iloc[0])

        alpha = alpha_map.get((agent, config), 1.0)
        has_real_alpha = (agent, config) in alpha_map

        # Metrics at configured alpha
        m = concentration_metrics(diag, alpha)
        batch_records.append(
            dict(
                agent=agent,
                configuration=config,
                seed=seed,
                phase=phase,
                actor=actor,
                batch_idx=b_idx,
                alpha=alpha,
                has_real_alpha=has_real_alpha,
                n_samples=len(diag),
                **m,
            )
        )

        # Alpha sweep
        for a in ALPHA_GRID:
            ms = concentration_metrics(diag, a)
            sweep_records.append(
                dict(
                    agent=agent,
                    configuration=config,
                    seed=seed,
                    phase=phase,
                    actor=actor,
                    batch_idx=b_idx,
                    alpha_sweep=float(a),
                    ess=ms["ess"],
                    saturation_mass=ms["saturation_mass"],
                    fraction_clipped=ms["fraction_clipped"],
                )
            )

        n_batches_seen += 1
        if (i + 1) % 1000 == 0:
            print(
                f"  row groups: {i + 1}/{n_groups}  diagonal batches: {n_batches_seen}"
            )

    print(f"  Done. {n_batches_seen} diagonal batches processed.")

    batch_df = pd.DataFrame(batch_records)
    sweep_df = pd.DataFrame(sweep_records)

    # ── aggregate across batches ──────────────────────────────────────────────
    agg_metrics = ["ess", "saturation_mass", "fraction_clipped", "top5_mass", "entropy"]
    meta_cols = [
        "agent",
        "configuration",
        "seed",
        "phase",
        "actor",
        "alpha",
        "has_real_alpha",
    ]

    conc = batch_df.groupby(meta_cols)[agg_metrics].agg(["mean", "std"]).round(6)
    conc.columns = [f"{m}_{s}" for m, s in conc.columns]
    conc = conc.reset_index()

    sweep_agg = (
        sweep_df.groupby(
            ["agent", "configuration", "seed", "phase", "actor", "alpha_sweep"]
        )[["ess", "saturation_mass", "fraction_clipped"]]
        .mean()
        .round(6)
        .reset_index()
    )

    # ── write ─────────────────────────────────────────────────────────────────
    env_dir.mkdir(parents=True, exist_ok=True)
    conc.to_csv(env_dir / "awr_concentration.csv", index=False)
    sweep_agg.to_csv(env_dir / "awr_alpha_sweep.csv", index=False)

    print(f"\nSaved → {env_dir / 'awr_concentration.csv'}  ({len(conc)} rows)")
    print(f"Saved → {env_dir / 'awr_alpha_sweep.csv'}  ({len(sweep_agg)} rows)")

    print("\nConcentration summary by agent (mean at configured alpha):")
    show = [
        c
        for c in [
            "ess_mean",
            "saturation_mass_mean",
            "fraction_clipped_mean",
            "top5_mass_mean",
        ]
        if c in conc.columns
    ]
    print(
        conc[conc["has_real_alpha"]].groupby("agent")[show].mean().round(3).to_string()
    )

    # ── correlation with return (antmaze-medium only) ─────────────────────────
    if args.env != "antmaze-medium" or not STATS_CSV.exists():
        return

    print("\nCorrelating with return (antmaze-medium) …")
    stats = pd.read_csv(STATS_CSV)
    nav = (
        stats[
            stats["env"].str.contains("navigate-v0") & ~stats["env"].str.contains(" ")
        ]
        .groupby(["algo", "config", "seed"])["success"]
        .mean()
        .reset_index()
        .rename(columns={"algo": "agent", "config": "configuration"})
    )

    # Collapse phase/actor within each (agent, config, seed)
    conc_merge = (
        conc[conc["has_real_alpha"]]
        .groupby(["agent", "configuration", "seed"])[
            [c for c in conc.columns if c.endswith("_mean")]
        ]
        .mean()
        .reset_index()
    )
    merged = conc_merge.merge(nav, on=["agent", "configuration", "seed"])

    corr_rows = []
    metric_cols = [
        "ess_mean",
        "saturation_mass_mean",
        "fraction_clipped_mean",
        "top5_mass_mean",
        "entropy_mean",
    ]
    for agent, g in merged.groupby("agent"):
        row: dict = {"agent": agent, "n": len(g)}
        for mc in metric_cols:
            if mc not in g.columns:
                continue
            valid = g[[mc, "success"]].dropna()
            if len(valid) < 3:
                continue
            r, p = pearsonr(valid[mc], valid["success"])
            star = (
                "***"
                if p < 0.001
                else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            )
            row[f"r_{mc}"] = round(float(r), 3)
            row[f"p_{mc}"] = round(float(p), 4)
            row[f"sig_{mc}"] = star
        corr_rows.append(row)

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(env_dir / "awr_vs_return.csv", index=False)

    # Pretty-print the key correlations
    print(
        f"\n{'Agent':<8}  {'ESS↔return':>14}  {'sat↔return':>14}  {'top5↔return':>14}"
    )
    print("-" * 56)
    for _, row in corr_df.iterrows():

        def fmt(metric):
            r = row.get(f"r_{metric}_mean", float("nan"))
            sig = row.get(f"sig_{metric}_mean", "")
            return f"{r:+.3f}{sig}" if not np.isnan(r) else "   n/a"

        print(
            f"{row['agent']:<8}  {fmt('ess'):>14}  "
            f"{fmt('saturation_mass'):>14}  {fmt('top5_mass'):>14}"
        )

    print("\n*** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")
    print(f"\nSaved → {env_dir / 'awr_vs_return.csv'}")


if __name__ == "__main__":
    main()
