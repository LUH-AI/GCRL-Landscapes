#!/usr/bin/env python3
"""Recompute AWR diagnostics for w_max ∈ {20, 50, 100, 200}.

Streams advantages.parquet (diagonal batches only) for each env and
recomputes ESS, saturation_mass, fraction_clipped, top5_mass at each
clip ceiling. Does not require retraining.

Outputs
-------
  ANALYSIS/metric_summaries/wmax_sensitivity.csv   — long-form: env × algo × w_max × metric
  ANALYSIS/metric_summaries/wmax_sensitivity_pivot.csv — agent×w_max pivot per metric×env

Usage
-----
  python ANALYSIS/wmax_sensitivity.py
  python ANALYSIS/wmax_sensitivity.py --env scene
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "metric_summaries"
OUT.mkdir(exist_ok=True)

AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]
WMAX_VALS = [20, 50, 100, 200]
ENVS = ["antmaze-medium", "antmaze-large", "cube", "scene"]

# Alpha map loaded from configuration JSONs (same LOGS_BASES as extract_env.py)
_RAW_DATA = Path("/Users/adityamohan/git/GCRL/plots/new-plots/raw-data")
LOGS_BASES = [
    _RAW_DATA / "logs-antmaze",
    _RAW_DATA / "logs-cube",
    _RAW_DATA / "logs-scene",
]


def load_alpha_map(env: str) -> dict:
    import json
    import re

    possible_tags = [
        env,
        f"{env}-navigate-v0",
        f"{env}-single-v0",
        f"{env}-single-play-v0",
        f"{env}-play-v0",
    ]
    for tag in possible_tags:
        alpha_map = {}
        for base in LOGS_BASES:
            for agent in AGENTS:
                cfg_dir = base / f"{agent}_{tag}_64c_lr-alpha" / "configurations"
                if not cfg_dir.exists():
                    continue
                for cjson in cfg_dir.glob("configuration_*.json"):
                    idx = int(re.search(r"(\d+)", cjson.name).group(1))
                    data = json.loads(cjson.read_text())
                    a = data.get("alpha", 0)
                    if a > 0:
                        alpha_map[(agent, idx)] = float(a)
        if alpha_map:
            return alpha_map
    return {}


def concentration(adv: np.ndarray, alpha: float, w_max: float) -> dict:
    raw = np.exp(np.clip(alpha * adv.astype(np.float64), -500, 500))
    clipped = np.minimum(raw, w_max)
    total = clipped.sum()
    w_sc = clipped / clipped.max()
    ess = float(w_sc.sum() ** 2 / (len(w_sc) * (w_sc**2).sum()))
    sat_mask = raw >= w_max
    sat_mass = float(clipped[sat_mask].sum() / total) if total > 0 else float("nan")
    frac_clip = float(sat_mask.mean())
    k = max(1, int(0.05 * len(clipped)))
    top5 = (
        float(np.partition(clipped, -k)[-k:].sum() / total)
        if total > 0
        else float("nan")
    )
    return dict(
        ess=ess, saturation_mass=sat_mass, fraction_clipped=frac_clip, top5_mass=top5
    )


def process_env(env: str) -> list[dict]:
    parquet = _ROOT / "ENVS" / env / "advantages.parquet"
    if not parquet.exists():
        print(f"  [skip] {env}: advantages.parquet not found")
        return []

    alpha_map = load_alpha_map(env)
    if not alpha_map:
        print(f"  [skip] {env}: no alpha map")
        return []

    print(f"  {env}  ({parquet.stat().st_size / 1e9:.1f} GB)  …")
    pf = pq.ParquetFile(parquet)
    N = 256

    # Accumulate per (agent, config) diagonal advantages
    diag_store: dict[tuple, list] = {}
    for batch in pf.iter_batches():
        if batch.num_rows != N * N:
            continue
        df = batch.to_pandas()
        agent = str(df["agent"].iloc[0])
        config = int(df["configuration"].iloc[0])
        diag = df.loc[df["is_positive"], "advantage"].to_numpy(dtype=np.float64)
        key = (agent, config)
        if key not in diag_store:
            diag_store[key] = []
        diag_store[key].append(diag)

    rows = []
    for (agent, config), chunks in diag_store.items():
        alpha = alpha_map.get((agent, config))
        if alpha is None:
            continue
        adv = np.concatenate(chunks)
        for w_max in WMAX_VALS:
            metrics = concentration(adv, alpha, w_max)
            rows.append(
                dict(
                    env=env,
                    algo=agent,
                    config=config,
                    alpha=alpha,
                    w_max=w_max,
                    **metrics,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=None)
    args = parser.parse_args()

    envs = [args.env] if args.env else ENVS
    all_rows = []
    for env in envs:
        all_rows.extend(process_env(env))

    if not all_rows:
        print("No data produced.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "wmax_sensitivity.csv", index=False)
    print(f"\nSaved → wmax_sensitivity.csv  ({len(df)} rows)")

    # Agent-level summary: mean per (env, algo, w_max)
    agg = (
        df.groupby(["env", "algo", "w_max"])[
            ["ess", "saturation_mass", "fraction_clipped", "top5_mass"]
        ]
        .mean()
        .round(4)
        .reset_index()
    )

    # Pretty print
    print("\n=== ESS by w_max (mean across configs) ===")
    for env in envs:
        print(f"\n{env}")
        sub = agg[agg["env"] == env].pivot(index="algo", columns="w_max", values="ess")
        sub = sub.reindex(AGENTS)
        sub.columns = [f"w={w}" for w in sub.columns]
        print(sub.round(3).to_string())

    print("\n=== saturation_mass by w_max ===")
    for env in envs:
        print(f"\n{env}")
        sub = agg[agg["env"] == env].pivot(
            index="algo", columns="w_max", values="saturation_mass"
        )
        sub = sub.reindex(AGENTS)
        sub.columns = [f"w={w}" for w in sub.columns]
        print(sub.round(3).to_string())

    # Pivot CSV for each metric
    for metric in ["ess", "saturation_mass", "fraction_clipped", "top5_mass"]:
        piv = agg.pivot_table(
            index=["env", "algo"], columns="w_max", values=metric
        ).round(4)
        piv.columns = [f"w_max={w}" for w in piv.columns]
        piv.to_csv(OUT / f"wmax_{metric}.csv")
        print(f"Saved → wmax_{metric}.csv")


if __name__ == "__main__":
    main()
