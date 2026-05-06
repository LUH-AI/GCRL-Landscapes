#!/usr/bin/env python3
"""Diagnostic-only advantage normalization sensitivity.

Using stored diagonal advantages, recomputes ESS and top-5 mass after four
normalization schemes:
  raw          — no normalization (current approach)
  zscore       — per-batch (adv - mean) / std
  rank         — per-batch rank / N (quantile within batch, in [0,1])
  clip_std     — clip to [-3σ, 3σ] then z-score

No retraining. Does NOT claim downstream AWR effects — diagnostic probe only.

Outputs
-------
  metric_summaries/adv_norm_sensitivity.csv       — long-form: env×algo×norm×metric
  metric_summaries/adv_norm_ess_pivot.csv         — ESS: algo × norm scheme per env
  metric_summaries/adv_norm_top5_pivot.csv        — top5_mass: algo × norm per env

Usage
-----
  python ANALYSIS/advantage_normalization.py
  python ANALYSIS/advantage_normalization.py --env scene
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
ENVS = ["antmaze-medium", "antmaze-large", "cube", "scene"]
CLIP_ALPHA = 100.0  # same w_max as main analysis

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


def normalize(adv: np.ndarray, scheme: str) -> np.ndarray:
    if scheme == "raw":
        return adv
    if scheme == "zscore":
        return (adv - adv.mean()) / (adv.std() + 1e-12)
    if scheme == "rank":
        order = np.argsort(adv)
        out = np.empty_like(adv, dtype=float)
        out[order] = np.arange(len(adv)) / (len(adv) - 1 + 1e-12)
        return out
    if scheme == "clip_std":
        mu, sigma = adv.mean(), adv.std() + 1e-12
        clipped = np.clip(adv, mu - 3 * sigma, mu + 3 * sigma)
        return (clipped - clipped.mean()) / (clipped.std() + 1e-12)
    raise ValueError(scheme)


def concentration(adv_norm: np.ndarray, alpha: float, scheme: str) -> dict:
    # For non-raw schemes, use alpha=1 (normalization absorbs scale)
    effective_alpha = alpha if scheme == "raw" else 1.0
    raw = np.exp(np.clip(effective_alpha * adv_norm.astype(np.float64), -500, 500))
    w = np.minimum(raw, CLIP_ALPHA)
    total = w.sum()
    w_sc = w / w.max()
    ess = float(w_sc.sum() ** 2 / (len(w_sc) * (w_sc**2).sum()))
    k = max(1, int(0.05 * len(w)))
    top5 = float(np.partition(w, -k)[-k:].sum() / total) if total > 0 else float("nan")
    sat = float(w[raw >= CLIP_ALPHA].sum() / total) if total > 0 else float("nan")
    return dict(ess=ess, top5_mass=top5, saturation_mass=sat)


SCHEMES = ["raw", "zscore", "rank", "clip_std"]


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
        for scheme in SCHEMES:
            adv_n = normalize(adv, scheme)
            m = concentration(adv_n, alpha, scheme)
            rows.append(
                dict(
                    env=env,
                    algo=agent,
                    config=config,
                    norm=scheme,
                    **{k: round(v, 4) for k, v in m.items()},
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
    df.to_csv(OUT / "adv_norm_sensitivity.csv", index=False)
    print(f"\nSaved → adv_norm_sensitivity.csv  ({len(df)} rows)")

    # Agent-level mean per (env, algo, norm)
    agg = (
        df.groupby(["env", "algo", "norm"])[["ess", "top5_mass", "saturation_mass"]]
        .mean()
        .round(4)
        .reset_index()
    )

    for metric in ["ess", "top5_mass"]:
        pivot_rows = []
        for env in envs:
            sub = agg[agg["env"] == env]
            piv = sub.pivot(index="algo", columns="norm", values=metric)
            piv = piv.reindex(index=AGENTS, columns=SCHEMES)
            piv.insert(0, "env", env)
            pivot_rows.append(piv.reset_index())
        pivot_df = pd.concat(pivot_rows, ignore_index=True)
        pivot_df.to_csv(OUT / f"adv_norm_{metric}_pivot.csv", index=False)
        print(f"Saved → adv_norm_{metric}_pivot.csv")

    print("\n=== ESS by normalization scheme ===")
    for env in envs:
        print(f"\n{env}")
        sub = agg[agg["env"] == env].pivot(index="algo", columns="norm", values="ess")
        sub = sub.reindex(index=AGENTS, columns=SCHEMES)
        print(sub.round(3).to_string())

    print("\n=== Top-5 mass by normalization scheme ===")
    for env in envs:
        print(f"\n{env}")
        sub = agg[agg["env"] == env].pivot(
            index="algo", columns="norm", values="top5_mass"
        )
        sub = sub.reindex(index=AGENTS, columns=SCHEMES)
        print(sub.round(3).to_string())


if __name__ == "__main__":
    main()
