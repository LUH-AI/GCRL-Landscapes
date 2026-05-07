"""Reproduce `tab:adv_norm_sensitivity` from `PAPER/APPENDIX/robustness.tex`.

Per (env, algo, normalisation scheme) reports ESS on diagonal advantages
after one of four diagnostic transforms:
  - raw         : no transform; effective alpha = trained alpha(lambda)
  - zscore      : (adv - mean) / std; alpha=1
  - clip_std    : clip to [mu - 3sigma, mu + 3sigma], then z-score; alpha=1
  - rank        : rank index / (N-1); alpha=1

Inputs:
  data/parquets/<env>/advantages.parquet
  data/zips/<env-fixed-batch>.zip
Output:
  outputs/a6_adv_norm_sensitivity.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from _common import AGENTS, alpha_map_from_zip, parquet_path, save_csv

N = 256
W_MAX = 100.0
SCHEMES = ["raw", "zscore", "clip_std", "rank"]
ENVS_THIS = ["antmaze-medium", "cube", "scene"]  # paper covers these three only


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


def ess_at(adv_norm: np.ndarray, alpha_eff: float) -> float:
    raw = np.exp(np.clip(alpha_eff * adv_norm.astype(np.float64), -500, 500))
    w = np.minimum(raw, W_MAX)
    if w.sum() <= 0:
        return float("nan")
    w_sc = w / w.max()
    return float(w_sc.sum() ** 2 / (len(w_sc) * (w_sc**2).sum()))


def process_env(env: str) -> list[dict]:
    print(f"  {env}: streaming parquet…")
    parquet = parquet_path(env)
    if not parquet.exists():
        raise FileNotFoundError(parquet)
    alpha_map = alpha_map_from_zip(env)

    diag_store: dict[tuple[str, int], list[np.ndarray]] = {}
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches():
        if batch.num_rows != N * N:
            continue
        df = batch.to_pandas()
        agent = str(df["agent"].iloc[0])
        config = int(df["configuration"].iloc[0])
        diag = df.loc[df["is_positive"].astype(bool), "advantage"].to_numpy(
            dtype=np.float64
        )
        diag_store.setdefault((agent, config), []).append(diag)

    rows = []
    for (agent, config), chunks in diag_store.items():
        alpha = alpha_map.get((agent, config))
        if alpha is None:
            continue
        adv = np.concatenate(chunks)
        for scheme in SCHEMES:
            adv_n = normalize(adv, scheme)
            alpha_eff = alpha if scheme == "raw" else 1.0
            rows.append(
                dict(
                    env=env,
                    agent=agent,
                    config=config,
                    scheme=scheme,
                    ess=ess_at(adv_n, alpha_eff),
                )
            )
    return rows


def main() -> None:
    all_rows = []
    for env in ENVS_THIS:
        all_rows.extend(process_env(env))
    df = pd.DataFrame(all_rows)

    agg = df.groupby(["env", "agent", "scheme"])["ess"].mean().round(3).reset_index()
    out = agg.pivot_table(
        index=["env", "agent"], columns="scheme", values="ess"
    ).reset_index()
    out = out.rename(columns={s: f"ess_{s}" for s in SCHEMES})
    out["env_order"] = out["env"].map({e: i for i, e in enumerate(ENVS_THIS)})
    out["agent_order"] = out["agent"].map({a: i for i, a in enumerate(AGENTS)})
    out = out.sort_values(["env_order", "agent_order"]).drop(
        columns=["env_order", "agent_order"]
    )

    save_csv(out, "a6_adv_norm_sensitivity")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
