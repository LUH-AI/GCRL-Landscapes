"""Reproduce `tab:wmax_sensitivity` from `PAPER/APPENDIX/robustness.tex`.

Per (env, algo, w_max) reports ESS and Top-5% mass on the diagonal advantages,
clipped at w_max ∈ {20, 50, 100, 200}, then averaged over configs. The paper
table only shows the two endpoints (w_max=20 and w_max=200); we emit all four.

Inputs:
  data/parquets/<env>/advantages.parquet
  data/zips/<env-fixed-batch>.zip   (alpha lookup from configurations)
Output:
  outputs/a5_wmax_sensitivity.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from _common import AGENTS, ENVS, alpha_map_from_zip, parquet_path, save_csv

N = 256
W_MAX_VALS = [20, 50, 100, 200]


def concentration(adv: np.ndarray, alpha: float, w_max: float) -> dict:
    raw = np.exp(np.clip(alpha * adv.astype(np.float64), -500, 500))
    w = np.minimum(raw, w_max)
    total = w.sum()
    if total <= 0:
        return dict(ess=float("nan"), top5_mass=float("nan"))
    w_sc = w / w.max()
    ess = float(w_sc.sum() ** 2 / (len(w_sc) * (w_sc**2).sum()))
    k = int(np.ceil(0.05 * len(w)))
    top5 = float(np.partition(w, -k)[-k:].sum() / total)
    return dict(ess=ess, top5_mass=top5)


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
        for w_max in W_MAX_VALS:
            m = concentration(adv, alpha, w_max)
            rows.append(dict(env=env, agent=agent, config=config, w_max=w_max, **m))
    return rows


def main() -> None:
    all_rows = []
    for env in ENVS:
        all_rows.extend(process_env(env))
    df = pd.DataFrame(all_rows)

    summary = (
        df.groupby(["env", "agent", "w_max"])[["ess", "top5_mass"]]
        .mean()
        .round(3)
        .reset_index()
    )
    # Pivot to a paper-style table: ESS@20, ESS@200, Top5@20, Top5@200
    pivots = []
    for metric in ["ess", "top5_mass"]:
        piv = summary.pivot_table(
            index=["env", "agent"], columns="w_max", values=metric
        ).reset_index()
        piv.columns = ["env", "agent"] + [
            f"{metric}_w{int(w)}" for w in piv.columns[2:]
        ]
        pivots.append(piv)

    out = pivots[0].merge(pivots[1], on=["env", "agent"])
    out["env_order"] = out["env"].map({e: i for i, e in enumerate(ENVS)})
    out["agent_order"] = out["agent"].map({a: i for i, a in enumerate(AGENTS)})
    out = out.sort_values(["env_order", "agent_order"]).drop(
        columns=["env_order", "agent_order"]
    )

    save_csv(out, "a5_wmax_sensitivity")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
