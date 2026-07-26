"""R-L6: filtered future-association check (7VBe-W4, fills {RESULT-FILTERED-FR}).

Sensitivity check, explicitly NOT an optimality oracle: FR-AUC/Gap/MRR treat
the sampled actor goal as the "positive"; anchors whose logged action does
not move toward that goal make the positive label questionable.  We filter
anchors whose action reduces xy-distance to their matched actor goal and
recompute the three metrics on the retained rows, next to the unfiltered
values.

Batch reconstruction reuses the exact generation code path
(`create_env_and_dataset` + `_sample_fixed_batch(val_ds, seed=batch_idx)`),
with the same configuration_0.json the sweep used (only lr/alpha vary across
configs, so goal-sampling params are identical for all 64 configs).

AntMaze only for now: obs[:2] is the agent xy (ogbench locomaze `ant.get_ob`
= concat(qpos, qvel), maze success = ||xy - goal_xy||).  Cube/scene need
object-index verification and (scene) a dataset download - deferred.

Inputs:  data/zips/{env}.zip (configuration_0.json per agent),
         RAW-ADV/ENVS/{env}/advantages.parquet, ~/.ogbench datasets
Outputs: outputs/rl6_filtered_fr.csv
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from ml_collections import ConfigDict

from _common import ENV_DATASET_TAG, save_csv, zip_path
from gcrl_landscapes.util.datasets import create_env_and_dataset
from generate_advantages import BATCH_SIZE, _sample_fixed_batch

RAW_ADV_ENVS = Path(
    os.environ.get(
        "RAW_ADV_ENVS",
        str(Path(__file__).resolve().parents[2] / "Analysis" / "RAW-ADV" / "ENVS"),
    )
)
ENVS = ["antmaze-medium", "antmaze-large", "cube", "scene"]
AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]
N_BATCHES = 10

# Goal-relevant position sub-vector per env, matching each env's own success
# criterion: locomaze success is agent-xy distance (obs = concat(qpos, qvel),
# qpos[:2] = xy); cube success is block position (obs tail per cube is
# [block_pos(3), quat(4), cos yaw, sin yaw], so block xyz = obs[-9:-6]).
# scene mirrors SceneEnv.compute_oracle_observation: cube xyz (obs[-21:-18]),
# button state one-hots (obs[-12:-10], obs[-8:-6]), drawer pos (obs[-4]),
# window pos (obs[-2]) — all already env-scaled to comparable magnitudes.
POS_SLICE = {
    "antmaze-medium": slice(0, 2),
    "antmaze-large": slice(0, 2),
    "cube": slice(-9, -6),
    "scene": np.r_[-21:-18, -12:-10, -8:-6, -4, -2],
}


def load_config0(env: str, agent: str) -> ConfigDict:
    zp = zip_path(env)
    with zipfile.ZipFile(zp) as zf:
        cands = [
            n
            for n in zf.namelist()
            if n.endswith("configurations/configuration_0.json")
            and f"{agent}_" in n.split("/")[-3]
        ]
        assert len(cands) == 1, (env, agent, cands)
        return ConfigDict(json.loads(zf.read(cands[0]).decode()))


def build_masks(env: str) -> dict[tuple[str, int], np.ndarray]:
    """{(agent, batch_idx): bool mask over 256 anchors} + determinism check."""
    tag = ENV_DATASET_TAG[env]
    masks = {}
    for agent in AGENTS:
        config = load_config0(env, agent)
        _, _, val_ds = create_env_and_dataset(tag, agent, config)
        # determinism check on batch 0
        b0a = _sample_fixed_batch(val_ds, seed=0)
        b0b = _sample_fixed_batch(val_ds, seed=0)
        assert np.array_equal(b0a["actor_goals"], b0b["actor_goals"]), (
            f"goal sampling not deterministic for {env}/{agent}"
        )
        for b in range(N_BATCHES):
            batch = _sample_fixed_batch(val_ds, seed=b)
            sl = POS_SLICE[env]
            obs_xy = batch["observations"][:, sl]
            next_xy = batch["next_observations"][:, sl]
            goal_xy = batch["actor_goals"][:, sl]
            d_now = np.linalg.norm(obs_xy - goal_xy, axis=1)
            d_next = np.linalg.norm(next_xy - goal_xy, axis=1)
            masks[(agent, b)] = d_next < d_now
        frac = np.mean([masks[(agent, b)].mean() for b in range(N_BATCHES)])
        print(f"  {env}/{agent}: mean retained fraction {frac:.3f}")
    return masks


def afr_metrics_masked(A: np.ndarray, keep: np.ndarray) -> dict:
    """FR-AUC / gap_mean / MRR over anchors selected by *keep* (bool, len N)."""
    N = A.shape[0]
    rows = np.where(keep)[0]
    if len(rows) < 2:
        return {
            "fr_auc": np.nan,
            "gap_mean": np.nan,
            "mrr": np.nan,
            "n_anchors": len(rows),
        }
    adv_plus = A[rows, rows]
    off = np.ones((len(rows), N), dtype=bool)
    off[np.arange(len(rows)), rows] = False
    gaps = (adv_plus[:, None] - A[rows])[off]
    order = np.argsort(-A[rows], axis=1)
    rank_pos = np.argmax(order == rows[:, None], axis=1) + 1
    return {
        "fr_auc": float(np.mean(gaps > 0.0)),
        "gap_mean": float(np.mean(gaps)),
        "mrr": float(np.mean(1.0 / rank_pos)),
        "n_anchors": len(rows),
    }


def process_env(env: str, masks: dict) -> list[dict]:
    pf = pq.ParquetFile(RAW_ADV_ENVS / env / "advantages.parquet")
    per_group = []
    all_keep = np.ones(BATCH_SIZE, dtype=bool)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(
            rg, columns=["agent", "batch_idx", "obs_idx", "goal_idx", "advantage"]
        )
        agent = str(t.column("agent")[0].as_py())
        b = int(str(t.column("batch_idx")[0].as_py()))
        adv = np.asarray(t.column("advantage"))
        if len(adv) != BATCH_SIZE * BATCH_SIZE:
            continue
        if rg == 0:  # layout check: rows = repeat(arange), cols = tile(arange)
            oi = np.asarray(t.column("obs_idx"))
            gi = np.asarray(t.column("goal_idx"))
            assert oi[0] == 0 and oi[-1] == BATCH_SIZE - 1 and gi[BATCH_SIZE] == 0
        A = adv.reshape(BATCH_SIZE, BATCH_SIZE)
        m_all = afr_metrics_masked(A, all_keep)
        m_kept = afr_metrics_masked(A, masks[(agent, b)])
        per_group.append(
            {
                "env": env,
                "agent": agent,
                "batch_idx": b,
                "retained_frac": float(masks[(agent, b)].mean()),
                **{f"{k}_all": v for k, v in m_all.items()},
                **{f"{k}_kept": v for k, v in m_kept.items()},
            }
        )
    return per_group


def main() -> None:
    envs = sys.argv[1:] or ENVS
    suffix = "_" + "-".join(envs) if sys.argv[1:] else ""
    all_rows = []
    for env in envs:
        print(f"Building masks for {env} …")
        masks = build_masks(env)
        print(f"Streaming {env}/advantages.parquet …")
        all_rows.extend(process_env(env, masks))

    df = pd.DataFrame(all_rows)
    agg = (
        df.groupby(["env", "agent"])[
            [
                "retained_frac",
                "fr_auc_all",
                "fr_auc_kept",
                "gap_mean_all",
                "gap_mean_kept",
                "mrr_all",
                "mrr_kept",
            ]
        ]
        .mean()
        .round(3)
        .reset_index()
    )
    save_csv(agg, f"rl6_filtered_fr{suffix}")
    print()
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
