"""Shared helpers for paper-metrics scripts.

Centralises path resolution and a few utilities that several scripts use.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ZIPS = DATA / "zips"
PARQUETS = DATA / "parquets"
EVAL_STATS = DATA / "eval_stats"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

ENVS = ["antmaze-medium", "antmaze-large", "cube", "scene"]
ENV_DATASET_TAG = {
    "antmaze-medium": "antmaze-medium-navigate-v0",
    "antmaze-large": "antmaze-large-navigate-v0",
    "cube": "cube-single-play-v0",
    "scene": "scene-play-v0",
}
AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]

ZIP_NAMES = {
    "antmaze-medium": "2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip",
    "antmaze-large": "2026-04-30-logs-advantage-antmaze-large-single-fixed-batch.zip",
    "cube": "2026-04-27-logs-advantage-cube-single-fixed-batch.zip",
    "scene": "2026-05-03-logs-advantage-scene-single-fixed-batch.zip",
}


def zip_path(env: str) -> Path:
    """Path to the fixed-batch result zip for a given env."""
    return ZIPS / ZIP_NAMES[env]


def parquet_path(env: str) -> Path:
    """Path to the cross-goal advantage matrix parquet for a given env."""
    return PARQUETS / env / "advantages.parquet"


def eval_stats_path(env: str) -> Path:
    """Path to the per-(algo, env, config, seed, phase) success CSV."""
    return EVAL_STATS / f"{env}.csv"


# ── Alpha lookup (reads configuration_*.json from inside the zip) ───────────


def alpha_map_from_zip(env: str) -> dict[tuple[str, int], float]:
    """Return {(agent, config_idx): alpha} for the fixed-batch run of *env*.

    Reads `<AGENT>_<dataset>_64c_lr-alpha/configurations/configuration_<i>.json`
    entries from the zip without unpacking the whole archive.
    """
    zp = zip_path(env)
    pattern = re.compile(
        r".*?(?P<agent>CRL|GCIQL|GCIVL|QRL)_[^/]+/configurations/configuration_(?P<idx>\d+)\.json$"
    )
    out: dict[tuple[str, int], float] = {}
    with zipfile.ZipFile(zp) as zf:
        for info in zf.infolist():
            m = pattern.match(info.filename)
            if not m:
                continue
            agent = m.group("agent")
            idx = int(m.group("idx"))
            data = json.loads(zf.read(info.filename).decode("utf-8"))
            alpha = data.get("alpha")
            if alpha is not None and float(alpha) > 0:
                out[(agent, idx)] = float(alpha)
    return out


# ── Eval-stats loader ───────────────────────────────────────────────────────


def load_all_eval_stats() -> pd.DataFrame:
    """Concat all 4 envs' eval_stats.csv into one long DataFrame.

    Columns: algo, env_short, config, seed, phase, alpha, lr, success.
    """
    parts = []
    for env in ENVS:
        df = pd.read_csv(eval_stats_path(env))
        df["env_short"] = env
        parts.append(
            df[
                [
                    "algo",
                    "env_short",
                    "config",
                    "seed",
                    "phase",
                    "alpha",
                    "lr",
                    "success",
                ]
            ]
        )
    return pd.concat(parts, ignore_index=True)


# ── Save helpers ────────────────────────────────────────────────────────────


def save_csv(df: pd.DataFrame, name: str) -> Path:
    """Save *df* to outputs/<name>.csv and return the path."""
    p = OUTPUTS / f"{name}.csv"
    df.to_csv(p, index=False)
    print(f"  → {p}")
    return p
