"""Shared helpers for paper-metrics scripts.

Centralises path resolution and a few utilities that several scripts use.

Campaigns are declared in ``data/manifest.json``. The module-level constants
(``ENVS``, ``AGENTS``, ``ZIP_NAMES``, ``ENV_DATASET_TAG``) are derived from the
manifest's ``default_campaign`` — the original AWR fixed-batch sweep — so
every existing table script reproduces unchanged. New result families
(DDPG+BC, n-step TD, FQL, rejection sampling, …) are onboarded by adding a
campaign entry to the manifest and passing ``campaign="<name>"`` to the
helpers here; no script edits required for eval-stats-driven analyses.
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

# ── Campaign manifest ────────────────────────────────────────────────────────

MANIFEST_PATH = DATA / "manifest.json"
_MANIFEST = json.loads(MANIFEST_PATH.read_text())
DEFAULT_CAMPAIGN = _MANIFEST["default_campaign"]
CAMPAIGNS: dict[str, dict] = _MANIFEST["campaigns"]


def campaign_spec(campaign: str | None = None) -> dict:
    """Return the manifest entry for *campaign* (default campaign if None)."""
    name = campaign or DEFAULT_CAMPAIGN
    if name not in CAMPAIGNS:
        raise KeyError(
            f"Unknown campaign {name!r}; known: {sorted(CAMPAIGNS)}. "
            f"Add it to {MANIFEST_PATH}."
        )
    return CAMPAIGNS[name]


# Backward-compatible constants, derived from the default campaign.
_DEFAULT = campaign_spec()
ENVS = list(_DEFAULT["envs"].keys())
ENV_DATASET_TAG = {e: c["dataset_tag"] for e, c in _DEFAULT["envs"].items()}
AGENTS = list(_DEFAULT["agents"])
ZIP_NAMES = {e: c["zip"] for e, c in _DEFAULT["envs"].items()}


def zip_path(env: str, campaign: str | None = None) -> Path:
    """Path to the result zip for a given env (and campaign)."""
    return ZIPS / campaign_spec(campaign)["envs"][env]["zip"]


def parquet_path(env: str, campaign: str | None = None) -> Path:
    """Path to the cross-goal advantage matrix parquet for a given env."""
    return PARQUETS / campaign_spec(campaign)["envs"][env]["parquet"]


def eval_stats_path(env: str, campaign: str | None = None) -> Path:
    """Path to the per-(algo, env, config, seed, phase) success CSV."""
    return EVAL_STATS / campaign_spec(campaign)["envs"][env]["eval_stats"]


# ── Alpha lookup (reads configuration_*.json from inside the zip) ───────────


def alpha_map_from_zip(
    env: str, campaign: str | None = None
) -> dict[tuple[str, int], float]:
    """Return {(agent, config_idx): alpha} for a campaign's run of *env*.

    Reads `<AGENT>_<dataset>_<suffix>/configurations/configuration_<i>.json`
    entries from the zip without unpacking the whole archive. The agent
    alternation comes from the campaign's agent list, so campaigns with
    other agents (e.g. FQL) resolve without touching this code.
    """
    spec = campaign_spec(campaign)
    zp = zip_path(env, campaign)
    agent_alt = "|".join(re.escape(a) for a in spec["agents"])
    pattern = re.compile(
        rf".*?(?P<agent>{agent_alt})_[^/]+/configurations/configuration_(?P<idx>\d+)\.json$"
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


def load_all_eval_stats(campaign: str | None = None) -> pd.DataFrame:
    """Concat a campaign's per-env eval_stats.csv into one long DataFrame.

    Columns: algo, env_short, config, seed, phase, alpha, lr, success —
    plus `variant` when the CSV carries it (new-campaign builds via
    `build_eval_stats.py --variant <tag>`).
    """
    spec = campaign_spec(campaign)
    base_cols = [
        "algo",
        "env_short",
        "config",
        "seed",
        "phase",
        "alpha",
        "lr",
        "success",
    ]
    parts = []
    for env in spec["envs"]:
        df = pd.read_csv(eval_stats_path(env, campaign))
        df["env_short"] = env
        cols = base_cols + (["variant"] if "variant" in df.columns else [])
        parts.append(df[cols])
    return pd.concat(parts, ignore_index=True)


# ── Save helpers ────────────────────────────────────────────────────────────


def save_csv(df: pd.DataFrame, name: str) -> Path:
    """Save *df* to outputs/<name>.csv and return the path."""
    p = OUTPUTS / f"{name}.csv"
    df.to_csv(p, index=False)
    print(f"  → {p}")
    return p
