#!/usr/bin/env python3
"""Orchestrate Q-vs-distance plots for all (env, agent) pairs in best_checkpoints.json.

Calls plot_q_vs_distance once per pair, producing PDF + PNG in analysis/plots/.

Usage
-----
    direnv exec . python analysis/run_q_vs_distance_all.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BEST = Path("analysis/best_checkpoints.json")
OUTDIR = Path("analysis/plots")
DATASET_MAP = {
    "cube": "cube-single-play-v0",
    "antmaze-medium": "antmaze-medium-navigate-v0",
    "antmaze-large": "antmaze-large-navigate-v0",
    "scene": "scene-play-v0",
}

data = json.loads(BEST.read_text())
OUTDIR.mkdir(parents=True, exist_ok=True)

for env_key, agents in data.items():
    dataset = DATASET_MAP[env_key]
    for agent_name, info in agents.items():
        output = OUTDIR / f"q_vs_distance_{env_key}_{agent_name}.pdf"
        cmd = [
            sys.executable, "-m", "analysis.plot_q_vs_distance",
            "--checkpoint", info["checkpoint"], agent_name,
            "--config-path", info["config"],
            "--dataset", dataset,
            "--output", str(output),
        ]
        print(f"\n=== {env_key} / {agent_name} ===")
        subprocess.run(cmd, check=True)
