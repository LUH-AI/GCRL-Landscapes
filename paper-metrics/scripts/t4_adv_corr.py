"""Reproduce `tab:adv_corr` from `PAPER/experiments.tex`.

Per agent reports two numbers at the final phase (AntMaze-medium only):
  - Seed Spearman r: pairwise rank correlation of the per-step training
    advantage arrays across seeds, averaged within (config, eval_step), then
    averaged across the (config, eval_step) population. Computed by
    `analysis/advantages/g_spearman.py` (Variant 1).
  - Oracle Spearman r: rank correlation between the per-step training advantage
    and a Euclidean-distance-progress oracle on a fixed validation batch.
    Computed by `analysis/advantages/h_oracle.py`.

Inputs:
  data/zips/2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip
Output:
  outputs/t4_adv_corr.csv

Requires the gcrl_landscapes package + the project venv. We delegate to the
two committed scripts in the repo so we use the same code that generated the
paper values.

This script is the slowest of the four (~10–20 minutes the first time
through, mostly for h_oracle.py downloading the OGBench validation dataset).
On a re-run with the same zip the merged-DataFrame cache makes it fast.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from _common import save_csv, zip_path

ENV = "antmaze-medium"
AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]

# Paths to the committed scripts in the gcrl_landscapes / analysis directory.
# This file lives at <repo>/paper-metrics/scripts/, so the repo root is two up.
REPO_ROOT = Path(__file__).resolve().parents[2]
G_SPEARMAN = REPO_ROOT / "analysis" / "advantages" / "g_spearman.py"
H_ORACLE = REPO_ROOT / "analysis" / "advantages" / "h_oracle.py"


def run(script: Path, zip_p: Path) -> str:
    """Run a g_spearman / h_oracle script and capture its stdout."""
    if not script.exists():
        raise FileNotFoundError(
            f"{script} not found — paper-metrics expects the GCRL-Landscapes repo "
            f"to live at {REPO_ROOT}."
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(script.parent) + os.pathsep + env.get("PYTHONPATH", "")
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_py.exists():
        raise FileNotFoundError(
            f"Project venv not found at {venv_py} — see paper-metrics/README.md."
        )
    cmd = [str(venv_py), str(script), "--zipfiles", str(zip_p)]
    print(f"  $ {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    return res.stdout


def parse_g_spearman(text: str, last_phase: int = 4) -> dict[str, float]:
    """Pull Variant-1 last-phase row per agent from g_spearman stdout."""
    block = re.search(
        r"Pairwise Spearman rank correlation of advantages across seeds \(same config\):"
        r"(.+?)Pairwise Spearman rank correlation of advantages \(positive only\)",
        text,
        flags=re.S,
    )
    if not block:
        raise RuntimeError("g_spearman.py output did not contain Variant 1 block.")
    out = {}
    for line in block.group(1).splitlines():
        m = re.match(
            r"\s+(crl|gciql|gcivl|qrl)\s+advantage/actor\s+(\d+)\s+([-\d.]+)", line
        )
        if m and int(m.group(2)) == last_phase:
            out[m.group(1).upper()] = float(m.group(3))
    return out


def parse_h_oracle(text: str, last_phase: int = 4) -> dict[str, float]:
    """Pull last-phase oracle Spearman r per agent."""
    block = re.search(
        r"Correlation between oracle advantage and predicted advantages per phase:"
        r"(.+?)(?:\n\n|Correlation between oracle advantage and predicted advantages per phase \(top-50)",
        text,
        flags=re.S,
    )
    if not block:
        raise RuntimeError("h_oracle.py output did not contain the per-phase block.")
    out = {}
    for line in block.group(1).splitlines():
        m = re.match(
            r"\s+(crl|gciql|gcivl|qrl)\s+advantage/actor\s+(\d+)\s+([-\d.]+)",
            line,
        )
        if m and int(m.group(2)) == last_phase:
            out[m.group(1).upper()] = float(m.group(3))
    return out


def main() -> None:
    zp = zip_path(ENV)
    if not zp.exists():
        raise FileNotFoundError(zp)

    print("Running analysis/advantages/g_spearman.py (Seed Spearman r)…")
    g_text = run(G_SPEARMAN, zp)
    seed_r = parse_g_spearman(g_text)

    print("Running analysis/advantages/h_oracle.py (Oracle r) — slow on first run…")
    h_text = run(H_ORACLE, zp)
    oracle_r = parse_h_oracle(h_text)

    rows = [
        {
            "agent": a,
            "Seed_r": round(seed_r.get(a, float("nan")), 3),
            "Oracle_r": round(oracle_r.get(a, float("nan")), 3),
        }
        for a in AGENTS
    ]
    out = pd.DataFrame(rows)
    save_csv(out, "t4_adv_corr")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
