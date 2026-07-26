"""Reproduce `tab:antmaze_ess_phase4` (main) and `tab:antmaze_ess_full` (appendix).

The typeset numbers were produced by `analysis/advantages/e_ess.py` on the
antmaze-medium fixed-batch zip (archived stdout:
`_malte_repro/e_ess_fixed_batch.log`). Its definition:

  - Per (config, seed, phase): pool the AWR weights w = exp(alpha * A) of
    every `advantage/actor` array logged at an eval step within the phase,
    clip to [1e-9, 100], scale by the pooled max, and compute Kish ESS on
    the pooled sample. n = 320 (64 configs x 5 seeds) per (agent, phase).
  - Correlate ESS against `mean_normalized_goal_distance_return`
    normalized by the per-(agent, phase) max.

Earlier versions of this script re-implemented the computation and
reproduced the r's but pooled ~5x more rows into the correlation, deflating
the p-values by orders of magnitude relative to the paper. To guarantee
definitional identity we now delegate to the committed `e_ess.py` (exactly
as t4 delegates to g_spearman/h_oracle) and parse its output.

Inputs:
  data/zips/2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip
Output:
  outputs/t3_antmaze_ess.csv  (one row per (agent, phase))

Requires the gcrl_landscapes package + the project venv.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from _common import save_csv, zip_path

ENV = "antmaze-medium"
AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]

REPO_ROOT = Path(__file__).resolve().parents[2]
E_ESS = REPO_ROOT / "analysis" / "advantages" / "e_ess.py"


def main() -> None:
    zp = zip_path(ENV)
    if not zp.exists():
        raise FileNotFoundError(zp)

    # Run e_ess in a temp dir but keep the tex it writes before cleanup.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(E_ESS.parent) + os.pathsep + env.get("PYTHONPATH", "")
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    for p, msg in ((E_ESS, "e_ess.py"), (venv_py, "project venv")):
        if not p.exists():
            raise FileNotFoundError(f"{msg} not found at {p}")
    with tempfile.TemporaryDirectory() as td:
        cmd = [str(venv_py), str(E_ESS), "--zipfiles", str(zp)]
        print(f"  $ {' '.join(cmd)}  (cwd={td})")
        res = subprocess.run(
            cmd, env=env, cwd=td, capture_output=True, text=True, check=True
        )
        stdout = res.stdout
        tex_files = list(Path(td).glob("plots/*/advantages/ess.tex"))
        tex = tex_files[0].read_text() if tex_files else ""

    # Full-precision r/p from stdout
    corr: dict[tuple[str, int], tuple[float, float]] = {}
    block = re.search(
        r"Correlation between ESS and normalized return, per phase:\n(.+?)(?:\n\s*\n|Saved )",
        stdout,
        re.S,
    )
    assert block, "correlation block not found in e_ess stdout"
    for m in re.finditer(
        r"^\s*(\w+) advantage/actor\s+(\d)\.0\s+(-?\d+\.\d+)\s+([\d.eE+-]+)\s*$",
        block.group(1),
        re.M,
    ):
        agent, phase, r, p = m.groups()
        corr[(agent.upper(), int(phase))] = (float(r), float(p))

    # mean ± std from the tex table
    stats: dict[tuple[str, int], tuple[float, float]] = {}
    for m in re.finditer(
        r"^(\w+) & advantage/actor & (\d)\.000 & (\d+\.\d+) \\\$\\pm\\\$ (\d+\.\d+)",
        tex,
        re.M,
    ):
        agent, phase, mean, std = m.groups()
        stats[(agent.upper(), int(phase))] = (float(mean), float(std))

    rows = []
    for agent in AGENTS:
        for phase in (1, 2, 3, 4):
            key = (agent, phase)
            assert key in corr, f"missing corr for {key}"
            assert key in stats, f"missing stats for {key}"
            r, p = corr[key]
            mean, std = stats[key]
            rows.append(
                {
                    "agent": agent,
                    "phase": phase,
                    "ess_mean": round(mean, 3),
                    "ess_std": round(std, 3),
                    "pearson_r": round(r, 3),
                    "p_value": round(p, 4),
                }
            )

    out = pd.DataFrame(rows)
    save_csv(out, "t3_antmaze_ess")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
