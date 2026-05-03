#!/usr/bin/env python3
"""Run all analysis/advantages/ evaluation scripts on the 10B advantage datasets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from datetime import datetime

REPO_DIR = Path(__file__).parent.parent

ENVIRONMENTS = {
    "antmaze-large": {
        "zipfile": Path(
            "2026-04-30-logs-advantage-antmaze-large-single-fixed-batch.zip"
        ),
        "parquet": Path("advantage-10b-antmaze-large.parquet"),
        "checkpoints": Path("checkpoints-10b-antmaze-large.csv"),
    },
    "antmaze-medium": {
        "zipfile": Path("2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip"),
        "parquet": Path("advantage-10b-antmaze-medium.parquet"),
        "checkpoints": Path("checkpoints-10b-antmaze-medium.parquet"),
    },
    "cube": {
        "zipfile": Path("2026-04-27-logs-advantage-cube-single-fixed-batch.zip"),
        "parquet": Path("advantage-10b-cube.parquet"),
        "checkpoints": Path("checkpoints-10b-cube.csv"),
    },
}

SCRIPTS_DIR = REPO_DIR / "analysis" / "advantages"
AFR_SCRIPT = REPO_DIR / "analysis" / "compute_afr_metrics.py"
OUT_DIR = Path("logs/evaluation")


def run(name: str, cmd: list[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"Logging to: {log_file}")
    print("=" * 60)
    with open(log_file, "w") as f:
        f.write(f"=== {name} ===\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write(f"Command: {' '.join(cmd)}\n\n")
        f.flush()
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        f.write(result.stdout or "")
        if result.returncode != 0:
            f.write(f"\n[FAILED with exit code {result.returncode}]\n")
        f.write(f"\nFinished: {datetime.now().isoformat()}\n")
    print(f"  -> exit {result.returncode}")
    return result.returncode


def run_afr_metrics(env: str, parquet: Path, output: Path) -> int:
    name = f"afr/{env}"
    cmd = [
        sys.executable,
        str(AFR_SCRIPT),
        "--input",
        str(parquet),
        "--output",
        str(output),
    ]
    log_file = OUT_DIR / env / f"{datetime.now().strftime('%Y%m%d')}_afr_{env}.log"
    return run(name, cmd, log_file)


def run_advantages_scripts(env: str, zipfile: Path) -> dict[str, int]:
    results: dict[str, int] = {}
    for script in [
        "a_distributions.py",
        "b_stats.py",
        "c_weights.py",
        "d_correlation.py",
        "e_ess.py",
        "g_spearman.py",
        "h_oracle.py",
    ]:
        name = f"advantages/{env}/{script}"
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / script),
            "--zipfiles",
            str(zipfile),
            "--top-k",
            "5",
        ]
        log_file = (
            OUT_DIR
            / env
            / f"{datetime.now().strftime('%Y%m%d')}_{script.replace('.py', '.log')}"
        )
        results[name] = run(name, cmd, log_file)
    return results


def main() -> int:
    all_results: dict[str, int] = {}
    for env, cfg in ENVIRONMENTS.items():
        (OUT_DIR / env).mkdir(parents=True, exist_ok=True)
        all_results[f"afr/{env}"] = run_afr_metrics(
            env, cfg["parquet"], Path(f"Analysis/afr_metrics-{env}.csv")
        )
        all_results.update(run_advantages_scripts(env, cfg["zipfile"]))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, code in all_results.items():
        status = "OK" if code == 0 else f"FAILED ({code})"
        print(f"  {name:<50} {status}")

    failed = [n for n, c in all_results.items() if c != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
