#!/usr/bin/env python3
"""Run all analysis/advantages/ evaluation scripts on the 10B advantage datasets.

For each environment, runs all scripts twice:
  1. With all configurations (no --top-k)
  2. With top-5 configurations per agent (--top-k 5)

Outputs are organized by filter variant under logs/evaluation/{env}/{filter}/.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from datetime import datetime

REPO_DIR = Path(__file__).parent.parent

ENVIRONMENTS = {
    "antmaze-large": {
        "zipfile": REPO_DIR
        / "evaluation_10b"
        / ("2026-04-30-logs-advantage-antmaze-large-single-fixed-batch.zip"),
        "parquet": REPO_DIR / "evaluation_10b" / "advantage-10b-antmaze-large.parquet",
        "checkpoints": REPO_DIR
        / "evaluation_10b"
        / "checkpoints-10b-antmaze-large.csv",
    },
    "antmaze-medium": {
        "zipfile": REPO_DIR
        / "evaluation_10b"
        / ("2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip"),
        "parquet": REPO_DIR / "evaluation_10b" / "advantage-10b-antmaze-medium.parquet",
        "checkpoints": REPO_DIR
        / "evaluation_10b"
        / "checkpoints-10b-antmaze-medium.parquet",
    },
    "cube": {
        "zipfile": REPO_DIR
        / "evaluation_10b"
        / ("2026-04-27-logs-advantage-cube-single-fixed-batch.zip"),
        "parquet": REPO_DIR / "evaluation_10b" / "advantage-10b-cube.parquet",
        "checkpoints": REPO_DIR / "evaluation_10b" / "checkpoints-10b-cube.csv",
    },
    "scene": {
        "zipfile": REPO_DIR
        / "evaluation_10b"
        / ("2026-05-03-logs-advantage-scene-single-fixed-batch.zip"),
        "parquet": REPO_DIR / "evaluation_10b" / "advantage-10b-scene.parquet",
        "checkpoints": REPO_DIR / "evaluation_10b" / "checkpoints-10b-scene.csv",
    },
}

SCRIPTS_DIR = REPO_DIR / "analysis" / "advantages"
EXPORT_SCRIPT = SCRIPTS_DIR / "export_top_k.py"
AFR_SCRIPT = REPO_DIR / "analysis" / "compute_afr_metrics.py"
OUT_DIR = Path("logs/evaluation")
TOP_K = 5

ADVANTAGE_SCRIPTS = [
    "a_distributions.py",
    "b_stats.py",
    "c_weights.py",
    "d_correlation.py",
    "e_ess.py",
    "g_spearman.py",
    "h_oracle.py",
]


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


def export_top_k(env: str, zipfile: Path) -> Path:
    """Export top-k config IDs to JSON for an environment.

    Returns the path where the JSON was written (next to the zip file).
    """
    cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--zipfiles",
        str(zipfile),
        "--top-k",
        str(TOP_K),
    ]
    log_file = OUT_DIR / env / f"{datetime.now().strftime('%Y%m%d')}_export_top_k.log"
    run(f"export/{env}", cmd, log_file)
    # export_top_k.py writes the JSON next to the zip, not in OUT_DIR
    return zipfile.parent / f"top_k_{TOP_K}_{zipfile.stem}.json"


def run_afr(
    env: str,
    parquet: Path,
    output: Path,
    config_ids: Path,
    top_k_filter: str,
) -> int:
    """Run AFR metrics. top_k_filter is 'all' or 'top_k_5'."""
    cmd = [
        sys.executable,
        str(AFR_SCRIPT),
        "--input",
        str(parquet),
        "--output",
        str(output),
    ]
    if top_k_filter == "top_k_5":
        cmd.extend(["--config-ids", str(config_ids)])
    log_file = (
        OUT_DIR
        / env
        / top_k_filter
        / f"{datetime.now().strftime('%Y%m%d')}_afr_{env}.log"
    )
    return run(f"afr/{env}/{top_k_filter}", cmd, log_file)


def run_advantages(
    env: str,
    zipfile: Path,
    top_k: int | None,
    filter_label: str,
) -> dict[str, int]:
    """Run all advantage scripts. top_k=None means all configs."""
    results: dict[str, int] = {}
    for script in ADVANTAGE_SCRIPTS:
        name = f"advantages/{env}/{filter_label}/{script}"
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / script),
            "--zipfiles",
            str(zipfile),
        ]
        if top_k is not None:
            cmd.extend(["--top-k", str(top_k)])
        log_file = (
            OUT_DIR
            / env
            / filter_label
            / f"{datetime.now().strftime('%Y%m%d')}_{script.replace('.py', '.log')}"
        )
        results[name] = run(name, cmd, log_file)
    return results


def run_environment(env: str, cfg: dict) -> dict[str, int]:
    results: dict[str, int] = {}
    zipfile = cfg["zipfile"]
    parquet = cfg["parquet"]

    # Export top-k config IDs once per environment
    config_ids_file = export_top_k(env, zipfile)

    # Ensure subdirectories exist
    (OUT_DIR / env / "all").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / env / "top_k_5").mkdir(parents=True, exist_ok=True)

    # === All configs ===
    print(f"\n{'#' * 70}")
    print(f"# {env} — ALL CONFIGS")
    print(f"{'#' * 70}")

    results[f"afr/{env}/all"] = run_afr(
        env,
        parquet,
        Path(f"Analysis/afr_metrics-{env}-all.csv"),
        config_ids_file,
        "all",
    )
    results.update(run_advantages(env, zipfile, None, "all"))

    # === Top-k configs ===
    print(f"\n{'#' * 70}")
    print(f"# {env} — TOP-{TOP_K} CONFIGS")
    print(f"{'#' * 70}")

    results[f"afr/{env}/top_k_{TOP_K}"] = run_afr(
        env,
        parquet,
        Path(f"Analysis/afr_metrics-{env}-top_k_{TOP_K}.csv"),
        config_ids_file,
        f"top_k_{TOP_K}",
    )
    results.update(run_advantages(env, zipfile, TOP_K, f"top_k_{TOP_K}"))

    return results


def main() -> int:
    all_results: dict[str, int] = {}
    for env, cfg in ENVIRONMENTS.items():
        all_results.update(run_environment(env, cfg))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, code in all_results.items():
        status = "OK" if code == 0 else f"FAILED ({code})"
        print(f"  {name:<60} {status}")

    failed = [n for n, c in all_results.items() if c != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
