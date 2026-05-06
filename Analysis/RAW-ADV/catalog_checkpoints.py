#!/usr/bin/env python3
"""Catalog model checkpoints across agent subdirectories in a log parent directory.

Scans all agent subdirectories (matched by agent name prefix) for the most recent
phase checkpoint (`params_{phase}.pkl`) and outputs a pandas DataFrame summarising
each discovered checkpoint with its metadata.

Usage
-----
    # CLI
    python analysis/catalog_checkpoints.py --logdir ./logs-antmaze-medium/
    python analysis/catalog_checkpoints.py --logdir ./logs-antmaze-medium/ --output checkpoints.csv

    # Programmatically
    from pathlib import Path
    from analysis.catalog_checkpoints import catalog_checkpoints
    df = catalog_checkpoints(Path("./logs-antmaze-medium/"))
"""

from __future__ import annotations

import argparse
import pathlib
import re
import warnings

import pandas as pd
import toml

# Agent name prefixes to recognise as valid agent directories.
_AGENT_PATTERN = re.compile(r"^(CRL|QRL|GCIQL|GCIVL|CMD|GCBC|HIQL|SAC)_")

_CHECKPOINT_FILENAME = "params_{phase}.pkl"


def catalog_checkpoints(logdir: pathlib.Path) -> pd.DataFrame:
    """Scan *logdir* for agent subdirectories and collect last-phase checkpoint paths.

    For each discovered agent directory the function:
        1. Reads ``info.toml`` to extract *agent*, *datasets*, *phases*, *n_configurations*.
        2. Determines ``last_phase = max(phases)``.
        3. Walks every ``configuration_X/phase_last_phase/seed_Y`` subtree.
        4. Yields a row whenever ``params_{last_phase}.pkl`` exists.

    Parameters
    ----------
    logdir : pathlib.Path
        Parent log directory expected to contain one or more agent subdirectories.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly seven columns:
        ``checkpoint_path``, ``config_path``, ``agent``, ``dataset``, ``phase``, ``seed``, ``configuration``.
    """
    logdir = pathlib.Path(logdir).resolve()
    if not logdir.is_dir():
        warnings.warn(f"logdir is not a directory: {logdir}")
        return pd.DataFrame(
            columns=[
                "checkpoint_path",
                "config_path",
                "agent",
                "dataset",
                "phase",
                "seed",
                "configuration",
            ]
        )

    # ── Discover agent subdirectories ──────────────────────────────────────
    agent_dirs = sorted(
        d
        for d in logdir.iterdir()
        if d.is_dir()
        and (d / "info.toml").is_file()
        and (d / "run_logs").is_dir()
        and _AGENT_PATTERN.match(d.name)
    )

    if not agent_dirs:
        warnings.warn(f"No valid agent directories found in {logdir}")
        return pd.DataFrame(
            columns=[
                "checkpoint_path",
                "config_path",
                "agent",
                "dataset",
                "phase",
                "seed",
                "configuration",
            ]
        )

    rows: list[dict] = []

    for agent_dir in agent_dirs:
        agent_name = agent_dir.name
        toml_path = agent_dir / "info.toml"

        try:
            info = toml.load(toml_path)
        except Exception as exc:
            warnings.warn(f"Could not parse {toml_path}: {exc}")
            continue

        arguments = info.get("arguments", {})
        agent = arguments.get("agent", agent_name.split("_", 1)[0])
        datasets: list[str] = list(arguments.get("datasets", []))
        phases: list[int] = list(arguments.get("phases", []))

        if not phases:
            warnings.warn(f"  {agent_dir}: no phases in info.toml")
            continue

        last_phase = max(phases)
        phase_index = phases.index(last_phase)

        # Dataset selection with fallback
        if 0 <= phase_index < len(datasets):
            dataset = datasets[phase_index]
        elif datasets:
            warnings.warn(
                f"  {agent_dir}: phase_index {phase_index} out of range for {len(datasets)} datasets; "
                "using last dataset as fallback"
            )
            dataset = datasets[-1]
        else:
            dataset = "<unknown>"

        # ── Walk configuration -> phase -> seed ────────────────────────────
        run_logs = agent_dir / "run_logs"
        config_dirs = sorted(
            d
            for d in run_logs.iterdir()
            if d.is_dir() and re.match(r"configuration_\d+", d.name)
        )

        for config_dir in config_dirs:
            try:
                config_idx = int(
                    re.search(r"configuration_(\d+)", config_dir.name).group(1)
                )
            except (AttributeError, ValueError):
                continue

            phase_dir = config_dir / f"phase_{last_phase}"
            if not phase_dir.is_dir():
                warnings.warn(f"  {agent_dir}: phase dir {phase_dir} does not exist")
                continue

            seed_dirs = sorted(
                d
                for d in phase_dir.iterdir()
                if d.is_dir() and re.match(r"seed_\d+", d.name)
            )

            for seed_dir in seed_dirs:
                try:
                    seed = int(re.search(r"seed_(\d+)", seed_dir.name).group(1))
                except (AttributeError, ValueError):
                    continue

                checkpoint_path = seed_dir / _CHECKPOINT_FILENAME.format(
                    phase=last_phase
                )
                if checkpoint_path.is_file():
                    config_path = str(
                        agent_dir
                        / "configurations"
                        / f"configuration_{config_idx}.json"
                    )
                    rows.append(
                        {
                            "checkpoint_path": str(checkpoint_path),
                            "config_path": config_path,
                            "agent": agent,
                            "dataset": dataset,
                            "phase": last_phase,
                            "seed": seed,
                            "configuration": config_idx,
                        }
                    )
                # If checkpoint doesn't exist — silently skip (incomplete training)

    if not rows:
        warnings.warn("No checkpoints found across all agents")

    df = pd.DataFrame(
        rows,
        columns=[
            "checkpoint_path",
            "config_path",
            "agent",
            "dataset",
            "phase",
            "seed",
            "configuration",
        ],
    )
    if not df.empty:
        df.sort_values(
            by=["agent", "dataset", "phase", "configuration", "seed"],
            ascending=True,
            ignore_index=True,
            inplace=True,
        )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Catalog model checkpoints across agent subdirectories."
    )
    parser.add_argument(
        "--logdir",
        type=pathlib.Path,
        required=True,
        help="Parent log directory containing agent subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional CSV file to write the DataFrame to.",
    )
    args = parser.parse_args()

    df = catalog_checkpoints(args.logdir)
    print(f"Found {df.shape[0]} checkpoints")
    if df.empty:
        print("No checkpoints to display.")
    else:
        print(df.to_string(index=False))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
