#!/usr/bin/env python3
"""Build per-configuration success CSV from raw run_logs directories.

Walks the directory structure:
  LOGS_BASE/
    {AGENT}_{env_tag}_64c_lr-alpha/
      configurations/
        configuration_{i}.json          ← contains alpha, lr
      run_logs/
        configuration_{i}/
          phase_{step}/
            seed_{j}/
              eval_log.csv              ← columns: success, step

Produces a CSV matching the schema of additional_stats_raw.csv:
  algo, env, config, seed, phase, alpha, lr, success

phase is the rank of the step checkpoint (1 = earliest, 4 = latest).

Usage
-----
    # antmaze-large
    python ANALYSIS/build_eval_stats.py \
        --logs /Users/adityamohan/git/GCRL/plots/new-plots/raw-data/logs-antmaze-large \
        --env antmaze-large-navigate-v0 \
        --out ENVS/antmaze-large/eval_stats.csv

    # cube
    python ANALYSIS/build_eval_stats.py \
        --logs /Users/adityamohan/git/GCRL/plots/new-plots/raw-data/logs-cube \
        --env cube-single-play-v0 \
        --out ENVS/cube/eval_stats.csv
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

DEFAULT_AGENTS = "CRL,GCIQL,GCIVL,QRL"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs",
        type=Path,
        required=True,
        help="Root of the logs directory (e.g. logs-antmaze-large)",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Env tag used in directory names (e.g. antmaze-large-navigate-v0)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument(
        "--agents",
        type=str,
        default=DEFAULT_AGENTS,
        help="Comma-separated agent-directory prefixes (default: %(default)s)",
    )
    parser.add_argument(
        "--dir-suffix",
        type=str,
        default="64c_lr-alpha",
        help="Suffix of the agent directories (default: %(default)s)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Optional variant tag (e.g. ddpgbc, nstep3, fql, rs32); when "
        "given, an extra 'variant' column is written so campaigns can be "
        "concatenated without ambiguity",
    )
    args = parser.parse_args()

    rows = []
    for agent in [a.strip() for a in args.agents.split(",") if a.strip()]:
        agent_dir = args.logs / f"{agent}_{args.env}_{args.dir_suffix}"
        if not agent_dir.exists():
            print(f"  [skip] {agent_dir.name} not found")
            continue

        # Read alpha and lr from configuration JSONs
        cfg_dir = agent_dir / "configurations"
        alpha_lr: dict[int, tuple[float, float]] = {}
        for cjson in sorted(cfg_dir.glob("configuration_*.json")):
            idx = int(re.search(r"(\d+)", cjson.name).group(1))
            data = json.loads(cjson.read_text())
            alpha_lr[idx] = (float(data["alpha"]), float(data["lr"]))

        # Walk run_logs
        run_logs = agent_dir / "run_logs"
        if not run_logs.exists():
            print(f"  [skip] {agent}: run_logs not found")
            continue

        # Collect all phase step values to build rank mapping
        step_vals: set[int] = set()
        for cfg_path in run_logs.iterdir():
            if not cfg_path.is_dir():
                continue
            for phase_path in cfg_path.iterdir():
                m = re.match(r"phase_(\d+)", phase_path.name)
                if m:
                    step_vals.add(int(m.group(1)))

        step_to_phase = {s: i + 1 for i, s in enumerate(sorted(step_vals))}
        print(f"  {agent}: phases {sorted(step_vals)} → {step_to_phase}")

        # Collect success values
        for cfg_path in sorted(
            run_logs.iterdir(), key=lambda p: int(re.search(r"\d+", p.name).group())
        ):
            if not cfg_path.is_dir():
                continue
            config_idx = int(re.search(r"(\d+)", cfg_path.name).group(1))
            if config_idx not in alpha_lr:
                continue
            alpha, lr = alpha_lr[config_idx]

            for phase_path in sorted(
                cfg_path.iterdir(), key=lambda p: int(re.search(r"\d+", p.name).group())
            ):
                m = re.match(r"phase_(\d+)", phase_path.name)
                if not m:
                    continue
                step = int(m.group(1))
                phase = step_to_phase[step]

                for seed_path in sorted(
                    phase_path.iterdir(),
                    key=lambda p: int(re.search(r"\d+", p.name).group()),
                ):
                    if not seed_path.is_dir():
                        continue
                    seed = int(re.search(r"(\d+)", seed_path.name).group(1))
                    eval_csv = seed_path / "eval_log.csv"
                    if not eval_csv.exists():
                        continue
                    try:
                        success = pd.read_csv(eval_csv)["success"].iloc[0]
                    except Exception:
                        continue

                    rows.append(
                        {
                            "algo": agent,
                            "env": args.env,
                            "config": config_idx,
                            "seed": seed,
                            "phase": phase,
                            "alpha": alpha,
                            "lr": lr,
                            "success": float(success),
                        }
                    )

    df = pd.DataFrame(rows)
    if args.variant:
        df["variant"] = args.variant
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows → {args.out}")
    print(
        df.groupby(["algo", "phase"])["success"]
        .agg(["mean", "std"])
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
