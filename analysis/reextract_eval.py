#!/usr/bin/env python3
"""Re-extract policies from frozen checkpoints under a non-AWR extractor.

No training happens here. Each checkpoint from an existing AWR campaign is
loaded, its value/critic networks are left untouched, and only *action
selection at evaluation time* is changed:

- ``awr``       — the trained Gaussian actor, greedy (the published protocol).
- ``rejection`` — sample N candidate actions from the same trained actor and
  execute the argmax of ``min(Q1, Q2)``. The zero-temperature mode is always
  included as one candidate, so under a perfect critic this can only match or
  beat greedy decoding.

This answers the "is the method ordering AWR-specific?" question from the
reviews without retraining anything: the value signal is held fixed and the
extractor is swapped.

Only agents with an action-conditioned double-Q critic are supported (GCIQL,
CRL, FQL). GCIVL has no such critic and QRL's is a quasimetric, so neither can
be re-extracted this way — that asymmetry is a property of the methods, not a
gap in this script, and should be stated wherever the results are reported.

Usage
-----
    # one (agent, env) cell
    python analysis/reextract_eval.py \
        --logdir /project/.../logs-advantage-cube-single-fixed-batch/GCIQL_cube-single-play-v0_64c_lr-alpha \
        --n-rejection 32 --seeds 0 1 2 --eval-episodes 10 \
        --out outputs/reextract_cube_GCIQL.csv

    # restrict to a slice of configurations (for Slurm array sharding)
    python analysis/reextract_eval.py ... --config-slice 0 16
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import flax.serialization
import numpy as np
import pandas as pd
from ml_collections import ConfigDict

from gcrl_landscapes.evaluate import evaluate_wrapper
from gcrl_landscapes.util.datasets import AGENT_CLASSES, create_env_and_dataset

# Agents whose critic is action-conditioned and double-headed.
_SUPPORTED = {"GCIQL", "CRL", "FQL"}


def _final_phase(logdir: Path) -> int:
    """Largest phase_<step> directory present under run_logs."""
    steps = {
        int(m.group(1))
        for p in logdir.glob("run_logs/configuration_*/phase_*")
        if (m := re.fullmatch(r"phase_(\d+)", p.name))
    }
    if not steps:
        raise SystemExit(f"no phase_* directories under {logdir}/run_logs")
    return max(steps)


def _load_agent(
    checkpoint: Path, config: ConfigDict, agent_name: str, env, train_ds
) -> Any:
    """Rebuild the agent and restore its trained parameters."""
    example_batch = train_ds.sample(1)
    agent = AGENT_CLASSES[agent_name].create(
        0, example_batch["observations"], example_batch["actions"], config
    )
    with open(checkpoint, "rb") as fh:
        raw = pickle.load(fh)
    return flax.serialization.from_state_dict(agent, raw["agent"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-rejection", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--config-slice",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "STOP"),
        help="Only evaluate configurations [START, STOP) — for array sharding.",
    )
    parser.add_argument(
        "--extractors",
        nargs="+",
        default=["awr", "rejection"],
        help=(
            "Which extractors to evaluate. 'awr' is the frozen-policy control, "
            "'rejection' reranks the trained actor's own proposals, and "
            "'sfbc' draws candidates from --proposal-ckpt instead (AWR-free)."
        ),
    )
    parser.add_argument(
        "--proposal-ckpt",
        type=Path,
        default=None,
        help=(
            "params_*.pkl of an independently trained proposal policy (e.g. GCBC). "
            "Required by the 'sfbc' extractor: candidates are drawn from THIS "
            "policy and selected by the swept agent's frozen critic — rejection "
            "sampling in the sense of Park et al. 2024, with no AWR component "
            "anywhere in the action-selection path."
        ),
    )
    parser.add_argument(
        "--proposal-agent",
        type=str,
        default="GCBC",
        help="Agent class of --proposal-ckpt (default: GCBC).",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=None,
        help="Candidate count for 'sfbc' (default: same as --n-rejection).",
    )
    args = parser.parse_args()

    if "sfbc" in args.extractors and args.proposal_ckpt is None:
        raise SystemExit("--proposal-ckpt is required when 'sfbc' is in --extractors")
    if args.proposal_ckpt is not None and not args.proposal_ckpt.is_file():
        raise SystemExit(f"proposal checkpoint not found: {args.proposal_ckpt}")
    n_candidates = args.n_candidates or args.n_rejection

    logdir: Path = args.logdir
    agent_name = logdir.name.split("_")[0].upper()
    if agent_name not in _SUPPORTED:
        raise SystemExit(
            f"{agent_name} has no action-conditioned double-Q critic; "
            f"rejection sampling is undefined for it (supported: {sorted(_SUPPORTED)})"
        )

    info = logdir / "info.toml"
    dataset = None
    if info.is_file():
        import toml

        datasets = toml.load(info)["arguments"]["datasets"]
        dataset = datasets[0] if isinstance(datasets, list) else datasets
    if dataset is None:
        # Fall back to the directory naming convention.
        dataset = "_".join(logdir.name.split("_")[1:-2])
    print(f"agent={agent_name} dataset={dataset}", flush=True)

    phase = _final_phase(logdir)
    print(f"final phase = {phase}", flush=True)

    config_paths = sorted(
        (logdir / "configurations").glob("configuration_*.json"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if args.config_slice:
        start, stop = args.config_slice
        config_paths = config_paths[start:stop]
    print(f"{len(config_paths)} configurations to evaluate", flush=True)

    env_cache: dict = {}
    rows: list[dict] = []
    t0 = time.time()

    for config_path in config_paths:
        config_idx = int(config_path.stem.split("_")[1])
        base_cfg = json.loads(config_path.read_text())

        for seed in args.seeds:
            checkpoint = (
                logdir
                / "run_logs"
                / f"configuration_{config_idx}"
                / f"phase_{phase}"
                / f"seed_{seed}"
                / f"params_{phase}.pkl"
            )
            if not checkpoint.is_file():
                print(f"  MISSING {checkpoint}", flush=True)
                continue

            for extractor in args.extractors:
                cfg = ConfigDict(dict(base_cfg))
                # 'rejection' reranks the agent's own AWR proposals; 'sfbc'
                # swaps the proposal for an independently trained policy and
                # keeps only the frozen critic in the selection path.
                cfg["rejection_sampling_n"] = {
                    "rejection": args.n_rejection,
                    "sfbc": n_candidates,
                }.get(extractor, 0)
                if extractor == "sfbc":
                    cfg["rejection_proposal_ckpt"] = str(args.proposal_ckpt)
                    cfg["rejection_proposal_agent"] = args.proposal_agent

                key = (dataset, agent_name)
                if key not in env_cache:
                    env_cache[key] = create_env_and_dataset(dataset, agent_name, cfg)
                env, train_ds, _ = env_cache[key]

                agent = _load_agent(checkpoint, cfg, agent_name, env, train_ds)
                _, eval_metrics, _, _ = evaluate_wrapper(
                    agent, env, args.eval_episodes, cfg
                )

                rows.append(
                    {
                        "agent": agent_name,
                        "dataset": dataset,
                        "extractor": extractor,
                        "n_rejection": cfg["rejection_sampling_n"],
                        "proposal": (
                            args.proposal_agent if extractor == "sfbc" else "self"
                        ),
                        "configuration": config_idx,
                        "seed": seed,
                        "phase": phase,
                        "lr": base_cfg.get("lr"),
                        "alpha": base_cfg.get("alpha"),
                        "success": float(eval_metrics["success"]),
                    }
                )
                print(
                    f"  cfg={config_idx:3d} seed={seed} {extractor:9s} "
                    f"success={eval_metrics['success']:.3f} "
                    f"({time.time() - t0:.0f}s elapsed)",
                    flush=True,
                )

        # Write incrementally so a walltime kill still leaves usable results.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    if not rows:
        raise SystemExit("no checkpoints evaluated")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} rows to {args.out}", flush=True)

    summary = (
        df.groupby(["agent", "dataset", "extractor"])["success"]
        .agg(["mean", "max", "count"])
        .round(4)
    )
    print("\n" + summary.to_string(), flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
