#!/usr/bin/env python3
"""Overwrite an experiment's sampled hyperparameters with a reference campaign's.

Why this exists
---------------
`configurations.py::_generate_configurations` builds ``Sobol(d=...)`` without a
seed. scipy's Sobol engine draws its scramble from its own default_rng, so the
``np.random.seed(seed)`` at the top of ``generate_configurations`` does not reach
it: **every ``main.py setup`` invocation produces a different 64-point design,
and the ``--seed`` flag is inert for the configuration draw.**

Consequently a freshly set-up arm cannot be compared per-configuration against
the published campaign — the two would sit on different points of the same
space. Injecting the published (lr, alpha) pairs turns the comparison from
"two independent samples of a landscape" into a paired one, where every
configuration index refers to the same hyperparameters in both arms.

Only the hyperparameter fields are copied. Everything else in the target config
(``n_step``, ``actor_loss``, agent architecture, dataset settings) is left
untouched, so the arm keeps the treatment it was set up with.

Usage
-----
    python analysis/inject_reference_configs.py \
        --target    logs-rebuttal/gciql-n3-antmaze-medium/GCIQL_..._64c_lr-alpha \
        --reference /project/NHWP25179/SSRL-Landscapes/logs-advantage-antmaze-medium-fixed-batch/GCIQL_..._64c_lr-alpha \
        --fields lr alpha

    # FQL: alpha is a distillation weight, not an AWR temperature, so only the
    # shared lr axis is paired.
    python analysis/inject_reference_configs.py --target ... --reference ... --fields lr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _configs_dir(path: Path) -> Path:
    """Accept either the experiment dir or its configurations/ subdirectory."""
    return path if path.name == "configurations" else path / "configurations"


def inject(
    target: Path, reference: Path, fields: list[str], dry_run: bool = False
) -> int:
    target_dir = _configs_dir(target)
    reference_dir = _configs_dir(reference)

    if not target_dir.is_dir():
        raise SystemExit(f"target configurations dir not found: {target_dir}")
    if not reference_dir.is_dir():
        raise SystemExit(f"reference configurations dir not found: {reference_dir}")

    target_files = sorted(
        target_dir.glob("configuration_*.json"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    reference_files = sorted(
        reference_dir.glob("configuration_*.json"),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    if not target_files:
        raise SystemExit(f"no configuration_*.json in {target_dir}")
    if len(target_files) != len(reference_files):
        raise SystemExit(
            f"configuration count mismatch: target has {len(target_files)}, "
            f"reference has {len(reference_files)} — refusing to inject"
        )

    for target_file, reference_file in zip(target_files, reference_files):
        if target_file.name != reference_file.name:
            raise SystemExit(
                f"index mismatch: {target_file.name} vs {reference_file.name}"
            )

    changed = 0
    for target_file, reference_file in zip(target_files, reference_files):
        target_cfg = json.loads(target_file.read_text())
        reference_cfg = json.loads(reference_file.read_text())

        missing = [f for f in fields if f not in reference_cfg]
        if missing:
            raise SystemExit(f"{reference_file} lacks field(s): {missing}")

        updates = {f: reference_cfg[f] for f in fields}
        if all(target_cfg.get(f) == v for f, v in updates.items()):
            continue

        target_cfg.update(updates)
        changed += 1
        if dry_run:
            print(f"  would patch {target_file.name}: {updates}")
        else:
            target_file.write_text(json.dumps(target_cfg, sort_keys=True))

    # Leave a breadcrumb so provenance is inspectable from the log tree itself.
    if not dry_run:
        (target_dir / "REFERENCE_CONFIGS.json").write_text(
            json.dumps(
                {
                    "reference": str(reference_dir),
                    "fields": fields,
                    "n_configurations": len(target_files),
                    "note": (
                        "Hyperparameters copied from the reference campaign so that "
                        "configuration_i denotes the same point in both. Required "
                        "because Sobol sampling in configurations.py is unseeded."
                    ),
                },
                indent=2,
            )
        )

    print(
        f"  injected {fields} from {reference_dir.parent.name} "
        f"into {len(target_files)} configs ({changed} changed)"
    )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fields", nargs="+", default=["lr", "alpha"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    inject(args.target, args.reference, args.fields, args.dry_run)
    sys.exit(0)


if __name__ == "__main__":
    main()
