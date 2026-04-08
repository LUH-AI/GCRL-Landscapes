"""Tests for the optimised CSV extraction in read_results_from_zip.

Builds a minimal in-memory zip (written to a temp file so the parallel
workers can open it), then asserts that:
  1. _csv_to_train_trajectory produces the same structure as the old
     TrainTrajectory.from_csv(pd.read_csv(...)) path.
  2. read_results_from_zip returns the expected keys and values.
"""

import io
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from gcrl_landscapes.util.data import (
    TrainTrajectory,
    _csv_to_train_trajectory,
    read_results_from_zip,
)

MOCK_CSV = "step;training/loss;time/total_time\n100;0.5;1.0\n200;0.3;2.0\n300;0.1;3.0\n"

MOCK_INFO_TOML = """
[arguments]
agent = "CRL"
datasets = ["antmaze-medium-navigate"]
hyperparameters = ["lr"]
n_configurations = 1
n_seeds = 1
"""

MOCK_CONFIG = {"lr": 0.001, "config_index": 0}


def _build_zip_bytes(
    *,
    n_phases: int = 2,
    n_seeds: int = 2,
    n_configs: int = 2,
) -> bytes:
    """Return bytes of a valid zip archive understood by read_results_from_zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        prefix = "logs/run1/"
        zf.writestr(f"{prefix}info.toml", MOCK_INFO_TOML)
        for cfg_idx in range(n_configs):
            config = dict(MOCK_CONFIG, config_index=cfg_idx, lr=0.001 * (cfg_idx + 1))
            zf.writestr(
                f"{prefix}configurations/configuration_{cfg_idx}.json",
                json.dumps(config),
            )
            for phase in range(n_phases):
                for seed in range(n_seeds):
                    path = f"{prefix}configuration_{cfg_idx}/phase_{phase}/seed_{seed}/train_log.csv"
                    zf.writestr(path, MOCK_CSV)
    return buf.getvalue()


def test_csv_to_train_trajectory_matches_pandas():
    """_csv_to_train_trajectory must produce the same step→metrics mapping as the old pd.read_csv path."""
    old = TrainTrajectory.from_csv(pd.read_csv(io.StringIO(MOCK_CSV), sep=";"))
    new = _csv_to_train_trajectory(MOCK_CSV)

    assert set(old.keys()) == set(new.keys()), "step keys differ"
    for step in old:
        old_row = old[step]
        new_row = new[step]
        assert set(old_row.keys()) == set(new_row.keys()), (
            f"column keys differ at step {step}"
        )
        for col in old_row:
            assert pytest.approx(old_row[col]) == new_row[col], (
                f"value mismatch at step={step} col={col}"
            )


def test_read_results_from_zip_structure():
    """read_results_from_zip must return one prefix with training logs for every (phase, config, seed)."""
    n_phases, n_seeds, n_configs = 2, 2, 2
    zip_bytes = _build_zip_bytes(
        n_phases=n_phases, n_seeds=n_seeds, n_configs=n_configs
    )

    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        tmp.write(zip_bytes)
        tmp.flush()
        results = read_results_from_zip(Path(tmp.name))

    assert len(results) == 1, "expected exactly one prefix (run1)"
    _run_info, phase_results, train_results = next(iter(results.values()))

    # training log results: one entry per phase
    assert set(train_results.keys()) == set(range(n_phases)), (
        f"expected phases {set(range(n_phases))}, got {set(train_results.keys())}"
    )
    for phase, phase_result in train_results.items():
        assert len(phase_result) == n_configs, (
            f"phase {phase}: expected {n_configs} configs, got {len(phase_result)}"
        )
        for config, seed_map in phase_result.items():
            assert len(seed_map) == n_seeds, (
                f"phase {phase} config {config}: expected {n_seeds} seeds, got {len(seed_map)}"
            )
            for seed, train_trajectory in seed_map.items():
                assert isinstance(train_trajectory, TrainTrajectory), (
                    f"expected TrainTrajectory, got {type(train_trajectory)}"
                )
                assert set(train_trajectory.keys()) == {100, 200, 300}, (
                    f"unexpected steps: {set(train_trajectory.keys())}"
                )


def test_parallel_matches_sequential():
    """The parallelised read_results_from_zip must produce the same CSV data as direct _csv_to_train_trajectory."""
    zip_bytes = _build_zip_bytes(n_phases=1, n_seeds=1, n_configs=1)
    expected = _csv_to_train_trajectory(MOCK_CSV)

    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        tmp.write(zip_bytes)
        tmp.flush()
        results = read_results_from_zip(Path(tmp.name))

    _run_info, _phase_results, train_results = next(iter(results.values()))
    # train_results: phase -> PhaseResult(config -> {seed -> TrainTrajectory})
    phase_result = next(iter(train_results.values()))
    seed_map = next(iter(phase_result.values()))
    train_trajectory = next(iter(seed_map.values()))

    assert set(train_trajectory.keys()) == set(expected.keys())
    for step in expected:
        for col in expected[step]:
            assert pytest.approx(expected[step][col]) == train_trajectory[step][col]
