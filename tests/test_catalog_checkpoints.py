"""Tests for analysis/catalog_checkpoints.py checkpoint discovery logic.

Tests the catalog_checkpoints() function using synthetic directory trees
written to temporary directories (tmp_path fixture from pytest).
"""

from pathlib import Path

import pandas as pd
import pytest
import toml

from analysis.catalog_checkpoints import catalog_checkpoints


# ── helpers ──────────────────────────────────────────────────────────────────


def mkinfo(
    agent: str = "CRL",
    datasets: list[str] | None = None,
    phases: list[int] | None = None,
    n_configurations: int = 1,
) -> str:
    """Build a minimal ``[arguments]`` block for info.toml."""
    info = {
        "arguments": {
            "agent": agent,
            "datasets": datasets or ["antmaze-medium-navigate-v0"],
            "phases": phases or [39772, 79545, 119318, 159090],
            "n_configurations": n_configurations,
        },
        "git": {"commit": "abc123", "branch": "main"},
        "time": "2026-01-01T00:00:00",
    }
    return toml.dumps(info)


def build_minimal_agent_dir(tmp_dir: Path, n_seeds: int = 2) -> Path:
    """Create one agent directory with minimal structure for checkpoint discovery.

    Returns the agent directory path.
    """
    agent_dir = tmp_dir / "CRL_antmaze-medium-navigate-v0_64c_lr-alpha"
    agent_dir.mkdir()

    # info.toml
    agent_dir.joinpath("info.toml").write_text(
        mkinfo(
            agent="CRL",
            datasets=["antmaze-medium-navigate-v0", "antmaze-medium-explore-v0"],
            phases=[39772, 79545, 119318, 159090],
        )
    )

    # run_logs / configuration_0 / phase_159090 / seed_0 | seed_1
    config_dir = agent_dir / "run_logs" / "configuration_0" / "phase_159090"
    for s in range(n_seeds):
        seed_dir = config_dir / f"seed_{s}"
        seed_dir.mkdir(parents=True)
        seed_dir.joinpath("params_159090.pkl").touch()

    return agent_dir


def build_multi_config_agent_dir(tmp_dir: Path) -> Path:
    """Create an agent directory with two configs and 3 seeds each."""
    agent_dir = tmp_dir / "QRL_antmaze-medium-explore-v0_64c"
    agent_dir.mkdir()

    agent_dir.joinpath("info.toml").write_text(
        mkinfo(
            agent="QRL",
            datasets=[
                "antmaze-medium-explore-v0",
                "antmaze-medium-explore80navigate-v0",
            ],
            phases=[50000, 100000],
        )
    )

    for cfg in range(2):
        config_dir = agent_dir / "run_logs" / f"configuration_{cfg}" / "phase_100000"
        for seed in range(3):
            seed_dir = config_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True)
            seed_dir.joinpath("params_100000.pkl").touch()

    return agent_dir


# ── tests ────────────────────────────────────────────────────────────────────


class TestCatalogCheckpoints:
    """Tests for the ``catalog_checkpoints`` function."""

    def test_returns_empty_dataframe_for_empty_dir(self, tmp_path: Path) -> None:
        """An empty logdir yields an empty DataFrame with the correct columns."""
        df = catalog_checkpoints(tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert list(df.columns) == [
            "checkpoint_path",
            "agent",
            "dataset",
            "phase",
            "seed",
            "configuration",
        ]

    def test_returns_empty_dataframe_for_no_agent_dirs(self, tmp_path: Path) -> None:
        """Directories without both info.toml + run_logs/ are ignored."""
        fake = tmp_path / "not_an_agent"
        fake.mkdir()
        fake.joinpath("info.toml").write_text('[arguments]\nagent = "blah"')
        df = catalog_checkpoints(tmp_path)
        assert df.empty

    def test_single_agent_single_config(self, tmp_path: Path) -> None:
        """One agent, one config, two seeds → 2 rows."""
        build_minimal_agent_dir(tmp_path, n_seeds=2)

        df = catalog_checkpoints(tmp_path)
        assert not df.empty
        assert len(df) == 2
        assert list(df.columns) == [
            "checkpoint_path",
            "agent",
            "dataset",
            "phase",
            "seed",
            "configuration",
        ]

        assert df["agent"].iloc[0] == "CRL"
        assert df["dataset"].iloc[0] == "antmaze-medium-navigate-v0"
        assert df["phase"].iloc[0] == 159090
        assert df["configuration"].iloc[0] == 0
        assert set(df["seed"].tolist()) == {0, 1}

    def test_single_agent_multi_config(self, tmp_path: Path) -> None:
        """One agent with 2 configs × 3 seeds each → 6 rows."""
        build_multi_config_agent_dir(tmp_path)

        df = catalog_checkpoints(tmp_path)
        assert not df.empty
        assert len(df) == 6
        assert df["agent"].iloc[0] == "QRL"
        assert df["phase"].iloc[0] == 100000
        assert df["configuration"].iloc[0] == 0
        assert set(df["configuration"].tolist()) == {0, 1}
        assert set(df["seed"].unique()) == {0, 1, 2}

    def test_deterministic_sort_order(self, tmp_path: Path) -> None:
        """Rows should be sorted by agent, dataset, phase, configuration, seed."""
        build_minimal_agent_dir(tmp_path, n_seeds=3)

        # Add a QRL agent with a different phase
        qrl_dir = tmp_path / "QRL_antmaze-medium-explore-v0_64c_test"
        qrl_dir.mkdir()
        qrl_dir.joinpath("info.toml").write_text(
            mkinfo(
                agent="QRL",
                datasets=["antmaze-medium-explore-v0"],
                phases=[99999],
            )
        )
        seed_dir = qrl_dir / "run_logs" / "configuration_0" / "phase_99999" / "seed_0"
        seed_dir.mkdir(parents=True)
        seed_dir.joinpath("params_99999.pkl").touch()

        df = catalog_checkpoints(tmp_path)
        assert not df.empty
        # CRL comes before QRL alphabetically
        assert df["agent"].iloc[0] == "CRL"
        assert df["agent"].iloc[-1] == "QRL"

    def test_partial_checkpoints_skipped(self, tmp_path: Path) -> None:
        """If a seed directory lacks the checkpoint file, that row is skipped."""
        agent_dir = build_minimal_agent_dir(tmp_path, n_seeds=3)
        # Delete one checkpoint
        ckpt = (
            agent_dir
            / "run_logs"
            / "configuration_0"
            / "phase_159090"
            / "seed_1"
            / "params_159090.pkl"
        )
        ckpt.unlink()

        df = catalog_checkpoints(tmp_path)
        assert len(df) == 2  # only seed_0 and seed_2 remain

    def test_phase_index_dataset_fallback(self, tmp_path: Path) -> None:
        """When ``datasets`` is shorter than ``phases``, uses the last dataset."""
        agent_dir = tmp_path / "SAC_antmaze-medium-navigate-v0_64c_test"
        agent_dir.mkdir()
        agent_dir.joinpath("info.toml").write_text(
            toml.dumps(
                {
                    "arguments": {
                        "agent": "SAC",
                        "datasets": ["antmaze-medium-navigate-v0"],  # only 1
                        "phases": [100, 200, 300],  # 3 phases
                    },
                }
            )
        )
        seed_dir = agent_dir / "run_logs" / "configuration_0" / "phase_300" / "seed_0"
        seed_dir.mkdir(parents=True)
        seed_dir.joinpath("params_300.pkl").touch()

        df = catalog_checkpoints(tmp_path)
        assert not df.empty
        assert df["dataset"].iloc[0] == "antmaze-medium-navigate-v0"
        assert df["phase"].iloc[0] == 300

    def test_no_run_logs_skipped(self, tmp_path: Path) -> None:
        """An agent dir without run_logs/ is ignored."""
        agent_dir = tmp_path / "HIQL_antmaze-medium-navigate-v0_test"
        agent_dir.mkdir()
        agent_dir.joinpath("info.toml").write_text(mkinfo(agent="HIQL"))
        # Deliberately no run_logs

        df = catalog_checkpoints(tmp_path)
        assert df.empty

    def test_no_phases_skipped(self, tmp_path: Path) -> None:
        """An agent dir with empty phases list is ignored."""
        agent_dir = tmp_path / "CMD_antmaze-medium-navigate-v0_test"
        agent_dir.mkdir()
        agent_dir.joinpath("info.toml").write_text(
            toml.dumps({"arguments": {"agent": "CMD", "datasets": [], "phases": []}})
        )
        df = catalog_checkpoints(tmp_path)
        assert df.empty

    def test_nonexistent_logdir(self) -> None:
        """A non-existent path produces an empty DataFrame."""
        df = catalog_checkpoints(Path("/tmp/does_not_exist_12345"))
        assert df.empty

    def test_agent_dir_regex_filter(self, tmp_path: Path) -> None:
        """Only directories matching the agent prefix regex are discovered."""
        # Valid prefix – discovered but no checkpoints
        valid = tmp_path / "CRL_antmaze-medium-navigate-v0_64c"
        valid.mkdir()
        valid.joinpath("info.toml").write_text(mkinfo(agent="CRL"))
        valid.joinpath("run_logs").mkdir()

        # Invalid prefix – ignored entirely
        invalid = tmp_path / "INVALID_agent_test"
        invalid.mkdir()
        invalid.joinpath("info.toml").write_text(mkinfo(agent="INVALID"))
        invalid.joinpath("run_logs").mkdir()

        df = catalog_checkpoints(tmp_path)
        assert df.empty  # valid agent had no checkpoints
