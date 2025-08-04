import flax
from pathlib import Path
import pickle
from ml_collections import FrozenConfigDict
from typing import Generic, TypeVar, Any
import json
import pandas as pd
import zipfile
import re
import toml
from tempfile import NamedTemporaryFile
from glob import glob
from scipy.stats import trim_mean
import numpy as np


T = TypeVar("T")


class ResultsPerStep(dict[int, T], Generic[T]):
    """Mapping of training step/phase step/... to some kind of result"""

    def get_final_result(self) -> T:
        return self[max(self.keys())]


class EvaluationResult:
    """Result of an evaluation run of an algorithm.
    Usually contains the aggregated values for the whole evaluation run.

    Attributes:
        success: Success rate
        metrics: all gathered metrics (usually mean)
        info: evaluation info dictionary
    """

    def __init__(self, success: float, metrics: dict[str, float], info: dict[str, Any]):
        self.success = success
        self.metrics = metrics
        self.info = info

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"success: {self.success}"

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "metrics": self.metrics, "info": self.info}


class EvalTrajectory(tuple[ResultsPerStep[EvaluationResult], ResultsPerStep[Path]]):
    """A mapping of training step to evaluation result and checkpoint path."""

    def to_json(self) -> str:
        def convert_evaluation_results(results: ResultsPerStep[EvaluationResult]):
            return {step: result.to_dict() for step, result in results.items()}

        def convert_paths(results: ResultsPerStep[Path]):
            return {step: str(path) for step, path in results.items()}

        return json.dumps((convert_evaluation_results(self[0]), convert_paths(self[1])))

    @classmethod
    def from_json(cls, eval_trajectory: tuple[dict, dict]) -> "EvalTrajectory":
        return EvalTrajectory(
            (
                ResultsPerStep(
                    {
                        int(step): EvaluationResult(**result)
                        for step, result in eval_trajectory[0].items()
                    }
                ),
                ResultsPerStep(
                    {int(step): Path(path) for step, path in eval_trajectory[1].items()}
                ),
            )
        )


class PhaseResult(dict[FrozenConfigDict, dict[int, EvalTrajectory]]):
    """Maps configurations to seed-EvalTrajectory pairs"""

    def to_dict(self) -> dict:
        return {
            config.to_json(): {
                seed: eval_trajectory.to_json()
                for seed, eval_trajectory in eval_trajectories.items()
            }
            for config, eval_trajectories in self.items()
        }

    @classmethod
    def from_dict(cls, raw_result: dict) -> "PhaseResult":
        return PhaseResult(
            {
                FrozenConfigDict(json.loads(config)): {
                    seed: EvalTrajectory.from_json(eval_trajectory)
                    for seed, eval_trajectory in eval_trajectories.items()
                }
                for config, eval_trajectories in raw_result.items()
            }
        )


def get_best_agent_path(
    result_pandas: pd.DataFrame, seed_mode: str = "iqm", eval_step: int | None = None
) -> Path:
    """Find the best agent for the given results for a phase.
    Currently uses success as metric for gauging "best"
    Uses highest evaluation step if not specified.

    Args:
        result_pandas: phase results as parsed by `get_phase_results`
        seed_mode: Which trained model (different seeds) to pick. Currently supports only "iqm", meaning picking the one closest to interquartile mean evaluation
        eval_step: For which evaluation step to get best agent. Will use highest available if not specified

    Returns:
         Path to best agent checkpoint
    """
    final_eval_step = eval_step if eval_step else result_pandas["eval_step"].max()
    final_eval_result = result_pandas[result_pandas["eval_step"] == final_eval_step]

    if seed_mode == "iqm":
        iqms = final_eval_result.groupby("config_index")["success"].apply(
            lambda df: trim_mean(df, proportiontocut=0.25)
        )

        best_config_index = iqms.idxmax()
        best_config_iqm = iqms.max()

        subset = final_eval_result[
            final_eval_result["config_index"] == best_config_index
        ]

        idx_closest = np.abs(subset["success"] - best_config_iqm).idxmin()
        best_config, phase, seed = subset.loc[
            idx_closest, ["config_index", "phase", "seed"]
        ]
    else:
        raise NotImplementedError(f"no seed mode {seed_mode}")

    best_df = result_pandas[
        (result_pandas["eval_step"] == phase)
        & (result_pandas["config_index"] == best_config)
        & (result_pandas["seed"] == seed)
    ]
    assert len(best_df) == 1
    best_path = Path(best_df.iloc[0]["path"])
    assert best_path.exists()
    return best_path


def get_phase_results(
    phase: int, logdir: Path, eval_step: int | None = None
) -> pd.DataFrame:
    logfiles = glob(str(logdir / "**"), recursive=True)
    nonbinary_logfiles = [
        file
        for file in logfiles
        if not re.fullmatch(r"^.*/submitit/.*$", file)
        and not re.fullmatch(r"^.*\.pkl$", file)
    ]
    phase_nonbinary_logfiles = [
        file
        for file in nonbinary_logfiles
        if re.fullmatch(f"^.*/phase_{phase}/.*$", file)
        or not re.fullmatch(r"^.*/phase_.*$", file)
    ]
    assert not [
        file
        for file in phase_nonbinary_logfiles
        if phase_nonbinary_logfiles.count(file) > 1
    ]

    # create temporary zip as it is currently needed to read results
    with NamedTemporaryFile(mode="w+b") as temp_f:
        with zipfile.ZipFile(temp_f, mode="w") as zip_f:
            for file in phase_nonbinary_logfiles:
                zip_f.write(file, arcname=re.sub(r"^.*/logs[^/]*/", "logs/", file))
        temp_f.flush()
        results_from_zip = read_results_from_zip(Path(temp_f.name))

    assert len(results_from_zip.values()) == 1
    result: ResultsPerStep[PhaseResult] = list(results_from_zip.values())[0][1]
    result_pandas = phase_results_to_pandas(result)

    return result_pandas


def restore_agent(agent, path: Path):
    """Restore agent from path

    Args:
        agent (): current agent object
        path: path of checkpoint

    Returns:
        agent with loaded checkpoint
    """
    with open(path, "rb") as f:
        load_dict = pickle.load(f)

    return flax.serialization.from_state_dict(agent, load_dict["agent"])


def phase_results_to_pandas(results: ResultsPerStep[PhaseResult]) -> pd.DataFrame:
    """Create pandas dataframe from phase results

    Args:
        results: experiment results, look at type hints and classes for more documentation

    Returns:
        pandas dataframe constructed from results
    """
    # [TODO: this code is a mess, rewrite it]
    df = pd.DataFrame()
    # results are unpacked here until every step has a column, all hyperparameters have a column and the performance has a column
    run_id = 0
    for phase_step, result in results.items():
        for config, eval_trajectories in result.items():
            for seed, eval_trajectory in eval_trajectories.items():
                for eval_step, (eval_result, path) in {
                    eval_step: (
                        eval_trajectory[0][eval_step],
                        eval_trajectory[1][eval_step]
                        if eval_step in eval_trajectory[1]
                        else None,
                    )
                    for eval_step in eval_trajectory[0].keys()
                }.items():  # In EvalTrajectory both ResultsPerStep have the same keys
                    new_df = pd.DataFrame.from_dict(
                        {
                            "run_id": run_id,
                            "seed": seed,
                            "phase": phase_step,
                            "eval_step": eval_step,
                            "success": eval_result.success,
                            "eval_result": eval_result,
                            "path": str(path),
                            "config_index": config["config_index"],
                        }
                        | {
                            f"hp.{key}": [
                                value,
                            ]
                            for key, value in config.to_dict().items()
                            if key != "config_index"
                        }
                    )
                    df = pd.concat(
                        [df, new_df],
                        ignore_index=True,
                    )
            run_id += 1  # every configuration per phase has a distinct run_id
    return df


def read_results_from_zip(
    zippath: Path,
) -> dict[str, tuple[dict, ResultsPerStep[PhaseResult]]]:
    """read results from zip that mirrors file structure as constructed by training scripts.
    This function heavily relies on regex and therefore proper file naming.

    Args:
        zippath: path to zip from experiment results. Root contains only folder "logs/"

    Returns:
        All experiment results inside of logs folder. Mapped by name
    """

    def get_prefix_runinfo_mappings(
        filenames: list[str], zip_file: zipfile.ZipFile
    ) -> dict[str, dict[str, Any]]:
        top_level_info_pattern = re.compile(r"^(logs[^/]*/[^/]*/)info.toml")
        top_level_info_names = [
            filename for filename in filenames if top_level_info_pattern.match(filename)
        ]
        info_contents: dict[str, str] = {}
        for info_name in top_level_info_names:
            with zip_file.open(info_name) as f:
                info_contents[info_name] = f.read().decode(encoding="utf-8")
        return {
            top_level_info_pattern.match(info_name).group(1): toml.loads(
                info_contents[info_name]
            )
            for info_name in top_level_info_names
        }

    def get_configurations(
        filenames: list[str], zip_file: zipfile.ZipFile
    ) -> dict[int, FrozenConfigDict]:
        configuration_regex = re.compile(r"^.*configurations/configuration_(\d+).json$")
        regex_matches = [configuration_regex.match(filename) for filename in filenames]
        configurations = {}
        for configuration_idx, configuration_path in {
            int(match.group(1)): match.group(0) for match in regex_matches if match
        }.items():
            with zip_file.open(configuration_path) as f:
                configurations[configuration_idx] = FrozenConfigDict(
                    json.loads(f.read().decode(encoding="utf-8"))
                    | {"config_index": configuration_idx}
                )
        return configurations

    def extract_results(
        filenames: list[str],
        configurations: dict[int, FrozenConfigDict],
        zip_file: zipfile.ZipFile,
    ) -> ResultsPerStep[PhaseResult]:
        def parse_filename(filename: str) -> tuple[int, int, int]:
            """Return (configuration_number, phase, seed)"""
            match = re.search(
                r".*configuration_(\d+)/phase_(\d+)/seed_(\d+).*", filename
            )
            return int(match.group(1)), int(match.group(2)), int(match.group(3))

        results: ResultsPerStep[PhaseResult] = ResultsPerStep({})
        for filename in [
            filename
            for filename in filenames
            if re.search(r".*eval_trajectory\.json$", filename)
        ]:
            with zip_file.open(filename) as f:
                file_content = f.read().decode(encoding="utf-8")
                config_num, phase, seed = parse_filename(filename)
                if phase not in results:
                    results[phase] = PhaseResult({})
                if configurations[config_num] not in results[phase]:
                    results[phase][configurations[config_num]] = {}
                results[phase][configurations[config_num]][seed] = (
                    EvalTrajectory.from_json(json.loads(file_content))
                )

        return results

    with zipfile.ZipFile(zippath, "r") as zip_file:
        filenames: list[str] = zip_file.namelist()
        prefix_run_mapping = get_prefix_runinfo_mappings(filenames, zip_file)
        prefix_filenames_mapping = {
            prefix: [
                filename for filename in filenames if re.match(f"^{prefix}.*", filename)
            ]
            for prefix in prefix_run_mapping.keys()
        }
        prefix_configuration_mapping: dict[str, dict[int, FrozenConfigDict]] = {
            prefix: get_configurations(filenames, zip_file)
            for prefix, filenames in prefix_filenames_mapping.items()
        }
        prefix_phase_results_mapping = {
            prefix: extract_results(
                prefix_filenames_mapping[prefix],
                prefix_configuration_mapping[prefix],
                zip_file,
            )
            for prefix in prefix_run_mapping.keys()
        }

    return {
        prefix: (prefix_run_mapping[prefix], prefix_phase_results_mapping[prefix])
        for prefix in prefix_run_mapping.keys()
    }
