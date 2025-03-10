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


T = TypeVar("T")


class ResultsPerStep(dict[int, T], Generic[T]):
    def get_final_result(self) -> T:
        return self[max(self.keys())]


class EvaluationResult:
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


class PhaseResult(dict[FrozenConfigDict, list[EvalTrajectory]]):
    def to_dict(self) -> dict:
        return {
            config.to_json(): [
                eval_trajectory.to_json() for eval_trajectory in eval_trajectories
            ]
            for config, eval_trajectories in self.items()
        }

    @classmethod
    def from_dict(cls, raw_result: dict) -> "PhaseResult":
        return PhaseResult(
            {
                FrozenConfigDict(json.loads(config)): [
                    EvalTrajectory.from_json(eval_trajectory)
                    for eval_trajectory in eval_trajectories
                ]
                for config, eval_trajectories in raw_result.items()
            }
        )


def restore_agent(agent, path: Path):
    with open(path, "rb") as f:
        load_dict = pickle.load(f)

    return flax.serialization.from_state_dict(agent, load_dict["agent"])


def phase_results_to_pandas(results: ResultsPerStep[PhaseResult]) -> pd.DataFrame:
    # [TODO: this code is a mess, rewrite it]
    df = pd.DataFrame()
    # results are unpacked here until every step has a column, all hyperparameters have a column and the performance has a column
    run_id = 0
    for phase_step, result in results.items():
        for config, eval_trajectories in result.items():
            for eval_trajectory_seed, eval_trajectory in enumerate(eval_trajectories):
                for eval_step, (eval_result, path) in {
                    eval_step: (
                        eval_trajectory[0][eval_step],
                        eval_trajectory[1][eval_step],
                    )
                    for eval_step in eval_trajectory[0].keys()
                }.items():  # In EvalTrajectory both ResultsPerStep have the same keys
                    new_df = pd.DataFrame.from_dict(
                        {
                            "run_id": run_id,
                            "eval_seed": eval_trajectory_seed,
                            "phase_start": phase_step,
                            "eval_step": eval_step,
                            "success": eval_result.success,
                            "eval_result": eval_result,
                            "path": str(path),
                        }
                        | {
                            f"hp.{key}": [
                                value,
                            ]
                            for key, value in config.to_dict().items()
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
    def get_prefix_run_mappings(
        filenames: list[str], zip_file: zipfile.ZipFile
    ) -> dict[str, dict[str, Any]]:
        top_level_info_pattern = re.compile(r"^(logs/[^/]*/)info.toml")
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
                    results[phase][configurations[config_num]] = []
                results[phase][configurations[config_num]].append(
                    EvalTrajectory.from_json(json.loads(file_content))
                )

        return results

    with zipfile.ZipFile(zippath, "r") as zip_file:
        filenames: list[str] = zip_file.namelist()
        prefix_run_mapping = get_prefix_run_mappings(filenames, zip_file)
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
