import flax
from pathlib import Path
import pickle
from ml_collections import FrozenConfigDict
from typing import Generic, TypeVar, Any
import json
import pandas as pd

type EvalTrajectory = tuple[ResultsPerStep[EvaluationResult], ResultsPerStep[Path]]

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


class PhaseResult(dict[FrozenConfigDict, EvalTrajectory]):
    def to_dict(self) -> dict:
        def convert_evaluation_results(results: ResultsPerStep[EvaluationResult]):
            return {step: result.to_dict() for step, result in results.items()}

        def convert_paths(results: ResultsPerStep[Path]):
            return {step: str(path) for step, path in results.items()}

        return {
            config.to_json(): (
                convert_evaluation_results(eval_trajectory[0]),
                convert_paths(eval_trajectory[1]),
            )
            for config, eval_trajectory in self.items()
        }

    @classmethod
    def from_dict(cls, raw_result: dict) -> "PhaseResult":
        return PhaseResult(
            {
                FrozenConfigDict(json.loads(config)): (
                    ResultsPerStep(
                        {
                            int(step): EvaluationResult(**result)
                            for step, result in eval_trajectory[0].items()
                        }
                    ),
                    ResultsPerStep(
                        {
                            int(step): Path(path)
                            for step, path in eval_trajectory[1].items()
                        }
                    ),
                )
                for config, eval_trajectory in raw_result.items()
            }
        )


def restore_agent(agent, path: Path):
    with open(path, "rb") as f:
        load_dict = pickle.load(f)

    return flax.serialization.from_state_dict(agent, load_dict["agent"])


def phase_results_to_pandas(results: ResultsPerStep[PhaseResult]) -> pd.DataFrame:
    df = pd.DataFrame()
    # results are unpacked here until every step has a column, all hyperparameters have a column and the performance has a column
    run_id = 0
    for phase_step, result in results.items():
        for config, eval_trajectory in result.items():
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
                        "phase": phase_step,
                        "eval_step": eval_step,
                        "success": eval_result.success,
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
