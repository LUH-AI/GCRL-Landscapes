import flax
from pathlib import Path
import pickle
from ml_collections import FrozenConfigDict
from typing import Generic, TypeVar, Any
import json

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
    def to_json(self) -> str:
        def convert_evaluation_results(results: ResultsPerStep[EvaluationResult]):
            return {step: result.to_dict() for step, result in results.items()}

        def convert_paths(results: ResultsPerStep[Path]):
            return {step: str(path) for step, path in results.items()}

        return json.dumps(
            {
                config.to_json(): (
                    convert_evaluation_results(eval_trajectory[0]),
                    convert_paths(eval_trajectory[1]),
                )
                for config, eval_trajectory in self.items()
            }
        )


def restore_agent(agent, path: Path):
    with open(path, "rb") as f:
        load_dict = pickle.load(f)

    return flax.serialization.from_state_dict(agent, load_dict["agent"])
