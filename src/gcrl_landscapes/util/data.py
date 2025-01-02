import flax
from pathlib import Path
import pickle
from ml_collections import FrozenConfigDict
from typing import Generic, TypeVar, Any

type EvalTrajectory = tuple[ResultsPerStep[EvaluationResult], ResultsPerStep[Path]]

T = TypeVar("T")


class ResultsPerStep(dict[int, T], Generic[T]):
    def get_final_result(self) -> T:
        return self[max(self.keys())]


class EvaluationResult:
    def __init__(self, success: float, metrics: dict[str, float], info: dict[str, Any]):
        self.success = success

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"success: {self.success}"


type PhaseResult = dict[FrozenConfigDict, EvalTrajectory]


def restore_agent(agent, path: Path):
    with open(path, "rb") as f:
        load_dict = pickle.load(f)

    return flax.serialization.from_state_dict(agent, load_dict["agent"])
