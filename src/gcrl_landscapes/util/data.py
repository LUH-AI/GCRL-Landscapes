import flax
from pathlib import Path
import pickle


class EvaluationResult:
    def __init__(self, reward: float):
        self.reward = reward


def restore_agent(agent, path: Path):
    with open(path, "rb") as f:
        load_dict = pickle.load(f)

    return flax.serialization.from_state_dict(agent, load_dict["agent"])
