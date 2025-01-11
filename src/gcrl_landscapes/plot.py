import argparse
from util.data import ResultsPerStep, PhaseResult
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfolder", type=str, required=True)
    args = parser.parse_args()

    with open(Path(args.logfolder) / "results.json", "r") as f:
        results_raw = json.load(f)

    results: ResultsPerStep[PhaseResult] = ResultsPerStep(
        {
            int(step): PhaseResult.from_dict(result)
            for step, result in results_raw.items()
        }
    )
