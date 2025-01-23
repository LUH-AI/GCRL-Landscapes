import argparse
from util.data import ResultsPerStep, PhaseResult, phase_results_to_pandas
import json
from pathlib import Path
from plots.triple_gp import TripleGPModel

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

    df = phase_results_to_pandas(results)

    model = TripleGPModel()
