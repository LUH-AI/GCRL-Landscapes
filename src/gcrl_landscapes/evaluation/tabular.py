import argparse
from gcrl_landscapes.util.data import (
    read_results_from_zip,
)
from pathlib import Path
import pandas as pd
import os
from .common import compute_additional_information, merge_experiments


def create_tables(results_pandas: pd.DataFrame, output_folder: Path):
    """Main code to generate the tabular data

    Args:
        results_pandas: pandas dataframe containing all results (all phases) for one experiment
        output_folder: folder to save plots in
    """
    per_config_folder = output_folder / "per_config"
    per_config_folder.mkdir(exist_ok=True)

    phases = sorted(results_pandas["phase"].unique().tolist())
    phase_results = [
        (
            phase,
            results_pandas[
                (results_pandas["eval_step"] == phase)
                & (results_pandas["phase"] == phase)
            ],
        )
        for phase in phases
    ]
    phase_results = compute_additional_information(phase_results)  # type: ignore

    for phase, phase_result in phase_results:
        per_config_phase_folder = per_config_folder / f"phase_{phase}"
        per_config_phase_folder.mkdir(exist_ok=True)
        phase_result_copy = phase_result.copy()
        phase_result_copy.loc[:, "run_id"], _ = pd.factorize(
            phase_result_copy["run_id"]
        )  # TripleGPModel needs continuous run-ids starting at 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zipfile", type=Path, required=True)
    args = parser.parse_args()

    # Parse results
    output_folder = Path("tables") / os.path.basename(args.zipfile)
    merged_results_df = merge_experiments(read_results_from_zip(args.zipfile))

    create_tables(merged_results_df, output_folder)
