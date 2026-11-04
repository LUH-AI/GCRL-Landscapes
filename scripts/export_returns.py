"""Export per-configuration goal-distance return for every (method, phase).

These are the observations the GP surfaces in fit_grids.py are fit to, used for
the pipeline walkthrough figure.

Usage: python export_returns.py <logs.zip> <returns.pkl>
"""
import pickle
import sys
from pathlib import Path

from gcrl_landscapes.evaluation.common import (
    compute_additional_information,
    resolve_alpha_sync,
)
from gcrl_landscapes.util.data import phase_results_to_pandas, read_results_from_zip

Y = "mean_normalized_goal_distance_return"
results = read_results_from_zip(Path(sys.argv[1]), load_training_logs=False)

out = {}
for _, (run_info, phase_results, _) in results.items():
    df = phase_results_to_pandas(phase_results)
    df = compute_additional_information(
        resolve_alpha_sync(df, run_info["arguments"]["hyperparameters"])
    )
    agent = str(df["hp.agent_name"].iloc[0]).lower()
    for phase, group in df.groupby("phase_num"):
        out[(agent, int(phase))] = group[["hp.lr", "hp.alpha", Y]].to_numpy()
        print(agent, phase, out[(agent, int(phase))].shape, flush=True)

with open(sys.argv[2], "wb") as f:
    pickle.dump(out, f)
