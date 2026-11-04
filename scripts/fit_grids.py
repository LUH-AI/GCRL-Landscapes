"""Per-(agent, phase) GP means on the same grid mobility_plot uses.

Usage: python fit_grids.py <logs.zip> <grids.pkl>
"""
import pickle
import sys
from pathlib import Path

import numpy as np

from gcrl_landscapes.configurations import get_bounds, sobol_codomain_to_hp
from gcrl_landscapes.evaluation.common import (
    compute_additional_information,
    resolve_alpha_sync,
)
from gcrl_landscapes.util.data import phase_results_to_pandas, read_results_from_zip
from gcrl_landscapes.util.eval import fit_model

zip_path, out_path = sys.argv[1], Path(sys.argv[2])
Y = "mean_normalized_goal_distance_return"
GRID = 100

results = read_results_from_zip(Path(zip_path), load_training_logs=False)
print("runs:", list(results), flush=True)

g = np.linspace(0, 1, GRID)
X0, X1 = np.meshgrid(g, g)
points = np.vstack([X0.ravel(), X1.ravel()]).T

out = {}
for prefix, (run_info, phase_results, _) in results.items():
    hps = run_info["arguments"]["hyperparameters"]
    df = phase_results_to_pandas(phase_results)
    df = compute_additional_information(resolve_alpha_sync(df, hps))
    df[Y] = df[Y].clip(0, 1)
    hp_list = [f"hp.{h}" for h in hps]
    agent = str(df["hp.agent_name"].iloc[0])
    loss = df["hp.actor_loss"].iloc[0] if "hp.actor_loss" in df.columns else None
    bx = get_bounds(hp_list[0].removeprefix("hp."), agent, loss)
    by = get_bounds(hp_list[1].removeprefix("hp."), agent, loss)
    print(agent, hp_list, bx, by, flush=True)
    px = sobol_codomain_to_hp(points[:, 0], *bx)
    py = sobol_codomain_to_hp(points[:, 1], *by)
    for phase, group in df.groupby("phase_num"):
        model = fit_model(group.copy().reset_index(), Y, hp_list)
        model.fit()
        Z = model.get_middle(points).clip(0, 1).squeeze()
        # Raw per-config peak, kept to sanity check the GP fit.
        cfg = group.groupby(hp_list)[Y].mean()
        out[(agent, int(phase))] = dict(
            Z=Z, px=px, py=py, hp_list=hp_list, bounds=(bx, by),
            raw_max=float(cfg.max()), n_cfg=int(len(cfg)),
        )
        print(agent, phase, "gp_max", round(float(Z.max()), 3), "raw_max",
              round(float(cfg.max()), 3), "n_cfg", len(cfg), flush=True)

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump(out, f)
print("saved", out_path)
