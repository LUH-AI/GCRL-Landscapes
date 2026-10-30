#!/usr/bin/env python3
"""#5/#6: n-step TD mobility plots at every threshold, GCIVL + GCIQL.

Camera-ready items #5 (GCIVL n=1,3,5 on Cube & AntMaze-L) and #6 (GCIQL n=1
vs n=3 on AntMaze-M). Landscape heatmaps for these cells already exist under
icrl-resubmission/plots/{gcivl,gciql}-nstep-*/n*/landscape-final.{png,pdf} —
this script (re)generates only the paired mobility plots, at every threshold
in MOBILITY_THRESHOLDS, reusing analysis/compare_nstep_breadth.py's CELLS and
load_cell() so the tree paths / seed handling stay identical to the numbers
already validated in outputs/nstep_breadth_final.csv.

Mobility here is single-layer, across the phases *within* one n-arm (by_col =
phase_num) — not a comparison across n values. GCIQL n=3 has only one phase
(820277; see analysis/compare_nstep_breadth.py docstring), so its mobility
plot is degenerate (one KDE layer, no drift to show) and is named accordingly.

Validation: final-phase max/mean per (agent, dataset, n_step) cell must match
icrl-resubmission/outputs/nstep_breadth_final.csv (tol 1e-3). Aborts on
mismatch.

Usage
-----
    python analysis/plot_nstep_mobility.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from scipy.stats import trim_mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_nstep_breadth import CELLS, load_cell  # noqa: E402

from gcrl_landscapes.evaluation.landscapes import mobility_plot  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "icrl-resubmission" / "plots"
REF_CSV = REPO_ROOT / "icrl-resubmission" / "outputs" / "nstep_breadth_final.csv"
HP_NAMES = ["hp.lr", "hp.alpha"]
MOBILITY_THRESHOLDS = [95, 90, 85, 80]

# (agent, dataset) -> output folder name, matching the existing plots/ layout.
DATASET_SHORT = {
    "antmaze-medium-navigate-v0": "antmaze-medium",
    "antmaze-large-navigate-v0": "antmaze-large",
    "cube-single-play-v0": "cube",
}


def clean_phases(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any phase without full 64-config coverage (stale/partial re-run
    artifacts left on disk, e.g. GCIQL n=3's old 43-config/615207 attempt
    coexisting with the fixed 64-config/820277 run in the same tree)."""
    covered = df.groupby("phase")["configuration"].nunique()
    good_phases = covered[covered == 64].index
    dropped = sorted(set(df["phase"]) - set(good_phases))
    if dropped:
        print(f"    dropping incomplete phases {dropped} (coverage: {covered.to_dict()})")
    return df[df["phase"].isin(good_phases)].copy()


def build_plot_df(df: pd.DataFrame, agent: str) -> pd.DataFrame:
    """Per-(config, phase) IQM frame in the shape mobility_plot expects."""
    df = clean_phases(df)
    per_cfg_phase = (
        df.groupby(["configuration", "phase"])
        .apply(
            lambda g: pd.Series(
                {
                    "success": trim_mean(g["success"], 0.25)
                    if len(g) >= 3
                    else g["success"].mean(),
                    "lr": g["lr"].iloc[0],
                    "alpha": g["alpha"].iloc[0],
                }
            )
        )
        .reset_index()
    )
    per_cfg_phase["hp.lr"] = per_cfg_phase["lr"]
    per_cfg_phase["hp.alpha"] = per_cfg_phase["alpha"]
    per_cfg_phase["hp.agent_name"] = agent
    per_cfg_phase["hp.actor_loss"] = "awr"
    per_cfg_phase["phase_num"] = per_cfg_phase["phase"].rank(method="dense").astype(int)
    per_cfg_phase["run_id"] = range(len(per_cfg_phase))
    return per_cfg_phase


def validate(agent: str, dataset: str, n_step: int, df: pd.DataFrame, ref: pd.DataFrame) -> None:
    row = ref[
        (ref["agent"] == agent) & (ref["dataset"] == dataset) & (ref["n_step"] == n_step)
    ]
    if row.empty:
        raise SystemExit(f"{agent} {dataset} n={n_step}: no reference row in {REF_CSV}")
    row = row.iloc[0]
    deepest = df[df["phase"] == df["phase"].max()]
    per_cfg = (
        deepest.groupby("configuration")["success"]
        .apply(lambda v: trim_mean(v, 0.25) if len(v) >= 3 else v.mean())
    )
    got_max, got_mean = float(per_cfg.max()), float(per_cfg.mean())
    if abs(got_max - row["max"]) > 1e-3 or abs(got_mean - row["mean"]) > 1e-3:
        raise SystemExit(
            f"{agent} {dataset} n={n_step}: max/mean {got_max:.4f}/{got_mean:.4f} "
            f"!= reference {row['max']}/{row['mean']} — pipeline mismatch"
        )
    print(f"  {agent} {dataset} n={n_step}: max/mean {got_max:.4f}/{got_mean:.4f} matches reference OK")


def main() -> None:
    ref = pd.read_csv(REF_CSV)

    for agent, dataset, n_step, expdir in CELLS:
        if not expdir.is_dir():
            print(f"MISSING tree: {expdir}")
            continue
        raw = load_cell(expdir)
        if raw is None:
            print(f"NO RESULTS: {agent} {dataset} n={n_step}")
            continue
        validate(agent, dataset, n_step, raw, ref)

        plot_df = build_plot_df(raw, agent)
        mob_df = plot_df.rename(columns={"success": "mean_normalized_goal_distance_return"})
        n_phases = mob_df["phase_num"].nunique()
        degenerate = n_phases < 2

        out_dir = OUT_ROOT / f"{agent.lower()}-nstep-{DATASET_SHORT[dataset]}" / f"n{n_step}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for pct in MOBILITY_THRESHOLDS:
            suffix = "-SINGLE-PHASE-DEGENERATE" if degenerate else ""
            mobility_plot(
                mob_df,
                agent,
                HP_NAMES,
                out_dir / f"mobility-{pct}{suffix}.png",
                by_col="phase_num",
                performance_threshold=pct / 100,
            )
        tag = " (single-phase, degenerate)" if degenerate else f" ({n_phases} phases)"
        print(f"  wrote {agent} {dataset} n={n_step} mobility at {MOBILITY_THRESHOLDS}{tag} -> {out_dir}")


if __name__ == "__main__":
    main()
