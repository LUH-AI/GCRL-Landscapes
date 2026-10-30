#!/usr/bin/env python3
"""A1: straight-through vs checkpoint-conditioned (phased) landscape plots.

Camera-ready item #7. For GCIQL and QRL on antmaze-medium-navigate-v0, plots
the (lr, alpha) success landscape for the straight-through protocol
(`logs-additional-exp/a1-straight-*`, one uninterrupted run per configuration,
evals logged as extra steps inside a single phase) next to the published
checkpoint-conditioned (phased) protocol (fixed-batch campaign).

Phase 1 and final phase are plotted SEPARATELY on purpose: phase-1 agreement
between protocols is the strongest number in the campaign (Spearman +0.94
GCIQL / +0.98 QRL), while the final-phase landscape identities diverge
(-0.221 / +0.511). Per A1_breadth_and_regret.md, quote breadth classes +
Spearman, never absolute levels across protocols (shared-checkpoint design
inflates the phased arm's absolute success mass).

Validation: the per-config trimmed-mean (3 seeds) values are checked against
the coauthor's aggregates before plotting — final-step values against
`A1_straight_vs_phased_per_config.csv` and per-step Spearman against
`A1_phase_alignment.csv`. The script aborts on mismatch.

Usage
-----
    python analysis/plot_a1_landscapes.py --out icrl-resubmission/plots/a1-straight-vs-phased
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, trim_mean

from gcrl_landscapes.evaluation.plot import plot_landscape
from gcrl_landscapes.evaluation.landscapes import mobility_plot
from gcrl_landscapes.util.eval import fit_model

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLISHED_ROOT = REPO_ROOT / "cluster-mount" / "SSRL-Landscapes"
ADDITIONAL_ROOT = REPO_ROOT / "cluster-mount" / "GCRL-Landscapes" / "logs-additional-exp"
CAMERA_READY = REPO_ROOT / "cluster-mount" / "GCRL-Landscapes" / "camera-ready"

HP_NAMES = ["hp.lr", "hp.alpha"]
MOBILITY_THRESHOLDS = [95, 90, 85, 80]

# (agent, straight tree, phased tree, eval steps [phase1, ..., final])
# Steps from info.toml / A1_phase_alignment.csv.
CELLS = [
    (
        "GCIQL",
        ADDITIONAL_ROOT
        / "a1-straight-gciql-antmaze-medium"
        / "GCIQL_antmaze-medium-navigate-v0_64c_lr-alpha",
        PUBLISHED_ROOT
        / "logs-advantage-antmaze-medium-fixed-batch"
        / "GCIQL_antmaze-medium-navigate-v0_64c_lr-alpha",
        [205069, 410138, 615208, 820277],
    ),
    (
        "QRL",
        ADDITIONAL_ROOT
        / "a1-straight-qrl-antmaze-medium"
        / "QRL_antmaze-medium-navigate-v0_64c_lr-alpha",
        PUBLISHED_ROOT
        / "logs-advantage-antmaze-medium-fixed-batch"
        / "QRL_antmaze-medium-navigate-v0_64c_lr-alpha",
        [77687, 155375, 233062, 310750],
    ),
]

# Spearman(phased, straight) per step, from the coauthor's A1_phase_alignment.csv.
EXPECTED_SPEARMAN = {
    "GCIQL": [0.94, -0.2378, -0.2052, -0.2212],
    "QRL": [0.9776, 0.3586, 0.4411, 0.5111],
}


def load_phased(expdir: Path, steps: list[int], max_seeds: int = 3) -> pd.DataFrame:
    """Per (config, seed, phase-step) success from a phased tree (last row per eval_log).

    The published fixed-batch campaign has 5 seeds; A1 compares at the straight
    arms' 3 seeds, so cap seeds to match (cf. compare_nstep_breadth --match-seeds).
    """
    rows = []
    for f in expdir.glob("run_logs/configuration_*/phase_*/seed_*/eval_log.csv"):
        m = re.search(r"configuration_(\d+)/phase_(\d+)/seed_(\d+)/eval_log\.csv", str(f))
        if not m:
            continue
        cfg, phase, seed = (int(g) for g in m.groups())
        if phase not in steps or seed >= max_seeds:
            continue
        d = pd.read_csv(f)
        rows.append(
            {"configuration": cfg, "step": phase, "seed": seed, "success": float(d["success"].iloc[-1])}
        )
    return pd.DataFrame(rows)


def load_straight(expdir: Path, steps: list[int]) -> pd.DataFrame:
    """Per (config, seed, eval-step) success from a straight tree.

    All evals live in the single final-phase directory as (success, step) rows.
    """
    rows = []
    for f in expdir.glob("run_logs/configuration_*/phase_*/seed_*/eval_log.csv"):
        m = re.search(r"configuration_(\d+)/phase_(\d+)/seed_(\d+)/eval_log\.csv", str(f))
        if not m:
            continue
        cfg, _phase, seed = (int(g) for g in m.groups())
        d = pd.read_csv(f)
        for step in steps:
            hit = d[d["step"] == step]
            if hit.empty:
                raise ValueError(f"step {step} missing in {f}")
            rows.append(
                {"configuration": cfg, "step": step, "seed": seed, "success": float(hit["success"].iloc[-1])}
            )
    return pd.DataFrame(rows)


def load_hps(expdir: Path) -> dict[int, tuple[float, float]]:
    hp = {}
    for cf in (expdir / "configurations").glob("configuration_*.json"):
        c = json.loads(cf.read_text())
        hp[int(cf.stem.split("_")[1])] = (c.get("lr"), c.get("alpha"))
    return hp


def seed_reduce(df: pd.DataFrame) -> pd.DataFrame:
    """trim_mean(v, 0.25) per (configuration, step) — the A1 convention (3 seeds)."""
    return (
        df.groupby(["configuration", "step"])["success"]
        .apply(lambda v: trim_mean(v, 0.25) if len(v) >= 3 else v.mean())
        .rename("iqm")
        .reset_index()
    )


def validate(agent: str, straight: pd.DataFrame, phased: pd.DataFrame, steps: list[int]) -> None:
    """Check against the coauthor's aggregates; abort on mismatch."""
    # (1) per-step Spearman vs A1_phase_alignment.csv
    s = seed_reduce(straight).set_index(["configuration", "step"])["iqm"]
    p = seed_reduce(phased).set_index(["configuration", "step"])["iqm"]
    for i, step in enumerate(steps):
        rho = spearmanr(p.xs(step, level="step"), s.xs(step, level="step")).statistic
        exp = EXPECTED_SPEARMAN[agent][i]
        if abs(rho - exp) > 0.005:
            raise SystemExit(
                f"{agent} step {step}: Spearman {rho:.4f} != expected {exp} — pipeline mismatch"
            )
        print(f"  {agent} step {step}: Spearman {rho:.4f} (expected {exp}) OK")

    # (2) final-step per-config values vs A1_straight_vs_phased_per_config.csv
    ref = pd.read_csv(CAMERA_READY / "results/additional-exp-A/A1/A1_straight_vs_phased_per_config.csv")
    ref = ref[ref["agent"] == agent].set_index("configuration")
    final = steps[-1]
    for cfg in ref.index:
        for col, src in (("success_straight", s), ("success_phased", p)):
            got = src.loc[(cfg, final)]
            want = ref.loc[cfg, col]
            if abs(got - want) > 1e-6:
                raise SystemExit(
                    f"{agent} cfg {cfg} {col}: {got} != {want} — pipeline mismatch"
                )
    print(f"  {agent} final-step per-config values match A1_straight_vs_phased_per_config.csv (64/64)")


def build_plot_df(iqm: pd.DataFrame, hp: dict[int, tuple[float, float]], agent: str) -> pd.DataFrame:
    """Per-config frame in the shape fit_model / mobility_plot expect."""
    df = iqm.copy()
    df["hp.lr"] = df["configuration"].map(lambda i: hp[i][0])
    df["hp.alpha"] = df["configuration"].map(lambda i: hp[i][1])
    df["hp.agent_name"] = agent
    df["hp.actor_loss"] = "awr"
    df["run_id"] = range(len(df))
    # Renumber steps 1..4 for by_col layers (mobility) / plot titles.
    df["phase_num"] = df["step"].rank(method="dense").astype(int)
    df = df.rename(columns={"iqm": "success"})
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "icrl-resubmission/plots/a1-straight-vs-phased"
    )
    args = ap.parse_args()

    for agent, straight_dir, phased_dir, steps in CELLS:
        print(f"=== {agent} ===")
        hp = load_hps(straight_dir)
        straight = load_straight(straight_dir, steps)
        phased = load_phased(phased_dir, steps)
        for name, df in (("straight", straight), ("phased", phased)):
            covered = df.groupby("step")["configuration"].nunique()
            seeds = df.groupby("step")["seed"].nunique()
            if not (covered == 64).all() or not (seeds == 3).all():
                raise SystemExit(
                    f"{agent} {name}: coverage {covered.to_dict()}, seeds {seeds.to_dict()}"
                )
        validate(agent, straight, phased, steps)

        straight_df = build_plot_df(seed_reduce(straight), hp, agent)
        phased_df = build_plot_df(seed_reduce(phased), hp, agent)

        for protocol, df in (("straight", straight_df), ("phased", phased_df)):
            out_dir = args.out / agent / protocol
            out_dir.mkdir(parents=True, exist_ok=True)
            # Landscape heatmaps, phase 1 and final phase separately.
            for step, phase_num in ((steps[0], 1), (steps[-1], df["phase_num"].max())):
                sub = df[df["phase_num"] == phase_num].reset_index(drop=True)
                model = fit_model(sub, "success", HP_NAMES)
                plot_landscape(
                    step,
                    model,
                    "success",
                    HP_NAMES,
                    agent,
                    out_dir,
                    "Success Rate",
                    actor_loss=sub["hp.actor_loss"].iloc[0],
                )
            # Mobility across all four eval steps (near-optimal-region drift),
            # at every threshold in MOBILITY_THRESHOLDS.
            mob_df = df.rename(columns={"success": "mean_normalized_goal_distance_return"})
            for pct in MOBILITY_THRESHOLDS:
                mobility_plot(
                    mob_df,
                    agent,
                    HP_NAMES,
                    out_dir / f"mobility-{pct}-4step-{agent}-{protocol}.png",
                    performance_threshold=pct / 100,
                )
            print(f"  wrote {protocol} landscapes (phase 1 + final) + mobility -> {out_dir}")


if __name__ == "__main__":
    main()
