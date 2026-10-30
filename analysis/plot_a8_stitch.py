#!/usr/bin/env python3
"""A8: stitch vs. navigate dataset landscapes, all four agents, AntMaze-M.

Camera-ready item #8. Each agent is trained on antmaze-medium-stitch-v0 at its
own navigate budget (`logs-additional-exp/a8-stitch-*`, single-phase trees
whose eval_log.csv holds the final-budget eval) and compared against the same
configuration grid trained on the published antmaze-medium-navigate dataset.
Per the convention, each heatmap is paired with a mobility plot: here a
two-layer KDE (navigate vs. stitch) showing how the near-optimal region moves
under the dataset swap.

FINDINGS.md framing to preserve in the write-up: GCIVL (+0.587) and QRL
(+0.439) CIs exclude zero; CRL (+0.270) spans zero (lower bound -0.005);
GCIQL (-0.117) is the lone null, leaning slightly negative. Every agent loses
roughly a third of its peak on stitch.

Validation: per-config seed-reduced success must reproduce
`A8_stitch_vs_navigate_per_config.csv` exactly, Spearman must match
`A8_summary.csv`, and max/mean must match the summary. The script aborts on
mismatch.

Usage
-----
    python analysis/plot_a8_stitch.py --out icrl-resubmission/plots/a8-stitch-vs-navigate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_a1_landscapes import (  # noqa: E402
    HP_NAMES,
    MOBILITY_THRESHOLDS,
    PUBLISHED_ROOT,
    ADDITIONAL_ROOT,
    CAMERA_READY,
    load_hps,
    load_phased,
    load_straight,
    seed_reduce,
    build_plot_df,
)

from gcrl_landscapes.evaluation.plot import plot_landscape  # noqa: E402
from gcrl_landscapes.evaluation.landscapes import mobility_plot  # noqa: E402
from gcrl_landscapes.util.eval import fit_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# (agent, stitch tree, navigate tree, final budget step)
CELLS = [
    (
        "GCIQL",
        ADDITIONAL_ROOT / "a8-stitch-gciql-antmaze-medium" / "GCIQL_antmaze-medium-stitch-v0_64c_lr-alpha",
        PUBLISHED_ROOT / "logs-advantage-antmaze-medium-fixed-batch" / "GCIQL_antmaze-medium-navigate-v0_64c_lr-alpha",
        820277,
    ),
    (
        "QRL",
        ADDITIONAL_ROOT / "a8-stitch-qrl-antmaze-medium" / "QRL_antmaze-medium-stitch-v0_64c_lr-alpha",
        PUBLISHED_ROOT / "logs-advantage-antmaze-medium-fixed-batch" / "QRL_antmaze-medium-navigate-v0_64c_lr-alpha",
        310750,
    ),
    (
        "GCIVL",
        ADDITIONAL_ROOT / "a8-stitch-gcivl-antmaze-medium" / "GCIVL_antmaze-medium-stitch-v0_64c_lr-alpha",
        PUBLISHED_ROOT / "logs-advantage-antmaze-medium-fixed-batch" / "GCIVL_antmaze-medium-navigate-v0_64c_lr-alpha",
        865793,
    ),
    (
        "CRL",
        ADDITIONAL_ROOT / "a8-stitch-crl-antmaze-medium" / "CRL_antmaze-medium-stitch-v0_64c_lr-alpha",
        PUBLISHED_ROOT / "logs-advantage-antmaze-medium-fixed-batch" / "CRL_antmaze-medium-navigate-v0_64c_lr-alpha",
        159090,
    ),
]


def validate(agent: str, stitch: pd.DataFrame, nav: pd.DataFrame, step: int) -> tuple[float, float, float]:
    """Check against the coauthor's A8 aggregates; abort on mismatch.

    Returns (spearman, stitch_max, stitch_mean) for reporting.
    """
    summary = pd.read_csv(CAMERA_READY / "results/additional-exp-A/A8/A8_summary.csv")
    row = summary[summary["agent"] == agent].iloc[0]

    s = seed_reduce(stitch).set_index("configuration")["iqm"]
    n = seed_reduce(nav).set_index("configuration")["iqm"]
    rho = spearmanr(n.sort_index(), s.sort_index()).statistic

    checks = [
        ("spearman", rho, row["spearman"], 5e-4),
        ("navigate_max", n.max(), row["navigate_max"], 1e-3),
        ("navigate_mean", n.mean(), row["navigate_mean"], 1e-3),
        ("stitch_max", s.max(), row["stitch_max"], 1e-3),
        ("stitch_mean", s.mean(), row["stitch_mean"], 1e-3),
    ]
    for name, got, want, tol in checks:
        if abs(got - want) > tol:
            raise SystemExit(f"{agent} {name}: {got} != {want} — pipeline mismatch")
    print(
        f"  {agent}: spearman {rho:.4f} [{row['spearman_lo']:.4f}, {row['spearman_hi']:.4f}]"
        f", navigate {n.max():.3f}/{n.mean():.3f}, stitch {s.max():.3f}/{s.mean():.3f} — OK"
    )

    # Per-config values vs the coauthor's per-config CSV.
    ref = pd.read_csv(CAMERA_READY / "results/additional-exp-A/A8/A8_stitch_vs_navigate_per_config.csv")
    ref = ref[ref["agent"] == agent].set_index("configuration")
    for cfg in ref.index:
        for col, src in (("success_navigate", n), ("success_stitch", s)):
            if abs(src.loc[cfg] - ref.loc[cfg, col]) > 1e-6:
                raise SystemExit(
                    f"{agent} cfg {cfg} {col}: {src.loc[cfg]} != {ref.loc[cfg, col]} — mismatch"
                )
    print(f"  {agent} per-config values match A8_stitch_vs_navigate_per_config.csv (64/64)")
    return rho, row["spearman_lo"], row["spearman_hi"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "icrl-resubmission/plots/a8-stitch-vs-navigate"
    )
    args = ap.parse_args()

    for agent, stitch_dir, nav_dir, step in CELLS:
        print(f"=== {agent} ===")
        hp = load_hps(stitch_dir)
        stitch = load_straight(stitch_dir, [step])
        nav = load_phased(nav_dir, [step])
        for name, df in (("stitch", stitch), ("navigate", nav)):
            covered = df.groupby("step")["configuration"].nunique()
            seeds = df.groupby("step")["seed"].nunique()
            if not (covered == 64).all() or not (seeds == 3).all():
                raise SystemExit(
                    f"{agent} {name}: coverage {covered.to_dict()}, seeds {seeds.to_dict()}"
                )
        rho, lo, hi = validate(agent, stitch, nav, step)

        nav_df = build_plot_df(seed_reduce(nav), hp, agent)
        stitch_df = build_plot_df(seed_reduce(stitch), hp, agent)

        for protocol, df in (("navigate", nav_df), ("stitch", stitch_df)):
            out_dir = args.out / agent / protocol
            out_dir.mkdir(parents=True, exist_ok=True)
            model = fit_model(df, "success", HP_NAMES)
            plot_landscape(
                step,
                model,
                "success",
                HP_NAMES,
                agent,
                out_dir,
                "Success Rate",
                actor_loss=df["hp.actor_loss"].iloc[0],
            )

        # Two-layer mobility: near-optimal region under the dataset swap,
        # at every threshold in MOBILITY_THRESHOLDS.
        combined = pd.concat([nav_df, stitch_df])
        combined["dataset"] = ["navigate"] * len(nav_df) + ["stitch"] * len(stitch_df)
        mob_df = combined.rename(columns={"success": "mean_normalized_goal_distance_return"})
        out_dir = args.out / agent
        for pct in MOBILITY_THRESHOLDS:
            mobility_plot(
                mob_df,
                agent,
                HP_NAMES,
                out_dir / f"mobility-{pct}-navigate-vs-stitch-{agent}.png",
                by_col="dataset",
                performance_threshold=pct / 100,
            )
        print(f"  wrote navigate+stitch heatmaps + mobility -> {out_dir}")


if __name__ == "__main__":
    main()
