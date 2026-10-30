#!/usr/bin/env python3
"""A3: DDPG+BC vs. AWR (published) landscape plots on cube-single-play-v0.

Camera-ready item #2. For CRL, GCIQL and QRL, renders the (lr, alpha) success
landscape of the DDPG+BC extractor (`SSRL-Landscapes/logs_ddpgbc/*_cube-*`,
64 cfg x 5 seeds — the source of the numbers posted to OpenReview on Jul-30)
next to the published AWR campaign (`logs-advantage-cube-single-fixed-batch`).
Each heatmap is paired with a mobility plot: a two-layer KDE (AWR -> DDPG+BC)
showing how the near-optimal region moves under the extractor swap.

The two grids are INDEPENDENT Sobol draws (verified: configuration i is a
different point in each), so — per the compare_ddpgbc.py convention — this
script emits NO Spearman and no paired ranking. Breadth class + levels only.
The comment-numbering caveat: the navigation sweep additionally used unequal
actor budgets; here every cell gets its per-agent budget verdict printed.

Budget verdicts follow compare_ddpgbc.py: clean if final-phase steps match
within 10%; otherwise conservative when the confound works against the
observed direction, confounded when the gain cannot be separated from extra
training (CRL: do not quote its DDPG+BC gain as an extractor effect).

Validation: DDPG+BC final-phase peak/mean must reproduce the posted numbers
(GCIQL 0.920/0.493, CRL 0.353/0.129, QRL 0.120/0.040 — FINDINGS.md, 7VBe
thread Jul-30). The script aborts on mismatch.

Usage
-----
    python analysis/plot_a3_ddpgbc_cube.py --out icrl-resubmission/plots/a3-ddpgbc-cube
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_a1_landscapes import (  # noqa: E402
    HP_NAMES,
    MOBILITY_THRESHOLDS,
    PUBLISHED_ROOT,
    load_hps,
    load_phased,
    seed_reduce,
    build_plot_df,
)

from gcrl_landscapes.evaluation.plot import plot_landscape  # noqa: E402
from gcrl_landscapes.evaluation.landscapes import mobility_plot  # noqa: E402
from gcrl_landscapes.util.eval import fit_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DDPGBC_ROOT = PUBLISHED_ROOT / "logs_ddpgbc"
AWR_ROOT = PUBLISHED_ROOT / "logs-advantage-cube-single-fixed-batch"

# (agent, ddpgbc final step, awr final step, posted ddpgbc peak, posted ddpgbc mean)
CELLS = [
    ("GCIQL", 819999, 949583, 0.920, 0.493),
    ("CRL", 995000, 315517, 0.353, 0.129),
    ("QRL", 900000, 809000, 0.120, 0.040),
]

# Posted values are rounded to 3 decimals in FINDINGS.md.
POSTED_TOL = 1e-3


def budget_verdict(ratio: float, ddpgbc_peak: float, awr_peak: float) -> str:
    if 0.9 <= ratio <= 1.1:
        return "clean"
    if ddpgbc_peak > awr_peak:
        return "conservative" if ratio < 1 else "confounded"
    return "conservative" if ratio > 1 else "confounded"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "icrl-resubmission/plots/a3-ddpgbc-cube"
    )
    args = ap.parse_args()

    for agent, ddbc_step, awr_step, posted_peak, posted_mean in CELLS:
        print(f"=== {agent} ===")
        ddbc_dir = DDPGBC_ROOT / f"{agent}_cube-single-play-v0_64c_lr-alpha"
        awr_dir = AWR_ROOT / f"{agent}_cube-single-play-v0_64c_lr-alpha"

        frames = {}
        for tag, expdir, step in (("ddpgbc", ddbc_dir, ddbc_step), ("awr", awr_dir, awr_step)):
            df = load_phased(expdir, [step], max_seeds=5)  # cube campaigns are 5-seed native
            covered = df.groupby("step")["configuration"].nunique()
            seeds = df.groupby("step")["seed"].nunique()
            if not (covered == 64).all() or not (seeds == 5).all():
                raise SystemExit(
                    f"{agent} {tag}: coverage {covered.to_dict()}, seeds {seeds.to_dict()}"
                )
            frames[tag] = (df, load_hps(expdir))

        iqm_ddbc = seed_reduce(frames["ddpgbc"][0]).set_index("configuration")["iqm"]
        iqm_awr = seed_reduce(frames["awr"][0]).set_index("configuration")["iqm"]

        # Validate against the posted numbers.
        if abs(iqm_ddbc.max() - posted_peak) > POSTED_TOL or abs(iqm_ddbc.mean() - posted_mean) > POSTED_TOL:
            raise SystemExit(
                f"{agent}: ddpgbc peak/mean {iqm_ddbc.max():.4f}/{iqm_ddbc.mean():.4f} "
                f"!= posted {posted_peak}/{posted_mean} — pipeline mismatch"
            )
        ratio = ddbc_step / awr_step
        verdict = budget_verdict(ratio, iqm_ddbc.max(), iqm_awr.max())
        print(
            f"  ddpgbc peak/mean {iqm_ddbc.max():.3f}/{iqm_ddbc.mean():.3f} matches posted "
            f"{posted_peak}/{posted_mean}; awr peak/mean {iqm_awr.max():.3f}/{iqm_awr.mean():.3f}; "
            f"budget ratio {ratio:.2f} -> {verdict}"
        )

        # Plots: heatmap per method + two-layer mobility (awr -> ddpgbc).
        for tag in ("awr", "ddpgbc"):
            raw, hp = frames[tag]
            df = build_plot_df(seed_reduce(raw), hp, agent)
            out_dir = args.out / agent / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            model = fit_model(df, "success", HP_NAMES)
            plot_landscape(
                ddbc_step if tag == "ddpgbc" else awr_step,
                model,
                "success",
                HP_NAMES,
                agent,
                out_dir,
                "Success Rate",
                actor_loss="ddpgbc" if tag == "ddpgbc" else "awr",
            )
            frames[tag] = (raw, hp, df)

        combined = pd.concat([frames["awr"][2], frames["ddpgbc"][2]])
        combined["method"] = ["awr"] * len(frames["awr"][2]) + ["ddpgbc"] * len(frames["ddpgbc"][2])
        mob_df = combined.rename(columns={"success": "mean_normalized_goal_distance_return"})
        for pct in MOBILITY_THRESHOLDS:
            mobility_plot(
                mob_df,
                agent,
                HP_NAMES,
                args.out / agent / f"mobility-{pct}-awr-vs-ddpgbc-{agent}.png",
                by_col="method",
                performance_threshold=pct / 100,
            )
        print(f"  wrote awr+ddpgbc heatmaps + mobility -> {args.out / agent}")


if __name__ == "__main__":
    main()
