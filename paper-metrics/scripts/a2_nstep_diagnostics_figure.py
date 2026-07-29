"""A2 figure: do FR-AUC / MRR track a value-side intervention (n-step returns)?

Two panels, both answering 7VBe Q5 directly:

  left   diagnostics and success against the TD horizon, per environment —
         does the value-quality signal move in the same direction as success?
  right  the paired per-configuration change, Delta FR-AUC against
         Delta success between n=1 and n in {3, 5} — is the relationship there
         configuration by configuration, or only in the aggregate?

Style matches rl9_new_result_figures.py: Okabe-Ito palette, marker as secondary
encoding, no dual axes (the two quantities on the left panel are both plotted
on a shared 0-1 scale, which is where FR-AUC, MRR and success all live).

    input   PAPER/rebuttal/additional-exp/results/A2/A2_{summary,nstep_diagnostics}.csv
    output  PAPER/rebuttal/figures/A2_nstep_diagnostics.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REBUTTAL = Path(__file__).resolve().parents[2] / "PAPER" / "rebuttal"
RESULTS = REBUTTAL / "additional-exp" / "results" / "A2"
FIGURES = REBUTTAL / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

ENV_LABEL = {"cube": "Cube", "antmaze-large": "AntMaze-L"}
ENV_MARKER = {"cube": "^", "antmaze-large": "o"}
# Okabe-Ito, same assignments as the other rebuttal figures.
METRIC_COLOR = {
    "success": "#000000",
    "fr_auc": "#009E73",
    "mrr": "#0072B2",
}
METRIC_LABEL = {"success": "success", "fr_auc": "FR-AUC", "mrr": "MRR"}
HORIZON_COLOR = {3: "#E69F00", 5: "#D55E00"}


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        p = FIGURES / f"{name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  -> {p}")
    plt.close(fig)


def main() -> None:
    summary = pd.read_csv(RESULTS / "A2_summary.csv")
    per_config = pd.read_csv(RESULTS / "A2_nstep_diagnostics.csv")

    envs = [e for e in ENV_LABEL if e in set(summary["env"])]
    fig, axes = plt.subplots(
        len(envs), 2, figsize=(10.5, 3.4 * len(envs)), squeeze=False
    )

    for row, env in enumerate(envs):
        # ── left: level against horizon ──────────────────────────────────
        ax = axes[row][0]
        s = summary[summary["env"] == env].sort_values("n_step")
        for metric in ("success", "fr_auc", "mrr"):
            col = "mean_success" if metric == "success" else f"{metric}_mean"
            ax.plot(
                s["n_step"],
                s[col],
                marker=ENV_MARKER[env],
                color=METRIC_COLOR[metric],
                linestyle="-" if metric == "success" else "--",
                label=METRIC_LABEL[metric],
            )
        ax.set_xticks(sorted(s["n_step"].unique()))
        ax.set_xlabel("TD horizon $n$")
        ax.set_ylabel("mean over 64 configurations")
        ax.set_ylim(0, 1)
        ax.set_title(f"{ENV_LABEL[env]} — level vs horizon")
        ax.grid(alpha=0.25, linewidth=0.6)
        if row == 0:
            ax.legend(frameon=False, fontsize=9, loc="upper left")

        # ── right: paired per-configuration change ───────────────────────
        ax = axes[row][1]
        base = per_config[
            (per_config["env"] == env) & (per_config["n_step"] == 1)
        ].set_index("configuration")
        for n in (3, 5):
            arm = per_config[
                (per_config["env"] == env) & (per_config["n_step"] == n)
            ].set_index("configuration")
            common = base.index.intersection(arm.index)
            if not len(common):
                continue
            d_frauc = arm.loc[common, "fr_auc"] - base.loc[common, "fr_auc"]
            d_success = arm.loc[common, "success"] - base.loc[common, "success"]
            ax.scatter(
                d_frauc,
                d_success,
                s=26,
                alpha=0.75,
                color=HORIZON_COLOR[n],
                marker=ENV_MARKER[env],
                edgecolor="none",
                label=f"$n$={n}",
            )
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.axvline(0, color="0.4", linewidth=0.8)
        ax.set_xlabel(r"$\Delta$ FR-AUC vs $n{=}1$")
        ax.set_ylabel(r"$\Delta$ success vs $n{=}1$")
        ax.set_title(f"{ENV_LABEL[env]} — paired change per configuration")
        ax.grid(alpha=0.25, linewidth=0.6)
        if row == 0:
            ax.legend(frameon=False, fontsize=9, loc="lower right")

        # The quadrant counts are the point of the panel: how many
        # configurations improve on the diagnostic while losing success.
        note = []
        for n in (3, 5):
            arm = per_config[
                (per_config["env"] == env) & (per_config["n_step"] == n)
            ].set_index("configuration")
            common = base.index.intersection(arm.index)
            if not len(common):
                continue
            up_down = int(
                np.sum(
                    (arm.loc[common, "fr_auc"] > base.loc[common, "fr_auc"])
                    & (arm.loc[common, "success"] < base.loc[common, "success"])
                )
            )
            note.append(f"$n$={n}: {up_down}/{len(common)}")
        ax.text(
            0.02,
            0.98,
            "FR-AUC up, success down\n" + "   ".join(note),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="0.25",
        )

    fig.suptitle(
        "GCIVL, 64 configurations x 3 seeds (trimmed centre), final phase — "
        "diagnostics respond to n-step; success does not follow",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "A2_nstep_diagnostics")


if __name__ == "__main__":
    main()
