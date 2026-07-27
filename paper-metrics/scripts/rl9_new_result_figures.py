"""Figures for the cluster rebuttal results R1 (re-extraction), R2
(consistency probe), and R3 (multi-step TD), in the rliable palette
(seaborn-colorblind) shared with the E-series figures.

Inputs default to <repo>/PAPER/rebuttal/results/ (override with RESULTS_DIR);
outputs land in <repo>/PAPER/rebuttal/figures/.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("RESULTS_DIR", REPO / "PAPER" / "rebuttal" / "results"))
FIGURES = RESULTS.parent / "figures"

AGENT_COLOR = {
    "CRL": "#0173b2",
    "GCIQL": "#de8f05",
    "GCIVL": "#029e73",
    "QRL": "#d55e00",
}
GREY = "#949494"
ENV_LABEL = {
    "antmaze-medium-navigate-v0": "AntMaze-M",
    "antmaze-large-navigate-v0": "AntMaze-L",
    "cube-single-play-v0": "Cube",
}


def _save(fig, name):
    for ext in ("png", "pdf"):
        p = FIGURES / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  → {p}")
    plt.close(fig)


def load_r1() -> pd.DataFrame:
    parts = [
        pd.read_csv(f) for f in sorted((RESULTS / "R1_reextract_raw").glob("*.csv"))
    ]
    d = pd.concat(parts, ignore_index=True)
    return (
        d.groupby(["agent", "dataset", "extractor", "configuration", "lr", "alpha"])[
            "success"
        ]
        .mean()
        .reset_index()
    )


def fig_r1_paired() -> None:
    d = load_r1()
    cells = [
        (e, a)
        for e in ["antmaze-medium-navigate-v0", "cube-single-play-v0"]
        for a in ["CRL", "GCIQL"]
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.6))
    for ax, (env, agent) in zip(axes, cells):
        sub = d[(d["dataset"] == env) & (d["agent"] == agent)]
        p = sub.pivot_table(
            index="configuration", columns="extractor", values="success"
        )
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="0.6")
        ax.scatter(
            p["awr"],
            p["rejection"],
            s=34,
            color=AGENT_COLOR[agent],
            alpha=0.85,
            edgecolors="none",
        )
        r = spearmanr(p["awr"], p["rejection"]).statistic
        ax.set_title(f"{ENV_LABEL[env]}  {agent}", fontsize=11)
        ax.text(0.05, 0.92, f"Spearman {r:.2f}", transform=ax.transAxes, fontsize=9)
        ax.set_xlabel("success, AWR", fontsize=9)
        ax.set_xlim(-0.03, 1.0)
        ax.set_ylim(-0.03, 1.0)
        ax.grid(True, alpha=0.15, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("success, rejection sampling", fontsize=9)
    fig.suptitle(
        "Per-configuration success under the two extractors (same frozen checkpoints)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "R1_reextract_paired")


def fig_r1_planes() -> None:
    d = load_r1()
    cells = [
        (e, a)
        for e in ["antmaze-medium-navigate-v0", "cube-single-play-v0"]
        for a in ["CRL", "GCIQL"]
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.4), sharex="col", sharey=True)
    for j, (env, agent) in enumerate(cells):
        for i, ext in enumerate(["awr", "rejection"]):
            ax = axes[i, j]
            sub = d[
                (d["dataset"] == env) & (d["agent"] == agent) & (d["extractor"] == ext)
            ]
            sc = ax.scatter(
                np.log10(sub["lr"]),
                sub["alpha"],
                c=sub["success"],
                cmap="viridis",
                vmin=0,
                vmax=1,
                s=42,
                edgecolors="none",
            )
            if i == 0:
                ax.set_title(f"{ENV_LABEL[env]}  {agent}", fontsize=11)
            if j == 0:
                ax.set_ylabel(
                    f"{'AWR' if ext == 'awr' else 'rejection'}\nalpha", fontsize=9
                )
            if i == 1:
                ax.set_xlabel("log10 lr", fontsize=9)
            ax.grid(True, alpha=0.15, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02, label="success")
    fig.suptitle(
        "The sampled (lr, alpha) plane under each extractor", fontsize=12, y=0.98
    )
    _save(fig, "R1_reextract_planes")


def fig_r2() -> None:
    d = pd.read_csv(RESULTS / "R2_consistency_per_config.csv")
    s = pd.read_csv(RESULTS / "R2_consistency_summary.csv")
    envs = ["antmaze-medium-navigate-v0", "cube-single-play-v0"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    for ax, env in zip(axes, envs):
        for agent in ["CRL", "GCIQL", "GCIVL", "QRL"]:
            sub = d[(d["dataset"] == env) & (d["agent"] == agent)]
            row = s[(s["dataset"] == env) & (s["agent"] == agent)].iloc[0]
            invalid = (
                bool(row["degenerate"])
                or bool(row["probe_invalid"])
                or (agent == "CRL" and env == "cube-single-play-v0")
            )
            color = GREY if invalid else AGENT_COLOR[agent]
            label = agent + (" (probe not applicable)" if invalid else "")
            ax.scatter(
                sub["violation_rate"],
                sub["success"],
                s=26,
                color=color,
                alpha=0.4 if invalid else 0.85,
                edgecolors="none",
                label=label,
            )
            if not invalid and np.isfinite(row["spearman_viol_success"]):
                ax.annotate(
                    f"{agent}: ρ = {row['spearman_viol_success']:.2f}",
                    xy=(
                        0.98,
                        0.08 + 0.075 * ["GCIQL", "GCIVL"].index(agent)
                        if agent in ("GCIQL", "GCIVL")
                        else 0.25,
                    ),
                    xycoords="axes fraction",
                    ha="right",
                    fontsize=8.5,
                    color=AGENT_COLOR[agent],
                )
        ax.set_title(ENV_LABEL[env], fontsize=11)
        ax.set_xlabel("triangle-inequality violation rate", fontsize=9)
        ax.grid(True, alpha=0.15, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("final success", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle(
        "Value-consistency probe: violation rate against success", fontsize=12, y=1.0
    )
    fig.tight_layout()
    _save(fig, "R2_consistency")


def fig_r3() -> None:
    b = pd.read_csv(RESULTS / "R3_nstep_breadth.csv")
    b = b[b["agent"] == "GCIVL"]
    envs = ["cube-single-play-v0", "antmaze-large-navigate-v0"]
    metrics = [("max", "peak"), ("mean", "mean"), ("rho_abs_0.25", "rho_abs(0.25)")]
    colors = ["#0173b2", "#de8f05", "#029e73"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, env in zip(axes, envs):
        sub = b[b["dataset"] == env].sort_values("n_step")
        for (col, lab), c in zip(metrics, colors):
            ax.plot(
                sub["n_step"], sub[col], marker="o", ms=5, lw=1.8, color=c, label=lab
            )
        ax.set_title(ENV_LABEL[env] + "  (GCIVL)", fontsize=11)
        ax.set_xlabel("TD horizon n", fontsize=9)
        ax.set_xticks([1, 3, 5])
        ax.set_ylim(-0.03, 1.0)
        ax.grid(True, alpha=0.15, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("value", fontsize=9)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Multi-step TD: the peak moves, absolute breadth does not follow",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "R3_nstep")


def fig_r3_planes() -> None:
    d = pd.read_csv(RESULTS / "R3_nstep_per_config.csv")
    d = d[d["agent"] == "GCIVL"]
    envs = ["cube-single-play-v0", "antmaze-large-navigate-v0"]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6), sharex=True, sharey=True)
    for i, env in enumerate(envs):
        for j, n in enumerate([1, 3, 5]):
            ax = axes[i, j]
            sub = d[(d["dataset"] == env) & (d["n_step"] == n)]
            per = (
                sub.groupby(["configuration", "lr", "alpha"])["success"]
                .mean()
                .reset_index()
            )
            sc = ax.scatter(
                np.log10(per["lr"]),
                per["alpha"],
                c=per["success"],
                cmap="viridis",
                vmin=0,
                vmax=1,
                s=42,
                edgecolors="none",
            )
            if i == 0:
                ax.set_title(f"n = {n}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{ENV_LABEL[env]}\nalpha", fontsize=9)
            if i == 1:
                ax.set_xlabel("log10 lr", fontsize=9)
            ax.grid(True, alpha=0.15, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02, label="success")
    fig.suptitle(
        "GCIVL landscapes in the (lr, alpha) plane at each TD horizon",
        fontsize=12,
        y=0.98,
    )
    _save(fig, "R3_nstep_planes")


def main() -> None:
    fig_r1_paired()
    fig_r1_planes()
    fig_r2()
    fig_r3()
    fig_r3_planes()


if __name__ == "__main__":
    main()
