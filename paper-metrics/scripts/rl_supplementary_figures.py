"""One figure per remaining rebuttal experiment (E1, E2, E4, E5, E6, E8).

E3 and E7 already have figures (rl3_selectivity_curves, rl7_diagnostic_scatter_*).
Style matches rl3: Okabe-Ito palette, fixed agent order, line style / marker as
secondary encoding, no dual axes.

Outputs: outputs/E{1,2,4,5,6,8}_*.png/.pdf
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import AGENTS, ENVS, OUTPUTS

AGENT_COLOR = {
    "CRL": "#0072B2",
    "GCIQL": "#E69F00",
    "GCIVL": "#009E73",
    "QRL": "#D55E00",
}
AGENT_MARKER = {"CRL": "o", "GCIQL": "s", "GCIVL": "^", "QRL": "D"}
AGENT_STYLE = {"CRL": "-", "GCIQL": "--", "GCIVL": "-.", "QRL": ":"}
ENV_LABEL = {
    "antmaze-medium": "AntMaze-M",
    "antmaze-large": "AntMaze-L",
    "cube": "Cube",
    "scene": "Scene",
}


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        p = OUTPUTS / f"{name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  → {p}")
    plt.close(fig)


def _parse_ci(s: str) -> tuple[float, float]:
    lo, hi = re.match(r"\[([-\d.]+), ([-\d.]+)\]", s).groups()
    return float(lo), float(hi)


def fig_e1() -> None:
    df = pd.read_csv(OUTPUTS / "rl1_abs_breadth.csv")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    width = 0.35
    for ax, env in zip(axes, ENVS):
        sub = df[df["env"] == env].set_index("agent").reindex(AGENTS)
        x = np.arange(len(AGENTS))
        for off, thr, shade in ((-width / 2, "0.25", 1.0), (width / 2, "0.5", 0.45)):
            vals = sub[f"rho_abs_{thr}"].values
            cis = [_parse_ci(s) for s in sub[f"rho_abs_{thr}_ci_cfg"]]
            err = np.array(
                [
                    [v - lo for v, (lo, _) in zip(vals, cis)],
                    [hi - v for v, (_, hi) in zip(vals, cis)],
                ]
            )
            colors = [AGENT_COLOR[a] for a in AGENTS]
            ax.bar(
                x + off,
                vals,
                width,
                yerr=err,
                capsize=2,
                color=colors,
                alpha=shade,
                edgecolor="none",
                error_kw=dict(lw=1, ecolor="0.3"),
                label=f"threshold {thr}" if env == ENVS[0] else None,
            )
        ax.set_xticks(x, AGENTS, fontsize=8)
        ax.set_title(ENV_LABEL[env], fontsize=11)
        ax.grid(True, axis="y", alpha=0.15, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel(r"$\rho^{\mathrm{abs}}$ (fraction of configs)", fontsize=9)
    # explain the two shades once (colors follow agents, shade follows threshold)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="0.35", alpha=1.0),
        plt.Rectangle((0, 0), 1, 1, fc="0.35", alpha=0.45),
    ]
    axes[1].legend(
        handles,
        ["success ≥ 0.25", "success ≥ 0.5"],
        frameon=False,
        fontsize=8,
        loc="upper center",
    )
    fig.suptitle(
        "E1  Absolute-threshold breadth, final phase (95% config-bootstrap CIs)",
        fontsize=11,
        y=1.04,
    )
    _save(fig, "E1_absolute_breadth")


def fig_e2() -> None:
    df = pd.read_csv(OUTPUTS / "rl2_restricted_domain.csv")
    variants = ["unrestricted", "alpha_le_10"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, env in zip(axes, ENVS):
        for i, agent in enumerate(AGENTS):
            sub = df[(df["env"] == env) & (df["agent"] == agent)].set_index("variant")
            xs = [i - 0.15, i + 0.15]
            mx = [sub.loc[v, "max_success"] for v in variants]
            mn = [sub.loc[v, "mean_success"] for v in variants]
            c = AGENT_COLOR[agent]
            ax.plot(xs, mx, "-", color=c, lw=1, alpha=0.6)
            ax.plot(xs, mn, "-", color=c, lw=1, alpha=0.6)
            ax.scatter(xs, mx, marker=AGENT_MARKER[agent], s=42, color=c, zorder=3)
            ax.scatter(
                xs,
                mn,
                marker=AGENT_MARKER[agent],
                s=42,
                facecolors="white",
                edgecolors=c,
                zorder=3,
            )
        ax.set_xticks(range(len(AGENTS)), AGENTS, fontsize=8)
        ax.set_title(ENV_LABEL[env], fontsize=11)
        ax.grid(True, axis="y", alpha=0.15, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("success", fontsize=9)
    h = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color="0.35",
            label="max (left: all 64, right: α ≤ 10)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="white",
            color="0.35",
            label="mean (left: all 64, right: α ≤ 10)",
        ),
    ]
    axes[0].legend(handles=h, frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle(
        "E2  Restricting to α ≤ 10: max success unchanged in 13/16 cells, ≤ 0.02 shift in the rest",
        fontsize=11,
        y=1.04,
    )
    _save(fig, "E2_restricted_alpha")


def fig_e4() -> None:
    df = pd.read_csv(OUTPUTS / "rl4_eps_sensitivity.csv")
    eps = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    cols = [f"rho_{e:.2f}" for e in eps]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, env in zip(axes, ENVS):
        for agent in AGENTS:
            row = df[(df["env"] == env) & (df["agent"] == agent)].iloc[0]
            ax.plot(
                eps,
                [row[c] for c in cols],
                AGENT_STYLE[agent],
                marker=AGENT_MARKER[agent],
                ms=4,
                lw=1.8,
                color=AGENT_COLOR[agent],
                label=agent,
            )
        ax.set_title(ENV_LABEL[env], fontsize=11)
        ax.set_xlabel(r"$\varepsilon$", fontsize=9)
        ax.grid(True, alpha=0.15, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"$\rho_\varepsilon$", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        r"E4  Relative breadth $\rho_\varepsilon$ across thresholds, final phase",
        fontsize=11,
        y=1.04,
    )
    _save(fig, "E4_threshold_sweep")


def fig_e5() -> None:
    df = pd.read_csv(OUTPUTS / "rl5_phase_null.csv")
    df["env_order"] = df["env"].map({e: i for i, e in enumerate(ENVS)})
    df = df.sort_values("env_order")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), width_ratios=[1.4, 1])
    x = np.arange(len(df))
    null_lo, null_hi = _parse_ci(df["null_95"].iloc[0])
    ax1.axhspan(null_lo, null_hi, color="0.85", zorder=0, label="chance 95% band")
    ax1.axhline(
        df["null_mean"].iloc[0], color="0.4", ls="--", lw=1, label="chance mean (0.062)"
    )
    cis = [_parse_ci(s) for s in df["jaccard_obs_ci"]]
    err = np.array(
        [
            [v - lo for v, (lo, _) in zip(df["jaccard_obs"], cis)],
            [hi - v for v, (_, hi) in zip(df["jaccard_obs"], cis)],
        ]
    )
    ax1.errorbar(
        x,
        df["jaccard_obs"],
        yerr=err,
        fmt="o",
        ms=7,
        capsize=3,
        color="#0072B2",
        lw=1.4,
        label="observed (seed CI)",
    )
    ax1.set_xticks(x, [ENV_LABEL[e] for e in df["env"]], fontsize=9)
    ax1.set_ylabel("adjacent-phase Jaccard of top-7 sets", fontsize=9)
    ax1.set_ylim(0, 0.32)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(True, axis="y", alpha=0.15, lw=0.5)

    # No whiskers here: the within-config seed bootstrap is attenuation-biased
    # for correlation statistics, so those intervals understate the point
    # estimate; the permutation band on the left panel carries the inference.
    ax2.scatter(x, df["spearman_obs"], marker="s", s=48, color="#009E73")
    ax2.axhline(0, color="0.4", ls="--", lw=1)
    ax2.set_xticks(x, [ENV_LABEL[e] for e in df["env"]], fontsize=9)
    ax2.set_ylabel("adjacent-phase Spearman (full ranking)", fontsize=9)
    ax2.set_ylim(-0.05, 0.8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(True, axis="y", alpha=0.15, lw=0.5)
    fig.suptitle("E5  Top-set turnover against the chance level", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, "E5_topset_null")


def fig_e6() -> None:
    df = pd.read_csv(OUTPUTS / "rl6_filtered_fr_all.csv")
    df["env_order"] = df["env"].map({e: i for i, e in enumerate(ENVS)})
    df = df.sort_values(["env_order", "agent"])
    metrics = [("fr_auc", "FR-AUC"), ("gap_mean", "Gap mean"), ("mrr", "MRR")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (m, label) in zip(axes, metrics):
        for i, env in enumerate(ENVS):
            for j, agent in enumerate(AGENTS):
                r = df[(df["env"] == env) & (df["agent"] == agent)].iloc[0]
                x = i * 5 + j
                c = AGENT_COLOR[agent]
                ax.annotate(
                    "",
                    xy=(x, r[f"{m}_kept"]),
                    xytext=(x, r[f"{m}_all"]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.6),
                )
                ax.scatter(
                    [x],
                    [r[f"{m}_all"]],
                    s=26,
                    facecolors="white",
                    edgecolors=c,
                    zorder=3,
                )
        centers = [i * 5 + 1.5 for i in range(len(ENVS))]
        ax.set_xticks(centers, [ENV_LABEL[e] for e in ENVS], fontsize=8)
        if m == "fr_auc":
            ax.axhline(0.5, color="0.4", ls="--", lw=1)
        if m == "gap_mean":
            ax.axhline(0.0, color="0.4", ls="--", lw=1)
        ax.set_title(label, fontsize=11)
        ax.grid(True, axis="y", alpha=0.15, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    h = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="white",
            color=AGENT_COLOR[a],
            label=a,
        )
        for a in AGENTS
    ]
    axes[0].legend(handles=h, frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        "E6  Metrics before (circle) and after (arrow tip) removing anchors "
        "whose action does not approach the paired goal",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout()
    _save(fig, "E6_anchor_filter")


def fig_e8() -> None:
    df = pd.read_csv(OUTPUTS / "rl8_lagged_partial_ess.csv")
    panels = [
        ("r_ess_success_t", "same phase:  r(ESS$_t$, success$_t$)", [1, 2, 3, 4]),
        ("r_ess_t_success_t+1", "lagged:  r(ESS$_t$, success$_{t+1}$)", [1, 2, 3]),
        ("r_partial_given_log_lr", "adjusted for log lr (partial r)", [1, 2, 3, 4]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), sharey=True)
    for ax, (col, title, phases) in zip(axes, panels):
        for agent in AGENTS:
            sub = df[(df["agent"] == agent) & (df["phase"].isin(phases))]
            ax.plot(
                sub["phase"],
                sub[col],
                AGENT_STYLE[agent],
                marker=AGENT_MARKER[agent],
                ms=5,
                lw=1.8,
                color=AGENT_COLOR[agent],
                label=agent,
            )
        ax.axhline(0, color="0.4", ls="--", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("phase", fontsize=9)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_ylim(-1, 0.6)
        ax.grid(True, alpha=0.15, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Pearson r", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(
        "E8  ESS–success association on AntMaze-Medium (n = 320 per point)",
        fontsize=11,
        y=1.04,
    )
    _save(fig, "E8_lagged_ess")


def main() -> None:
    fig_e1()
    fig_e2()
    fig_e4()
    fig_e5()
    fig_e6()
    fig_e8()


if __name__ == "__main__":
    main()
