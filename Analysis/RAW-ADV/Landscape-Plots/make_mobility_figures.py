#!/usr/bin/env python3
"""Regenerate mobility figures from raw eval data for NeurIPS submission.

Main paper  (main_mobility_top10.{pdf,png})
  3 × 4: (AntMaze-M, Cube, Scene) × (CRL, GCIQL, GCIVL, QRL), top-10%

Appendix
  appendix_mobility_top10.{pdf,png}  — 4 × 4 top-10%
  appendix_mobility_top05.{pdf,png}  — 4 × 4 top-5%

Styling (NeurIPS ready)
  - Shared log-x (lr) and linear-y (α) axes across all panels
  - Tick labels only on bottom row and left column
  - Single phase legend outside the grid
  - No per-panel axis labels; one global x-label and y-label
  - KDE filled contours + numbered centroid per phase
  - Colourblind-safe viridis phase colours
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", category=UserWarning)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "xtick.minor.width": 0.4,
        "xtick.minor.visible": True,
        "ytick.minor.visible": False,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "pdf.fonttype": 42,  # TrueType in PDF (crisp in Acrobat)
        "ps.fonttype": 42,
    }
)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
_ROOT = BASE.parent

STATS_CSV = Path(
    "/Users/adityamohan/git/GCRL/plots/new-plots/"
    "advantage-dist-phases/basin-analysis/additional_stats_raw.csv"
)
EVAL_ENVS = {
    "antmaze-large": _ROOT / "ENVS" / "antmaze-large" / "eval_stats.csv",
    "cube": _ROOT / "ENVS" / "cube" / "eval_stats.csv",
    "scene": _ROOT / "ENVS" / "scene" / "eval_stats.csv",
}

# ── constants ─────────────────────────────────────────────────────────────────
AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]
ENV_LABELS = {
    "antmaze-medium": "AntMaze-M",
    "antmaze-large": "AntMaze-L",
    "cube": "Cube",
    "scene": "Scene",
}

# Viridis samples — perceptually uniform, colourblind-safe
_vc = plt.cm.viridis(np.linspace(0.05, 0.88, 4))
PHASE_COLORS = {i + 1: _vc[i] for i in range(4)}

# Shared axis limits
LR_LO, LR_HI = 6e-7, 1.5e-2  # slight padding beyond data range
ALPHA_LO, ALPHA_HI = -0.5, 31.5

# KDE evaluation grid (in log10-lr × alpha space)
_xg = np.linspace(np.log10(LR_LO), np.log10(LR_HI), 120)
_yg = np.linspace(ALPHA_LO, ALPHA_HI, 120)
XGRID, YGRID = np.meshgrid(_xg, _yg)
GRID_PTS = np.vstack([XGRID.ravel(), YGRID.ravel()])

OUT_DIR = BASE / "combined"
OUT_DIR.mkdir(exist_ok=True)


# ── data ──────────────────────────────────────────────────────────────────────
def load_all() -> pd.DataFrame:
    parts = []
    if STATS_CSV.exists():
        df = pd.read_csv(STATS_CSV)
        df = df[df["env"] == "antmaze-medium-navigate-v0"].copy()
        df["env_key"] = "antmaze-medium"
        parts.append(
            df[["algo", "env_key", "config", "seed", "phase", "lr", "alpha", "success"]]
        )
    for env_key, p in EVAL_ENVS.items():
        if p.exists():
            df = pd.read_csv(p)
            df["env_key"] = env_key
            parts.append(
                df[
                    [
                        "algo",
                        "env_key",
                        "config",
                        "seed",
                        "phase",
                        "lr",
                        "alpha",
                        "success",
                    ]
                ]
            )
    return pd.concat(parts, ignore_index=True)


def build_phase_cfg(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean success per (algo, env, config, phase); keep lr/alpha."""
    return (
        raw.groupby(["algo", "env_key", "config", "phase", "lr", "alpha"])["success"]
        .mean()
        .reset_index()
    )


# ── KDE per phase ─────────────────────────────────────────────────────────────
def draw_phase(ax, lr_vals, alpha_vals, color, phase_num):
    """KDE filled contours + numbered centroid for one phase."""
    n = len(lr_vals)
    r, g, b, _ = color

    cx_log = float(np.mean(np.log10(lr_vals)))
    cy = float(np.mean(alpha_vals))

    if n < 4:
        # Too few points for KDE — just mark centroid
        ax.scatter(
            [10**cx_log],
            [cy],
            s=48,
            c=[(r, g, b, 1.0)],
            edgecolors="white",
            linewidth=0.7,
            zorder=5,
        )
        ax.text(
            10**cx_log,
            cy,
            str(phase_num),
            ha="center",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color="white",
            zorder=6,
        )
        return

    pts = np.vstack([np.log10(lr_vals), alpha_vals])
    try:
        kde = gaussian_kde(pts, bw_method=0.45)
        Z = kde(GRID_PTS).reshape(XGRID.shape)
    except Exception:
        return
    if Z.max() == 0:
        return

    z_pos = Z[Z > 0]
    z_outer = np.percentile(z_pos, 18)  # outer contour boundary
    z_inner = np.percentile(z_pos, 58)  # inner / denser region

    # Outer fill
    cmap_o = mcolors.LinearSegmentedColormap.from_list(
        "", [(r, g, b, 0.0), (r, g, b, 0.20)], N=32
    )
    ax.contourf(
        10**XGRID,
        YGRID,
        Z,
        levels=np.linspace(z_outer, z_inner, 3),
        cmap=cmap_o,
        extend="neither",
    )

    # Inner fill
    cmap_i = mcolors.LinearSegmentedColormap.from_list(
        "", [(r, g, b, 0.20), (r, g, b, 0.52)], N=32
    )
    ax.contourf(
        10**XGRID,
        YGRID,
        Z,
        levels=np.linspace(z_inner, Z.max() * 1.001, 3),
        cmap=cmap_i,
        extend="neither",
    )

    # Boundary contour line
    ax.contour(
        10**XGRID,
        YGRID,
        Z,
        levels=[z_inner],
        colors=[(r, g, b, 0.88)],
        linewidths=0.9,
        linestyles="-",
    )

    # Numbered centroid
    ax.scatter(
        [10**cx_log],
        [cy],
        s=52,
        c=[(r, g, b, 1.0)],
        edgecolors="white",
        linewidth=0.7,
        zorder=5,
    )
    ax.text(
        10**cx_log,
        cy,
        str(phase_num),
        ha="center",
        va="center",
        fontsize=5.5,
        fontweight="bold",
        color="white",
        zorder=6,
    )


# ── single panel ──────────────────────────────────────────────────────────────
def fill_panel(ax, cfg_df, algo, env_key, top_frac):
    sub = cfg_df[(cfg_df["algo"] == algo) & (cfg_df["env_key"] == env_key)]
    phases = sorted(sub["phase"].unique())
    for phase in phases:
        grp = sub[sub["phase"] == phase]
        k = max(1, int(np.ceil(top_frac * len(grp))))
        top = grp.nlargest(k, "success")
        color = PHASE_COLORS.get(phase, PHASE_COLORS[4])
        draw_phase(ax, top["lr"].values, top["alpha"].values, color, phase)

    ax.set_xscale("log")
    ax.set_xlim(LR_LO, LR_HI)
    ax.set_ylim(ALPHA_LO, ALPHA_HI)
    ax.set_yticks([0, 10, 20, 30])
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_facecolor("#fafafa")  # very subtle off-white background


# ── assemble figure ───────────────────────────────────────────────────────────
def make_figure(env_list, top_frac, stem, fig_w, fig_h):
    raw = load_all()
    cfg = build_phase_cfg(raw)

    n_rows, n_cols = len(env_list), len(AGENTS)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.10, "wspace": 0.05},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, env_key in enumerate(env_list):
        for c, algo in enumerate(AGENTS):
            ax = axes[r][c]
            fill_panel(ax, cfg, algo, env_key, top_frac)

            # Suppress interior tick labels
            if c > 0:
                ax.tick_params(labelleft=False)
            if r < n_rows - 1:
                ax.tick_params(labelbottom=False)

            # Column header: algorithm name (top row only)
            if r == 0:
                ax.set_title(algo, fontsize=9.5, fontweight="semibold", pad=5)

            # Row label: env name (left column only, as y-axis label)
            if c == 0:
                ax.set_ylabel(
                    ENV_LABELS[env_key], fontsize=9, fontweight="semibold", labelpad=5
                )

    # ── shared axis descriptors ───────────────────────────────────────────────
    fig.supxlabel("Learning rate  $\\eta$", fontsize=8.5, y=-0.01)
    fig.supylabel("AWR temperature  $\\alpha$", fontsize=8.5, x=-0.02)

    # ── phase legend (right of grid) ──────────────────────────────────────────
    handles = [
        mpatches.Patch(
            facecolor=PHASE_COLORS[p],
            edgecolor="none",
            label=f"Phase {p}",
        )
        for p in sorted(PHASE_COLORS)
    ]
    fig.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(1.13, 0.50),
        ncol=1,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        fontsize=8,
        title="Training\nphase",
        title_fontsize=7.5,
        handlelength=1.4,
        handleheight=1.1,
        borderpad=0.8,
    )

    for fmt in ("pdf", "png"):
        out = OUT_DIR / f"{stem}.{fmt}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Saved → {out.name}")
    plt.close(fig)


def main():
    print("Main figure (3 × 4, top-10%) …")
    make_figure(
        env_list=["antmaze-medium", "cube", "scene"],
        top_frac=0.10,
        stem="main_mobility_top10",
        fig_w=7.0,
        fig_h=5.5,
    )

    print("Appendix figure (4 × 4, top-10%) …")
    make_figure(
        env_list=["antmaze-medium", "antmaze-large", "cube", "scene"],
        top_frac=0.10,
        stem="appendix_mobility_top10",
        fig_w=7.0,
        fig_h=7.25,
    )

    print("Appendix figure (4 × 4, top-5%) …")
    make_figure(
        env_list=["antmaze-medium", "antmaze-large", "cube", "scene"],
        top_frac=0.05,
        stem="appendix_mobility_top05",
        fig_w=7.0,
        fig_h=7.25,
    )


if __name__ == "__main__":
    main()
