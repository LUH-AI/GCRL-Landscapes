#!/usr/bin/env python3
"""Individual scatter plots: success vs FR-AUC and success vs MRR for Cube and Scene.

Outputs (in metric_summaries/)
  scatter_cube_fr_auc.{pdf,png}
  scatter_scene_fr_auc.{pdf,png}
  scatter_cube_mrr.{pdf,png}
  scatter_scene_mrr.{pdf,png}
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
    }
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

_ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "metric_summaries"
OUT.mkdir(exist_ok=True)

AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]
COLORS = {"CRL": "#1f77b4", "GCIQL": "#ff7f0e", "GCIVL": "#2ca02c", "QRL": "#d62728"}
MARKER = {"CRL": "o", "GCIQL": "s", "GCIVL": "^", "QRL": "D"}

ENV_LABELS = {"cube": "Cube", "scene": "Scene"}
METRIC_LABELS = {"fr_auc": "FR-AUC", "mrr": "MRR"}


def load_data(env: str) -> pd.DataFrame:
    env_dir = _ROOT / "ENVS" / env
    mat = pd.read_parquet(env_dir / "advantage_matrix_metrics.parquet")
    diag = (
        mat.groupby(["agent", "config"])[["fr_auc", "mrr"]]
        .mean()
        .reset_index()
        .rename(columns={"agent": "algo"})
    )

    p = env_dir / "eval_stats.csv"
    df = pd.read_csv(p)
    final = df[df["phase"] == df["phase"].max()]
    succ = final.groupby(["algo", "config"])["success"].mean().reset_index()

    return diag.merge(succ, on=["algo", "config"])


def make_plot(df: pd.DataFrame, env: str, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    for algo in AGENTS:
        sub = df[df["algo"] == algo]
        ax.scatter(
            sub[metric],
            sub["success"],
            color=COLORS[algo],
            marker=MARKER[algo],
            s=28,
            alpha=0.82,
            edgecolors="none",
            label=algo,
            zorder=3,
        )

    # Pooled OLS trend line
    x_all = df[metric].values
    y_all = df["success"].values
    mask = np.isfinite(x_all) & np.isfinite(y_all)
    if mask.sum() > 3:
        m, b = np.polyfit(x_all[mask], y_all[mask], 1)
        xs = np.linspace(x_all[mask].min(), x_all[mask].max(), 60)
        ax.plot(xs, m * xs + b, color="#666666", lw=1.0, ls="--", alpha=0.7, zorder=2)

        pr, _ = pearsonr(x_all[mask], y_all[mask])
        sr, _ = spearmanr(x_all[mask], y_all[mask])
        ax.text(
            0.97,
            0.04,
            f"$r={pr:+.2f}$\n$\\rho={sr:+.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#444444",
        )

    ax.set_xlabel(METRIC_LABELS[metric], fontsize=9)
    ax.set_ylabel("Success (final phase)", fontsize=9)
    ax.set_title(ENV_LABELS[env], fontsize=9.5, fontweight="semibold")
    ax.tick_params(labelsize=8)
    ax.legend(
        fontsize=7.5,
        markerscale=1.1,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        handletextpad=0.4,
        borderpad=0.5,
    )
    ax.set_ylim(-0.04, 1.04)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    stem = f"scatter_{env}_{metric}"
    for fmt in ("pdf", "png"):
        out = OUT / f"{stem}.{fmt}"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  → {out.name}")
    plt.close(fig)


def main() -> None:
    for env in ["cube", "scene"]:
        df = load_data(env)
        for metric in ["fr_auc", "mrr"]:
            make_plot(df, env, metric)


if __name__ == "__main__":
    main()
