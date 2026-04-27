#!/usr/bin/env python3
"""AFR (Future-Random Advantage Separation) metrics — plotting and summary.

Produces visualizations and summary statistics from AFR metrics computed by
`scripts/afr_metrics.py` or directly via `afr_recomputation.py`.

Usage:
    python analysis/advantages/f_metrics_afr.py \\
        --input afr_metrics.pkl \\
        --output-dir afr_metrics_plots
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.gridspec import GridSpec

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore  # noqa: F811
    matplotlib = None  # type: ignore  # noqa: F811
    matplotlib.pyplot = None  # type: ignore  # noqa: F811

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot AFR (Future-Random Advantage Separation) metrics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("afr_metrics.pkl"),
        help="Path to AFR metrics pickle/JSON file (default: afr_metrics.pkl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("afr_plots"),
        help="Directory for output plots (default: afr_plots).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["fr_auc", "ei"],
        help="Metrics to plot (default: fr_auc ei).",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="phase",
        help="Column to group plots by (default: phase).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="A FR Metrics by Phase",
        help="Plot title for summary figure (default: 'AFR Metrics by Phase').",
    )
    parser.add_argument(
        "--n-batches-filter",
        type=int,
        default=None,
        help="Only include rows with n_batches_sampled >= this value.",
    )
    return parser.parse_args()


def _load_metrics(input_path: Path) -> pd.DataFrame:
    """Load AFR metrics from pickle or JSON file.

    Args:
        input_path: Path to the metrics file.

    Returns:
        DataFrame with AFR metrics.
    """
    if input_path.suffix == ".json":
        import json

        data = json.loads(input_path.read_text())
        return pd.DataFrame(data)

    with open(input_path, "rb") as f:
        import pickle

        data = pickle.load(f)

    # Ensure the loaded object is a DataFrame
    if isinstance(data, pd.DataFrame):
        return data
    raise ValueError(
        f"Expected a DataFrame in {input_path}, but got {type(data).__name__}. "
        "Please ensure the metrics file contains a pickled DataFrame."
    )


def _compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics across seeds for each phase.

    Args:
        df: DataFrame with columns (phase, seed, fr_auc, ei, ...)

    Returns:
        DataFrame with summary stats.
    """
    summary_rows: List[Dict[str, Any]] = []

    for phase in df["phase"].unique():
        phase_df = df[df["phase"] == phase]
        summary: Dict[str, Any] = {"phase": phase, "n_seeds": len(phase_df)}

        for metric in [
            "fr_auc",
            "ei",
            "gap_mean",
            "log_weight_ratio_mean",
            "frac_weight_ratio_gt_10",
            "frac_weight_ratio_gt_100",
        ]:
            if metric in phase_df.columns:
                vals = phase_df[metric].dropna()
                if len(vals) > 0:
                    summary[f"{metric}_mean"] = vals.mean()
                    summary[f"{metric}_std"] = vals.std() if len(vals) > 1 else 0.0
                    summary[f"{metric}_median"] = vals.median()
                else:
                    summary[f"{metric}_mean"] = np.nan
                    summary[f"{metric}_std"] = 0.0

        summary_rows.append(summary)

    return pd.DataFrame(summary_rows)


def _plot_fr_auc_by_phase(
    df: pd.DataFrame,
    output_dir: Path,
    title: str = "FR-AUC by Phase",
) -> None:
    """Plot FR-AUC (Pr(A+ > A-)) per seed and phase.

    Args:
        df: DataFrame with phase, seed, fr_auc columns.
        output_dir: Directory to save the plot.
        title: Plot title.
    """
    if not HAS_MATPLOTLIB:  # type: ignore[arg-type]
        logger.warning("matplotlib not available — skipping FR-AUC plot")
        return

    assert plt is not None
    fig, ax = plt.subplots(figsize=(10, 6))

    phases = sorted(df["phase"].unique())

    # Plot per-seed values as points
    for phase in phases:
        phase_df = df[df["phase"] == phase]
        seeds = sorted(phase_df["seed"])
        fr_aucs = [phase_df[phase_df["seed"] == s]["fr_auc"].values[0] for s in seeds]

        if not fr_aucs:
            continue

        ax.errorbar(
            [phase] * len(fr_aucs),
            fr_aucs,
            yerr=0,  # per-seed, no error bar for individual points
            fmt="o",
            markersize=4,
            alpha=0.5,
            color="steelblue",
        )

    # Compute and plot mean ± std
    means, stds = [], []
    for phase in phases:
        phase_df = df[df["phase"] == phase]
        fr_aucs = phase_df["fr_auc"].dropna()
        if len(fr_aucs) > 0:
            means.append(fr_aucs.mean())
            stds.append(fr_aucs.std() if len(fr_aucs) > 1 else 0.0)
        else:
            means.append(np.nan)
            stds.append(0.0)

    ax.errorbar(
        phases,
        means,
        yerr=stds,
        fmt="-o",
        linewidth=2,
        markersize=8,
        color="darkred",
        ecolor="darkred",
        capsize=4,
    )

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)")
    ax.set_xlabel("Phase", fontsize=12)
    ax.set_ylabel("FR-AUC", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(phases)
    ax.set_xticklabels([f"P{k}" for k in phases])
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "fr_auc_by_phase.png", dpi=150)
    plt.close(fig)
    logger.info("Saved FR-AUC plot → %s", output_dir / "fr_auc_by_phase.png")


def _print_fr_auc_summary(df: pd.DataFrame) -> None:
    """Print FR-AUC summary to stdout.

    Args:
        df: DataFrame with phase, seed, fr_auc, ei columns.
    """
    print("=" * 60)
    print("FR-AUC SUMMARY")
    print("=" * 60)

    for phase in sorted(df["phase"].unique()):
        phase_df = df[df["phase"] == phase]
        fr_aucs = phase_df["fr_auc"].dropna()
        eis = (
            phase_df["ei"].dropna()
            if "ei" in phase_df.columns
            else pd.Series(dtype=float)
        )

        if len(fr_aucs) == 0:
            print(f"\nPhase {phase}: No FR-AUC data")
            continue

        print(f"\nPhase {phase}: {len(fr_aucs)} seeds")
        print(
            f"  FR-AUC:  mean={fr_aucs.mean():+.4f}  std={fr_aucs.std():.4f}  median={fr_aucs.median():+.4f}"
        )
        if len(eis) > 0:
            print(
                f"  EI:      mean={eis.mean():+.4f}  std={eis.std():.4f}  median={eis.median():+.4f}"
            )

        # High-quality rows have FR-AUC > 0.6 and EI > 0.1
        high = phase_df[
            (phase_df["fr_auc"] > 0.6) & (phase_df["ei"] > 0.1)
            if "ei" in phase_df.columns
            else phase_df["fr_auc"] > 0.6
        ]
        if len(high) > 0:
            print(f"  High-quality: {len(high)} / {len(fr_aucs)} seeds")
    print("=" * 60)


def _plot_gap_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    n_bins: int = 50,
) -> None:
    """Plot gap distributions across phases.

    Args:
        df: DataFrame with gap_mean, gap_std, phase columns.
        output_dir: Directory to save the plot.
        n_bins: Number of histogram bins.
    """
    if not HAS_MATPLOTLIB:  # type: ignore[arg-type]
        logger.warning("matplotlib not available — skipping gap distribution plot")
        return

    assert plt is not None
    fig, axes = plt.subplots(
        1, len(df["phase"].unique()), figsize=(5 * len(df["phase"].unique()), 4)
    )
    if len(axes) == 1:
        axes = [axes]

    for ax, phase in zip(axes, sorted(df["phase"].unique())):
        phase_df = df[df["phase"] == phase]
        gap_means = phase_df["gap_mean"].dropna().values
        gap_stds = phase_df["gap_std"].dropna().values

        # Approximate KDE from summary stats (mean ± std)
        x = np.linspace(-3, 3, 200)
        y = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

        ax.fill_between(
            x * gap_stds[0],
            y * gap_stds[0] / 4,
            y * gap_stds[0] / 2,
            alpha=0.3,
            label="Approx Gaussian",
        )
        ax.axvline(
            gap_means[0], color="red", linestyle="--", label=f"μ={gap_means[0]:+.3f}"
        )
        ax.set_title(
            f"\nPhase {phase}\nμ={gap_means[0]:+.3f} σ={gap_stds[0]:.3f}", fontsize=10
        )
        ax.set_xlabel("Gap (normalized)")
        ax.set_ylabel("Density (approx.)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "gap_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved gap distribution plot → %s", output_dir / "gap_distribution.png")


def _plot_all_metrics(
    df: pd.DataFrame,
    output_dir: Path,
    title: str = "AFR Metrics by Phase",
) -> None:
    """Plot all metrics (FR-AUC, EI, log_weight_ratio) for all phases.

    Args:
        df: DataFrame with AFR metrics.
        output_dir: Directory to save plots.
        title: Figure title.
    """
    if not HAS_MATPLOTLIB:  # type: ignore[arg-type]
        logger.warning("matplotlib not available — skipping all-metrics plot")
        return

    assert plt is not None

    phases = sorted(df["phase"].unique())
    n_metrics = len(df.columns)
    if n_metrics == 0:
        logger.warning("No metrics to plot")
        return

    fig = plt.figure(
        figsize=(min(12, 4 * n_metrics), 4 * len(phases) if n_metrics > 1 else 4)
    )
    gs = GridSpec(min(len(phases), 5), n_metrics, figure=fig)

    for m_idx, metric in enumerate(df.columns):
        if metric in ("phase", "seed", "n_batches_sampled", "fr_auc_std", "ei_std"):
            continue  # skip non-numeric or metadata columns
        ax = fig.add_subplot(gs[: len(phases), m_idx])

        means, stds = [], []
        for phase in phases:
            phase_df = df[df["phase"] == phase]
            vals = phase_df[metric].dropna()
            if len(vals) > 0:
                means.append(vals.mean())
                stds.append(vals.std() if len(vals) > 1 else 0.0)
            else:
                means.append(np.nan)
                stds.append(0.0)

        ax.errorbar(
            [f"Phase {k}" for k in phases],
            means,
            yerr=stds,
            fmt="-o",
            linewidth=2,
            markersize=6,
            capsize=4,
        )
        ax.set_title(f"{metric}", fontsize=10)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    fig.savefig(output_dir / "all_metrics_by_phase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved all-metrics plot → %s", output_dir / "all_metrics_by_phase.png")


def main() -> None:
    """Main plotting entry point."""
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    df = _load_metrics(args.input)

    if args.n_batches_filter is not None:
        df = df[df["n_batches_sampled"] >= args.n_batches_filter]

    if df is None or df.empty:
        print(f"[ERROR] No metrics loaded from {args.input}")
        return

    # Compute summary
    summary_df = _compute_summary(df)

    # Print summary
    _print_fr_auc_summary(df)

    # Save summary
    summary_path = args.output_dir / "afr_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    # Plot FR-AUC
    _plot_fr_auc_by_phase(df, args.output_dir, title=args.title)

    # Plot gap distributions
    if "gap_mean" in df.columns:
        _plot_gap_distribution(df, args.output_dir)

    # Plot all metrics
    _plot_all_metrics(df, args.output_dir, title=args.title)

    print("Done.")


if __name__ == "__main__":
    main()
