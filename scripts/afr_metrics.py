#!/usr/bin/env python3
"""CLI for running AFR (Future-Random Advantage Separation) metrics computation.

Loads trained model checkpoints from phase directories, recomputes cross-trajectory
advantage gaps using the OG-Bench dataset, and computes FR-AUC, EI, log_weight_ratio
metrics.

Usage:
    # Auto-discover latest phase in run_logs
    python scripts/afr_metrics.py --latest-phase /path/to/run_logs --plots

    # Explicit phase directory
    python scripts/afr_metrics.py \
      --phase-dir /path/to/phase \
      --alpha 0.5 \
      --n-batches 5 \
      --plots \
      --output afr_metrics.pkl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict


# Import with fallback for robustness
try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore  # noqa: F811


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute AFR (Future-Random Advantage Separation) metrics from trained checkpoints."
    )
    # Input specification
    parser.add_argument(
        "--phase-dir",
        nargs="+",
        type=Path,
        default=None,
        help="Phase directory (or directories) containing seed_*/ subdirectories with "
        "params_*.pkl files. If not provided, use --latest-phase.",
    )
    parser.add_argument(
        "--latest-phase",
        type=Path,
        default=None,
        help="Auto-discover the latest phase directory within a run_logs/ directory.",
    )
    parser.add_argument(
        "--configuration",
        type=Path,
        default=None,
        help="Path to run_logs/ directory with configuration_*.json files. "
        "If not specified, configuration is inferred from parent directories.",
    )

    # Algorithm parameters
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="AWR temperature parameter (default: None, skips log_weight_ratio metrics).",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=5,
        help="Number of dataset batches to sample per seed (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for advantage computation (default: 256, matches CONST_VAL_BATCH_SIZE).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed random seed for reproducibility (default: None).",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("afr_metrics.pkl"),
        help="Output file for metrics (default: afr_metrics.pkl).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Also dump results as JSON to this path (default: None).",
    )

    # Post-processing
    parser.add_argument(
        "--plots",
        dest="generate_plots",
        action="store_true",
        default=False,
        help="Generate AFR diagnostic plots (FR-AUC, gap distributions, etc.).",
    )
    parser.add_argument(
        "--no-plots",
        dest="generate_plots",
        action="store_false",
        help="Disable plot generation (default if no --plots provided).",
    )
    parser.set_defaults(generate_plots=False)

    return parser.parse_args()


def _find_latest_phase(base_dir: Path) -> Path | None:
    """Auto-discover the latest phase with trained checkpoints (params_*.pkl)."""
    if not base_dir.is_dir():
        print(f"[ERROR] Directory not found: {base_dir}")
        return None

    latest_phase = None
    for phase_dir in sorted(base_dir.glob("phase_*")):
        if phase_dir.is_dir():
            pkls = list(phase_dir.glob("params_*.pkl"))
            if pkls:
                latest_phase = phase_dir

    if latest_phase:
        print(f"[INFO] Auto-discovered latest phase: {latest_phase}")
        return latest_phase

    print(f"[ERROR] No trained checkpoints found in {base_dir}")
    return None


def _load_afr_module():
    """Import and return the afr_recomputation module."""
    try:
        from gcrl_landscapes.util import afr_recomputation

        return afr_recomputation
    except ImportError as e:
        print(f"[ERROR] Failed to import afr_recomputation: {e}")
        print("[HINT] Make sure gcrl_landscapes is installed: pip install -e .")
        sys.exit(1)


def _generate_plots(metrics_path: Path, output_dir: Path) -> bool:
    """Run the plotting module to visualize metrics."""
    try:
        from analysis.advantages.f_metrics_afr import main as plot_main
    except ImportError:
        print("[WARN] Could not import f_metrics_afr. Skipping plots.")
        return False

    # Save metrics to a temporary pkl if the path itself is not a pkl,
    # or just pass the existing path if it is one.
    metrics_file = metrics_path
    if metrics_path.suffix != ".pkl":
        # If the user provided a different output format, pickle it temporarily
        metrics_file = metrics_path.parent / "temp_afr_metrics.pkl"
        with open(metrics_file, "wb") as f:
            pickle.dump(
                pd.read_pickle(metrics_file)
                if isinstance(metrics_file, pd.DataFrame)
                else metrics_path,
                f,
            )

    # Use a simple wrapper to pass arguments to the plotting main
    import sys as _sys

    plot_args = ["--input", str(metrics_file), "--output-dir", str(output_dir)]

    original_argv = _sys.argv
    _sys.argv = ["f_metrics_afr.py"] + plot_args
    try:
        plot_main()
        print(f"[OK] Plots generated in: {output_dir}")
        return True
    except Exception as e:
        print(f"[WARN] Plot generation failed: {e}")
        return False
    finally:
        _sys.argv = original_argv


def _format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics dict as a string for printing."""
    lines = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"  {key}: {value:+.4f}")
        elif isinstance(value, int):
            lines.append(f"  {key}: {value}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def main() -> None:
    """Main CLI entry point."""
    args = _parse_args()

    # 1. Resolve phase directory
    phase_dirs: list[Path] = []
    if args.latest_phase:
        found = _find_latest_phase(args.latest_phase)
        if found:
            phase_dirs.append(found)
    elif args.phase_dir:
        phase_dirs.extend(args.phase_dir)
    else:
        print("[ERROR] Please provide either --phase-dir or --latest-phase.")
        sys.exit(1)

    if not phase_dirs:
        print("[ERROR] No phase directories resolved.")
        sys.exit(1)

    print("=" * 60)
    print("AFR (Future-Random Advantage Separation) Metrics Computation")
    print("=" * 60)
    print(f"Phase directories: {len(phase_dirs)}")
    for p in phase_dirs:
        print(f"  - {p}")
    if args.alpha is not None:
        print(f"AWR alpha: {args.alpha}")
    print(f"n_batches: {args.n_batches}")
    print(f"batch_size: {args.batch_size}")
    print(f"plots: {args.generate_plots}")
    print(f"output: {args.output}")
    print("=" * 60)

    # 2. Run computation
    afr = _load_afr_module()
    all_metrics_rows: list[pd.DataFrame] = []

    for phase_dir in phase_dirs:
        print(f"\n--- Processing phase: {phase_dir} ---")
        config_dir = args.configuration if args.configuration else None

        try:
            df, configs = afr.run_afr_for_phase(
                phase_dir,
                alpha=args.alpha,
                n_batches=args.n_batches,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {phase_dir}: {e}")
            import traceback

            traceback.print_exc()
            continue

        if df is not None and not df.empty:
            for seed_num, config in configs.items():
                mask = df["seed"] == seed_num
                for hp_key in [
                    "agent_name",
                    "dataset_name",
                    "dataset",
                    "actor_loss",
                    "alpha",
                    "obs_dim",
                    "act_dim",
                ]:
                    if hp_key in config:
                        df.loc[mask, f"hp.{hp_key}"] = config[hp_key]

            all_metrics_rows.append(df)
            print(f"[OK] {phase_dir}: {len(df)} seeds processed")
            for _, row in df.iterrows():
                print(f"\n  Seed {row['seed']}:")
                print(
                    f"    {_format_metrics({k: v for k, v in row.to_dict().items() if k.startswith(('fr_auc', 'ei', 'gap', 'log_weight_ratio'))})}"
                )
        else:
            print(f"[WARN] {phase_dir}: No metrics computed")

    # 3. Output
    if all_metrics_rows:
        final_df = pd.concat(all_metrics_rows, ignore_index=True)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for phase in final_df.get("phase", pd.Series(dtype=object)).unique():
            phase_df = final_df[final_df["phase"] == phase]
            print(f"\nPhase {phase}:")
            print(
                f"  Seeds: {int(phase_df['seed'].min())}-{int(phase_df['seed'].max())}"
            )

            for metric in ["fr_auc", "ei"]:
                if metric in final_df.columns:
                    mean_val = phase_df[metric].mean()
                    std_val = phase_df[metric].std() if len(phase_df) > 1 else 0.0
                    print(f"  {metric}: {mean_val:+.4f} ± {std_val:.4f}")

            if "log_weight_ratio_mean" in final_df.columns:
                mw = phase_df["log_weight_ratio_mean"].mean()
                mstd = (
                    phase_df["log_weight_ratio_std"].mean()
                    if len(phase_df) > 1
                    else 0.0
                )
                print(f"  log_weight_ratio: {mw:+.4f} ± {mstd:.4f}")

        # Save pickle
        with open(args.output, "wb") as f:
            pickle.dump(final_df, f)
        print(f"\nMetrics saved to: {args.output}")

        if args.json_output:
            final_dict = final_df.to_dict(orient="records")
            for row in final_dict:
                for k, v in row.items():
                    if hasattr(v, "item"):
                        row[k] = v.item()
                    elif hasattr(v, "to_dict"):
                        row[k] = v.to_dict()
                    elif hasattr(v, "tolist"):
                        row[k] = v.tolist()
            with open(args.json_output, "w") as f:
                json.dump(final_dict, f, indent=2, default=str)
            print(f"Metrics saved to: {args.json_output}")

        # 4. Plots
        if args.generate_plots:
            print("\nGenerating plots...")
            plots_dir = args.output.parent / "afr_plots"
            _generate_plots(args.output, plots_dir)

    else:
        print("[ERROR] No metrics were computed for any phase.")
        sys.exit(1)


if __name__ == "__main__":
    main()
