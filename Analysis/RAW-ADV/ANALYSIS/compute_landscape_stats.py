#!/usr/bin/env python3
"""Compute ρ_0.9, ρ_0.8, mean/max success and generate landscape plots.

For each (algo, env, phase) from eval_stats.csv:
  - Averages success over seeds per config
  - Computes ρ_0.9 / ρ_0.8 = fraction of configs at ≥ 90% / 80% of agent's max
  - Saves summary CSVs and (lr, alpha) scatter/contour landscape PNGs

Sources
-------
  ENVS/{env}/eval_stats.csv   — built by build_eval_stats.py
  (also picks up antmaze-medium from additional_stats_raw.csv)

Outputs
-------
  ANALYSIS/metric_summaries/landscape_stats.csv       — ρ_0.9/ρ_0.8/mean/max per algo×env×phase
  ANALYSIS/metric_summaries/landscape_stats_last.csv  — final phase only, agent×env pivot
  ENVS/{env}/landscapes/landscape_{algo}_phase{p}.png — one plot per algo×phase

Usage
-----
  python ANALYSIS/compute_landscape_stats.py               # all envs
  python ANALYSIS/compute_landscape_stats.py --env scene   # one env
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]
_ROOT = Path(__file__).parent.parent
OUT_CSV = Path(__file__).parent / "metric_summaries"
OUT_CSV.mkdir(exist_ok=True)

STATS_CSV = Path(
    "/Users/adityamohan/git/GCRL/plots/new-plots/"
    "advantage-dist-phases/basin-analysis/additional_stats_raw.csv"
)

EVAL_ENVS = {
    "antmaze-large": (
        "antmaze-large-navigate-v0",
        _ROOT / "ENVS" / "antmaze-large" / "eval_stats.csv",
    ),
    "cube": ("cube-single-play-v0", _ROOT / "ENVS" / "cube" / "eval_stats.csv"),
    "scene": ("scene-play-v0", _ROOT / "ENVS" / "scene" / "eval_stats.csv"),
}


# ── data loading ─────────────────────────────────────────────────────────────


def load_all(env_filter: str | None = None) -> pd.DataFrame:
    parts = []

    # antmaze-medium from legacy CSV
    if STATS_CSV.exists() and (env_filter is None or env_filter == "antmaze-medium"):
        df = pd.read_csv(STATS_CSV)
        df = df[df["env"] == "antmaze-medium-navigate-v0"][
            ["algo", "env", "config", "seed", "phase", "lr", "alpha", "success"]
        ].copy()
        df["env_short"] = "antmaze-medium"
        parts.append(df)

    for env_short, (env_tag, csv_path) in EVAL_ENVS.items():
        if env_filter is not None and env_filter != env_short:
            continue
        if not csv_path.exists():
            print(f"  [skip] {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df["env_short"] = env_short
        parts.append(df)

    if not parts:
        raise RuntimeError("No data found.")
    return pd.concat(parts, ignore_index=True)


# ── ρ computation ─────────────────────────────────────────────────────────────


def compute_rho(df: pd.DataFrame) -> pd.DataFrame:
    """Per (algo, env_short, phase): ρ_0.9, ρ_0.8, mean_success, max_success."""
    # Average over seeds first
    agg = (
        df.groupby(["algo", "env_short", "config", "phase", "lr", "alpha"])["success"]
        .mean()
        .reset_index()
    )

    rows = []
    for (algo, env_short, phase), grp in agg.groupby(["algo", "env_short", "phase"]):
        s = grp["success"].values
        mx = s.max()
        rows.append(
            {
                "algo": algo,
                "env": env_short,
                "phase": phase,
                "n_configs": len(s),
                "mean_success": round(float(s.mean()), 4),
                "max_success": round(float(mx), 4),
                "rho_0.9": round(float((s >= 0.9 * mx).mean()), 4)
                if mx > 0
                else float("nan"),
                "rho_0.8": round(float((s >= 0.8 * mx).mean()), 4)
                if mx > 0
                else float("nan"),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["env", "algo", "phase"]).reset_index(drop=True)
    )


# ── landscape plots ───────────────────────────────────────────────────────────


def plot_landscapes(df: pd.DataFrame, env_short: str) -> None:
    out_dir = _ROOT / "ENVS" / env_short / "landscapes"
    out_dir.mkdir(exist_ok=True)

    agg = (
        df[df["env_short"] == env_short]
        .groupby(["algo", "config", "phase", "lr", "alpha"])["success"]
        .mean()
        .reset_index()
    )
    if agg.empty:
        print(f"  [skip] no data for {env_short}")
        return

    phases = sorted(agg["phase"].unique())

    for phase in phases:
        sub = agg[agg["phase"] == phase]
        fig, axes = plt.subplots(
            1, len(AGENTS), figsize=(4 * len(AGENTS), 4), sharey=False
        )
        fig.suptitle(f"{env_short}  —  phase {phase}", fontsize=13)

        vmin = sub["success"].min()
        vmax = sub["success"].max()
        if vmax <= vmin:
            vmax = vmin + 1e-6

        for ax, agent in zip(axes, AGENTS):
            g = sub[sub["algo"] == agent]
            if g.empty:
                ax.set_title(agent)
                ax.axis("off")
                continue

            lx = np.log10(g["lr"].values)
            la = np.log10(g["alpha"].values)
            s = g["success"].values

            # Scatter coloured by success
            sc = ax.scatter(
                lx,
                la,
                c=s,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                s=60,
                edgecolors="none",
                alpha=0.85,
            )

            # Interpolated contour overlay (if enough unique points)
            if len(np.unique(lx)) > 3 and len(np.unique(la)) > 3:
                xi = np.linspace(lx.min(), lx.max(), 60)
                yi = np.linspace(la.min(), la.max(), 60)
                XI, YI = np.meshgrid(xi, yi)
                try:
                    ZI = griddata((lx, la), s, (XI, YI), method="linear")
                    ax.contourf(
                        XI,
                        YI,
                        ZI,
                        levels=8,
                        cmap="viridis",
                        alpha=0.35,
                        vmin=vmin,
                        vmax=vmax,
                    )
                except Exception:
                    pass

            ax.set_title(agent, fontsize=11)
            ax.set_xlabel("log₁₀(lr)", fontsize=9)
            if ax is axes[0]:
                ax.set_ylabel("log₁₀(α)", fontsize=9)
            plt.colorbar(
                sc,
                ax=ax,
                fraction=0.046,
                pad=0.04,
                label="success" if ax is axes[-1] else "",
            )

        plt.tight_layout()
        out = out_dir / f"landscape_phase{phase}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → {out.relative_to(_ROOT)}")


# ── pivot table for paper ─────────────────────────────────────────────────────


def pivot_last_phase(stats: pd.DataFrame) -> pd.DataFrame:
    last = stats.loc[stats.groupby(["algo", "env"])["phase"].idxmax()].copy()
    rows = []
    for env in last["env"].unique():
        for algo in AGENTS:
            r = last[(last["env"] == env) & (last["algo"] == algo)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append(
                {
                    "algo": algo,
                    "env": env,
                    "mean_success": r["mean_success"],
                    "max_success": r["max_success"],
                    "rho_0.9": r["rho_0.9"],
                    "rho_0.8": r["rho_0.8"],
                }
            )
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", default=None, help="Single env to process (e.g. scene)"
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    df = load_all(args.env)
    print(f"Loaded {len(df):,} rows  |  envs: {sorted(df['env_short'].unique())}")

    stats = compute_rho(df)
    stats.to_csv(OUT_CSV / "landscape_stats.csv", index=False)
    print(f"\nSaved → landscape_stats.csv  ({len(stats)} rows)")

    last = pivot_last_phase(stats)
    last.to_csv(OUT_CSV / "landscape_stats_last.csv", index=False)
    print("Saved → landscape_stats_last.csv")

    # Print final-phase table
    print("\n=== Final phase — ρ_0.9 / ρ_0.8 / mean / max success ===")
    print(f"{'Algo':<8} {'Env':<22} {'ρ_0.9':>6} {'ρ_0.8':>6} {'Mean':>7} {'Max':>7}")
    print("-" * 60)
    for _, row in last.iterrows():
        print(
            f"{row['algo']:<8} {row['env']:<22} "
            f"{row['rho_0.9']:>6.3f} {row['rho_0.8']:>6.3f} "
            f"{row['mean_success']:>7.3f} {row['max_success']:>7.3f}"
        )

    if not args.no_plots:
        print("\nGenerating landscape plots …")
        for env_short in df["env_short"].unique():
            print(f"  {env_short}")
            plot_landscapes(df, env_short)


if __name__ == "__main__":
    main()
