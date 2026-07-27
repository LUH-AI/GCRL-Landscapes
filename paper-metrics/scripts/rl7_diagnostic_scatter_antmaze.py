"""R-L7: AntMaze counterparts of the cube/scene diagnostic scatters (7VBe-Q4).

Adapted from RAW-ADV/ANALYSIS/diagnostic_scatter.py with two changes:
  - envs: antmaze-medium, antmaze-large
  - success comes from data/eval_stats/{env}.csv (RAW-ADV has no
    eval_stats.csv for antmaze-medium), final phase, seed-mean per config,
    matching the original's aggregation.

Colors/layout match the existing cube/scene figures so the four scatters can
be shown side by side.

Outputs: outputs/rl7_diagnostic_scatter_{env}.png/.pdf
         outputs/rl7_diagnostic_correlations.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from _common import OUTPUTS, eval_stats_path  # noqa: E402

RAW_ADV_ENVS = Path(
    os.environ.get(
        "RAW_ADV_ENVS",
        str(Path(__file__).resolve().parents[2] / "Analysis" / "RAW-ADV" / "ENVS"),
    )
)

AGENTS = ["CRL", "GCIQL", "GCIVL", "QRL"]
COLORS = {"CRL": "#0173b2", "GCIQL": "#de8f05", "GCIVL": "#029e73", "QRL": "#d55e00"}
ENVS = ["antmaze-medium", "antmaze-large"]
METRICS = ["fr_auc", "gap_mean", "mrr", "ess", "top5_mass"]
METRIC_LABELS = {
    "fr_auc": "FR-AUC",
    "gap_mean": "Gap mean",
    "mrr": "MRR",
    "ess": "ESS",
    "top5_mass": "Top-5 mass",
}


def load_diagnostics(env: str) -> pd.DataFrame:
    env_dir = RAW_ADV_ENVS / env

    mat = pd.read_parquet(env_dir / "advantage_matrix_metrics.parquet")
    mat_cfg = (
        mat.groupby(["agent", "config"])[["fr_auc", "gap_mean", "mrr"]]
        .mean()
        .reset_index()
        .rename(columns={"agent": "algo"})
    )

    ess_df = pd.read_parquet(env_dir / "ess_per_batch.parquet")
    ess_cfg = (
        ess_df.groupby(["agent", "config"])["ess"]
        .mean()
        .reset_index()
        .rename(columns={"agent": "algo"})
    )

    conc = pd.read_csv(env_dir / "awr_concentration.csv")
    if "has_real_alpha" in conc.columns:
        conc = conc[conc["has_real_alpha"]]
    col = "top5_mass_mean" if "top5_mass_mean" in conc.columns else "top5_mass"
    conc_col = "configuration" if "configuration" in conc.columns else "config"
    conc_cfg = (
        conc.groupby(["agent", conc_col])[col]
        .mean()
        .reset_index()
        .rename(columns={"agent": "algo", conc_col: "config", col: "top5_mass"})
    )

    return mat_cfg.merge(ess_cfg, on=["algo", "config"]).merge(
        conc_cfg, on=["algo", "config"]
    )


def load_success(env: str) -> pd.DataFrame:
    df = pd.read_csv(eval_stats_path(env))
    final = df[df["phase"] == df["phase"].max()]
    return final.groupby(["algo", "config"])["success"].mean().reset_index()


def correlations(df: pd.DataFrame, metric: str, env: str) -> list[dict]:
    rows = []
    x, y = df[metric].values, df["success"].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 3:
        pr, _ = pearsonr(x[mask], y[mask])
        sr, _ = spearmanr(x[mask], y[mask])
        rows.append(
            dict(
                env=env,
                method="pooled",
                metric=metric,
                pearson=round(pr, 3),
                spearman=round(sr, 3),
                n=int(mask.sum()),
            )
        )
    for agent in AGENTS:
        sub = df[df["algo"] == agent]
        x2, y2 = sub[metric].values, sub["success"].values
        mask2 = np.isfinite(x2) & np.isfinite(y2)
        if mask2.sum() > 3:
            pr, _ = pearsonr(x2[mask2], y2[mask2])
            sr, _ = spearmanr(x2[mask2], y2[mask2])
        else:
            pr = sr = float("nan")
        rows.append(
            dict(
                env=env,
                method=agent,
                metric=metric,
                pearson=round(pr, 3) if np.isfinite(pr) else float("nan"),
                spearman=round(sr, 3) if np.isfinite(sr) else float("nan"),
                n=int(mask2.sum()),
            )
        )
    return rows


def main() -> None:
    all_corr = []
    for env in ENVS:
        df = load_diagnostics(env).merge(load_success(env), on=["algo", "config"])
        print(f"{env}: {len(df)} (algo, config) rows after merge")

        fig, axes = plt.subplots(1, len(METRICS), figsize=(4.5 * len(METRICS), 4.5))
        fig.suptitle(f"{env} — Success vs diagnostic metrics", fontsize=13)

        for ax, metric in zip(axes, METRICS):
            for agent in AGENTS:
                sub = df[df["algo"] == agent]
                ax.scatter(
                    sub[metric],
                    sub["success"],
                    label=agent,
                    color=COLORS[agent],
                    alpha=0.75,
                    s=45,
                    edgecolors="none",
                )
            x_all, y_all = df[metric].values, df["success"].values
            mask = np.isfinite(x_all) & np.isfinite(y_all)
            if mask.sum() > 3:
                m, b = np.polyfit(x_all[mask], y_all[mask], 1)
                xs = np.linspace(x_all[mask].min(), x_all[mask].max(), 60)
                ax.plot(xs, m * xs + b, "k--", lw=1, alpha=0.5)
            ax.set_xlabel(METRIC_LABELS[metric], fontsize=10)
            ax.set_ylabel("Success (final phase)", fontsize=10)
            ax.legend(fontsize=8)
            all_corr.extend(correlations(df, metric, env))

        plt.tight_layout()
        for fmt in ("png", "pdf"):
            out = OUTPUTS / f"rl7_diagnostic_scatter_{env}.{fmt}"
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  → {out.name}")
        plt.close()

    corr_df = pd.DataFrame(all_corr)
    corr_df.to_csv(OUTPUTS / "rl7_diagnostic_correlations.csv", index=False)
    print("  → rl7_diagnostic_correlations.csv")
    print("\n=== Pooled Pearson / Spearman (success vs metric) ===")
    print(
        corr_df[corr_df["method"] == "pooled"][
            ["env", "metric", "pearson", "spearman", "n"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
