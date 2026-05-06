"""
Reproduce the ESS table (ess.tex) from additional_stats_raw.csv.

The table shows per-(agent, actor, phase) ESS mean ± std and its Pearson
correlation with normalised return.  This corresponds to Table 3 in the paper
and requires the pre-computed `additional_stats_raw.csv` from the antmaze
basin-analysis pipeline (only antmaze-medium data is currently available there).

ESS values in that CSV use the formula (Σw)²/(N·Σw²) without max-scaling.
The correlation and formatting match the original e_ess.py pipeline.

For ESS from the raw parquet (last checkpoint only), see ess_per_batch.parquet
inside each ENVS/<env>/ folder — produced by extract_env.py.

Usage
-----
    cd RAW-ADV
    python ANALYSIS/compute_ess_table.py
    python ANALYSIS/compute_ess_table.py --out ENVS/antmaze-medium/ess_table.tex
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

_ROOT = Path(__file__).parent.parent  # RAW-ADV/
STATS_CSV = Path(
    "/Users/adityamohan/git/GCRL/plots/new-plots/"
    "advantage-dist-phases/basin-analysis/additional_stats_raw.csv"
)


def _pearson_r(g: pd.DataFrame, x: str, y: str) -> pd.Series:
    valid = g[[x, y]].dropna()
    if len(valid) < 3:
        return pd.Series({"r": float("nan"), "p": float("nan")})
    r, p = pearsonr(valid[x], valid[y])
    return pd.Series({"r": float(r), "p": float(p)})


def _tex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    col_fmt = "l" + "r" * len(df.columns)
    header = " & ".join(df.columns) + " \\midrule\n"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for v in row:
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        rows.append(" & ".join(vals) + " \\")
    body = "\n".join(rows)
    return (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\begin{{tabular}}{{{col_fmt}}}\n\\toprule\n"
        + header
        + body
        + "\n\\bottomrule\n\\end{tabular}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ESS table (tex) from additional_stats_raw.csv."
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=STATS_CSV,
        help=f"Path to additional_stats_raw.csv (default: {STATS_CSV})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "ENVS" / "antmaze-medium" / "ess_table.tex",
        help="Output .tex path",
    )
    args = parser.parse_args()

    if not args.stats.exists():
        raise FileNotFoundError(
            f"Stats CSV not found: {args.stats}\n"
            "This file is produced by ANALYSIS/basin-analysis/additional_plots.py "
            "and is only available for antmaze-medium."
        )

    df = pd.read_csv(args.stats)

    # Normalise return per (algo, env, phase) — match e_ess.py
    grp_max = df.groupby(["algo", "env", "phase"])["success"].transform("max")
    df["return_normalized"] = df["success"] / grp_max.replace(0, np.nan)
    df["actor"] = "advantage/actor"

    ess_stats = (
        df.groupby(["algo", "actor", "phase"])["ess"].agg(["mean", "std"]).reset_index()
    )
    corr = (
        df.groupby(["algo", "actor", "phase"])
        .apply(_pearson_r, x="ess", y="return_normalized", include_groups=False)
        .reset_index()
    )

    combined = ess_stats.merge(corr, on=["algo", "actor", "phase"])
    combined["algo"] = combined["algo"].str.lower()
    combined["ESS"] = (
        combined["mean"].map("{:.3f}".format)
        + r" \$\pm\$ "
        + combined["std"].map("{:.3f}".format)
    )
    combined = combined.rename(
        columns={
            "algo": "Agent",
            "actor": "Actor",
            "phase": "Phase",
            "r": "Corr",
            "p": "p-value",
        }
    )
    result = combined[
        ["Agent", "Actor", "Phase", "ESS", "Corr", "p-value"]
    ].sort_values(["Agent", "Phase"])

    print(result.to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        _tex_table(
            result,
            caption="Effective sample size statistics with correlation between ESS "
            "and normalized return per agent, actor, and phase.",
            label="tab:ess",
        )
    )
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
