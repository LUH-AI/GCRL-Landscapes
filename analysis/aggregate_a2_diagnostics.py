#!/usr/bin/env python3
"""A2 Step 1 — do the advantage diagnostics respond to a value-side intervention?

7VBe Q5 asked for exactly this test: take an intervention known to change value
quality (n-step returns), and check whether FR-AUC / MRR move with it. This
script joins the per-checkpoint diagnostics computed by compute_afr_metrics.py
against the per-configuration success rates of the same arms, and reports the
paired change per configuration between n=1 and n in {3, 5}.

Conventions follow the paper pipeline exactly:
  - final phase only;
  - seeds aggregated per configuration with trim_mean(0.25) (the paper's IQM);
  - the n=1 baseline is restricted to seeds 0-2 so it is compared at the same
    3-seed protocol as the n-step arms, never against the published 5-seed
    tables;
  - bootstrap CIs over configurations, 2000 draws, percentile 95%.

Inputs:  outputs-a2/afr_gcivl-n{1,3,5}-{cube,antmaze-large}.csv
         PAPER/rebuttal/results/R3_nstep_per_config.csv  (success per config)
Outputs: PAPER/rebuttal/additional-exp/results/A2/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, trim_mean

REPO = Path(__file__).resolve().parent.parent
AFR_DIR = REPO / "outputs-a2"
# Seed-matched success: the posted R3_nstep_per_config.csv aggregates the n=1
# baseline over 5 seeds while the n-step arms have 3, which would put a
# protocol change inside every reported Delta. Regenerate it with
#   compare_nstep_breadth.py --match-seeds --dump-per-config
# so all horizons are 3-seed trimmed centres.
SUCCESS_CSV = REPO / "outputs-a2/nstep_per_config_3seed.csv"
OUT_DIR = REPO / "PAPER/rebuttal/additional-exp/results/A2"

# (env key used in filenames, dataset name in the success table)
ENVS = {
    "cube": "cube-single-play-v0",
    "antmaze-large": "antmaze-large-navigate-v0",
}
HORIZONS = [1, 3, 5]
SEEDS = [0, 1, 2]
N_BOOT = 2000
RNG = np.random.default_rng(0)

# Columns compute_afr_metrics.py emits (mean across validation batches; with a
# single batch the *_std columns are NaN by construction).
DIAG_COLS = {
    "fr_auc_mean": "fr_auc",
    "gap_mean_mean": "gap",
    "mrr_mean": "mrr",
    "saturation_mass_mean": "saturation",
    "extractability_index_mean": "extractability",
}

# Below this max success a cell is a floor: relative orderings inside it are
# noise, so no correlation is quoted from it (standing provenance rule).
FLOOR_MAX_SUCCESS = 0.10


def load_diagnostics() -> pd.DataFrame:
    """Per (env, n, configuration) diagnostics, seed-aggregated by trimmed centre."""
    frames = []
    for env_key in ENVS:
        for n in HORIZONS:
            path = AFR_DIR / f"afr_gcivl-n{n}-{env_key}.csv"
            if not path.is_file():
                raise SystemExit(f"missing diagnostics file: {path}")
            df = pd.read_csv(path)
            # The n=1 baseline parquet holds all four agents and five seeds.
            df = df[df["agent"] == "GCIVL"]
            df = df[df["seed"].astype(int).isin(SEEDS)]
            if df.empty:
                raise SystemExit(f"no GCIVL/seed-{SEEDS} rows in {path}")
            df = df.rename(columns=DIAG_COLS)
            df["n_step"] = n
            df["env"] = env_key
            df["configuration"] = df["configuration"].astype(int)
            frames.append(df[["env", "n_step", "configuration", "seed", *DIAG_COLS.values()]])

    raw = pd.concat(frames, ignore_index=True)
    agg = (
        raw.groupby(["env", "n_step", "configuration"])[list(DIAG_COLS.values())]
        .agg(lambda v: trim_mean(v, 0.25))
        .reset_index()
    )
    agg["n_seeds"] = (
        raw.groupby(["env", "n_step", "configuration"])["seed"].nunique().values
    )
    return agg


def load_success() -> pd.DataFrame:
    df = pd.read_csv(SUCCESS_CSV)
    df = df[df["agent"] == "GCIVL"]
    inverse = {v: k for k, v in ENVS.items()}
    df = df[df["dataset"].isin(inverse)]
    df["env"] = df["dataset"].map(inverse)
    return df[["env", "n_step", "configuration", "success", "lr", "alpha"]]


def boot_ci(values: np.ndarray, statistic) -> tuple[float, float]:
    if len(values) < 3:
        return (float("nan"), float("nan"))
    draws = [
        statistic(RNG.choice(values, size=len(values), replace=True))
        for _ in range(N_BOOT)
    ]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def boot_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Spearman rho with a bootstrap CI over configurations (paired resampling)."""
    rho = float(spearmanr(x, y).statistic)
    idx = np.arange(len(x))
    draws = []
    for _ in range(N_BOOT):
        take = RNG.choice(idx, size=len(idx), replace=True)
        if len(np.unique(x[take])) < 2 or len(np.unique(y[take])) < 2:
            continue
        draws.append(spearmanr(x[take], y[take]).statistic)
    if not draws:
        return rho, float("nan"), float("nan")
    return rho, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    diag = load_diagnostics()
    success = load_success()
    merged = diag.merge(success, on=["env", "n_step", "configuration"], how="inner")

    expected = len(ENVS) * len(HORIZONS) * 64
    if len(merged) != expected:
        print(
            f"WARNING: {len(merged)} joined rows, expected {expected} "
            "— some (env, n, config) cells are missing on one side of the join"
        )

    per_config = args.out_dir / "A2_nstep_diagnostics.csv"
    merged.sort_values(["env", "n_step", "configuration"]).to_csv(per_config, index=False)
    print(f"wrote {per_config}  ({len(merged)} rows)")

    # ---- per-arm summary -------------------------------------------------
    rows = []
    for (env, n), grp in merged.groupby(["env", "n_step"]):
        max_success = grp["success"].max()
        on_floor = max_success < FLOOR_MAX_SUCCESS
        rho, lo, hi = (
            boot_spearman(grp["fr_auc"].to_numpy(), grp["success"].to_numpy())
            if not on_floor
            else (float("nan"),) * 3
        )
        mrr_rho, mrr_lo, mrr_hi = (
            boot_spearman(grp["mrr"].to_numpy(), grp["success"].to_numpy())
            if not on_floor
            else (float("nan"),) * 3
        )
        row = {
            "env": env,
            "n_step": n,
            "n_configs": len(grp),
            "max_success": max_success,
            "mean_success": grp["success"].mean(),
            "on_floor": on_floor,
            "spearman_frauc_success": rho,
            "spearman_frauc_success_lo": lo,
            "spearman_frauc_success_hi": hi,
            "spearman_mrr_success": mrr_rho,
            "spearman_mrr_success_lo": mrr_lo,
            "spearman_mrr_success_hi": mrr_hi,
        }
        for col in DIAG_COLS.values():
            row[f"{col}_mean"] = grp[col].mean()
            lo_c, hi_c = boot_ci(grp[col].to_numpy(), np.mean)
            row[f"{col}_lo"], row[f"{col}_hi"] = lo_c, hi_c
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["env", "n_step"])
    summary_path = args.out_dir / "A2_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")

    # ---- paired change vs the n=1 baseline -------------------------------
    delta_rows = []
    base = merged[merged["n_step"] == 1].set_index(["env", "configuration"])
    for n in [h for h in HORIZONS if h != 1]:
        arm = merged[merged["n_step"] == n].set_index(["env", "configuration"])
        common = base.index.intersection(arm.index)
        for env in ENVS:
            sel = [c for c in common if c[0] == env]
            if not sel:
                continue
            b, a = base.loc[sel], arm.loc[sel]
            d_success = (a["success"] - b["success"]).to_numpy()
            entry = {
                "env": env,
                "n_step": n,
                "n_configs": len(sel),
                "mean_delta_success": float(np.mean(d_success)),
                "max_success_n1": float(b["success"].max()),
                "max_success_n": float(a["success"].max()),
                "on_floor": bool(
                    max(b["success"].max(), a["success"].max()) < FLOOR_MAX_SUCCESS
                ),
            }
            for col in ("fr_auc", "mrr", "gap"):
                d_diag = (a[col] - b[col]).to_numpy()
                entry[f"mean_delta_{col}"] = float(np.mean(d_diag))
                if entry["on_floor"]:
                    entry[f"spearman_d{col}_dsuccess"] = float("nan")
                    entry[f"spearman_d{col}_dsuccess_lo"] = float("nan")
                    entry[f"spearman_d{col}_dsuccess_hi"] = float("nan")
                else:
                    r, lo, hi = boot_spearman(d_diag, d_success)
                    entry[f"spearman_d{col}_dsuccess"] = r
                    entry[f"spearman_d{col}_dsuccess_lo"] = lo
                    entry[f"spearman_d{col}_dsuccess_hi"] = hi
            delta_rows.append(entry)

    deltas = pd.DataFrame(delta_rows)
    delta_path = args.out_dir / "A2_nstep_deltas.csv"
    deltas.to_csv(delta_path, index=False)
    print(f"wrote {delta_path}")

    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n=== per-arm summary ===")
    print(
        summary[
            [
                "env",
                "n_step",
                "max_success",
                "mean_success",
                "fr_auc_mean",
                "mrr_mean",
                "gap_mean",
                "spearman_frauc_success",
            ]
        ].to_string(index=False)
    )
    print("\n=== paired change vs n=1 (same 64 configurations, 3 seeds) ===")
    print(
        deltas[
            [
                "env",
                "n_step",
                "mean_delta_success",
                "mean_delta_fr_auc",
                "mean_delta_mrr",
                "spearman_dfr_auc_dsuccess",
                "spearman_dfr_auc_dsuccess_lo",
                "spearman_dfr_auc_dsuccess_hi",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
