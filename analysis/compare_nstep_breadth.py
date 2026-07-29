#!/usr/bin/env python3
"""Compare landscape breadth and peak across n-step variants.

Produces the table behind {RESULT-NSTEP}: for each (agent, environment, n),
the peak success, mean success, and the breadth measures the paper reports,
computed at the deepest phase available in each tree.

The n=1 baseline comes from the published fixed-batch campaign; n=3 and n=5
come from the rebuttal arms, which reuse the published (lr, alpha) design via
analysis/inject_reference_configs.py. Configuration i therefore denotes the
same hyperparameters in every arm, so the comparison is paired.

Conventions follow paper-metrics/README.md:
  - seed reduction by IQM, scipy.stats.trim_mean(v, 0.25)
  - rho_eps  = fraction of configs with seed-IQM success >= eps * max
  - rho_abs  = fraction of configs with seed-IQM success >= an absolute bar
Note the rebuttal arms use 3 seeds where the paper used 5; with 3 values
trim_mean(., 0.25) is effectively a trimmed centre, not the paper's 5-seed IQM.
That difference is stated rather than hidden — do not present these as
identical estimators.

Usage
-----
    python analysis/compare_nstep_breadth.py --out outputs/nstep_breadth.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

PUBLISHED_ROOT = Path("/project/NHWP25179/SSRL-Landscapes")
REBUTTAL_ROOT = Path("/bigwork/nhwpmoha/GCRL-Landscapes")

# (label, n_step, tree, experiment dir)
CELLS = [
    # --- GCIQL, antmaze-medium ---
    ("GCIQL", "antmaze-medium-navigate-v0", 1,
     PUBLISHED_ROOT / "logs-advantage-antmaze-medium-fixed-batch"
     / "GCIQL_antmaze-medium-navigate-v0_64c_lr-alpha"),
    ("GCIQL", "antmaze-medium-navigate-v0", 3,
     REBUTTAL_ROOT / "logs-rb-gciql-n3-antmaze-medium"
     / "GCIQL_antmaze-medium-navigate-v0_64c_lr-alpha"),
    # --- GCIVL, cube ---
    ("GCIVL", "cube-single-play-v0", 1,
     PUBLISHED_ROOT / "logs-advantage-cube-single-fixed-batch"
     / "GCIVL_cube-single-play-v0_64c_lr-alpha"),
    ("GCIVL", "cube-single-play-v0", 3,
     REBUTTAL_ROOT / "logs-rb-gcivl-n3-cube-single"
     / "GCIVL_cube-single-play-v0_64c_lr-alpha"),
    ("GCIVL", "cube-single-play-v0", 5,
     REBUTTAL_ROOT / "logs-rb-gcivl-n5-cube-single"
     / "GCIVL_cube-single-play-v0_64c_lr-alpha"),
    # --- GCIVL, antmaze-large ---
    ("GCIVL", "antmaze-large-navigate-v0", 1,
     PUBLISHED_ROOT / "logs-advantage-antmaze-large-single-fixed-batch"
     / "GCIVL_antmaze-large-navigate-v0_64c_lr-alpha"),
    ("GCIVL", "antmaze-large-navigate-v0", 3,
     REBUTTAL_ROOT / "logs-rb-gcivl-n3-antmaze-large"
     / "GCIVL_antmaze-large-navigate-v0_64c_lr-alpha"),
    ("GCIVL", "antmaze-large-navigate-v0", 5,
     REBUTTAL_ROOT / "logs-rb-gcivl-n5-antmaze-large"
     / "GCIVL_antmaze-large-navigate-v0_64c_lr-alpha"),
]


def load_cell(expdir: Path, max_seeds: int | None = None) -> pd.DataFrame | None:
    """Read (configuration, seed, phase, success, lr, alpha) from a run tree."""
    runs = list(expdir.glob("run_logs/configuration_*/phase_*/seed_*/eval_log.csv"))
    if not runs:
        return None

    rows = []
    for f in runs:
        m = re.search(
            r"configuration_(\d+)/phase_(\d+)/seed_(\d+)/eval_log\.csv", str(f)
        )
        if not m:
            continue
        cfg, phase, seed = (int(g) for g in m.groups())
        if max_seeds is not None and seed >= max_seeds:
            continue
        try:
            d = pd.read_csv(f)
            success = float(d["success"].iloc[-1])
        except Exception:
            continue
        rows.append(
            {"configuration": cfg, "phase": phase, "seed": seed, "success": success}
        )
    if not rows:
        return None

    df = pd.DataFrame(rows)
    # Hyperparameters, for reporting where the optimum sits.
    hp = {}
    for cf in (expdir / "configurations").glob("configuration_*.json"):
        c = json.loads(cf.read_text())
        hp[int(cf.stem.split("_")[1])] = (c.get("lr"), c.get("alpha"))
    df["lr"] = df["configuration"].map(lambda i: hp.get(i, (np.nan, np.nan))[0])
    df["alpha"] = df["configuration"].map(lambda i: hp.get(i, (np.nan, np.nan))[1])
    return df


def summarise(df: pd.DataFrame) -> dict:
    """Breadth and peak at the deepest phase present."""
    phase = df["phase"].max()
    d = df[df["phase"] == phase]
    per_cfg = (
        d.groupby("configuration")["success"]
        .apply(lambda v: trim_mean(v, 0.25) if len(v) >= 3 else v.mean())
        .rename("iqm")
        .reset_index()
    )
    s = per_cfg["iqm"].to_numpy()
    mx = float(s.max()) if len(s) else float("nan")
    out = {
        "phase": int(phase),
        "n_configs": int(per_cfg.shape[0]),
        "n_seeds": int(d.groupby("configuration")["seed"].nunique().max()),
        "max": round(mx, 4),
        "mean": round(float(s.mean()), 4),
    }
    for eps in (0.9, 0.8):
        out[f"rho_{eps}"] = (
            round(float((s >= eps * mx).mean()), 4) if mx > 0 else float("nan")
        )
    for bar in (0.25, 0.5):
        out[f"rho_abs_{bar}"] = round(float((s >= bar).mean()), 4)
    best = per_cfg.loc[per_cfg["iqm"].idxmax(), "configuration"] if len(per_cfg) else -1
    row = d[d["configuration"] == best]
    out["best_config"] = int(best)
    out["best_lr"] = float(row["lr"].iloc[0]) if len(row) else float("nan")
    out["best_alpha"] = float(row["alpha"].iloc[0]) if len(row) else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("outputs/nstep_breadth.csv"))
    ap.add_argument(
        "--match-seeds",
        action="store_true",
        help="Restrict the published n=1 baseline to 3 seeds, matching the arms.",
    )
    ap.add_argument(
        "--dump-per-config",
        type=Path,
        default=None,
        help="Also write per-configuration (lr, alpha, seed-reduced success) rows "
        "at the deepest phase, for plotting the landscape planes.",
    )
    args = ap.parse_args()

    per_config_rows = []
    records = []
    for agent, env, n, expdir in CELLS:
        if not expdir.is_dir():
            print(f"MISSING tree: {expdir}")
            continue
        df = load_cell(expdir, max_seeds=3 if (args.match_seeds and n == 1) else None)
        if df is None:
            print(f"NO RESULTS YET: {agent} {env} n={n}")
            continue
        rec = {"agent": agent, "dataset": env, "n_step": n, **summarise(df)}
        records.append(rec)
        if args.dump_per_config is not None:
            deepest = df[df["phase"] == df["phase"].max()]
            pc = (
                deepest.groupby(["configuration", "lr", "alpha"])["success"]
                .apply(lambda v: trim_mean(v, 0.25) if len(v) >= 3 else v.mean())
                .rename("success")
                .reset_index()
            )
            pc.insert(0, "n_step", n)
            pc.insert(0, "dataset", env)
            pc.insert(0, "agent", agent)
            pc["phase"] = int(deepest["phase"].max())
            per_config_rows.append(pc)
        print(
            f"{agent:6s} {env:28s} n={n}  phase={rec['phase']:>7d}  "
            f"max={rec['max']:.3f} mean={rec['mean']:.3f} "
            f"rho0.9={rec['rho_0.9']} rho_abs0.25={rec['rho_abs_0.25']} "
            f"({rec['n_configs']} cfg x {rec['n_seeds']} seeds)"
        )

    if not records:
        raise SystemExit("no cells produced results yet")

    out = pd.DataFrame(records).sort_values(["agent", "dataset", "n_step"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    if per_config_rows:
        pc = pd.concat(per_config_rows, ignore_index=True)
        args.dump_per_config.parent.mkdir(parents=True, exist_ok=True)
        pc.to_csv(args.dump_per_config, index=False)
        print(f"wrote {args.dump_per_config} ({len(pc)} rows)")
    print(
        out[
            ["agent", "dataset", "n_step", "phase", "max", "mean", "rho_0.9",
             "rho_abs_0.25", "n_configs", "n_seeds"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
