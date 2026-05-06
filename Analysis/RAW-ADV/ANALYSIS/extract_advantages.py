"""
Efficient extraction of advantages.parquet (838M rows, 4.1GB).

Structure: 4 agents × 4 phases × 5 seeds × 2 configurations = 160 groups,
~5.24M advantages per group.

Outputs
-------
advantages_summary.parquet  — one row per group, statistics + ESS
advantages_histograms.npz   — per-group histogram arrays (optional, set SAVE_HISTS)
"""

import pyarrow.dataset as ds
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

PARQUET = Path("advantages.parquet")
OUT_SUMMARY = Path("advantages_summary.parquet")
OUT_HISTS = Path("advantages_histograms.npz")

KEY_COLS = ["agent", "actor", "phase", "seed", "configuration"]
VAL_COLS = ["advantage", "is_positive"]

SAVE_HISTS = True  # set False to skip the histogram file
N_HIST_BINS = 200
BATCH_SIZE = 2**17  # 131072 rows — comfortably in RAM, not too many batches


def ess_fraction(adv: np.ndarray) -> float:
    """
    ESS/N via the Kish approximation using softmax weights over advantages.
    Returns a value in (0, 1]: 1 = uniform, approaching 0 = degenerate.
    """
    logw = adv.astype(np.float64)
    logw -= logw.max()  # numerical stability
    w = np.exp(logw)
    w /= w.sum()
    return float(1.0 / (len(w) * (w**2).sum()))


def main():
    dataset = ds.dataset(PARQUET, format="parquet")
    print(f"Columns: {dataset.schema.names}")

    # ------------------------------------------------------------------ #
    # Streaming aggregation — one batch at a time, O(groups) peak memory  #
    # ------------------------------------------------------------------ #
    class GroupAccum:
        __slots__ = ("adv_chunks", "n_pos", "n_total", "hist_counts", "bin_edges")

        def __init__(self):
            self.adv_chunks = []
            self.n_pos = 0
            self.n_total = 0
            self.hist_counts = None
            self.bin_edges = None

    groups: dict[tuple, GroupAccum] = defaultdict(GroupAccum)

    scanner = dataset.scanner(
        columns=KEY_COLS + VAL_COLS,
        batch_size=BATCH_SIZE,
    )

    total_rows = 0
    report_every = BATCH_SIZE * 32

    for batch in scanner.to_batches():
        df = batch.to_pandas()
        total_rows += len(df)

        for key, grp in df.groupby(KEY_COLS, sort=False):
            g = groups[key]
            adv = grp["advantage"].to_numpy(dtype=np.float32)
            g.adv_chunks.append(adv)
            g.n_pos += int(grp["is_positive"].sum())
            g.n_total += len(grp)

        if total_rows % report_every < BATCH_SIZE:
            print(f"  {total_rows:>12,} / 838,860,800 rows processed …")

    print(f"\nDone. {total_rows:,} rows across {len(groups)} groups.\n")

    # ------------------------------------------------------------------ #
    # Build summary + histograms                                          #
    # ------------------------------------------------------------------ #
    summary_rows = []
    hist_data = {}

    for key, g in groups.items():
        adv = np.concatenate(g.adv_chunks)

        ess = ess_fraction(adv)
        mean = float(adv.mean())
        std = float(adv.std())
        pcts = np.percentile(adv, [5, 25, 50, 75, 95]).tolist()

        row = dict(zip(KEY_COLS, key))
        row.update(
            dict(
                n=g.n_total,
                n_positive=g.n_pos,
                pos_fraction=g.n_pos / g.n_total,
                ess=ess,
                adv_mean=mean,
                adv_std=std,
                adv_p5=pcts[0],
                adv_p25=pcts[1],
                adv_median=pcts[2],
                adv_p75=pcts[3],
                adv_p95=pcts[4],
            )
        )
        summary_rows.append(row)

        if SAVE_HISTS:
            counts, edges = np.histogram(adv, bins=N_HIST_BINS)
            tag = "_".join(str(k) for k in key)
            hist_data[f"{tag}_counts"] = counts.astype(np.int32)
            hist_data[f"{tag}_edges"] = edges.astype(np.float32)

    summary = pd.DataFrame(summary_rows).sort_values(KEY_COLS).reset_index(drop=True)

    summary.to_parquet(OUT_SUMMARY, index=False)
    print(f"Summary → {OUT_SUMMARY}  ({len(summary)} rows)")
    print(
        summary[
            [
                "agent",
                "phase",
                "seed",
                "configuration",
                "n",
                "pos_fraction",
                "ess",
                "adv_mean",
                "adv_std",
            ]
        ].to_string()
    )

    if SAVE_HISTS:
        np.savez_compressed(OUT_HISTS, **hist_data)
        print(f"\nHistograms → {OUT_HISTS}")


if __name__ == "__main__":
    main()
