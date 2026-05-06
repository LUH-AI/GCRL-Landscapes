# ANALYSIS — Scripts and Outputs

Scripts for extracting, processing, and analysing the advantage data in `ENVS/`.

```
ANALYSIS/
├── extract_env.py                  ← primary extraction pipeline
├── compute_afr_metrics.py          ← standalone AFR metrics (alternative to extract_env)
├── compute_ess_table.py            ← reproduce the ESS table from the paper
├── extract_advantages.py           ← earlier single-env version (superseded by extract_env.py)
├── advantages/                     ← zip-file pipeline (uses gcrl_landscapes package)
│   ├── _common.py
│   ├── a_distributions.py … h_oracle.py
├── aggregates/                     ← training-log fractions (frac_neg / frac_mid / frac_clip)
├── advantage-trustworthiness/      ← stability, α-sensitivity, return correlation plots
├── basin-analysis/                 ← regime maps, GCIQL basin analysis
└── advantage-pred/                 ← advantage prediction experiments
```

---

## `extract_env.py` — full extraction pipeline

Runs all three extraction steps on one environment's `advantages.parquet` and writes results into `ENVS/<env>/`. Takes ~15–20 min per environment (streaming, ~4 GB).

```bash
cd RAW-ADV
python ANALYSIS/extract_env.py --env antmaze-medium
python ANALYSIS/extract_env.py --env antmaze-large
python ANALYSIS/extract_env.py --env cube
```

**Step 1 — summary + histograms**
Streams the full parquet, aggregates per (agent, actor, phase, seed, configuration), writes:
- `advantages_summary.parquet` — distribution stats (mean, std, percentiles)
- `advantages_histograms.npz` — 200-bin histograms per group

**Step 2 — AFR + ranking metrics**
Reconstructs each 256×256 advantage matrix from its row group, computes:
- `advantage_matrix_metrics.parquet` — fr\_auc, gap, MRR, prec@1, prec@5, extractability index

**Step 3 — ESS from AWR weights**
Uses per-config `alpha` values from raw log JSON files to compute normalised ESS on the diagonal advantages (matched pairs only):
- `ess_per_batch.parquet` — one ESS value per row group

> **Note:** Step 3 requires config JSONs under
> `plots/new-plots/raw-data/logs-antmaze/<AGENT>_<env>_64c_lr-alpha/configurations/`.
> Cube has no config JSONs so ESS is skipped there.

**Alpha lookup path** (hardcoded in `extract_env.py`):
```python
LOGS_BASE = Path(".../plots/new-plots/raw-data/logs-antmaze")
```
Update this constant if logs move.

---

## `compute_afr_metrics.py` — standalone AFR metrics

Earlier, standalone version of the AFR step. Accepts explicit `--input` / `--output` paths and optionally filters by batch index.

```bash
python ANALYSIS/compute_afr_metrics.py \
    --input  ENVS/antmaze-medium/advantages.parquet \
    --output ENVS/antmaze-medium/afr_metrics.csv

# Only batches 0–4
python ANALYSIS/compute_afr_metrics.py \
    --input ENVS/antmaze-large/advantages.parquet \
    --batches 0-4 \
    --no-aggregate
```

Outputs one row per (checkpoint, actor) with `fr_auc`, `gap_mean`, `gap_std`, `extractability_index`, `adv_plus_mean`, `adv_minus_mean`.

---

## `compute_ess_table.py` — reproduce the ESS table

Produces a LaTeX table matching the paper's `tab:ess` in two ways:

**A) From `advantages.parquet`** — ESS from diagonal AWR weights (one phase per agent, corresponds to Phase 4 in the paper).

**B) From `additional_stats_raw.csv`** — full 4-phase table using pre-computed per-group ESS values from the antmaze training logs.

```bash
cd ENVS/antmaze-medium
python ../../ANALYSIS/compute_ess_table.py
# writes: ess_per_batch.parquet, ess_from_csv.tex
```

> This script has hardcoded paths for antmaze-medium. For other environments, update `STATS_CSV` and `PARQUET` at the top of the file.

**ESS formula** (matches `_common.py`):
```python
w = clip(exp(alpha * adv), 1e-9, 100)
w_scaled = w / w.max()
ESS = (sum(w_scaled)^2) / (N * sum(w_scaled^2))
```

---

## `advantages/` — zip-file pipeline

Scripts `a_` through `h_` operate on zip archives of training logs (loaded via the `gcrl_landscapes` package) rather than on `advantages.parquet` directly. They produce the full multi-phase analysis seen in the paper.

```bash
cd ANALYSIS/advantages
python e_ess.py --zipfiles /path/to/logs.zip
```

`_common.py` provides shared helpers: `parse_args`, `build_adv_df`, `_ess`, `_pearson_r`.

| Script | Output |
|---|---|
| `a_distributions.py` | Advantage distribution plots per phase |
| `b_stats.py` | Summary statistics table |
| `c_weights.py` | AWR weight mass plots |
| `d_correlation.py` | Advantage–return correlation |
| `e_ess.py` | ESS table (`ess.tex`) + correlation with normalised return |
| `g_spearman.py` | Spearman rank correlation analysis |
| `h_oracle.py` | Oracle advantage comparisons |

---

## `aggregates/compute_and_plot.py`

Reads raw training log CSVs from `plots/new-plots/raw-data/logs-antmaze/` and produces stacked-bar plots of advantage fractions (negative / active / clipped) per agent and phase.

```bash
python ANALYSIS/aggregates/compute_and_plot.py
# reads from: plots/new-plots/raw-data/logs-antmaze/
# writes to:  plots/new-plots/advantage-dist-phases/
```

---

## `advantage-trustworthiness/stability_sensitivity_correlation.py`

Three analyses from pre-computed CSVs in `plots/new-plots/advantage-dist-phases/basin-analysis/`:

1. **Distribution stability** — Wasserstein-1 between consecutive-phase z-distributions
2. **α-sensitivity** — how frac\_sat shifts when α is scaled
3. **Advantage→return correlation** — Spearman correlation between critic geometry (ESS, p⁺, W\_mid) and success rate

```bash
python ANALYSIS/advantage-trustworthiness/stability_sensitivity_correlation.py
```

---

## `basin-analysis/additional_plots.py`

Reads raw training log CSVs and computes per-(agent, config, seed, phase):
- ESS, p⁺, saturation, W\_neg / W\_mid / W\_sat
- Success rate from eval logs
- z-quantiles for stability analysis

Writes `additional_stats_raw.csv` and `z_quantiles_raw.csv` to `plots/new-plots/advantage-dist-phases/basin-analysis/`.

```bash
python ANALYSIS/basin-analysis/additional_plots.py
```
