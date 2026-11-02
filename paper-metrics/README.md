# paper-metrics

A self-contained, one-stop reference for reproducing every numeric table in the
current draft of the GCRL-Landscapes paper. Each script in `scripts/` reads
data from `data/` and writes a CSV to `outputs/`. A single bash driver,
`reproduce_all.sh`, regenerates everything in dependency order.

The numbers in this directory are produced from the **fixed-batch** evaluation
zips (one per environment), the same artefacts the published tables are based
on. Reproducing the paper does *not* require the cluster — the data under
`data/` is the canonical input. That input is **not** shipped with this
repository, however; see *Data availability* below before running anything.

---

## Table → script mapping

| Paper LaTeX label                          | Section            | Script                                | Output CSV                                  |
| ------------------------------------------ | ------------------ | ------------------------------------- | ------------------------------------------- |
| `tab:final_landscape_ci`                   | Main results       | `t1_final_landscape_ci.py`            | `outputs/t1_final_landscape_ci.csv`         |
| `tab:compact_diagnostics`                  | Main results       | `t2_compact_diagnostics.py`           | `outputs/t2_compact_diagnostics.csv`        |
| `tab:antmaze_ess_phase4`, `*_full`         | Main results       | `t3_antmaze_ess.py`                   | `outputs/t3_antmaze_ess.csv`                |
| `tab:adv_corr`                             | Main results       | `t4_adv_corr.py`                      | `outputs/t4_adv_corr.csv`                   |
| `tab:seed_variance_summary`                | Robustness appx.   | `a2_seed_variance.py`                 | `outputs/a2_seed_variance.csv`              |
| `tab:phase_mobility`                       | Robustness appx.   | `a3_phase_mobility.py`                | `outputs/a3_phase_mobility.csv`             |
| `tab:subsample_stability_close_pairs`      | Robustness appx.   | `a4_subsample_stability.py`           | `outputs/a4_subsample_stability.csv`        |
| `tab:wmax_sensitivity`                     | Robustness appx.   | `a5_wmax_sensitivity.py`              | `outputs/a5_wmax_sensitivity.csv`           |
| `tab:adv_norm_sensitivity`                 | Robustness appx.   | `a6_adv_norm_sensitivity.py`          | `outputs/a6_adv_norm_sensitivity.csv`       |
| `tab:sensitivity_summary`                  | Robustness appx.   | `a7_sensitivity_summary.py`           | `outputs/a7_sensitivity_summary.csv`        |

`scripts/_common.py` is a shared module (paths, alpha lookup, eval-stats
loader) — not a table.

---

## Layout

```
paper-metrics/
├── .gitignore                      ignores the large inputs and generated CSVs
├── README.md                       this file
├── PROVENANCE.md                   how every artefact is derived
├── reproduce_all.sh                run every script, write outputs/*.csv
├── data/
│   ├── manifest.json               campaign registry (zips, eval stats, parquets)
│   ├── zips/                       fixed-batch eval zips, one per env (contents not checked in)
│   ├── parquets/<env>/             advantages.parquet per env (contents not checked in)
│   └── eval_stats/                 <env>.csv success tables (contents not checked in)
├── scripts/
│   ├── _common.py                  shared helpers
│   ├── build_eval_stats.py         helper: builds data/eval_stats/<env>.csv from a zip
│   ├── t1_final_landscape_ci.py
│   ├── t2_compact_diagnostics.py
│   ├── t3_antmaze_ess.py
│   ├── t4_adv_corr.py
│   ├── a2_seed_variance.py
│   ├── a3_phase_mobility.py
│   ├── a4_subsample_stability.py
│   ├── a5_wmax_sensitivity.py
│   ├── a6_adv_norm_sensitivity.py
│   ├── a7_sensitivity_summary.py
│   ├── rl1_abs_breadth.py
│   ├── rl2_restricted_domain.py
│   ├── rl3_realized_selectivity.py
│   ├── rl4_eps_sensitivity.py
│   ├── rl5_phase_mobility_null.py
│   ├── rl6_filtered_fr.py
│   ├── rl7_diagnostic_scatter_antmaze.py
│   ├── rl8_lagged_partial_ess.py
│   ├── rl9_new_result_figures.py   figure-only, not called by the driver
│   └── rl_supplementary_figures.py additional-analysis figures
└── outputs/                        regenerated CSVs land here (created at runtime)
```

The `data/` subdirectories ship empty (only `.gitkeep` files): the four zips
(~6 GB total), four parquets (~17 GB total), and four eval-stats CSVs are not
checked in — see *Data availability* below.

---

## Data availability

The **fixed-batch zips and the advantage parquets are not included** in this
repository (they total ~23 GB). Steps 1–2 below expect them under
`data/zips/` and `data/parquets/` with the filenames listed in
`data/manifest.json`. The commands further down only *copy* those files from a
location that you must supply yourself.

Only the derived `data/eval_stats/<env>.csv` tables are small (~400 KB each);
step 3 rebuilds them from the zips.

---

## Fetching the data from the cluster

Everything under `data/` originates from one project directory of the training
campaign, which is not part of this repository. If you have access to such a
copy, fetch the inputs as below; total transfer is ~23 GB (≈6 GB zips + ~17 GB
parquets). Otherwise see *Data availability* above.

The commands below use `user@cluster` and `${CLUSTER_PROJECT_DIR}` as
placeholders — set them to the host and directory that hold your copy of the
campaign artefacts. `${CLUSTER_SCRATCH_DIR}` (mentioned further down) is an
open placeholder as well.

**Canonical location on the cluster (`${CLUSTER_PROJECT_DIR}`):**

```
${CLUSTER_PROJECT_DIR}/
├── 2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip          ( 696 MB)
├── 2026-04-27-logs-advantage-cube-single-fixed-batch.zip             ( 687 MB)
├── 2026-04-30-logs-advantage-antmaze-large-single-fixed-batch.zip    (4.2 GB)
├── 2026-05-03-logs-advantage-scene-single-fixed-batch.zip            ( 691 MB)
└── logs-advantage-<env>-fixed-batch/                                  (unzipped trees)
    ├── advantages.parquet                                             (~4.2 GB each)
    ├── checkpoints.csv
    └── <AGENT>_<env>_64c_lr-alpha/                                    (run_logs for eval_stats)
```

You need: 4 zips, 4 `advantages.parquet`, and per-env `eval_stats/<env>.csv`
(rebuilt locally from the unzipped trees).

**Step 1 — fetch the four fixed-batch zips into `data/zips/`:**

```bash
cd paper-metrics
mkdir -p data/zips
rsync -avhP \
  user@cluster:${CLUSTER_PROJECT_DIR}/2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip \
  user@cluster:${CLUSTER_PROJECT_DIR}/2026-04-27-logs-advantage-cube-single-fixed-batch.zip \
  user@cluster:${CLUSTER_PROJECT_DIR}/2026-04-30-logs-advantage-antmaze-large-single-fixed-batch.zip \
  user@cluster:${CLUSTER_PROJECT_DIR}/2026-05-03-logs-advantage-scene-single-fixed-batch.zip \
  data/zips/
```

The filenames must match `_common.py::ZIP_NAMES` exactly — don't rename.

**Step 2 — fetch the four `advantages.parquet` into `data/parquets/<env>/`:**

```bash
mkdir -p data/parquets/{antmaze-medium,antmaze-large,cube,scene}

rsync -avhP user@cluster:${CLUSTER_PROJECT_DIR}/logs-advantage-antmaze-medium-fixed-batch/advantages.parquet      data/parquets/antmaze-medium/
rsync -avhP user@cluster:${CLUSTER_PROJECT_DIR}/logs-advantage-antmaze-large-single-fixed-batch/advantages.parquet data/parquets/antmaze-large/
rsync -avhP user@cluster:${CLUSTER_PROJECT_DIR}/logs-advantage-cube-single-fixed-batch/advantages.parquet         data/parquets/cube/
rsync -avhP user@cluster:${CLUSTER_PROJECT_DIR}/logs-advantage-scene-single-fixed-batch/advantages.parquet        data/parquets/scene/
```

These four parquets are the cross-goal 256×256 advantage matrices used by
T2 and A4–A6. Each is ~4 GB.

**Step 3 — build `eval_stats/<env>.csv`.**

The eval-stats CSVs are derived from the per-seed `eval_log.csv` files
inside each unzipped tree. Use the vendored `scripts/build_eval_stats.py`:

```bash
mkdir -p data/eval_stats
ENVS=(
  "antmaze-medium  antmaze-medium-navigate-v0  2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip"
  "antmaze-large   antmaze-large-navigate-v0   2026-04-30-logs-advantage-antmaze-large-single-fixed-batch.zip"
  "cube            cube-single-play-v0         2026-04-27-logs-advantage-cube-single-fixed-batch.zip"
  "scene           scene-play-v0               2026-05-03-logs-advantage-scene-single-fixed-batch.zip"
)
for row in "${ENVS[@]}"; do
  read -r env tag zip <<<"$row"
  tmp=$(mktemp -d) && unzip -q "data/zips/$zip" -d "$tmp"
  python scripts/build_eval_stats.py \
    --logs "$tmp" --env "$tag" --out "data/eval_stats/${env}.csv"
  rm -rf "$tmp"
done
```

Each CSV is ~400 KB. If you already have a pre-built copy on another host,
`rsync` it directly into `data/eval_stats/` instead.

**Step 4 — verify everything resolves.**

```bash
ls -lL data/zips/                    # 4 entries, all real files (no broken links)
ls -lL data/parquets/*/advantages.parquet
ls -l  data/eval_stats/*.csv         # 4 files, each ~400 KB
```

**(Optional) Use symlinks instead of copies.** If the zips or parquets
already exist elsewhere on the same host, point at them with symlinks
rather than re-rsync'ing:

```bash
ln -s /abs/path/to/2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip \
      data/zips/2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip
ln -s /abs/path/to/advantages.parquet \
      data/parquets/cube/advantages.parquet
```

The scripts read through symlinks transparently. Only the *names* under
`data/zips/` matter (they must match `_common.py::ZIP_NAMES`).

---

## How to use

```bash
# from the project root
cd paper-metrics

# reproduce everything (≈10 min; the RF fits in a7 dominate)
./reproduce_all.sh          # or: ./reproduce_all.sh all

# only the main-paper tables (t1..t4)
./reproduce_all.sh main

# only the robustness-appendix tables (a2..a7)
./reproduce_all.sh appendix

# additional analyses (rl1..rl8 plus the supplementary figures)
./reproduce_all.sh rebuttal

# run an individual script
.venv/bin/python scripts/t1_final_landscape_ci.py
```

`reproduce_all.sh` accepts exactly four modes: `all` (the default), `main`,
`appendix` and `rebuttal`. The script names the last one `rebuttal`; it covers
the additional analyses rather than the main tables and runs `rl1`–`rl8` plus
`rl_supplementary_figures.py`. `rl9` is figure-only and must be called
directly.

The driver picks up `PAPER_METRICS_PYTHON` if set, else uses `python3` from
`PATH`. Point it at a venv that has the `gcrl_landscapes` package installed.

After a run, `outputs/` looks like:

```
outputs/
├── t1_final_landscape_ci.csv
├── t2_compact_diagnostics.csv
├── ...
└── a7_sensitivity_summary.csv
```

Each CSV is a tidy long/wide table whose rows match the LaTeX table; the
script's docstring tells you which columns map to which paper cells.

---

## Dependencies

The scripts import from the project venv. From the repo root:

```bash
uv pip install -e .
uv pip install pyarrow scikit-learn scipy pandas numpy
# transitive (already pulled in by the editable install):
#   flax, ml_collections, ogbench, tomli/tomllib, tqdm
```

`t3` and `t4` use `gcrl_landscapes.evaluation.tabular.compute_merged_df`,
which transitively imports `flax` / `ml_collections` / `ogbench`. If any of
those are missing, the t3 / t4 / a2 / a3 / a4 scripts will fail at import
time.

---

## Conventions used inside the scripts

These are the exact knobs the paper tables use; each script comments on them
in its docstring. Listed here as a quick reference:

- **Seed reduction:** IQM via `scipy.stats.trim_mean(v, 0.25)` (interquartile
  mean — trim 25% from each tail). Bootstrap CIs are over configs.
- **AWR weights:** `w = min(exp(α · A), w_max)` with `w_max = 100` everywhere
  except `a5_wmax_sensitivity.py`, which sweeps `w_max ∈ {20, 50, 100, 200}`.
- **ESS:** Kish formula, `ESS = (Σw)² / (B · Σw²)`, on the *scaled* weights
  `w / max(w)` (so it is invariant to a positive multiplicative constant).
- **Top-5 mass:** `k = ceil(0.05 · 256) = 13` — Malte's convention. (Aditya's
  earlier scripts used `int(0.05·256)=12`; the paper uses 13.)
- **MRR:** argsort-based, the formulation in `g_spearman.py`. (The
  count-based variant `(M ≥ A_plus[:, None]).sum(axis=1)` differs only on
  ties and would change a couple of QRL cells — do not switch.)
- **AntMaze ESS:** computed via `compute_merged_df(...)` so that landscape
  filters (`min_seeds=3`, `min_configurations`) match the rest of the paper.
- **Phase mobility:** Jaccard overlap of the top-10% configs across
  consecutive phases, plus centroid-drift in the (log lr, log α) plane.

---

## Provenance / "why does cell X match the paper exactly"

The fixed-batch zips are the artefacts that produced the paper. They are:

```
2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip
2026-04-27-logs-advantage-cube-single-fixed-batch.zip
2026-04-30-logs-advantage-antmaze-large-single-fixed-batch.zip
2026-05-03-logs-advantage-scene-single-fixed-batch.zip
```

`advantages.parquet` for each env was generated on the training cluster by
`analysis/generate_advantages.py` from the `params_*.pkl` checkpoints in the
unzipped training tree. The zips themselves contain NO checkpoints (verified:
zero `.pkl` entries), so the parquets are primary data that cannot be
regenerated offline. See `PROVENANCE.md` for the full derivation chain.

The eval-stats CSVs in `data/eval_stats/` are the per-step success-rate dumps
used by `t1` and `a7`; they are derived from the same zips.

If a CSV under `outputs/` disagrees with the paper:

1. Check that `data/zips/` and `data/parquets/<env>/` resolve to the
   fixed-batch versions (`ls -L data/zips/` and `ls -L
   data/parquets/<env>/`).
2. Check that `_common.py::ZIP_NAMES` still matches the filenames in
   `data/zips/`.
3. Check that `wmax`, top-k, and MRR conventions in the script match the
   ones in the bullet list above.

---

## Known issues

- `g_spearman.py --top-k` has a latent bug (filters `adv_data` but the
  analysis loop reads `merged_training_df` directly). `t4_adv_corr.py` works
  around this by not relying on `--top-k` filtering.

## Extending the benchmark to incoming results

Campaigns are declared in `data/manifest.json`; the paper's AWR fixed-batch
sweep is the `default_campaign`, and all constants in `scripts/_common.py`
derive from it, so existing tables never change. To onboard a new result
family (DDPG+BC, n-step TD, FQL, rejection sampling, ...):

1. Drop the campaign zip(s) in `data/zips/`. The expected layout is the
   same as the existing zips: `<AGENT>_<dataset>_<suffix>/` with
   `configurations/configuration_*.json` and
   `run_logs/configuration_*/phase_*/seed_*/eval_log.csv`.
2. Add a campaign entry to `data/manifest.json`: agents, dir suffix,
   per-env dataset tag / zip / eval_stats filename / parquet path.
3. Build its eval stats with a variant tag so rows are self-describing:

   ```bash
   python scripts/build_eval_stats.py \
       --logs /tmp/trees/<unzipped-tree> --env <dataset-tag> \
       --agents GCIQL,CRL,QRL --dir-suffix 32c_lr-alpha \
       --variant ddpgbc \
       --out data/eval_stats/<env>-ddpgbc.csv
   ```

4. Eval-stats-driven analyses (t1-style landscape tables, rl1/rl2/rl4/rl5)
   run on the new campaign via `load_all_eval_stats(campaign="<name>")` —
   no script edits.
5. Advantage-side analyses additionally need the campaign's
   `advantages.parquet`, generated on the cluster by
   `analysis/catalog_checkpoints.py` + `analysis/generate_advantages.py`
   against the checkpoint tree. Supported there: AWR and DDPG+BC variants
   of GCIQL/CRL/GCIVL/QRL/HIQL, MQE, n-step GCIQL/GCIVL (same class names),
   and FQL (GCIQL-identical critic). The parquet's `actor_loss` column
   distinguishes extractor families.

Semantic caution when comparing across campaigns: `alpha` is the AWR
temperature in the default campaign, a BC coefficient under DDPG+BC, and a
distillation weight under FQL — same column name, different quantity. Keep
the `variant` column in any concatenated table.
