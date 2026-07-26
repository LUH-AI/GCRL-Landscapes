# paper-metrics

A self-contained, one-stop reference for reproducing every numeric table in the
current draft of the GCRL-Landscapes paper. Each script in `scripts/` reads
data from `data/` and writes a CSV to `outputs/`. A single bash driver,
`reproduce_all.sh`, regenerates everything in dependency order.

The numbers in this directory are produced from the **fixed-batch** evaluation
zips (one per environment), the same artefacts the published tables are based
on. Reproducing the paper does *not* require the cluster — the data here is
the canonical input.

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
├── README.md                  this file
├── reproduce_all.sh           run every script, write outputs/*.csv
├── data/
│   ├── zips/                  fixed-batch eval zips, one per env
│   ├── parquets/<env>/advantages.parquet
│   ├── eval_stats/<env>.csv   per-(algo, env, config, seed, phase) success
│   └── configurations/        (reserved; alphas are read from the zips)
├── scripts/
│   ├── _common.py             shared helpers
│   ├── build_eval_stats.py    helper: builds data/eval_stats/<env>.csv from a zip
│   ├── t1_final_landscape_ci.py
│   ├── t2_compact_diagnostics.py
│   ├── t3_antmaze_ess.py
│   ├── t4_adv_corr.py
│   ├── a2_seed_variance.py
│   ├── a3_phase_mobility.py
│   ├── a4_subsample_stability.py
│   ├── a5_wmax_sensitivity.py
│   ├── a6_adv_norm_sensitivity.py
│   └── a7_sensitivity_summary.py
└── outputs/                   regenerated CSVs land here
```

`data/` ships empty: the four zips (~6 GB total), four parquets (~17 GB
total), and four eval-stats CSVs are not checked in — fetch them from the
cluster as described below.

---

## Fetching the data from Luis

Everything under `data/` originates from one project directory on the LUIS
cluster. If you cloned the repo on a fresh host, fetch the inputs as below;
total transfer is ~23 GB (≈6 GB zips + ~17 GB parquets).

**Canonical location on luis:**

```
/project/NHWP25179/SSRL-Landscapes/
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
  luis:/project/NHWP25179/SSRL-Landscapes/2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip \
  luis:/project/NHWP25179/SSRL-Landscapes/2026-04-27-logs-advantage-cube-single-fixed-batch.zip \
  luis:/project/NHWP25179/SSRL-Landscapes/2026-04-30-logs-advantage-antmaze-large-single-fixed-batch.zip \
  luis:/project/NHWP25179/SSRL-Landscapes/2026-05-03-logs-advantage-scene-single-fixed-batch.zip \
  data/zips/
```

The filenames must match `_common.py::ZIP_NAMES` exactly — don't rename.

**Step 2 — fetch the four `advantages.parquet` into `data/parquets/<env>/`:**

```bash
mkdir -p data/parquets/{antmaze-medium,antmaze-large,cube,scene}

rsync -avhP luis:/project/NHWP25179/SSRL-Landscapes/logs-advantage-antmaze-medium-fixed-batch/advantages.parquet      data/parquets/antmaze-medium/
rsync -avhP luis:/project/NHWP25179/SSRL-Landscapes/logs-advantage-antmaze-large-single-fixed-batch/advantages.parquet data/parquets/antmaze-large/
rsync -avhP luis:/project/NHWP25179/SSRL-Landscapes/logs-advantage-cube-single-fixed-batch/advantages.parquet         data/parquets/cube/
rsync -avhP luis:/project/NHWP25179/SSRL-Landscapes/logs-advantage-scene-single-fixed-batch/advantages.parquet        data/parquets/scene/
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
./reproduce_all.sh

# only the main-paper tables
./reproduce_all.sh main

# only the robustness-appendix tables
./reproduce_all.sh appendix

# run an individual script
.venv/bin/python scripts/t1_final_landscape_ci.py
```

The driver picks up `PAPER_METRICS_PYTHON` if set, else uses the project venv
at `GCRL-Landscapes/.venv/bin/python`.

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
uv pip install -e GCRL-Landscapes
uv pip install pyarrow scikit-learn scipy pandas numpy
# transitive (already pulled in by GCRL-Landscapes' install):
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

`advantages.parquet` for each env was generated on the LUIS cluster by
`analysis/generate_advantages.py` from the `params_*.pkl` checkpoints in the
unzipped training tree (`/bigwork/nhwptoem/gcrl/code/...`). The zips
themselves contain NO checkpoints (verified: zero `.pkl` entries), so the
parquets are primary released data that cannot be regenerated offline — see
PROVENANCE.md for the full derivation chain and audit. The eval-stats CSVs in `data/eval_stats/` are the per-step
success-rate dumps used by `t1` and `a7`; they are derived from the same
zips. (The `_malte_repro/eval_stats_v2/` versions of the medium CSV were
copied here unchanged.)

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
- `compute_ess_table.py` is incompatible with newer pandas (uses
  `include_groups`). `t3_antmaze_ess.py` does not use it; it computes ESS
  directly via `compute_merged_df`.
