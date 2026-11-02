# PROVENANCE — how the paper's numbers are produced

Last verified 2026-07-26.

This document is the single source of truth for: what the four fixed-batch
zips contain, how each artefact in `paper-metrics/` derives from them, which
script produces every table, and exactly what can and cannot be reconstructed
offline. A reader with the four zips, the four advantage parquets, this
repository, and its Python environment can regenerate every table (see §5).

## 1. The primary data: four "fixed-batch" zips

One zip per environment, produced by the training campaign on the training
cluster (April–May 2026):

| Zip | Environment | Entries | Uncompressed |
|---|---|---|---|
| `2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip` | antmaze-medium-navigate-v0 | 27,162 | 1.88 GB |
| `2026-04-30-logs-advantage-antmaze-large-single-fixed-batch.zip` | antmaze-large-navigate-v0 | 27,165 | 6.10 GB |
| `2026-04-27-logs-advantage-cube-single-fixed-batch.zip` | cube-single-play-v0 | 27,162 | 1.89 GB |
| `2026-05-03-logs-advantage-scene-single-fixed-batch.zip` | scene-play-v0 | 27,162 | 1.90 GB |

Identical structure in each:

```
logs-advantage-<ENV>-fixed-batch/
  {CRL,GCIQL,GCIVL,QRL}_<dataset>_64c_lr-alpha/
    info.toml                      # agent, phases (4 steps at 25/50/75/100%),
                                   # n_configurations=64, hyperparameters=[lr,alpha],
                                   # actor_loss="awr", eval_episodes=10, git commit
    configurations/
      configuration_{0..63}.json   # the sampled (lr, alpha) per configuration
    run_logs/configuration_{0..63}/phase_{step}/seed_{0..4}/
      eval_log.csv                 # one row: success at this checkpoint
      train_log.csv                # 10 rows x 1708 cols; col 'advantage/actor'
                                   # holds the 256-length advantage vector on the
                                   # fixed validation batch at each logged step
      eval_trajectory.json
      info.toml
```

Counts, verified per zip: 4 agents × 64 configurations × 4 phases × 5 seeds
= 5,120 runs; 5,120 `eval_log.csv` and 5,120 `train_log.csv` each. No run
is missing in any environment.

Two important facts:

- **The zips contain no model checkpoints.** Zero `.pkl` entries in all
  four (verified). The `params_{step}.pkl` checkpoints stayed in the
  unzipped trees on the cluster (`${CLUSTER_SCRATCH_DIR}/...`) and are
  the one input that cannot be re-derived offline (see §3).
- The antmaze-large zip additionally carries three derived files at its
  root (`advantages.parquet`, a per-batch `afr_metrics.csv`,
  `checkpoints.csv`) that the other three zips do not; they are convenience
  copies, not extra primary data.

## 2. How each artefact is produced

### Derivation chain

```
zips (4)                                     cluster checkpoints (params_*.pkl)
   |                                                      |
   | scripts/build_eval_stats.py      analysis/catalog_checkpoints.py
   |                                 + analysis/generate_advantages.py
   v                                                      v
data/eval_stats/<env>.csv                data/parquets/<env>/advantages.parquet
(5,120 rows: algo, env, config, seed,    (838M rows = 4 agents x 64 configs x 5
phase, alpha, lr, success)               seeds x 10 fixed batches; one 256x256
   |                                     cross-goal matrix per row group;
   |                                     FINAL phase per agent only)
   +--> t1, a2, a3, a4, rl1, rl2, rl4, rl5
   +--> t2, a5, a6              (alpha map read from the zips' configurations/)
   +--> t3, rl8                  (train_log.csv 'advantage/actor' arrays)
   +--> t4                      (zip arrays; oracle column needs the OGBench dataset)
```

`t3` delegates the ESS computation to `analysis/advantages/e_ess.py`; `t4`
delegates to `analysis/advantages/g_spearman.py` and
`analysis/advantages/h_oracle.py`. Both delegate to the same committed code
that produced the paper values, so definitional identity is guaranteed.

### Table ledger

| LaTeX label | Script | Inputs |
|---|---|---|
| `tab:final_landscape_ci` | `t1_final_landscape_ci.py` | eval_stats |
| `tab:compact_diagnostics` | `t2_compact_diagnostics.py` | parquets + zip alphas |
| `tab:antmaze_ess_phase4` | `t3_antmaze_ess.py` | medium zip (train_log advantage arrays) |
| `tab:adv_corr` | `t4_adv_corr.py` | medium zip (+ OGBench dataset for oracle) |
| `tab:seed_variance_summary` | `a2_seed_variance.py` | eval_stats |
| `tab:phase_mobility` | `a3_phase_mobility.py` | eval_stats |
| `tab:subsample_stability_close_pairs` | `a4_subsample_stability.py` | eval_stats |
| `tab:wmax_sensitivity` | `a5_wmax_sensitivity.py` | parquets + zip alphas |
| `tab:adv_norm_sensitivity` | `a6_adv_norm_sensitivity.py` | parquets + zip alphas |
| `tab:sensitivity_summary` | `a7_sensitivity_summary.py` | eval_stats |

All scripts live in `paper-metrics/scripts/`; outputs land in
`paper-metrics/outputs/`. `scripts/_common.py` is the shared helper module
(paths from `data/manifest.json`, alpha lookup from the zip
`configurations/`, eval-stats loader) and produces no table.

### Aggregation conventions

- `t1` reduces seeds per configuration by IQM (`trim_mean(·, 0.25)`) and
  reports the maximum, mean and ρ over the 64 configurations.
- `t2` averages each per-matrix diagnostic over all configurations, seeds and
  validation batches. It selects no single checkpoint; the `Max Success`
  column is taken from `t1`.
- `t3`, `a5` and `a6` recompute AWR weights post hoc from the advantage
  arrays with the per-configuration alpha read from the zip, and never
  retrain.

## 3. Reconstructibility classes

**(a) From the zips alone:** all four `eval_stats` CSVs (verified: rebuilt
from freshly extracted zips, 5,120 rows each, zero numeric deltas against
the copies used for the paper), and therefore t1, a2, a3, a4, rl1, rl2's
alpha-restriction half, rl4, rl5; t3 (advantage arrays are inside
`train_log.csv`); t4's seed-consistency column; rl8.

**(b) From zips + the four `advantages.parquet`:** t2, a5, a6, rl2's
saturation half, rl3, rl6, rl7, and the derived metrics tables built from
the parquets (per-checkpoint × per-batch 256×256 matrix metrics, ESS per
batch, per-batch AFR metrics). t4's oracle column additionally needs the
public OGBench dataset download.

**(c) Requires cluster-only checkpoints (NOT reconstructible offline):**
the four `advantages.parquet` themselves. They were produced by
`analysis/generate_advantages.py` loading `params_{step}.pkl` from the
cluster trees; the zips deliberately exclude checkpoints. Treat the
parquets as primary released data. They cover the final phase per agent
only, and are short 30 (antmaze-large) and 10 (cube) QRL row groups out of
12,800 — advantage-generation failures on the cluster; the training runs
themselves are complete.

## 4. Additional analyses (rl1–rl9)

`paper-metrics/scripts/rl1_*.py`–`rl9_new_result_figures.py` produce
supplementary figures and metrics.
They follow the same conventions as `t1` (final phase, seed-IQM via
`trim_mean(·, 0.25)`) except where noted in the scripts.

## 5. Verification status

- 2026-07-24: `./reproduce_all.sh` (all 10 tables) — outputs byte-identical
  to the 2026-05-07 archive.
- 2026-07-26: all four zips freshly extracted; `build_eval_stats.py` re-run
  on the extracted trees; the rebuilt eval_stats are identical to the copies
  used for the paper (0 numeric deltas, 5,120 rows each).
- 2026-07-26: clean-room pipeline (scratch checkout of the scripts, eval_stats
  taken only from the from-zip rebuilds, zips and parquets linked)
  regenerated all 10 tables byte-identical to the canonical outputs.

## 6. How a newcomer reconstructs everything

```bash
# 0. clone repo, create the venv (README), download the 4 zips and 4
#    parquets into paper-metrics/data/ (README lists the rsync sources)
# 1. eval stats from the zips
for env in antmaze-medium antmaze-large cube scene; do
  unzip -q data/zips/<zip-for-$env>.zip -d /tmp/trees/
  python scripts/build_eval_stats.py \
    --logs /tmp/trees/<tree-for-$env> --env <dataset-tag> \
    --out data/eval_stats/$env.csv
done
# 2. every paper table
./reproduce_all.sh          # t1..t4 + a2..a7 -> outputs/
# 3. supplementary analyses
./reproduce_all.sh rebuttal # rl1..rl8 + supplementary figures -> outputs/
```

Only step 0's parquets are non-reproducible inputs (class (c) above);
everything else is derived in front of you.
