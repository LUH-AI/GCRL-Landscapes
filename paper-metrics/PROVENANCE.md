# PROVENANCE — how every number in the paper is produced

Audited 2026-07-26 (four independent audit passes + a clean-room rebuild).
This document is the single source of truth for: what the release zips
contain, how each artifact derives from them, which script produces every
table and inline number, and exactly what can and cannot be reconstructed
offline. A reader with the four zips, the four advantage parquets, this
repository, and its Python environment can regenerate every paper table
byte-for-byte; this was verified end to end (see §6).

## 1. The primary data: four "fixed-batch" zips

One zip per environment, produced by the training campaign on the LUIS
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
  unzipped trees on the cluster (`/bigwork/nhwptoem/gcrl/code/...`) and are
  the one input that cannot be re-derived offline (see §4).
- The antmaze-large zip additionally carries three derived files at its
  root (`advantages.parquet`, a per-batch `afr_metrics.csv`,
  `checkpoints.csv`) that the other three zips do not; they are convenience
  copies, not extra primary data.

## 2. The derivation chain

```
                    zips (4)                          cluster checkpoints (params_*.pkl)
                      |                                        |
   build_eval_stats.py|                     catalog_checkpoints.py + generate_advantages.py
                      v                                        v
        data/eval_stats/<env>.csv  (5120 rows:        advantages.parquet  (per env, ~4.2 GB;
        algo, env, config, seed, phase,               838M rows = 4 agents x 64 configs x 5
        alpha, lr, success)                           seeds x 10 fixed batches; one 256x256
                      |                               cross-goal matrix per row group;
                      |                               FINAL phase per agent only)
                      |                                        |
        +-------------+--------------+          +--------------+----------------+
        v                            v          v                               v
  t1, a2, a3, a4          (alpha from zip) extract_env.py /            compute_awr_concentration.py
  rl1, rl2*, rl4, rl5              t2, a5, a6  compute_afr_metrics.py         (alpha from zip)
                                               -> advantage_matrix_metrics,   -> awr_concentration.csv,
        train_log.csv 'advantage/actor'           ess_per_batch, afr_metrics     awr_alpha_sweep.csv
                      |                                   |                          |
                      v                                   v                          v
                 t3, rl8                              rl6, rl7                  rl2, rl3
```

t4 shells out to `analysis/advantages/{g_spearman,h_oracle}.py`, which read
the antmaze-medium zip directly (plus the OGBench dataset for the oracle).

## 3. Script → table ledger

Main-paper tables (`content/experiments.tex`):

| LaTeX label | Script | Inputs | Audit status (typeset vs regenerated) |
|---|---|---|---|
| `tab:final_landscape_ci` | `t1_final_landscape_ci.py` | eval_stats | 63/64 cells match; 1 typo: AntMaze-L GCIVL ρ₀.₈ typeset 0.12, correct 0.11 (7/64 = 0.109) |
| `tab:compact_diagnostics` | `t2_compact_diagnostics.py` | parquets + zip alphas | FR-AUC/Gap/MRR/Max all match; the ESS column was typeset from the older G1 run (7 cells differ ≤ 0.015 from the pipeline); 4 CRL Top-5 cells match no archived artifact |
| `tab:antmaze_ess_phase4` | `t3_antmaze_ess.py` | medium zip (train_log advantage arrays) | ESS mean±std 16/16 match. The script was fixed 2026-07-26 to use the final eval step per phase (n = 320), which the typeset p-values assume; before the fix the r's matched but p-values were deflated ~10⁴× |
| `tab:adv_corr` | `t4_adv_corr.py` | medium zip (+ OGBench dataset for oracle) | 8/8 match |

Appendix tables (`content/Appendix/robustness-2.tex` — note: `robustness.tex`
is NOT compiled; it is a stale predecessor and holds the only two
`\texttodo` markers plus an outdated sensitivity table):

| LaTeX label | Script | Inputs | Audit status |
|---|---|---|---|
| `tab:seed_variance_summary` | `a2_seed_variance.py` | eval_stats | 12/12 match |
| `tab:phase_mobility` | `a3_phase_mobility.py` | eval_stats | 8/8 match |
| `tab:subsample_stability_close_pairs` | `a4_subsample_stability.py` | eval_stats | typeset values come from `_malte_repro/recompute_robustness.py` (same algorithm, different RNG stream); 3/4 differ by ≤ 1.6 pp from the pipeline. Neither implementation uses the "common index set across algorithms" the appendix text claims — text/code mismatch to fix in revision |
| `tab:wmax_sensitivity` | `a5_wmax_sensitivity.py` | parquets + zip alphas | typeset AntMaze rows used a stale alpha map (config JSONs from the pre-fixed-batch campaign paired with fixed-batch advantages) — 25/32 AntMaze cells differ; the pipeline (zip-derived alphas) is the correct version. Cube/Scene unaffected |
| `tab:adv_norm_sensitivity` | `a6_adv_norm_sensitivity.py` | parquets + zip alphas | same stale-alpha issue, confined to the 4 AntMaze-M "Raw ESS" cells; all normalized columns match |
| `tab:sensitivity_summary` | `a7_sensitivity_summary.py` | eval_stats | 87/96 cells match; 9 ±half-width cells are ±0.01 leftovers from a different RNG-generation run; 2 match no artifact. Compiled table's R²_α max is 0.07 |

Inline prose numbers: all spot-checked. Known inconsistencies (revision fix
list): `experiments.tex:413/414/418` quote seed-mean maxima (0.904 / 0.728 /
0.868) while the table prints IQM (0.907 / 0.733 / 0.873); `:424` "0.218"
for Scene QRL max matches no artifact (t1 says 0.220); `tab:hps:defaults`'
AWR temperature row (3.0/3.0/3.0/10.0) is not traceable to code in this
tree. `tab:convergence:phases` values verified against the zips' `info.toml`
phase steps, but the exact generator invocation is not archived.

## 4. Reconstructibility classes

**(a) From the zips alone:** all four `eval_stats` CSVs (verified: rebuilt
from freshly extracted zips, 5,120 rows each, zero numeric deltas against
the copies used for the paper), and therefore t1, a2, a3, a4, rl1, rl2's
alpha-restriction half, rl4, rl5; t3 (advantage arrays are inside
`train_log.csv`); t4's seed-consistency column; rl8.

**(b) From zips + the four `advantages.parquet`:** t2, a5, a6, rl2's
saturation half, rl3, rl6, rl7, and every derived file in `RAW-ADV/ENVS/`
(`advantage_matrix_metrics.parquet`, `ess_per_batch.parquet`,
`afr_metrics.csv`, `awr_concentration.csv`, `awr_alpha_sweep.csv`, …).
t4's oracle column additionally needs the public OGBench dataset download.

**(c) Requires cluster-only checkpoints (NOT reconstructible offline):**
the four `advantages.parquet` themselves. They were produced by
`analysis/generate_advantages.py` loading `params_{step}.pkl` from the
cluster trees; the zips deliberately exclude checkpoints. Treat the
parquets as primary released data. They cover the final phase per agent
only, and are short 30 (antmaze-large) and 10 (cube) QRL row groups out of
12,800 — advantage-generation failures on the cluster; the training runs
themselves are complete.

**(d) Separate lineage (older campaign, not fixed-batch):**
`additional_stats_raw.csv` and everything derived from it (`ess_table.tex`,
`awr_vs_return.csv`, the antmaze-medium rows of the G1
`compute_landscape_stats.py` outputs). None of these feed the compiled
paper; do not cite them next to fixed-batch numbers.

## 5. Rebuttal analyses (rl1–rl8, 2026-07-24)

`scripts/rl*.py` here are the rebuttal analyses (E1–E8 in
`PAPER/rebuttal/experiments-explained.md`). They follow the same
conventions as t1 (final phase, seed-IQM via `trim_mean(·, 0.25)`) except
where they deliberately mirror an older figure (rl7 uses seed-mean to match
the existing cube/scene scatters; rl5 uses seed-mean to match a3). An
independent adversarial review (2026-07-26) recomputed every headline
number exactly; two presentation notes: rl5's Spearman seed-bootstrap CIs
are attenuation-biased (quote point estimates; inference rests on the
permutation band), and rl2's "max unchanged" holds in 13/16 cells with
≤ 0.02 shifts in the rest.

## 6. Verification log (what was actually re-run)

- 2026-07-24: `./reproduce_all.sh` (all 10 tables) — outputs byte-identical
  to the 2026-05-07 archive.
- 2026-07-26: all four zips freshly extracted; `build_eval_stats.py` re-run
  on the extracted trees; rebuilt eval_stats ≡ the paper's copies (0 deltas,
  5,120 rows each). Clean-room pipeline (scratch checkout of the scripts,
  eval_stats taken only from the from-zip rebuilds, zips and parquets
  linked) regenerated all 10 tables byte-identical to the canonical
  outputs.
- 2026-07-26: t3 final-eval-step fix validated against the archived
  campaign log (`_malte_repro/e_ess_fixed_batch.log`): r and p per cell.

## 7. How a newcomer reconstructs everything

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
# 3. rebuttal analyses
./reproduce_all.sh rebuttal # rl1..rl8 -> outputs/
```

Only step 0's parquets are non-reproducible inputs (class (c) above);
everything else is derived in front of you.
