# Cube — What Data Is Missing and Where to Get It

## What we already have

| File | Status |
|---|---|
| `ENVS/cube/advantages.parquet` | ✓ 3.9 GB, 838M rows, all 4 agents |
| `ENVS/cube/advantage_matrix_metrics.parquet` | ✓ fr_auc, gap, MRR, prec@k per batch |
| `ENVS/cube/advantages_summary.parquet` | ✓ distribution stats per group |
| `ENVS/cube/ess_per_batch.parquet` | ✗ **missing** |

The fr_auc patterns already tell an interesting story (see below), but we cannot close the
loop to *success rates* or compute ESS without the two missing pieces.

---

## What is missing

### 1. Configuration JSONs → needed for ESS

ESS requires `alpha` per (agent, config), which lives in:

```
/bigwork/nhwptoem/gcrl/code/
  logs-advantage-cube-single-fixed-batch/
    {AGENT}_cube-single-play-v0_64c_lr-alpha/
      configurations/
        configuration_0.json    ← contains {"alpha": ..., "lr": ...}
        configuration_1.json
        ...
        configuration_63.json
```

**What to pull** (4 agents × 64 files = 256 small JSONs, a few KB total):

```bash
rsync -av \
  bigwork:/bigwork/nhwptoem/gcrl/code/logs-advantage-cube-single-fixed-batch/\
{CRL,GCIQL,GCIVL,QRL}_cube-single-play-v0_64c_lr-alpha/configurations/ \
  /Users/adityamohan/git/GCRL/plots/new-plots/raw-data/logs-cube/
```

Once local, `extract_env.py` will pick them up automatically if `LOGS_BASE` is pointed at the
right parent directory (or add a second search path for cube in the script).

### 2. Eval logs → needed for success rates

The success-rate correlations (Findings 2–3 for antmaze) require per-(agent, config, seed, phase)
success rates from evaluation runs. These live at:

```
/bigwork/.../logs-advantage-cube-single-fixed-batch/
  {AGENT}_cube-single-play-v0_64c_lr-alpha/
    run_logs/
      configuration_{i}/
        phase_{step}/
          seed_{j}/
            eval_log.csv    ← first column is success rate
```

**Minimum pull** — only the last phase per (agent, config, seed), ~1280 small CSVs:

```bash
# Pull only eval_log.csv from the last phase of each config/seed
rsync -av --include="*/" --include="eval_log.csv" --exclude="*" \
  bigwork:/bigwork/nhwptoem/gcrl/code/\
logs-advantage-cube-single-fixed-batch/ \
  /Users/adityamohan/git/GCRL/plots/new-plots/raw-data/logs-cube/
```

### 3. Train logs (optional) → needed only for basin-analysis scripts

`additional_plots.py` and `aggregates/compute_and_plot.py` read `train_log.csv` for advantage
arrays from training batches. These are ~MB each and there are ~1280 of them. Only needed if
you want to extend the z-quantile / basin-analysis to cube.

---

## What cube already shows (without the missing pieces)

Even without alpha or success rates, the fr_auc patterns are informative:

| Agent | fr_auc (cube) | fr_auc (medium) | fr_auc (large) |
|---|---|---|---|
| GCIVL | **0.644** | 0.564 | 0.575 |
| CRL   | 0.562 | 0.530 | 0.527 |
| QRL   | 0.523 | 0.546 | 0.582 |
| **GCIQL** | **0.531** | **0.471** | **0.472** |

Two notable shifts relative to antmaze:

1. **GCIQL is above 0.5 on cube (0.531)** — it is no longer inverted. The Q−V margin
   correctly ranks matched pairs above random on the cube manipulation task. If GCIQL's
   inversion on antmaze is the explanation for its brittleness there, cube gives us a
   natural test: does GCIQL show a broader trainability landscape on cube compared to antmaze?
   We cannot answer this without success data, but it is the key hypothesis to test.

2. **CRL has a large gap_mean on cube (0.875 vs 0.169 on antmaze)** — the matched-pair
   advantage margin is 5× larger on cube. CRL's contrastive objective may be much more
   task-aligned for cube-style manipulation than for maze navigation.

3. **GCIVL is most discriminative across all environments** (fr_auc 0.564 → 0.575 → 0.644),
   suggesting value-progress advantages consistently identify the right transitions.

---

## Priority

To complete the cube analysis, pull items in this order:

1. **Config JSONs** (256 small files, minutes to rsync) → unlocks ESS computation
2. **Eval logs, last phase only** (~1280 small CSVs) → unlocks success-rate correlations
3. **Train logs** (large, ~GB) → only if basin-analysis is needed for cube
