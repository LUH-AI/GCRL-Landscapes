# Quickstart — one node, no Slurm

A minimal end-to-end run on a single machine: create a small experiment, train
one configuration locally, then evaluate and plot the result. Everything here
uses the real CLI flags of `gcrl_landscapes.main` (see
[`cli-reference.md`](cli-reference.md) for the complete tables).

The walkthrough needs a CUDA-capable GPU (training runs on JAX with CUDA 12).
Pick a smaller `--final_phase` if you just want to see the pipeline move.

---

## 0. Prerequisites

```bash
# from the repository root
python -c "import jax; print(jax.__version__, jax.default_backend())"   # expect: 0.6.x gpu
```

Installation (Python 3.10, JAX/CUDA 12, `uv sync --extra dev`) is described in
the [README installation section](../README.md#installation).

Use a directory whose top-level name starts with `logs`: the evaluation step
below zips it, and the zip reader expects `logs*/<experiment>/info.toml`
inside the archive.

```bash
mkdir -p logs-quickstart
```

## 1. Create the experiment

```bash
python -m gcrl_landscapes.main setup \
  --agent CRL \
  --datasets antmaze-medium-navigate-v0 \
  --n_configurations 4 \
  --hyperparameters lr alpha \
  --eval_episodes 2 \
  --final_phase 20000 \
  --phase_percentages 50 100 \
  --final_step_is_phase \
  --actor_loss awr \
  --seed 0 \
  --logdir ./logs-quickstart/CRL_antmaze-medium-navigate-v0
```

What this does:

- `--n_configurations 4` — Sobol design over `lr` and `alpha`; the count must be
  a power of two (`1` would use the agent's adapted default config unchanged).
- `--final_phase 20000 --phase_percentages 50 100` — no convergence archive
  needed; the two phases end at step 10000 and 20000
  (`int(pct/100 * final_phase)`).
- `--final_step_is_phase` — each phase trains only up to its own end. Without
  it every run trains to the final step, so phase 0 alone would consume the
  whole budget (`train_all.sh` always sets it).
- `--actor_loss awr` — required for the advantage pipeline in step 4 (CRL only
  builds a value network for `awr`).

Check the result:

```bash
cat ./logs-quickstart/CRL_antmaze-medium-navigate-v0/info.toml
ls ./logs-quickstart/CRL_antmaze-medium-navigate-v0/configurations/
```

`info.toml` holds the parsed arguments plus the derived `phases`
(`[10000, 20000]`) and the current git commit. Re-running `setup` against the
same `--logdir` fails, because `configurations/` is created with
`mkdir(exist_ok=False)` — remove the directory or pick a new one.

## 2. Train one configuration locally (no Slurm)

`run` is what the Slurm tasks call, and it works without submitit:

```bash
python -m gcrl_landscapes.main run \
  --logdir ./logs-quickstart/CRL_antmaze-medium-navigate-v0 \
  --phase_idx 0 \
  --configuration 0 \
  --seed 0
```

Optional second phase (resumes from the best checkpoint of phase 0):

```bash
python -m gcrl_landscapes.main run \
  --logdir ./logs-quickstart/CRL_antmaze-medium-navigate-v0 \
  --phase_idx 1 \
  --configuration 0 \
  --seed 0
```

Notes:

- Run from the repository root — `run` records the git commit/branch of the
  enclosing repository.
- `run` sets `MUJOCO_GL=egl` before importing JAX/MuJoCo. That is fine on GPU
  nodes; on headless hosts without an EGL-capable driver it can fail.
- Each (configuration, phase, seed) triple needs a fresh output directory.
  Starting the same triple twice fails on `mkdir(exist_ok=False)`.
- Output: `run_logs/configuration_0/phase_<steps>/seed_0/` with
  `eval_log.csv`, `train_log.csv` (`;`-separated), `eval_trajectory.json`,
  `info.toml` and `params_<steps>.pkl`.

## 3. Evaluate and plot

Package the logs with the same exclusions as `zip_and_plot.sh` (no
checkpoints, no `submitit/`):

```bash
zip -r quickstart.zip logs-quickstart -x '*.pkl' -x '*/submitit/*'
```

Landscape/curve plots (`--zipfile`, singular):

```bash
python -m gcrl_landscapes.evaluation.plot \
  --zipfile quickstart.zip \
  --plot_eval_curves \
  --plot_return_distributions
```

Tables (`--zipfiles`, one or more, plus `--output_folder`):

```bash
python -m gcrl_landscapes.evaluation.tabular \
  --zipfiles quickstart.zip \
  --output_folder tables/quickstart
```

Outputs:

```
plots/quickstart.zip/CRL_antmaze-medium-navigate-v0/…      # per-experiment PDFs
plots/quickstart.zip/grid_plots/…                          # cross-experiment grids
tables/quickstart/table_all_phases.{md,tex,csv}            # aggregated tables
tables/quickstart/table_last_phase.{md,tex,csv}
```

GP landscape plots (`igpr-*.pdf`) and phase-mobility plots are produced by
default; disable them with `--no_plot_landscapes` / `--no_plot_mobility` if you
only want the curves. Landscape plotting requires exactly two hyperparameters
(`NotImplementedError` otherwise) — with `--n_configurations 4` expect very
sparse fits, so this is only a smoke test.

## 4. Optional: cross-goal advantages

Works with the AWR-type agents (here CRL with `--actor_loss awr`):

```bash
python analysis/catalog_checkpoints.py \
  --logdir ./logs-quickstart \
  --output ./logs-quickstart/checkpoints.csv

python analysis/generate_advantages.py \
  --catalog ./logs-quickstart/checkpoints.csv \
  --output ./logs-quickstart/advantages.parquet \
  --chunk-size 4

python analysis/compute_afr_metrics.py \
  --input ./logs-quickstart/advantages.parquet \
  --output ./logs-quickstart/afr_metrics.csv
```

`catalog_checkpoints.py` picks the deepest phase that has a
`params_<phase>.pkl`, so run step 2 (phase 1) first if you want the 20000-step
checkpoints.

## 5. Next steps

| Goal | Where |
| ---- | ----- |
| Run the full campaign on Slurm | `bash submit_train_all.sh`, then `CLUSTER=pc2 bash submit_train_all.sh` |
| Full flag reference | [`cli-reference.md`](cli-reference.md) |
| Landscape/HPO machinery | `bash hpo.sh <agent>`, `configs/hpo_*.yaml` |
| Reproduce the paper tables | [`../paper-metrics/README.md`](../paper-metrics/README.md) |
| Architecture diagram | [`figures/actor-loop.md`](figures/actor-loop.md) |
