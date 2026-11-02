# GCRL-Landscapes

GCRL-Landscapes explores how the hyperparameter landscape of offline
goal-conditioned reinforcement learning agents evolves during training. It
samples hyperparameter configurations (Sobol design), trains each
configuration in consecutive phases on a Slurm cluster (each phase resumes from
the best checkpoint of the previous phase), evaluates every
(configuration, phase, seed) run, fits GP-based optimality landscapes, and
computes cross-goal advantage matrices from the trained checkpoints. Agents,
environments and datasets come from [OGBench](https://github.com/seohongpark/ogbench).

> **Paper:** *Beyond Success Rates: Trainability and Extractability for Offline
> GCRL* — Jan Malte Töpperwien, Aditya Mohan, Marius Lindauer. NeurIPS 2026.
> [`CITATION.cff`](CITATION.cff) describes the software (title, authors, version,
> license), not the publication.

**License:** MIT — see [`LICENSE`](LICENSE) (Copyright (c) 2026 Institute of AI
at Leibniz University Hannover).

## Installation

**Python 3.10 only** — `pyproject.toml` pins `requires-python = ">=3.10, <3.11"`.
Training runs on **JAX/Flax with CUDA 12** (`jax[cuda12] >= 0.6.2, < 0.7.0` and
`flax >= 0.10.6`, both Linux-only). `ogbench` is installed from the
[`CrunchyFlakes/ogbench`](https://github.com/CrunchyFlakes/ogbench) fork and
`smac` from `automl/SMAC3` branch `v2.4.0` (see `[tool.uv.sources]`).

With [uv](https://docs.astral.sh/uv/) (a lockfile, `uv.lock`, is tracked):

```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync --extra dev
```

With pip:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Then install the git hooks:

```bash
pre-commit install
```

Nix alternative: `nix develop` uses the `devShells.default` from
[`flake.nix`](flake.nix); [`shell.nix`](shell.nix) builds an FHS environment
with the NVIDIA driver (`nix-shell shell.nix`).

Datasets are not shipped with the repository: they are loaded through
`ogbench.make_env_and_datasets`. A convergence log archive for
`--convergence_zip` is not part of the repository either (see
[Quickstart](#quickstart)).

## Quickstart

Training runs in three steps: **setup → submit → run**.

### 1. `setup` — create the experiment

Generates the folder structure, samples the configurations and writes
`info.toml`, `configurations/configspace.json` and
`configurations/configuration_<i>.json` into `--logdir`.

```bash
python -m gcrl_landscapes.main setup \
  --agent CRL \
  --datasets antmaze-medium-navigate-v0 \
  --n_configurations 64 \
  --hyperparameters lr alpha \
  --eval_episodes 10 \
  --final_phase 1000000 \
  --phase_percentages 25 50 75 100 \
  --final_step_is_phase \
  --actor_loss awr \
  --logdir ./logs/CRL_antmaze-medium-navigate-v0_64c_lr-alpha
```

Required flags: `--agent --datasets --n_configurations --hyperparameters
--eval_episodes --logdir`, plus exactly one of `--convergence_zip` or
`--final_phase`.

### 2. `submit` — launch the Slurm array jobs

```bash
python -m gcrl_landscapes.main submit \
  --logdir ./logs/CRL_antmaze-medium-navigate-v0_64c_lr-alpha \
  --n_seeds 5 \
  --tasks_per_node_total 16 \
  --tasks_per_node_parallel 4 \
  --jobname CRL-antmaze-medium \
  --partition ai,tnt \
  --mem_per_cpu 3G
```

Required flags: `--logdir --n_seeds --tasks_per_node_total
--tasks_per_node_parallel --jobname`. The command prints
`SUBMITTED_ARRAY_ID=<id>`; `train_all.sh` parses that line.

### 3. `run` — execute one (configuration, phase, seed) triple

This is what the Slurm tasks call. You can also run it directly on a single
machine, without Slurm:

```bash
python -m gcrl_landscapes.main run \
  --logdir ./logs/CRL_antmaze-medium-navigate-v0_64c_lr-alpha \
  --phase_idx 0 \
  --configuration 0 \
  --seed 0
```

Required flags: `--logdir --phase_idx --configuration --seed`. Run it from the
repository root: `setup` and `run` record the current git commit and branch, so
a git repository must be reachable from the working directory.

### Flag reference

Full tables with types, defaults and meanings are in
[`docs/cli-reference.md`](docs/cli-reference.md). The flags you are most likely
to change:

| Flag | Meaning |
| ---- | ------- |
| `--agent` | Agent to run; one of the supported agents listed below |
| `--datasets` | Which dataset/environment to run (one name per phase, or one name for all phases) |
| `--n_configurations` | How many configurations to run/sample across the configuration space (must be a power of two; `1` uses the adapted default config unsampled) |
| `--hyperparameters` | Which configuration space to use — exactly two names out of `lr`, `discount`, `actor_p_trajgoal`, `alpha`, `low_alpha`, `high_alpha`, `eps`, `tau` |
| `--phase_percentages` | At which percentages phases should be run (default `25 50 100`) |
| `--final_phase` | How many steps the final phase has (alternative to `--convergence_zip`) |
| `--convergence_zip` | Zip file with logs used to derive the phase boundaries from training results (reaches `--final_performance_percentage`, default 95%, of the benchmarked performance) |
| `--actor_loss` | Sets `actor_loss` in the configuration (`awr`, `ddpgbc`, …); may be ignored by some algorithms |
| `--n_step` | n-step TD target horizon for GCIQL/GCIVL (default `1`); ignored by other agents — `setup` rejects `n_step > 1` for them |
| `--rejection_sampling_n` | Number of candidates for eval-time rejection sampling (`0` = disabled, `1` rejected); only GCIQL/CRL/FQL |
| `--normalize_advantages` | Enable advantage normalization before AWR weighting; may be ignored by some algorithms |
| `--eval_episodes` | How many evaluation episodes to run per evaluation |
| `--extra_eval_steps` | At what (additional) steps to evaluate |
| `--extra_save_steps` | At what (additional) steps to save a checkpoint |
| `--seed` | Which seed to use for configuration sampling (default `0`) |
| `--final_step_is_phase` | Train configs in a phase only up to the end of that phase, instead of to the final step (`train_all.sh` always sets it) |

## Supported agents

`SUPPORTED_AGENTS` and `AGENT_CLASSES` in
[`src/gcrl_landscapes/submission.py`](src/gcrl_landscapes/submission.py) and
[`src/gcrl_landscapes/util/datasets.py`](src/gcrl_landscapes/util/datasets.py)
define ten agents:

| Agent | Class | Notes |
| ----- | ----- | ----- |
| CRL | `CRLAgent` | |
| CMD | `CMDAgent` | |
| FQL | `FQLAgent` (in-repo, `src/gcrl_landscapes/agents/fql.py`) | `alpha` is the distillation weight, not an AWR temperature |
| GCBC | `GCBCAgent` | |
| GCIQL | `NStepGCIQLAgent` (in-repo subclass) | falls back to base GCIQL behaviour when `n_step <= 1` |
| GCIVL | `NStepGCIVLAgent` (in-repo subclass) | falls back to base GCIVL behaviour when `n_step <= 1` |
| MQE | `MQEAgent` | |
| QRL | `QRLAgent` | |
| HIQL | `HIQLAgent` | |
| SAC | `SACAgent` | |

The n-step variants only change behaviour when the batch carries per-sample
`discounts`, which `NStepGCDataset` produces for `n_step > 1`; `n_step = 1`
therefore reproduces the OGBench baseline exactly. MQE and FQL are first-class
agents: both have their own configuration entry and an advantage formulation in
`analysis/generate_advantages.py`. The static tests in
`tests/test_new_agents_static.py` cover FQL, the n-step GCIQL/GCIVL targets and
eval-time rejection sampling; they do not run MQE. MQE additionally has an HPO config
(`configs/hpo_mqe.yaml`) and search space
(`configs/hydra/sweeper/search_space/mqe.yaml`); FQL has no dedicated HPO config.

## Supported environments and datasets

Datasets are addressed by their OGBench name and loaded through
`ogbench.make_env_and_datasets`; `setup --datasets` has no `choices` list, so
any OGBench dataset name works. The families exercised by the repository:

| Family | Examples used in this repository | Where |
| ------ | -------------------------------- | ----- |
| AntMaze (locomaze) | `antmaze-medium-navigate-v0`, `-explore-v0`, `-explore{40,80,90}navigate-v0`, `-stitch-v0`, `antmaze-large-*`, `antmaze-teleport-*`, `antmaze-giant-*` | `train_all.sh`, `convergence.sh`, `SUPPORTED_DATASETS`, paper-metrics |
| Humanoid maze | `humanoidmaze-medium-navigate-v0`, `humanoidmaze-large-*` | `convergence.sh`, `train_all.sh` (commented) |
| Point maze | `pointmaze-medium-navigate-v0` | `convergence.sh` |
| Cube (manipspace) | `cube-single-play-v0`, `cube-double-play-v0` | `analysis/plot_a3_ddpgbc_cube.py`, paper-metrics |
| Scene (manipspace) | `scene-v0`, `scene-singletask-v0`, `scene-singletask-task1..5-v0` | `SUPPORTED_DATASETS` |
| Scene (paper campaign) | Scene *play* dataset — the exact name is `DATASET_MAP["scene"]` in `analysis/run_q_vs_distance_all.py` | paper-metrics, `analysis/plot_q_vs_distance.py` |
| Powderworld | `powderworld-easy-play-v0` | `train_all.sh` (commented); visual + discrete config adaption |

`src/gcrl_landscapes/evaluate.py` imports `HumanoidEnv`, `CubeEnv`, `SceneEnv`
and `PowderworldEnv` for goal-distance bookkeeping, and
`configurations.py::_adapt_base_config` switches on the dataset name:
`visual` for `*visual*`/`*powderworld*` (Impala encoder, HIQL gradient
propagation) and `discrete` for `*powderworld*`. `SUPPORTED_DATASETS` in
`submission.py` is a curated list used by `evaluation/tabular.py` for its
default agent/dataset combinations, not a validation gate.

**Explore mixing.** A dataset name matching
`.*explore<NN><type>.*` (for example `antmaze-medium-explore80navigate-v0`) is
split into a base dataset (`antmaze-medium-navigate-v0`) and an explore dataset
(`antmaze-medium-explore-v0`); `util/datasets.py::MixedDataset` then mixes
`NN`% explore transitions into every batch.

## Orchestration scripts

The shell scripts are Slurm drivers around `main.py`. Every script sources the
cluster profile at the top of the file — see
[`CLUSTER` and `configs/cluster/`](#cluster-and-configscluster).
`profiling_sample.sh` submits nothing: it runs `setup` and one `cProfile` run in
the foreground.

| Script | Purpose |
| ------ | ------- |
| `submit_train_all.sh` | Submits `train_all.sh` to Slurm with the partition/reservation of the selected cluster (`sbatch` wrapper) |
| `train_all.sh` | Full landscape campaign: loops over the agents and environments declared at the top of the file, runs `setup` + `submit` per pair, then chains `zip_and_plot.sh` after all arrays finish |
| `hpo.sh` | Phase-based Bayesian HPO: submits `hpo_single.sh` once per phase (25/50/75/100), each `afterok` the previous one |
| `hpo_single.sh` | One HPO phase — runs `python -m gcrl_landscapes.hpo --multirun --config-name hpo_<agent> …` with Hydra/SMAC + submitit |
| `convergence.sh` | Convergence testing: one configuration per (agent, environment) with a long `--extra_eval_steps` list; the log zip of that campaign is what `--convergence_zip` consumes |
| `zip_and_plot.sh` | Zips the log directory (without `*.pkl` and `submitit/`) and runs the plotting, tabular, `evaluate.py` and `evaluate_advantages.py` stages on it |
| `collect_advantages.sh` | Catalog checkpoints → generate advantage parquet → compute AFR metrics for a log directory |
| `profiling_sample.sh` | Profiles a single local run: `setup` plus `cProfile -m gcrl_landscapes.main run` for configuration 0 / phase 0 / seed 0 |

```bash
bash submit_train_all.sh                    # default cluster: luh
CLUSTER=pc2 bash submit_train_all.sh        # Paderborn PC2
CLUSTER=kisski bash submit_train_all.sh     # KISSKI
CLUSTER=pc2 bash hpo.sh gciql               # HPO for one agent
sbatch collect_advantages.sh /path/to/logs  # catalog + advantages + AFR
```

### `CLUSTER` and `configs/cluster/`

All eight root scripts — `submit_train_all.sh`, `train_all.sh`, `hpo.sh`,
`hpo_single.sh`, `convergence.sh`, `zip_and_plot.sh`, `collect_advantages.sh`
and `profiling_sample.sh` — source the cluster profile before they run any
command:

```bash
source "${SCRIPT_DIR}/configs/cluster/${CLUSTER:-luh}.sh"
```

A profile exports
`CLUSTER_PARTITION` (GPU partition passed to `submit --partition`),
`CLUSTER_SETUP_PARTITION` (CPU partition for orchestration jobs),
`CLUSTER_RESERVATION`, `CLUSTER_GRES` and `CLUSTER_ZIP_CMD`, and defines
`cluster_load_env` / `cluster_load_plot_env` (module loads, conda activation).

| Profile | Partition | Reservation | GRES |
| ------- | --------- | ----------- | ---- |
| `luh` (default) | `ai,tnt` | `ai,tnt` | none (1 GPU per node) |
| `pc2` | `gpu` (setup: `normal`) | none | `gpu:h100:1` |
| `kisski` | `kisski-inference` | none | none |

The profiles are plain shell scripts; no other file in the repository reads
`configs/cluster/`. `submit_train_all.sh`, `hpo.sh`, the `submit` calls of
`train_all.sh` and `convergence.sh`, and their chained `sbatch` calls pass the
`CLUSTER_*` partition and reservation on the command line. `zip_and_plot.sh`
uses `CLUSTER_ZIP_CMD` and `cluster_load_plot_env`, `collect_advantages.sh`
calls `cluster_load_env`, and `hpo_single.sh` turns `CLUSTER_PARTITION`,
`CLUSTER_GRES` and `CLUSTER_RESERVATION` into `hydra.launcher.*` overrides.
`profiling_sample.sh` sources the profile but submits nothing.

A job script that you submit yourself with `sbatch` (`convergence.sh`,
`zip_and_plot.sh`, `collect_advantages.sh`, `profiling_sample.sh`) takes its
partition and reservation from the `sbatch` command line, because none of these
scripts has a `#SBATCH --partition` or `#SBATCH --reservation` line.

`train_all.sh` and `convergence.sh` also default `convergencezip` to
`./convergence.zip` in the repository root. Provide that archive before
running them, or pass `--final_phase` instead of `--convergence_zip` in a
modified copy.

A convergence archive must satisfy the constraints of
`phase_splitting.py::get_data`: exactly one experiment directory whose name
contains the agent, the dataset and the actor loss (`awr`/`ddpgbc`), with
exactly one configuration and one phase. Phase boundaries are then
interpolated from the success curve of that experiment
(`get_all_phases`, mode `target_ratio`, linear interpolation) up to
`--final_performance_percentage` (default 95%) of its best success.

### Phased HPO

`hpo.sh <agent>` runs [SMAC](https://github.com/automl/SMAC3) through
Hydra/hypersweeper per phase, so the incumbent of phase *k* is the starting
point of phase *k+1*. Agent configs live in `configs/hpo_<agent>.yaml`
(`crl`, `gciql`, `gcivl`, `hiql`, `mqe`, `qrl`), search spaces in
`configs/hydra/sweeper/search_space/` (`awr`, `crl`, `ddpgbc`, `gciql`,
`gcivl`, `hiql`, `mqe`, `qrl`), and the Slurm launcher in
`configs/launcher.yaml`. Importance data can be inspected with
[DeepCAVE](https://github.com/automl/DeepCAVE). HPO also derives its phase
boundaries from a convergence archive: `configs/defaults.yaml` sets
`convergence_zip_path: ./convergence.zip` (resolved by
`gcrl_landscapes.hpo` → `phase_splitting.get_all_phases`).

## Evaluation

Both entry points read the log zip archives produced by `zip_and_plot.sh`
(or any zip with the `logs*/<experiment>/info.toml` layout).

`evaluation.plot` takes a **singular** `--zipfile` and writes into
`plots/<zip name>/` (landscape grids into `plots/<zip name>/grid_plots/`):

```bash
python -m gcrl_landscapes.evaluation.plot \
  --zipfile 2026-04-26-logs-advantage-antmaze-medium-fixed-batch.zip \
  --plot_return_distributions \
  --plot_eval_curves \
  --plot_gp_fits \
  --plot_regret \
  --plot_advantages
```

| Flag | Meaning |
| ---- | ------- |
| `--zipfile` | (required) log zip to plot |
| `--plot_return_distributions` | Per-configuration return/success distributions |
| `--plot_eval_curves` | Success-versus-training-step curves |
| `--plot_gp_fits` | GP fit quality (MAE/RMSE) plots |
| `--plot_regret` | Regret curves (first/second/… phase picks) |
| `--plot_advantages` | Advantage and AWR-weight distributions from `train_log.csv` (loads training logs) |
| `--no_plot_landscapes` | Skip the IGPR landscape plots (on by default) |
| `--no_plot_mobility` | Skip the phase-mobility plots (on by default) |
| `--no_multiprocessing` | Run serially instead of a spawn pool |

`evaluation.tabular` takes **`--zipfiles` (one or more)** plus an
`--output_folder`:

```bash
python -m gcrl_landscapes.evaluation.tabular \
  --zipfiles LOG1.zip LOG2.zip \
  --output_folder tables/combined
```

It writes `table_all_phases.{md,tex,csv}`, `table_last_phase.*`,
`igpr_fit_table.*`, regret tables and per-configuration CSVs. Use
`--hp_importance_data` for HPO importance archives and `--convergence_data`
for convergence archives.

Landscapes are fitted with IGPR: three independent GPs for the lower, IQM
(interquartile mean) and upper confidence bound
(`src/gcrl_landscapes/plots/triple_gp.py`, GPFlow).

## Analysis pipeline

### Cross-goal advantages

Three steps, also bundled in `collect_advantages.sh`:

```bash
# 1. Catalog the checkpoints of every agent directory
python analysis/catalog_checkpoints.py \
  --logdir ./logs-antmaze-medium/ \
  --output checkpoints.csv

# 2. Compute cross-goal advantage matrices (GPU)
python analysis/generate_advantages.py \
  --catalog checkpoints.csv \
  --output advantages.parquet

# 3. Compute AFR (future-random advantage) metrics
python analysis/compute_afr_metrics.py \
  --input advantages.parquet \
  --output analysis/afr_metrics.csv
```

- `catalog_checkpoints.py` scans agent subdirectories (prefixes `CRL`, `QRL`,
  `GCIQL`, `GCIVL`, `CMD`, `FQL`, `GCBC`, `HIQL`, `MQE`, `SAC`), reads
  `info.toml`, and yields one row per existing
  `run_logs/configuration_*/phase_<max phase>/seed_*/params_<phase>.pkl`:
  `checkpoint_path, config_path, agent, dataset, phase, seed, configuration`.
- `generate_advantages.py` loads each checkpoint with a fixed 256-sample
  validation batch (`seed=0`) and computes an NxN matrix for every
  (obs_i, action_i, goal_j) triple. Supported: the AWR-type agents GCIQL, CRL,
  GCIVL, QRL, HIQL, MQE and FQL. Per-agent formulas:
  - GCIQL / FQL: `min(Q1,Q2)(s_i,a_i,g_j) - V(s_i,g_j)`
  - CRL: same, but without the value network (`has_value=False`) when
    `actor_loss != "awr"`, i.e. raw `min(Q1,Q2)` is the ranking signal
  - GCIVL: `mean(V(s'_i,g_j)) - mean(V(s_i,g_j))` (ensemble mean)
  - QRL: `V(s_i,g_j) - V(s'_i,g_j)` (negated quasimetric distances)
  - MQE: `min(Q1,Q2)(s_i,a_i,g_j) - V(s_i,g_j)`, mirroring `MQEAgent.actor_loss`
  - HIQL: two actor rows — low actor `mean(V(s'_i,g_j)) - mean(V(s_i,g_j))`,
    high actor `mean(V(target_i,g_j)) - mean(V(s_i,g_j))`
  The parquet has one row per (obs_idx, goal_idx) plus
  `checkpoint_path, agent, dataset, phase, seed, configuration, actor,
  actor_loss, batch_idx, is_positive`.
- `compute_afr_metrics.py` derives `fr_auc`, `gap_mean`, `gap_std`,
  `extractability_index`, `adv_plus_mean` and `adv_minus_mean` per
  (checkpoint, actor), aggregating over batches unless `--no-aggregate`.

### Advantage and AWR-weight analyses

`analysis/advantages/` contains the per-question scripts. All of them accept
`--zipfiles` (one or more log zips) and `--top-k` (restrict to the top-k
configurations per agent), and write figures below
`plots/<zip name>/advantages/`:

| Script | Content |
| ------ | ------- |
| `a_distributions.py` | Advantage distributions |
| `b_stats.py` | Descriptive statistics over advantages |
| `c_weights.py` | AWR weight distributions |
| `d_correlation.py` | Advantage / return correlations |
| `e_ess.py` | Effective sample size of the AWR weights |
| `g_spearman.py` | Spearman rank correlation of advantages across seeds |
| `h_oracle.py` | Oracle (distance-based) advantage comparison |
| `export_top_k.py` | Exports the top-k configuration IDs per agent to JSON (`--top-k` required) |

```bash
python analysis/advantages/c_weights.py --zipfiles logs.zip --top-k 5
```

### Figure scripts

| Script | Content |
| ------ | ------- |
| `analysis/plot_a1_landscapes.py` | A1: straight-through vs. checkpoint-conditioned (phased) landscapes, GCIQL/QRL on `antmaze-medium-navigate-v0` |
| `analysis/plot_a3_ddpgbc_cube.py` | A3: DDPG+BC vs. AWR landscapes, CRL/GCIQL/QRL on `cube-single-play-v0` |
| `analysis/plot_a8_stitch.py` | A8: stitch vs. navigate datasets, four agents on AntMaze-Medium |
| `analysis/plot_nstep_mobility.py` | n-step TD mobility plots for GCIVL/GCIQL at every mobility threshold |
| `analysis/compare_nstep_breadth.py` | Breadth/peak table across n-step variants (`--out`, `--match-seeds`, `--dump-per-config`) |
| `analysis/plot_q_vs_distance.py` | Q-value vs. goal distance for one checkpoint |
| `analysis/run_q_vs_distance_all.py` | Runs `plot_q_vs_distance.py` for every entry in `analysis/best_checkpoints.json` |
| `figures/make_overview.py` | Paper teaser figure (Fig. 1) → `figures/approach-overview.{svg,pdf}` (needs a LaTeX installation) |

The A-series and n-step scripts read campaign trees from
`cluster-mount/{SSRL,GCRL}-Landscapes` and `icrl-resubmission/` (paths are
constants at the top of each script) and abort if their results do not
reproduce the validated reference numbers. The Q-vs-distance scripts read the
checkpoint/config pairs listed in `analysis/best_checkpoints.json` (or the
`--checkpoint`/`--config-path` arguments).

Run `analysis/run_q_vs_distance_all.py` from the repository root with a
directory named `gcrl-cluster/` in it: every `checkpoint` and `config` value in
`analysis/best_checkpoints.json` starts with `gcrl-cluster/…` and is resolved
relative to the working directory, and the script also opens
`analysis/best_checkpoints.json` relative to the working directory. Without
that directory the run stops at the first missing checkpoint.

## paper-metrics

[`paper-metrics/`](paper-metrics/README.md) is the self-contained,
cluster-free reproduction pipeline for the numeric tables of the paper: each
script in `paper-metrics/scripts/` reads data from `paper-metrics/data/` and
writes one CSV per table to `paper-metrics/outputs/`.

- Main results: `t1_final_landscape_ci.py` (`tab:final_landscape_ci`),
  `t2_compact_diagnostics.py` (`tab:compact_diagnostics`),
  `t3_antmaze_ess.py` (`tab:antmaze_ess_phase4`, `tab:antmaze_ess_full`),
  `t4_adv_corr.py` (`tab:adv_corr`).
- Robustness appendix: `a2_seed_variance.py` (`tab:seed_variance_summary`),
  `a3_phase_mobility.py` (`tab:phase_mobility`),
  `a4_subsample_stability.py` (`tab:subsample_stability_close_pairs`),
  `a5_wmax_sensitivity.py` (`tab:wmax_sensitivity`),
  `a6_adv_norm_sensitivity.py` (`tab:adv_norm_sensitivity`),
  `a7_sensitivity_summary.py` (`tab:sensitivity_summary`).

```bash
cd paper-metrics
./reproduce_all.sh            # all scripts
./reproduce_all.sh main       # t1..t4 only
./reproduce_all.sh appendix   # a2..a7 only
./reproduce_all.sh rebuttal   # rl1..rl8 + supplementary figures
```

The data prerequisites (four fixed-batch zips, four advantage parquets, four
eval-stats CSVs), the exact conventions (IQM seed reduction, `w_max = 100`,
Kish ESS, top-5 mass `k = 13`, MRR) and the known issues are documented in
[`paper-metrics/README.md`](paper-metrics/README.md).
[`paper-metrics/PROVENANCE.md`](paper-metrics/PROVENANCE.md) is the audit
trail: what the release zips contain, how each artefact derives from them, and
what cannot be regenerated offline (the `params_*.pkl` checkpoints are not in
the zips).

## Tests and quality gates

```bash
pytest tests -q
ruff check src tests analysis paper-metrics/scripts
```

`ruff check` and `pytest` are the enforced gates: CI runs
`ruff check src tests analysis paper-metrics/scripts` and `pytest -q` on
Python 3.10 — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

`tests/` contains `test_catalog_checkpoints.py` (checkpoint discovery on
synthetic trees), `test_extract_training_log.py` (optimised CSV extraction in
`read_results_from_zip`) and `test_new_agents_static.py` (shape/finiteness
tests for FQL, the n-step targets and rejection sampling; no dataset download
needed).

Two further checks exist, but neither is a gate:

- `hatch run types:check` (mypy over `src/gcrl_landscapes` and `tests`) is
  advisory: type checking is not yet enforced in CI, so mypy findings do not
  block a merge.
- `ruff format --check` is not part of CI either; the `ruff-format`
  pre-commit hook rewrites files instead of gating them.

`pre-commit run --all-files` therefore modifies files when it runs
`ruff-format`; it also checks for large files and whitespace, and it checks the
commit message (conventional commits via `git-conventional-commits.yaml`).

## Repository layout

| Path | Content |
| ---- | ------- |
| `src/gcrl_landscapes/` | Package: `main.py` (CLI), `submission.py` (setup/submit/run), `training.py`, `configurations.py`, `phase_splitting.py`, `evaluate.py`, `hpo.py` |
| `src/gcrl_landscapes/evaluation/` | `plot.py`, `tabular.py`, `landscapes.py`, `common.py` |
| `src/gcrl_landscapes/plots/` | IGPR triple-GP model (`triple_gp.py`) |
| `src/gcrl_landscapes/agents/` | In-repo agents: `fql.py`, `nstep.py` |
| `src/gcrl_landscapes/util/` | Dataset wrappers, data model, evaluation helpers |
| `src/gcrl_landscapes/visualization/` | Maze rollout rendering, dataset heatmaps |
| `analysis/` | Checkpoint catalog, advantage generation, AFR metrics, figure scripts, `analysis/advantages/` |
| `paper-metrics/` | Paper table reproduction pipeline (`README.md`, `PROVENANCE.md`, `scripts/`, `reproduce_all.sh`) |
| `figures/` | `make_overview.py` (writes `figures/approach-overview.svg` and `.pdf`) |
| `configs/` | `defaults.yaml`, `hpo_<agent>.yaml`, `smac.yaml`, `launcher.yaml`, `cluster/`, `hydra/sweeper/search_space/` |
| `docs/` | `quickstart.md`, `cli-reference.md`, `figures/` |
| `tests/` | pytest suite |
| `*.sh` (root) | Slurm orchestration scripts (see above) |
| `evaluate.py`, `evaluate_advantages.py` | Notebook-style evaluation scripts driven by `zip_and_plot.sh` (`--zipfiles`) |

## Documentation

- [`docs/quickstart.md`](docs/quickstart.md) — single-node, small-scale
  walkthrough: setup → one local run → evaluate → plot.
- [`docs/cli-reference.md`](docs/cli-reference.md) — every subcommand and
  entry point with a full flag table.
- [`docs/figures/actor-loop.md`](docs/figures/actor-loop.md) — the
  advantage → actor → test loop diagram.
- [`CHANGELOG.md`](CHANGELOG.md) — release history.

## Citation

If you use this software, please cite it as described in
[`CITATION.cff`](CITATION.cff):

```bibtex
@software{gcrl_landscapes,
  title  = {GCRL-Landscapes},
  author = {Töpperwien, Jan Malte and Mohan, Aditya},
  year   = {2026},
  url    = {https://github.com/LUH-AI/GCRL-Landscapes},
  note   = {Institute of AI, Leibniz University Hannover}
}
```

## License

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Institute of AI at Leibniz
University Hannover.
