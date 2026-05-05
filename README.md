# GCRL-Landscapes

A hyperparameter landscape exploration framework for offline goal-conditioned RL agents. It generates diverse hyperparameter configurations, trains agents in phases across Slurm clusters, and analyzes how optimality landscapes evolve during training.

Supported agents: **CRL, GCIQL, GCIVL, QRL, HIQL** (via [OGBench](https://github.com/seohongpark/ogbench)).  
Environments: antmaze, humanoid, cube, scene (via [Gymnasium](https://gymnasium.farama.org/)).

Already generated convergence data required for phase splitting can be found on [Huggingface](https://huggingface.co/datasets/jmtoepperwien/GCRL-Landscapes). Precomputed log files are available there as well. All of this can be reproduced given the code

## Installation

```bash
pip install -e ".[dev]"
pre-commit install
```

A reproducible [Nix](https://nixos.org/) environment is also provided via `flake.nix`.

**Requirements:** Python 3.10, JAX/Flax with CUDA 12.

## Workflow

Training runs in three steps: **setup → submit → run**. All steps are orchestrated via `main.py`.

### 1. Setup

Generates hyperparameter configurations (Sobol sampling) and computes phase boundaries from convergence data.

```bash
python -m gcrl_landscapes.main setup \
  --agent CRL \
  --datasets antmaze-medium-navigate \
  --n_configurations 100 \
  --logdir /path/to/logs
```

### 2. Submit

Launches Slurm array jobs and chains zip/evaluation jobs as dependencies.

```bash
python -m gcrl_landscapes.main submit \
  --logdir /path/to/logs \
  --n_seeds 5
```

### 3. Run (called by Slurm)

Executes a single (config, phase, seed) triple.

```bash
python -m gcrl_landscapes.main run \
  --logdir /path/to/logs \
  --phase_idx 0 \
  --configuration 0 \
  --seed 0
```

For non-Slurm usage, modify the submitter in `src/gcrl_landscapes/submission.py`.

## Orchestration Scripts

Higher-level Slurm scripts covering common experiment types:

| Script                | Purpose |
| ------                 | ------- |
| `submit_train_all.sh` | Submit full landscape experiments across all agents/environments (auto-selects cluster partition) |
| `train_all.sh`        | Main training orchestration (called by submit_train_all.sh)                                       |
| `hpo.sh`              | Phase-based Bayesian HPO via SMAC                                                                 |
| `convergence.sh`      | Convergence testing                                                                               |
| `zip_and_plot.sh`     | Post-job zip + evaluation                                                                         |

Submit training via `bash submit_train_all.sh` (default: LUH cluster) or `CLUSTER=pc2 bash submit_train_all.sh` (PC2 cluster). This wrapper automatically selects the correct Slurm partition and reservation. Comment out agents/environments in `train_all.sh` you don't need.

## Phased HPO

Runs [SMAC](https://github.com/automl/SMAC3) on a Slurm cluster to perform phase-based Bayesian hyperparameter optimization. This yields both a tuned configuration and importance data across training phases.

```bash
bash hpo.sh
```

Configuration lives in `configs/hpo_*.yaml` and `configs/hydra/sweeper/search_space/`. For non-Slurm usage, modify the launcher in `configs/hpo_algorithm.yaml`.

Analyze gathered importance data with [DeepCAVE](https://github.com/automl/DeepCAVE).

## Evaluation

Generate plots and tables from a log zip archive:

```bash
python -m gcrl_landscapes.evaluation.plot \
  --plot_return_distributions \
  --plot_eval_curves \
  --plot_gp_fits \
  --zipfile LOGFILEPATH

python -m gcrl_landscapes.evaluation.tabular --zipfile LOGFILEPATH
```

Output goes to `plots/` and `tables/`. The `plots/grid_plots/` subdirectory gives a quick overview across all experiments.

Landscape visualization uses GP-based fitting (IGPR: three independent GPs for lower/IQM/upper confidence bounds via GPFlow).

## Analysis Pipeline

Compute cross-goal advantage landscapes from trained checkpoints via a two-step pipeline:

```bash
# Step 1: Catalog all available checkpoints
python analysis/catalog_checkpoints.py --logdir ./logs-antmaze-medium/ --output checkpoints.csv

# Step 2: Generate cross-goal advantage matrices
python analysis/generate_advantages.py --catalog checkpoints.csv --output advantages.parquet
```

`catalog_checkpoints.py` scans agent subdirectories (matched by prefix: CRL, QRL, GCIQL, GCIVL, CMD, GCBC, HIQL, SAC) for `params_{phase}.pkl` checkpoints in each agent's `run_logs/configuration_*/phase_N/seed_M/` tree, producing a CSV of (agent, dataset, phase, configuration, seed, checkpoint_path, config_path) entries.

`generate_advantages.py` iterates over cataloged checkpoints, loads each agent with a fixed 256-sample validation batch, and computes a cross-goal advantage matrix (NxN) for every (obs_i, action_i, goal_j) triple. Advantage computation per agent type:

- **GCIQL / CRL**: `advantage[i,j] = min(Q1,Q2)(s_i,a_i,g_j) - V(s_i,g_j)`
- **GCIVL**: `advantage[i,j] = mean(V(s'_i,g_j)) - mean(V(s_i,g_j))` (ensemble mean)
- **QRL**: `advantage[i,j] = V(s_i,g_j) - V(s'_i,g_j)` (negated quasimetric distances)
- **HIQL**: separate low_actor (`V(s'_i,g_j) - V(s_i,g_j)`) and high_actor (`V(target_i,g_j) - V(s_i,g_j)`) variants

The resulting parquet contains one row per (obs_idx, goal_idx) pair plus metadata (checkpoint_path, agent, dataset, phase, seed, configuration, actor, is_positive).

## Development

```bash
ruff check src/       # linting
ruff format src/      # formatting
hatch run types:check # mypy type checking
pre-commit run --all-files
```

Conventional commits are enforced via pre-commit hook. There are no formal test files; `ruff` and `mypy` are the main quality gates.
