# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GCRL-Landscapes is a hyperparameter landscape exploration framework for offline goal-conditioned RL agents. It generates diverse hyperparameter configurations, trains agents in phases across Slurm clusters, and analyzes how optimality landscapes evolve during training.

## Commands

### Linting & Formatting
```bash
ruff check src/
ruff format src/
hatch run types:check   # mypy type checking
pre-commit run --all-files
```

### Running the main workflows
```bash
# 1. Setup experiment (generate configs and phases)
python -m gcrl_landscapes.main setup --agent CRL --datasets antmaze-medium-navigate --n_configurations 100 --logdir /path/to/logs

# 2. Submit Slurm jobs
python -m gcrl_landscapes.main submit --logdir /path/to/logs --n_seeds 5

# 3. Run a single configuration (called by Slurm jobs)
python -m gcrl_landscapes.main run --logdir /path/to/logs --phase_idx 0 --configuration 0 --seed 0

# Evaluate results
python -m gcrl_landscapes.evaluation.plot --plot_return_distributions --plot_eval_curves --plot_gp_fits --zipfile LOGFILEPATH
python -m gcrl_landscapes.evaluation.tabular --zipfile LOGFILEPATH
```

### Orchestration scripts (Slurm)
```bash
bash train_all.sh      # Full landscape experiments across all agents/environments
bash hpo.sh            # Phase-based Bayesian HPO via SMAC
bash convergence.sh    # Convergence testing
bash zip_and_plot.sh   # Post-job zip + evaluation
```

## Architecture

### Three-Phase Workflow

**Setup → Submit → Run**

1. `main.py setup` — Creates folder structure, generates hyperparameter configs via Sobol sampling, computes phase splits from convergence data (`configs/defaults.yaml` → `convergence_zip_path`); phase boundaries computed in `phase_splitting.py` by interpolating performance curves to reach 95% of target performance
2. `main.py submit` — Wraps `submission.py` to launch Slurm array jobs, chains zip/evaluation jobs as dependencies
3. `main.py run` — Executes one (config, phase, seed) triple via `training.py::train()`

### Training Loop (`training.py`)

- JAX/Flax-based; loads agents from **OGBench** agent classes
- Datasets loaded via `util/datasets.py::MixedDataset` (blends explore/navigate splits)
- Evaluates at specified steps, saves metrics to CSV and checkpoints
- Supported agents: CRL, CMD, GCBC, GCIQL, GCIVL, QRL, HIQL, SAC

### Configuration Space (`configurations.py`)

Hyperparameters with bounds (log-scale where appropriate):
- `lr` (1e-6 to 1e-2), `discount`, `actor_p_trajgoal`, `alpha`, `eps`, `low_alpha`, `high_alpha`, `tau`
- Agent-specific subsets defined per agent type
- `low_alpha`/`high_alpha` sync logic: if only `alpha` is set, both are synced to it

### Results & Data Model (`util/data.py`)

- `PhaseResult`: `config_id → (seed → EvalTrajectory)`
- `EvalTrajectory`: evaluation metrics + checkpoint path
- `EvaluationResult`: success rate, per-step metrics, info dict
- Results serialized to JSON per phase inside zip archives

### Evaluation Pipeline

- `evaluation/common.py` — Computes CVaR, normalized goal distances, regret metrics; seeds aggregated via **IQM** (interquartile mean: trim 25% each tail); regret computation requires ≤2 varied HPs
- `evaluation/plot.py` — GP-based landscape visualization using `plots/triple_gp.py::TripleGPModel` (IGPR: three independent GPs for lower/IQM/upper confidence bounds via GPFlow)
- `evaluation/tabular.py` — Aggregated tables across phases/seeds
- `visualization/maze.py` — Renders agent trajectories overlaid on maze environments
- `visualization/dataset_heatmap.py` — KDE-based heatmaps of dataset coverage

### Hydra + SMAC Integration

- `configs/hpo_*.yaml` — Agent-specific HPO configs (compose base defaults + search spaces)
- `configs/hydra/sweeper/search_space/*.yaml` — Per-agent HP bounds for SMAC
- `hpo.py` — Incremental phase-based HPO with incumbent tracking
- Submitit launcher for distributed SMAC evaluations

## Key Dependencies

- **JAX/Flax** (CUDA 12) — training backend
- **OGBench** — offline RL environments and agent implementations
- **Gymnasium** — environment interface (antmaze, humanoid, cube, powderworld)
- **SMAC 2.3+** — Bayesian hyperparameter optimization
- **Hydra** — configuration management
- **Submitit** — Slurm job submission
- **GPFlow** — Gaussian processes for landscape fitting
- **Hatchling** — build system; Python 3.10 required (3.10 ≤ version < 3.11)

## Development Environment

Uses Nix (`flake.nix`) for reproducible environments. Alternatively, install via:
```bash
pip install -e ".[dev]"
pre-commit install
```

Conventional commits are enforced via pre-commit hook.

There are no formal test files; `ruff` + `mypy` are the main quality gates.
