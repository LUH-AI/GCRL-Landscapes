# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-30

First public release, prepared from the `develop` line for publication.

### Added
- **Agents**: MQE and FQL support in the landscape framework
- **Training**: N-step returns and rejection sampling, exposed as `setup` CLI options
- **Paper metrics**: `paper-metrics/` pipeline that reproduces the paper tables and figures from raw results
- **Analysis**: checkpoint catalog, cross-goal advantage generation and advantage/AFR analysis pipeline
- **Analysis**: figure-reproduction scripts for the camera-ready plots
- **Documentation**: full project overview, usage guide and reproduction instructions

### Changed
- History reunified: the `develop` line is merged onto `main`
- Version scheme reset to `1.0.0` for the public release

---

## [0.5.2] - 2026-04-29

### Added
- Advantage and AWR weight distribution analysis, with per-sample advantage logging on the held-out validation batch
- BFS maze-graph goal distances for locomaze environments and scene-environment support for goal distances
- DDPGBC actor loss support and a GCIQL HPO config
- `evaluate.py` replaces the `evaluate.md` notebook and accepts `--zipfiles`
- `submit_train_all.sh` cluster-aware submission wrapper and a kisski cluster profile
- Tests for `extract_training_log`

### Changed
- Held-out validation batch is now identical across evaluations
- Plots are written as PDF instead of PNG
- Pin `jax < 0.7.0` and `setuptools < 81`, update `ogbench`

### Fixed
- Out-of-memory failures in training metrics via `chunked_vmap` and earlier gradient deallocation
- Alternative advantage/held-out column handling in the tabular and evaluate pipelines
- Dataset seeding, `MixedDataset` parameter propagation and ambiguous truth value in `_split_indices`
- Cluster conda activation in Slurm batch jobs

---

## [0.5.1] - 2026-04-08

### Added
- Cluster profiles, replacing branch-based cluster configuration
- Reservation support for the `luis` cluster

### Changed
- Phase boundaries use maximum performance for the phase setting

### Fixed
- Reservation is passed to Slurm via `additional_parameters`
- Submission scripts use `SLURM_SUBMIT_DIR` for the cluster config path
- `evaluate.md` is runnable as a script

### Documentation
- Rewrote `README.md` with a full project overview and usage guide
- Added `CLAUDE.md` project guidance

---

## [0.5.0] - 2026-04-02

### Added
- **Evaluation**: Jupyter notebook (`evaluate.md`) for fast evaluation iteration, now self-contained with jupytext setup cells
- **Evaluation**: Combined landscape analysis with gradient plots and seed marginalization
- **Evaluation**: Importance uniformity and stability plots
- **Evaluation**: Ability to start kernels for HP importance data from notebook
- **Evaluation**: Parse and display training logs in evaluation pipeline
- **Evaluation**: Plot alpha instead of discount in mobility analysis
- Notebook analysis: epsilon-optimality with seed marginalization, CDF for cosine similarity, bad configuration inspection, quantile melting
- `feat(tabular)`: allow starting Jupyter kernels from tabular module

### Changed
- Sync `low_alpha` and `high_alpha` to `alpha` when only `alpha` is set in configurations

### Fixed
- Drop synced `low_alpha`/`high_alpha` columns before `compute_additional_information` to avoid duplicate column errors
- GP fitting robustness improvements
- Pandas indexing, GP thread count, and tabular pipeline bugs
- Debug label and hardcoded thread count in evaluation
- `del args` NameError in evaluation entrypoint
- Return empty dict instead of error when training logs are not present in data module

### Style
- Apply ruff formatting to configurations and data modules

---

## [0.4.0] - 2025

### Added
- **Training**: Gradient interference computation
- **Training**: Gradient scale logging
- **Training**: Feature embedding rank calculation
- **Training**: Pairwise metrics with deduplication (pairs from same trajectory/task removed)
- **Training**: Log target-value-network values
- **Training**: Configurable amount of training logs (instead of fixed interval)
- **Training**: Goal distance calculation for PointEnv
- **Training**: Held-out batch metrics for actual Adam updates
- **Training**: Log 100 quantiles during training
- **Evaluation**: Phase-indexed plotting
- **Evaluation**: Extract phase settings utility
- **HPO**: GCIVL agent configuration
- **Convergence**: Convergence script for new environments
- **Profiling**: Profiling script
- NixOS/CUDA support via `shell.nix`

### Fixed
- GCIVL HPO config fixes (extra alpha, missing config)
- Training: correct duplicate mask, remove forgotten import, reduce VRAM consumption
- Gradient magnitude similarity computation
- `load_or_compute` was only identifying cached results by filepath
- Zip binary path

### Changed
- Reduced evaluation episodes per task to 10

---

## [0.3.0]

### Added
- **Evaluation**: Regret tables (cumulative regret using IQM, normalized regret, multi-column regret)
- **Evaluation**: Merged regret table and mean diff table
- **Evaluation**: Table aggregations (regret, IGPR fit, phased general, extra function)
- **Evaluation**: Tables without pure-explore baseline
- **Evaluation**: Phase-indexed optimum-shift tables (many combinations)
- **Evaluation**: Performance optimizations and caching of parsed results
- **Plotting**: Regret plot for picking configuration at phase *i*
- **Plotting**: Epsilon-optimality with discrete levels
- **Plotting**: Option to disable multiprocessing

### Fixed
- Epsilon-optimality shifted bucket boundaries
- Grid plots incorrectly recognized as experiments
- Cumulative regret used inner-join, dropping unmatched eval steps
- Tables: recognize max columns correctly
- Tables: add forgotten constant dataset aggregation column
- Tables: compute additional information before merging
- Tables: pair merged experiments by HP setup

---

## [0.2.2]

### Added
- **Evaluation**: Hyperparameter importance table using cosine similarity
- **Evaluation**: Parser for hyperparameter importance data
- **Evaluation**: Optimum-shift analysis
- **Evaluation**: Variable scaling before fitting GP model (log-scaled variables)
- **Evaluation**: Use predicted values for scaling
- **Evaluation**: Add 0.95 epsilon-optimality bin
- **Evaluation**: Allow table creation from multiple zip files
- **Evaluation**: Refactored and documented table functions
- **Visualization**: Dataset heatmaps (KDE-based, multiprocessing, ice color palette)
- **Configurations**: Bounds and Sobol codomain helper; tau support for HIQL

### Fixed
- HP importance output folder not created automatically
- HPO: add agent name to log filename to prevent overwrites; fix script log names

---

## [0.2.1]

### Added
- Pick seed closest to IQM for phased HPO and landscape pipelines
- Improved README (WIP)

---

## [0.2.0]

Initial tagged release.
