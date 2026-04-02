# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
