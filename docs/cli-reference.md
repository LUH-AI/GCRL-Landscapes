# CLI reference

Every command-line entry point of the project, generated from the `argparse`
definitions in the source. Flag names, types and defaults are quoted verbatim
from the parsers; the "Meaning" column follows the `help=` strings.

- [Training CLI: `gcrl_landscapes.main`](#training-cli-gcrl_landscapesmain)
  - [`setup`](#setup)
  - [`submit`](#submit)
  - [`run`](#run)
  - [Validation rules and derived behaviour](#validation-rules-and-derived-behaviour)
- [Evaluation entry points](#evaluation-entry-points)
  - [`gcrl_landscapes.evaluation.plot`](#gcrl_landscapesevaluationplot)
  - [`gcrl_landscapes.evaluation.tabular`](#gcrl_landscapesevaluationtabular)
  - [`evaluate.py`](#evaluatepy)
  - [`evaluate_advantages.py`](#evaluate_advantagespy)
- [Analysis entry points](#analysis-entry-points)
  - [`analysis/catalog_checkpoints.py`](#analysiscatalog_checkpointspy)
  - [`analysis/generate_advantages.py`](#analysisgenerate_advantagespy)
  - [`analysis/compute_afr_metrics.py`](#analysiscompute_afr_metricspy)
  - [`analysis/advantages/*.py`](#analysisadvantagespy)
  - [`analysis/plot_q_vs_distance.py`](#analysisplot_q_vs_distancepy)
  - [`analysis/run_q_vs_distance_all.py`](#analysisrun_q_vs_distance_allpy)
  - [`analysis/compare_nstep_breadth.py`](#analysiscompare_nstep_breadthpy)
  - [`analysis/plot_a1_landscapes.py`, `plot_a3_ddpgbc_cube.py`, `plot_a8_stitch.py`](#analysisplot_a1_landscapespy-plot_a3_ddpgbc_cubepy-plot_a8_stitchpy)
  - [`analysis/plot_nstep_mobility.py`](#analysisplot_nstep_mobilitypy)
  - [`figures/make_overview.py`](#figuresmake_overviewpy)
- [Shell drivers](#shell-drivers)
- [HPO entry point: `gcrl_landscapes.hpo`](#hpo-entry-point-gcrl_landscapeshpo)

---

## Training CLI: `gcrl_landscapes.main`

```bash
python -m gcrl_landscapes.main {setup,submit,run} [flags]
```

Source: `src/gcrl_landscapes/main.py`. Subcommands are required
(`subparsers(required=True)`); `setup` dispatches to
`submission.run_setup`, `submit` to `submission.submit`, `run` to
`submission.run_config_wrapper`.

Run all commands from the repository root: `setup` and `run` record
`git.Repo(".", search_parent_directories=True)` (commit and branch) into
`info.toml`, so a git repository must be reachable from the working directory.

### `setup`

Creates the folder structure with metadata for an experiment. Writes
`<logdir>/info.toml`, `<logdir>/configurations/configspace.json` and
`<logdir>/configurations/configuration_<i>.json`.

| Flag | Type | Default | Meaning |
| ---- | ---- | ------- | ------- |
| `--agent` | str, choice of `SUPPORTED_AGENTS` | required | Agent to run |
| `--datasets` | str, `nargs="+"` | required | Which dataset/environment to run (one name per phase, or one name that is repeated for every phase) |
| `--n_configurations` | int | required | How many configurations to run/sample across the configuration space |
| `--hyperparameters` | str, `nargs="+"` | required | Which configuration space to use (exactly two names) |
| `--eval_episodes` | int | required | How many evaluation episodes to run per evaluation |
| `--logdir` | Path | required | In which directory to save results |
| `--convergence_zip` | Path | required¹ | Zip file which contains logs to determine phases using training results |
| `--final_phase` | int | required¹ | How many steps the final phase has |
| `--final_performance_percentage` | int | `95` | How much of the benchmarked performance the final evaluation should reach; used for setting phase steps, only with `--convergence_zip` |
| `--phase_percentages` | int, `nargs="+"` | `[25, 50, 100]` | At which percentages phases should be run |
| `--extra_eval_steps` | int, `nargs="+"` | `[]` | At what steps to evaluate |
| `--extra_save_steps` | int, `nargs="+"` | `None` | At what steps to save a checkpoint; checkpoints are saved for the best configurations |
| `--seed` | int | `0` | Which seed to use for configuration sampling |
| `--final_step_is_phase` | flag | `False` | Train configs in a phase only till end of phase |
| `--actor_loss` | str | `None` | Sets `actor_loss` in the configuration; may be ignored by some algorithms |
| `--normalize_advantages` | flag | `False` | Enable advantage normalization before AWR weighting; may be ignored by some algorithms |
| `--n_step` | int | `1` | n-step TD target horizon for GCIQL/GCIVL; ignored by other agents |
| `--rejection_sampling_n` | int | `0` | Number of candidates for eval-time rejection sampling (`0` = disabled); only GCIQL/CRL/FQL |

¹ `--convergence_zip` and `--final_phase` are a required, mutually exclusive
group: give exactly one.

### `submit`

Submits the jobs, chained per phase, and prints
`SUBMITTED_ARRAY_ID=<array id>`. Reads `<logdir>/info.toml`, so `setup` must
have run first.

| Flag | Type | Default | Meaning |
| ---- | ---- | ------- | ------- |
| `--logdir` | Path | required | In which directory to save results; has to be already initialized using `setup` |
| `--n_seeds` | int | required | How many seeds to run (seeds `0 .. n_seeds-1`) |
| `--tasks_per_node_total` | int | required | How many tasks/configurations should run per node in total |
| `--tasks_per_node_parallel` | int | required | How many tasks/configurations should run per node in parallel |
| `--jobname` | str | required | How to name the job in Slurm |
| `--agent_path` | Path | `None` | Which checkpoint to load; searches the best checkpoint itself if not given. **Parsed but unused by `submit`** — the generated per-task arguments do not include it |
| `--partition` | str | `"ai"` | On which Slurm partition to schedule the jobs |
| `--reservation` | str | `""` | Slurm reservation name (empty string = no reservation) |
| `--gres` | str | `""` | Slurm GRES spec for GPU (e.g. `gpu:h100:1`); if empty, uses `gpus_per_node=1` |
| `--mem_per_cpu` | str | `"3G"` | Memory given to node per allocated CPU |
| `--min_per_mill_steps` | int | `300` | How many minutes should be allocated on Slurm as limit per million steps taken |
| `--basetime` | int | `10` | Time allocation per task independent from training steps |
| `--phase_indices` | int, `nargs="+"` | `None` | Index of which phases to run; runs all if not given |

Fixed in code (not flags): `cpus_per_task=3`, `slurm_array_parallelism=50`,
and `XLA_PYTHON_CLIENT_MEM_FRACTION` derived from
`--tasks_per_node_parallel`. Job mail (`slurm_mail_type="END,FAIL"`) is only
configured when the `SLURM_MAIL_USER` environment variable is set. All
inherited `SLURM*` environment variables are deleted before submission.

### `run`

Executes one (configuration, phase, seed) triple. The Slurm tasks call this;
it also works locally without Slurm.

| Flag | Type | Default | Meaning |
| ---- | ---- | ------- | ------- |
| `--logdir` | Path | required | In which directory to save results; has to be already initialized using `setup` |
| `--phase_idx` | int | required | Which phase to run (index into `info.toml:arguments.phases`) |
| `--configuration` | int | required | Which configuration (index) to run |
| `--seed` | int | required | Which seed to use for training |
| `--agent_path` | Path | `None` | Which checkpoint to load; searches the best checkpoint if not given |
| `--tasks_per_node_parallel` | int | `1` | Limits JAX GPU pre-allocation in case of multiple jobs running; see `submit` |

Writes `<logdir>/run_logs/configuration_<i>/phase_<steps>/seed_<s>/` with
`eval_log.csv`, `train_log.csv`, `eval_trajectory.json`, `info.toml` and
`params_<steps>.pkl`. Sets `MUJOCO_GL=egl` and
`XLA_PYTHON_CLIENT_MEM_FRACTION` before importing JAX.

### Validation rules and derived behaviour

Checked in `submission.py` and `configurations.py`:

| Rule | Behaviour |
| ---- | --------- |
| `--n_step > 1` with an agent other than GCIQL/GCIVL | `ValueError: --n_step > 1 only supported for GCIQL/GCIVL` |
| `--rejection_sampling_n == 1` | `ValueError: --rejection_sampling_n must be 0 or >1` |
| `--rejection_sampling_n > 1` with an agent other than GCIQL/CRL/FQL | `ValueError: --rejection_sampling_n only supported for GCIQL/CRL/FQL` |
| `len(--datasets) != 1` and `!= len(--phase_percentages)` | `ValueError: Number of datasets (…) does not match number of phase percentages (…)` |
| `--n_configurations` not a power of two and `> 1` | `ValueError: Only supports powers of 2` (Sobol design) |
| `--n_configurations 1` | No sampling; the adapted default config of the agent is used as configuration 0 |
| `--hyperparameters` not exactly two names from `lr`, `discount`, `actor_p_trajgoal`, `alpha`, `low_alpha`, `high_alpha`, `eps`, `tau` | `NotImplementedError` (`SUPPORTED_HPS` in `configurations.py`; `Only supports two hyperparameters at the moment`) |
| `--final_phase` given without `--convergence_zip` | `phases = [int(pct/100 * final_phase) for pct in phase_percentages]` |
| `--convergence_zip` given | Phase boundaries are interpolated from the log zip until `--final_performance_percentage` of the best success is reached (`phase_splitting.py::get_all_phases`, mode `target_ratio`, linear interpolation); the returned list holds cumulative training steps |
| `--hyperparameters` contains `actor_p_trajgoal` | `actor_p_randomgoal = 1 - actor_p_trajgoal`, `actor_p_curgoal = 0` |
| Only `alpha` in `--hyperparameters` | `low_alpha` / `high_alpha` are synced to `alpha` if the agent config has them |

---

## Evaluation entry points

### `gcrl_landscapes.evaluation.plot`

```bash
python -m gcrl_landscapes.evaluation.plot --zipfile <zip> [flags]
```

Reads a log zip with the `logs*/<experiment>/info.toml` layout and writes to
`plots/<zip file name>/`, with cross-experiment grids in
`plots/<zip file name>/grid_plots/`.

| Flag | Type | Default | Meaning |
| ---- | ---- | ------- | ------- |
| `--zipfile` | Path | required | The (single) log zip to plot |
| `--plot_return_distributions` | flag | `False` | Plot per-configuration return distributions |
| `--plot_eval_curves` | flag | `False` | Plot success/eval curves over training steps |
| `--plot_gp_fits` | flag | `False` | Plot GP fit quality |
| `--plot_regret` | flag | `False` | Plot regret curves |
| `--plot_advantages` | flag | `False` | Plot advantage / AWR-weight distributions (loads `train_log.csv`) |
| `--no_plot_landscapes` | flag | `False` | Disable the IGPR landscape plots (enabled by default) |
| `--no_plot_mobility` | flag | `False` | Disable the phase-mobility plots (enabled by default) |
| `--no_multiprocessing` | flag | `False` | Process experiments serially instead of in a spawn pool (`SLURM_CPUS_ON_NODE // 4` or `cpu_count() // 4` workers) |

### `gcrl_landscapes.evaluation.tabular`

```bash
python -m gcrl_landscapes.evaluation.tabular --zipfiles <zip> [<zip> …] --output_folder <dir>
```

Note the plural `--zipfiles` (one or more), unlike `plot`'s singular
`--zipfile`.

| Flag | Type | Default | Meaning |
| ---- | ---- | ------- | ------- |
| `--zipfiles` | Path, `nargs="+"` | required | Log zips to aggregate |
| `--output_folder` | Path | required | Directory for the generated tables |
| `--hp_importance_data` | flag | `False` | Zip contains HP importance data; only create tables based on HP importance |
| `--convergence_data` | flag | `False` | Zip contains convergence data; only create tables based on convergence data |
| `--no_multiprocessing` | flag | `False` | Process experiments serially |
| `--start_kernel` | flag | `False` | Drop into an IPython kernel (`IPython.start_kernel(user_ns=locals())`) after loading the data instead of running the tables inline |

Outputs `table_all_phases.{md,tex,csv}`, `table_last_phase.{md,tex,csv}`,
`igpr_fit_table.*`, `<regret>_table_{first,second}_phase.*`,
`<regret>_table.*`, mean-difference tables and `per_config/` CSVs.

### `evaluate.py`

Notebook-style script (cell markers `# %%`), driven by `zip_and_plot.sh`.

| Flag | Type | Default | Meaning |
| ---- | ---- | ------- | ------- |
| `--zipfiles` | Path, `nargs="+"` | required | Log zips; output goes to `plots/<first zip name>/` and `tables/<first zip name>/` |

### `evaluate_advantages.py`

Notebook-style script for advantage / AWR-weight analysis; output goes to
`plots/<first zip name>/advantages/`.

| Flag | Type | Default | Meaning |
| ---- | ---- | ------- | ------- |
| `--zipfiles` | Path, `nargs="+"` | required | Log zips to analyse |

---

## Analysis entry points

### `analysis/catalog_checkpoints.py`

| Flag | Type | Default | Meaning |
| ---- | ---- | ------- | ------- |
| `--logdir` | Path | required | Parent log directory containing agent subdirectories |
| `--output` | Path | `None` | Optional CSV file to write the DataFrame to |

Prints the table and, with `--output`, writes a CSV with the columns
`checkpoint_path, config_path, agent, dataset, phase, seed, configuration`.

### `analysis/generate_advantages.py`

| Flag | Type | Default | Meaning |
| ---- | ---- | ------- | ------- |
| `--catalog` | Path | required | CSV produced by `catalog_checkpoints.py` (must include the `config_path` column) |
| `--output` | Path | required | Output parquet file path |
| `--chunk-size` | int | `32` | Checkpoints per vmap batch; memory scales with `B × N² × hidden_dim × layers × 4` bytes, `32` is safe for 20 GB VRAM |
| `--num-workers` | int | `min(8, cpu_count)` | Worker threads for parallel checkpoint I/O (does not affect VRAM) |
| `--num-batches` | int | `1` | Number of independent validation batches per checkpoint |

### `analysis/compute_afr_metrics.py`

| Flag | Type | Default | Meaning |
| ---- | ---- | ------- | ------- |
| `--input` | Path | `advantages.parquet` | Path to the advantages parquet |
| `--output` | Path | `analysis/afr_metrics.csv` | Output CSV path |
| `--no-aggregate` | flag | `False` | Skip aggregation across batches (one row per batch instead of mean±std) |
| `--batches` | str | `None` | Comma-separated batch indices to include (`0,1,2` or `0-5`); all batches if unset |
| `--config-ids` | Path | `None` | JSON file with `{agent: [config_ids]}` to filter to |

### `analysis/advantages/*.py`

`a_distributions.py`, `b_stats.py`, `c_weights.py`, `d_correlation.py`,
`e_ess.py`, `g_spearman.py`, `h_oracle.py` and `export_top_k.py` share the
parser in `analysis/advantages/_common.py` (`parse_known_args`, so unknown
flags are ignored):

| Flag | Type | Default | Meaning |
| ---- | ---- | ------- | ------- |
| `--zipfiles` | Path, `nargs="+"` | required | One or more evaluation zip archives to analyse |
| `--top-k` | int | `None` | Restrict the analysis to the top-k configurations per agent; required by `export_top_k.py` |

`export_top_k.py` writes `top_k_<k>_<zip stem>.json` next to the first zip.

### `analysis/plot_q_vs_distance.py`

| Flag | Type | Default | Meaning |
| ---- | ---- | ------- | ------- |
| `--checkpoint` | two values: `PATH AGENT_NAME`, required | — | Checkpoint path and agent name |
| `--config-path` | Path | required | Config JSON path for the checkpoint |
| `--dataset` | str | `cube-single-play-v0` | OGBench dataset name |
| `--output` | Path | `analysis/plots/q_vs_distance.pdf` | Output figure path (PDF; a PNG is saved at the same path) |

### `analysis/run_q_vs_distance_all.py`

No flags. Reads `analysis/best_checkpoints.json` and writes
`analysis/plots/q_vs_distance_<env>_<agent>.pdf` for every entry.

### `analysis/compare_nstep_breadth.py`

| Flag | Type | Default | Meaning |
| ---- | ---- | ------- | ------- |
| `--out` | Path | `outputs/nstep_breadth.csv` | Output CSV |
| `--match-seeds` | flag | `False` | Restrict the published n=1 baseline to 3 seeds, matching the arms |
| `--dump-per-config` | Path | `None` | Also write per-configuration (lr, alpha, seed-reduced success) rows at the deepest phase |

### `analysis/plot_a1_landscapes.py`, `plot_a3_ddpgbc_cube.py`, `plot_a8_stitch.py`

| Flag | Type | Default | Meaning |
| ---- | ---- | ------- | ------- |
| `--out` | Path | `icrl-resubmission/plots/a1-straight-vs-phased` (a1), `…/a3-ddpgbc-cube` (a3), `…/a8-stitch-vs-navigate` (a8); `REPO_ROOT`-relative | Output directory for the generated PNG/PDF panels |

The input trees are constants at the top of each script
(`cluster-mount/SSRL-Landscapes`, `cluster-mount/GCRL-Landscapes`) and are not
flags.

### `analysis/plot_nstep_mobility.py`

No flags. Reads the same campaign trees, writes
`icrl-resubmission/plots/<agent>-nstep-<dataset>/n<n>/mobility-<threshold>.png`.

### `figures/make_overview.py`

No flags. Writes `figures/approach-overview.svg` and
`figures/approach-overview.pdf`; requires a LaTeX installation (`text.usetex`).

---

## Shell drivers

| Command | Meaning |
| ------- | ------- |
| `bash submit_train_all.sh [sbatch args…]` | `sbatch` wrapper for `train_all.sh`; honours `CLUSTER` |
| `bash train_all.sh` | Loops over the agents/environments declared in the file and runs `setup` + `submit` per pair; chains `zip_and_plot.sh` |
| `bash hpo.sh <agent>` | Submits `hpo_single.sh` for phases 25/50/75/100, each `afterok` the previous one |
| `bash hpo_single.sh <agent> <logdir> <datasets_str> <phases_str> <phase>` | One HPO phase (`hpo.sh` passes the arguments) |
| `bash convergence.sh` | One configuration per (agent, environment) with a long `--extra_eval_steps` list |
| `bash zip_and_plot.sh <logdir> <python>` | Zip without `*.pkl`/`submitit/`, then `evaluation.plot`, `evaluation.tabular`, `evaluate.py`, `evaluate_advantages.py` |
| `sbatch collect_advantages.sh [LOGDIR]` | Catalog → advantages → AFR metrics; env overrides `LOGDIR`, `CHUNK_SIZE` (default 8), `NUM_BATCHES` (10), `NUM_WORKERS` (4) |
| `bash profiling_sample.sh` | `setup` plus `cProfile -m gcrl_landscapes.main run` for one configuration |

All eight root scripts (`submit_train_all.sh`, `train_all.sh`, `hpo.sh`,
`hpo_single.sh`, `convergence.sh`, `zip_and_plot.sh`, `collect_advantages.sh`,
`profiling_sample.sh`) source `configs/cluster/${CLUSTER:-luh}.sh`. The
`sbatch` and `submit` calls that the drivers make themselves pass the
`CLUSTER_*` partition and reservation; a script you submit yourself with
`sbatch` takes them from your `sbatch` command line, because no script has a
`#SBATCH --partition` line. See the README section
[“`CLUSTER` and `configs/cluster/`”](../README.md#cluster-and-configscluster).

---

## HPO entry point: `gcrl_landscapes.hpo`

Hydra application (not an `argparse` CLI), started by `hpo_single.sh`:

```bash
python -m gcrl_landscapes.hpo --multirun \
  --config-name hpo_<agent> \
  +logdir=<dir> +datasets=<list> +phases=<list> +phase=<int> \
  hydra.launcher.partition=<partition>
```

Configs: `configs/hpo_{crl,gciql,gcivl,hiql,mqe,qrl}.yaml` (each composes
`defaults`, `hydra/sweeper/search_space/<agent>`, `smac`, `launcher`),
`configs/smac.yaml`, `configs/launcher.yaml`. The sweeper is
`hydra/sweeper: HyperSMAC` with the submitit Slurm launcher
(`hydra/launcher: submitit_slurm`).

## Paper-metrics pipeline

`paper-metrics/scripts/*.py` are single-table scripts; their flags and the
table mapping are documented in
[`paper-metrics/README.md`](../paper-metrics/README.md). The driver takes one
positional mode:

```bash
cd paper-metrics
./reproduce_all.sh            # all scripts
./reproduce_all.sh main       # t1..t4
./reproduce_all.sh appendix   # a2..a7
./reproduce_all.sh rebuttal   # rl1..rl8 + supplementary figures
```
