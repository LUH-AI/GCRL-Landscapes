#!/usr/bin/env bash

#SBATCH --partition=ai,tnt
#SBATCH --job-name=TrainSetup
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-user=m.toepperwien@ai.uni-hannover.de
#SBATCH --mail-type=END,FAIL
#SBATCH --get-user-env

logdir="./logs"
hyperparameters="lr alpha"
actorloss="awr"
numconfigurations="64"
convergencezip="./convergence.zip"
phasepercentages="5 50 75 100"
finalperformancepercentage="95"
numevalepisodes="1"
nseeds="5"
taskspernodetotal="16"
taskspernodeparallel="4"
partitions="ai,tnt"
mempercpu="3G"

# module load Miniforge3
# conda activate gcrl

declare -a -r agents=(
  # "CRL"
  # "GCBC"
  # "GCIQL"
  # "GCIVL"
  # "HIQL"
  "QRL"
)
declare -A -r agent_min_per_mill_steps=(
  ["CRL"]="300"
  ["GCBC"]="200"
  ["GCIQL"]="300"
  ["GCIVL"]="300"
  ["HIQL"]="300"
  ["QRL"]="500"
)

declare -a -r environments=(
  "antmaze-medium-explore-v0 antmaze-medium-explore80navigate-v0 antmaze-medium-explore40navigate-v0 antmaze-medium-navigate-v0"
  # "antmaze-medium-explore-v0"
  # "antmaze-medium-navigate-v0"
  # "antmaze-medium-explore10navigate-v0"
  # "antmaze-medium-explore20navigate-v0"
  # "antmaze-medium-explore40navigate-v0"
  # "antmaze-medium-explore80navigate-v0"
  # "antmaze-medium-stitch-v0"
  # "antmaze-medium-explore10stitch-v0"
  # "antmaze-medium-explore20stitch-v0"
  # "antmaze-medium-explore40stitch-v0"
  # "antmaze-medium-explore80stitch-v0"
  # "antmaze-large-navigate-v0"
  # "antmaze-large-stitch-v0"
  # "antmaze-large-explore-v0"
  # "antmaze-teleport-navigate-v0"
  # "antmaze-teleport-stitch-v0"
  # "antmaze-teleport-explore-v0"
  # "humanoidmaze-medium-navigate-v0"
  # "humanoidmaze-medium-stitch-v0"
  # "humanoidmaze-large-navigate-v0"
  # "humanoidmaze-large-stitch-v0"
  # "cube-single-play-v0"
  # "cube-double-play-v0"
  # "powderworld-easy-play-v0"
)

# cache autotuning results to not having to recompile them on every run
# !DEACTIVATED for now due to strange caching bugs
# if [ -z ${BIGWORK+x} ]; then
#   export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"
# else
#   export JAX_COMPILATION_CACHE_DIR="${BIGWORK}/jax_cache"
# fi

for environment in "${environments[@]}"; do
  echo "Starting jobs for ${environment}"
  for agent in "${agents[@]}"; do
    echo "Starting job for ${agent}"
    full_log_dir="${logdir}/${agent}_${environment}_${numconfigurations}c_${hyperparameters// /-}"

    python -m gcrl_landscapes.main setup \
      --agent "$agent" \
      --datasets $environment \
      --n_configurations "$numconfigurations" \
      --convergence_zip "$convergencezip" \
      --phase_percentages $phasepercentages \
      --final_performance_percentage $finalperformancepercentage \
      --eval_episodes "$numevalepisodes" \
      --hyperparameters $hyperparameters \
      --logdir "$full_log_dir" \
      --final_step_is_phase \
      --actor_loss "$actorloss"
    python -m cProfile -o profile.pstats -m gcrl_landscapes.main run \
      --logdir "$full_log_dir" \
      --phase_idx 0 \
      --configuration 0 \
      --seed 0
  done
done
