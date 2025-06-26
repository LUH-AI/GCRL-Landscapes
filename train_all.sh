#!/usr/bin/env bash

logdir="./logs"
hyperparameters="discount actor_p_trajgoal"
actorloss="awr"
numconfigurations="32"
convergencezip="./convergence.zip"
phasepercentages="25 50 100"
finalperformancepercentage="90"
numevalepisodes="50"
nseeds="5"
taskspernodetotal="16"
taskspernodeparallel="4"
partitions="ai,tnt"
mempercpu="3G"

declare -a -r agents=(
  "CRL"
  "GCBC"
  "GCIQL"
  "GCIVL"
  "HIQL"
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
  "antmaze-medium-explore-v0"
  "antmaze-medium-navigate-v0"
  "antmaze-medium-explore10navigate-v0"
  "antmaze-medium-explore20navigate-v0"
  "antmaze-medium-explore40navigate-v0"
  "antmaze-medium-explore80navigate-v0"
  "antmaze-medium-stitch-v0"
  "antmaze-medium-explore10stitch-v0"
  "antmaze-medium-explore20stitch-v0"
  "antmaze-medium-explore40stitch-v0"
  "antmaze-medium-explore80stitch-v0"
  "antmaze-large-navigate-v0"
  "antmaze-large-stitch-v0"
  "antmaze-large-explore-v0"
  "antmaze-teleport-navigate-v0"
  "antmaze-teleport-stitch-v0"
  "antmaze-teleport-explore-v0"
  "humanoidmaze-medium-navigate-v0"
  "humanoidmaze-medium-stitch-v0"
  "humanoidmaze-large-navigate-v0"
  "humanoidmaze-large-stitch-v0"
  "cube-single-play-v0"
  "cube-double-play-v0"
  "powderworld-easy-play-v0"
)

# cache autotuning results to not having to recompile them on every run
if [ -z ${BIGWORK+x} ]; then
  export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"
else
  export JAX_COMPILATION_CACHE_DIR="${BIGWORK}/jax_cache"
fi

for environment in "${environments[@]}"; do
  echo "Starting jobs for ${environment}"
  for agent in "${agents[@]}"; do
    echo "Starting job for ${agent}"
    full_log_dir="${logdir}/${agent}_${environment}_${numconfigurations}c_${hyperparameters// /-}"

    python -m gcrl_landscapes.main setup \
      --agent "$agent" \
      --dataset "$environment" \
      --n_configurations "$numconfigurations" \
      --convergence_zip "$convergencezip" \
      --phase_percentages $phasepercentages \
      --final_performance_percentage $finalperformancepercentage \
      --eval_episodes "$numevalepisodes" \
      --hyperparameters $hyperparameters \
      --logdir "$full_log_dir" \
      --final_step_is_phase \
      --actor_loss "$actorloss"
    python -m gcrl_landscapes.main submit \
      --logdir "$full_log_dir" \
      --n_seeds "$nseeds" \
      --tasks_per_node_total "$taskspernodetotal" \
      --tasks_per_node_parallel "$taskspernodeparallel" \
      --mem_per_cpu "$mempercpu" \
      --jobname "${agent \
      -${environment}" \
      --partition "$partitions" \
      --min_per_mill_steps "${agent_min_per_mill_steps[${agent}]}"
  done
done

# Automatically zip and plot after all runs done
# Currently this waits for all jobs belonging to user and not only the ones submitted here
dependencies=$(squeue --me -l -r | tail -n +3 | tr -s ' ' | cut -d ' ' -f2 | cut -d '_' -f1 | sort | uniq | paste -s -d':')
sbatch --output "${logdir}/zip_and_plot_log.txt" -d "afterok:${dependencies}" ./zip_and_plot.sh "$logdir" $(which python)
