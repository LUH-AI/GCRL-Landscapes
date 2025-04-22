#!/usr/bin/env bash

logdir="./logs"
hyperparameters="discount actor_p_trajgoal"
numconfigurations="32"
phases="50000 100000 200000"
evalsteps="${phases} 250000"
numevalepisodes="50"
nseeds="5"
taskspernode="5"
partitions="ai,tnt"

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
  ["QRL"]="300"
)

declare -a -r environments=(
  "antmaze-medium-navigate-v0"
  "antmaze-medium-stitch-v0"
  "antmaze-large-navigate-v0"
  "antmaze-large-stitch-v0"
  "humanoidmaze-medium-navigate-v0"
  "humanoidmaze-medium-stitch-v0"
  "humanoidmaze-large-navigate-v0"
  "humanoidmaze-large-stitch-v0"
  "cube-single-play-v0"
  "cube-double-play-v0"
  "powderworld-easy-play-v0"
  "visual-antmaze-medium-navigate-v0"
  "visual-antmaze-medium-stitch-v0"
  "visual-antmaze-large-navigate-v0"
  "visual-antmaze-large-stitch-v0"
  "visual-humanoidmaze-medium-navigate-v0"
  "visual-humanoidmaze-medium-stitch-v0"
  "visual-humanoidmaze-large-navigate-v0"
  "visual-humanoidmaze-large-stitch-v0"
  "visual-cube-single-play-v0"
  "visual-cube-double-play-v0"
)

# cache autotuning results to not having to recompile them on every run
if [ -z ${BIGWORK+x}]; then
  export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"
else
  export JAX_COMPILATION_CACHE_DIR="${BIGWORK}/jax_cache"
fi

for environment in "${environments[@]}"; do
  for agent in "${agents[@]}"; do
    full_log_dir="${logdir}/${agent}_${environment}_${numconfigurations}c_${hyperparameters// /-}"

    python -m gcrl_landscapes.main setup --agent "$agent" --dataset "$environment" --n_configurations "$numconfigurations" --phases $phases --eval_steps $evalsteps --eval_episodes "$numevalepisodes" --hyperparameters $hyperparameters --logdir "$full_log_dir"
    python -m gcrl_landscapes.main submit --logdir "$full_log_dir" --n_seeds "$n_seeds" --tasks_per_node "$taskspernode" --jobname "${agent}-${environment}" --partition "$partitions" --min_per_mill_steps "${agent_min_per_mill_steps[${agent}]}"
  done
done
