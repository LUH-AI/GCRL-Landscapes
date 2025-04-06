#!/usr/bin/env bash

export JAX_COMPILATION_CACHE_DIR="${BIGWORK}/jax_cache"

for environment in "navigate" "explore" "stitch"; do
  ./setup_run.sh "$algorithm" "antmaze-medium-${environment}-v0" "discount actor_p_trajgoal" "./logs/CRL_${environment}_32c_disc-acttraj"
  ./setup_run.sh "$algorithm" "antmaze-medium-${environment}-v0" "discount actor_p_trajgoal" "./logs/GCBC_${environment}_32c_disc-acttraj"
  python -m gcrl_landscapes.main submit --logdir "./logs/CRL_${environment}_32c_disc-acttraj" --n_seeds 5 --tasks_per_node 5 --jobname "${algorithm}_${environment}" --partition "ai,tnt" --min_per_mill_steps 150
  python -m gcrl_landscapes.main submit --logdir "./logs/GCBC_${environment}_32c_disc-acttraj" --n_seeds 5 --tasks_per_node 5 --jobname "${algorithm}_${environment}" --partition "ai,tnt" --min_per_mill_steps 300
done
