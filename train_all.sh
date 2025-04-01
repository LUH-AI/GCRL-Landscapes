#!/usr/bin/env bash

export JAX_COMPILATION_CACHE_DIR="${BIGWORK}/jax_cache"

for algorithm in "CRL" "GCBC"; do
  for environment in "navigate" "explore" "stitch"; do
    ./setup_run.sh "$algorithm" "antmaze-medium-${environment}-v0" "discount actor_p_trajgoal" "./logs/${algorithm}_${environment}_32c_disc-acttraj"
    python -m gcrl_landscapes.main submit --logdir "./logs/${algorithm}_${environment}_32c_disc-acttraj" --n_seeds 3 --tasks_per_node 4 --jobname "${algorithm}_${environment}" --partition "ai,tnt"
  done
done
