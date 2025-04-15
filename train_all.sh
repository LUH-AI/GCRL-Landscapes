#!/usr/bin/env bash

export JAX_COMPILATION_CACHE_DIR="${BIGWORK}/jax_cache"

for environment in "navigate" "stitch"; do
  ./setup_run.sh "CRL" "antmaze-medium-${environment}-v0" "discount actor_p_trajgoal" "./logs/CRL_${environment}_32c_disc-acttraj"
  ./setup_run.sh "GCBC" "antmaze-medium-${environment}-v0" "discount lr" "./logs/GCBC_${environment}_32c_disc-acttraj"
   python -m gcrl_landscapes.main submit --logdir "./logs/CRL_${environment}_32c_disc-acttraj" --n_seeds 3 --tasks_per_node 5 --jobname "CRL_${environment}" --partition "ai,tnt" --min_per_mill_steps 300
  python -m gcrl_landscapes.main submit --logdir "./logs/GCBC_${environment}_32c_disc-acttraj" --n_seeds 3 --tasks_per_node 5 --jobname "GCBC_${environment}" --partition "ai,tnt" --min_per_mill_steps 200
done
