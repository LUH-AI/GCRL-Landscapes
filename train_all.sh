#!/usr/bin/env bash

for algorithm in "CRL" "GCBC"; do
  for environment in "navigate" "explore" "stitch"; do
    ./setup_run.sh "$algorithm" "antmaze-medium-${environment}-v0" "discount actor_p_trajgoal" "./logs/${algorithm}_${environment}_32c_disc-acttraj"
    python -m gcrl_landscapes.main submit --logdir "./logs/${algorithm}_${environment}_32c_disc-acttraj" --phase 0 --n_seeds 5 --tasks_per_node 8 --jobname "${algorithm}_${environment}" --partition "ai,tnt,gpu"
  done
done
