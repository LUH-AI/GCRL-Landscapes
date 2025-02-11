#!/usr/bin/env bash

for algorithm in "CRL" "GCBC"; do
  for environment in "antmaze-medium-navigate-v0" "antmaze-medium-explore-v0" "antmaze-medium-stitch-v0"; do
    ./setup_run.sh "$algorithm" "$environment" "discount actor_p_trajgoal" "./logs/2025-02-11_${algorithm}_${environment}_32configurations_discount-actorptrajgoal-5seeds"
    python -m gcrl_landscapes.main submit --logdir "./logs/2025-02-11_${algorithm}_${environment}_32configurations_discount-actorptrajgoal-5seeds" --phase 0 --n_seeds 5 --tasks_per_node 5 --jobname "${algorithm}_${environment}_5seeds" --partition "ai,tnt,gpu"
  done
done
