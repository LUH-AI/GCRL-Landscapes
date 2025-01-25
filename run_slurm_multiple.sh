#!/usr/bin/env bash

for dataset in "antmaze-medium-navigate-v0" "antmaze-medium-explore-v0" "antmaze-medium-stitch-v0"; do
  for agent in "CRL" "GCBC" "QRL" "HIQL"; do
    sbatch run_slurm.sh "$agent" "$dataset"
  done
done
