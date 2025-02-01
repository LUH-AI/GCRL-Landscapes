#!/usr/bin/env bash

for dataset in "antmaze-large-navigate-v0" "antmaze-large-explore-v0" "antmaze-large-stitch-v0"; do
  for agent in "CRL" "GCBC" "QRL" "HIQL"; do
    sbatch run_slurm.sh "$agent" "$dataset"
  done
done
