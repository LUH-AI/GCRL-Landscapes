#!/bin/bash

logdir="./logs_hpo"
declare -a -r datasets=(
  antmaze-medium-explore-v0
  antmaze-medium-explore80navigate-v0
  antmaze-medium-explore40navigate-v0
  antmaze-medium-navigate-v0
)
declare -a -r phases=(
  25
  50
  75
  100
)
phases_str=$(printf '%s,' "${phases[@]}")
phases_str="[${phases_str%?}]"
datasets_str=$(printf '%s,' "${datasets[@]}")
datasets_str="[${datasets_str%?}]"


if [ -z "$1" ]
  then
    echo "No agent supplied"
fi

for phase in "${phases[@]}"; do
  if [ -z "$job_id" ]
    then
      output=$(sbatch --parsable --output "${logdir}/hpo_log_$1_${phase}.txt" hpo_single.sh "$1" "$logdir" "$datasets_str" "$phases_str" "$phase")
    else
      output=$(sbatch --parsable --output "${logdir}/hpo_log_$1_${phase}.txt" --dependency="afterok:${job_id}" hpo_single.sh "$1" "$logdir" "$datasets_str" "$phases_str" "$phase")
  fi
  if [[ "$output" == *";"* ]]; then
    IFS=';' read -r job_id cluster_name <<< "$output"
  else
    job_id="$output"
    cluster_name="unknown"
  fi
  echo "Submitted job $job_id to cluster $cluster_name for phase $phase"
done
