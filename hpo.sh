#!/bin/bash

#SBATCH --partition=ai,tnt
#SBATCH --job-name=HPO
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=m.toepperwien@stud.uni-hannover.de
#SBATCH --mail-type=END,FAIL
#SBATCH --get-user-env

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

module load Miniforge3
conda activate gcrl

for phase in "${phases[@]}"; do
  if [ -z "$job_id" ]
    then
      output=$(sbatch --parsable --output "${logdir}/hpo_log_${phase}.txt" hpo_single.sh "$1" "$logdir" "$datasets_str" "$phases_str" "$phase")
    else
      output=$(sbatch --parsable --output "${logdir}/hpo_log_${phase}.txt" --dependency="afterok:${job_id}" hpo_single.sh "$1" "$logdir" "$datasets_str" "$phases_str" "$phase")
  fi
  env IFS=';' read -r job_id cluster_name <<< "$output"
  echo "Submitted job $job_id to cluster $cluster_name for phase $phase"
done
