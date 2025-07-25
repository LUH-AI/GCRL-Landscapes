#!/bin/bash

#SBATCH --partition=ai,tnt
#SBATCH --job-name=HPO
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=m.toepperwien@stud.uni-hannover.de
#SBATCH --mail-type=END,FAIL
#SBATCH --get-user-env

phases_str=$(printf '%s,' "${phases[@]}")
phases_str="[${phases_str%?}]"
datasets_str=$(printf '%s,' "${datasets[@]}")
datasets_str="[${datasets_str%?}]"


if [ -z "$1" ]
  then
    echo "No agent supplied"
fi

if [ -z "$2" ]
  then
    echo "No logdir supplied"
fi

if [ -z "$3" ]
  then
    echo "No datasets_str supplied"
fi

if [ -z "$4" ]
  then
    echo "No phases_str supplied"
fi

if [ -z "$5" ]
  then
    echo "No phase supplied"
fi


module load Miniforge3
conda activate gcrl

echo python -m gcrl_landscapes.hpo --multirun --config-name "hpo_$1" +logdir="$2" +datasets="$3" +phases="$4" +phase="$5"
python -m gcrl_landscapes.hpo --multirun --config-name "hpo_$1" +logdir="$2" +datasets="$3" +phases="$4" +phase="$5"
