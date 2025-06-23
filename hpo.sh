#!/bin/bash

#SBATCH --partition=ai,tnt
#SBATCH --job-name=HPO
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=m.toepperwien@stud.uni-hannover.de
#SBATCH --mail-type=END,FAIL
#SBATCH --get-user-env

if [ -z "$1" ]
  then
    echo "No agent supplied"
fi

if [ -z "$2" ]
  then
    echo "No environment supplied"
fi

module load Miniforge3
conda activate gcrl

python -m gcrl_landscapes.hpo --multirun --config-name "hpo_$1" +env="$2" hydra.run.dir="./smac_log/crl/$2" hydra.sweep.dir="./smac_log/crl/$2"
