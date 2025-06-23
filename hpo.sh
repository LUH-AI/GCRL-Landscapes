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

module load Miniforge3
conda activate gcrl

python -m gcrl_landscapes.hpo --multirun +env="antmaze-medium-navigate-v0"
