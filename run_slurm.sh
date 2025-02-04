#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH -J GCRL
#SBATCH --mem=32G
#SBATCH -o slurm_outputs/slurm-%j.out
#SBATCH --partition=ai
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --mail-user=m.toepperwien@stud.uni-hannover.de
#SBATCH --mail-type=FAIL,END

if [ -z "$1" ]
  then
    echo "Agent empty"
    exit 1
fi
if [ -z "$2" ]
  then
    echo "Dataset empty"
    exit 1
fi
if [ -z "$3" ]
  then
    echo "no hyperparameters given"
    exit 1
fi

cd $SLURM_SUBMIT_DIR

module load Miniforge3

source /home/nhwptopm/.bashrc

conda activate /bigwork/nhwptopm/.conda/envs/gcrl

export WANDB_MODE=offline

./train.sh "$1" "$2" "$3"
