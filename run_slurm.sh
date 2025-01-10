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
cd $SLURM_SUBMIT_DIR

module load Miniforge3

source /home/nhwptopm/.bashrc

conda activate /bigwork/nhwptopm/.conda/envs/gcrl

export WANDB_MODE=offline

bash /bigwork/nhwptopm/GCRL-Landscapes/train.sh > log.txt
