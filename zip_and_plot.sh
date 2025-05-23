#!/bin/bash

#SBATCH --partition=ai,tnt
#SBATCH --job-name=ZipAndPlot
#SBATCH --output=ZipAndPlot.out
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=m.toepperwien@stud.uni-hannover.de
#SBATCH --mail-type=END,FAIL

ZIPNAME=`date -u +%Y-%m-%dT%H:%M:%S%Z`.zip
zip -r "$ZIPNAME" "$1" -x 'logs*/**/*.pkl' -x 'logs*/**/submitit/*'

module load Miniforge3
python -m gcrl_landscapes.evaluation.plot --zipfile "$ZIPNAME"
python -m gcrl_landscapes.evaluation.tabular --zipfile "$ZIPNAME"
