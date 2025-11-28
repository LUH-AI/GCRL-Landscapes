#!/bin/bash

#SBATCH --partition=ai,tnt
#SBATCH --job-name=ZipAndPlot
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3G
#SBATCH --mail-user=m.toepperwien@ai.uni-hannover.de
#SBATCH --mail-type=END,FAIL
#SBATCH --get-user-env

DATE=`date -u +%Y-%m-%d`
ZIPNAME="${DATE}-${1/\.\//}.zip"
/home/nhwptoem/bin/zip -r "$ZIPNAME" "$1" -x 'logs*/**/*.pkl' -x 'logs*/**/submitit/*'

module load GCC/12.2.0 OpenMPI/4.1.4 Armadillo
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${BIGWORK}/usr/lib"
$2 -m gcrl_landscapes.evaluation.plot --plot_eval_curves --plot_return_distributions --plot_gp_fits --zipfile "$ZIPNAME"
$2 -m gcrl_landscapes.evaluation.tabular --zipfiles "$ZIPNAME" --output_folder "tables/${ZIPNAME}"
