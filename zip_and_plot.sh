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

# Load cluster-specific settings. CLUSTER env var is inherited from the submitting shell.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/configs/cluster/${CLUSTER:-luh}.sh"

DATE=$(date -u +%Y-%m-%d)
ZIPNAME="${DATE}-${1/\.\//}.zip"
"$CLUSTER_ZIP_CMD" -r "$ZIPNAME" "$1" -x 'logs*/**/*.pkl' -x 'logs*/**/submitit/*'

cluster_load_plot_env
$2 -m gcrl_landscapes.evaluation.plot --plot_eval_curves --plot_return_distributions --plot_gp_fits --zipfile "$ZIPNAME"
$2 -m gcrl_landscapes.evaluation.tabular --zipfiles "$ZIPNAME" --output_folder "tables/${ZIPNAME}"
