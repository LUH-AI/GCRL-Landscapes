#!/bin/bash

#SBATCH --partition=ai,tnt
#SBATCH --job-name=HPO
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=m.toepperwien@ai.uni-hannover.de
#SBATCH --mail-type=END,FAIL
#SBATCH --get-user-env

# Load cluster-specific settings. CLUSTER env var is inherited from the submitting shell.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/configs/cluster/${CLUSTER:-luh}.sh"

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


cluster_load_env

# Build cluster-specific Hydra launcher overrides
launcher_overrides="hydra.launcher.partition='${CLUSTER_PARTITION}'"
if [ -n "$CLUSTER_GRES" ]; then
  launcher_overrides="${launcher_overrides} hydra.launcher.gres='${CLUSTER_GRES}'"
else
  launcher_overrides="${launcher_overrides} hydra.launcher.gpus_per_node=1"
fi
if [ -n "$CLUSTER_RESERVATION" ]; then
  launcher_overrides="${launcher_overrides} ++hydra.launcher.additional_parameters.reservation='${CLUSTER_RESERVATION}'"
fi

echo python -m gcrl_landscapes.hpo --multirun --config-name "hpo_$1" +logdir="$2" +datasets="$3" +phases="$4" +phase="$5" "$launcher_overrides"
python -m gcrl_landscapes.hpo --multirun --config-name "hpo_$1" +logdir="$2" +datasets="$3" +phases="$4" +phase="$5" $launcher_overrides
