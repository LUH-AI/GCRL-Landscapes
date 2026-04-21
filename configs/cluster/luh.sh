#!/usr/bin/env bash
# Cluster profile for LUH (Leibniz University Hannover) — ai,tnt partitions
#
# Usage: source configs/cluster/${CLUSTER:-luh}.sh
#
# Defines:
#   CLUSTER_PARTITION        — GPU compute partition (used by submission.py)
#   CLUSTER_SETUP_PARTITION  — CPU-only setup/orchestration partition
#   CLUSTER_RESERVATION      — Slurm reservation name (empty = no reservation)
#   CLUSTER_GRES             — GPU GRES spec (empty = use gpus_per_node=1)
#   CLUSTER_ZIP_CMD          — zip binary to use

export CLUSTER_PARTITION="ai,tnt"
export CLUSTER_SETUP_PARTITION="ai,tnt"
export CLUSTER_RESERVATION="ai,tnt"
export CLUSTER_GRES=""
export CLUSTER_ZIP_CMD="zip"

cluster_load_env() {
    module load Miniforge3
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    conda deactivate 2>/dev/null || true
    conda activate gcrl
}

cluster_load_plot_env() {
    module load GCC/12.2.0 OpenMPI/4.1.4 Armadillo
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${BIGWORK}/usr/lib"
}
