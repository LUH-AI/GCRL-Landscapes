#!/usr/bin/env bash
# Cluster profile for PC2 (Paderborn Center for Parallel Computing)
#
# Usage: source configs/cluster/${CLUSTER:-luh}.sh
#
# Defines:
#   CLUSTER_PARTITION        — GPU compute partition (used by submission.py)
#   CLUSTER_SETUP_PARTITION  — CPU-only setup/orchestration partition
#   CLUSTER_RESERVATION      — Slurm reservation name (empty = no reservation)
#   CLUSTER_GRES             — GPU GRES spec (empty = use gpus_per_node=1)
#   CLUSTER_ZIP_CMD          — zip binary to use

export CLUSTER_PARTITION="gpu"
export CLUSTER_SETUP_PARTITION="normal"
export CLUSTER_RESERVATION=""
export CLUSTER_GRES="gpu:h100:1"
export CLUSTER_ZIP_CMD="zip"

cluster_load_env() {
    module load Miniforge3
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda deactivate 2>/dev/null || true
    conda activate gcrl
}

cluster_load_plot_env() {
    # Adjust module names for PC2 as needed
    module load GCC OpenMPI
}
