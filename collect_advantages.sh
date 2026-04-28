#!/usr/bin/env bash
# Catalog checkpoints + compute advantage matrices.
# Usage:
#   sbatch [--partition=...] collect_advantages.sh [LOGDIR]
#
# LOGDIR defaults to ./logs. Override via positional arg or env:
#   LOGDIR=/path/to/logs sbatch collect_advantages.sh
#   sbatch collect_advantages.sh /path/to/logs
#
# Outputs:
#   ${LOGDIR}/checkpoints.csv
#   ${LOGDIR}/advantages.parquet
#   ${LOGDIR}/afr_metrics.csv

#SBATCH --job-name=CollectAdvantages
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --mail-user=m.toepperwien@ai.uni-hannover.de
#SBATCH --mail-type=END,FAIL
#SBATCH --get-user-env
#SBATCH --output=collect_advantages_%j.log

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/configs/cluster/${CLUSTER:-luh}.sh"
cluster_load_env

LOGDIR="${1:-${LOGDIR:-${SCRIPT_DIR}/logs}}"
CATALOG="${LOGDIR}/checkpoints.csv"
PARQUET="${LOGDIR}/advantages.parquet"
AFR_CSV="${LOGDIR}/afr_metrics.csv"

# --- Safety: chunk-size 4 avoids GCIVL OOM (32 GiB at chunk=32).
# --- Once per-agent chunk-size override is implemented, raise this to 32
# --- and pass --agent-chunk-size GCIVL=4 instead.
CHUNK_SIZE="${CHUNK_SIZE:-8}"
NUM_BATCHES="${NUM_BATCHES:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "=== collect_advantages.sh ==="
echo "LOGDIR       : ${LOGDIR}"
echo "CATALOG      : ${CATALOG}"
echo "PARQUET      : ${PARQUET}"
echo "AFR_CSV      : ${AFR_CSV}"
echo "CHUNK_SIZE   : ${CHUNK_SIZE}"
echo "NUM_BATCHES  : ${NUM_BATCHES}"
echo "NUM_WORKERS  : ${NUM_WORKERS}"
echo "Running on   : $(hostname)"
echo "Date         : $(date)"
echo ""

# Step 1: Catalog checkpoints (CPU)
echo "=== Step 1: catalog_checkpoints ==="
python analysis/catalog_checkpoints.py \
    --logdir "${LOGDIR}" \
    --output "${CATALOG}"

# Step 2: Generate advantage matrices (GPU)
echo ""
echo "=== Step 2: generate_advantages ==="
python analysis/generate_advantages.py \
    --catalog  "${CATALOG}" \
    --output   "${PARQUET}" \
    --chunk-size  "${CHUNK_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --num-batches "${NUM_BATCHES}"

# Step 3: Compute AFR metrics (CPU)
echo ""
echo "=== Step 3: compute_afr_metrics ==="
python analysis/compute_afr_metrics.py \
    --input  "${PARQUET}" \
    --output "${AFR_CSV}"

echo ""
echo "=== Done: $(date) ==="
