#!/bin/bash

#SBATCH --partition=ai,tnt
#SBATCH --job-name=AFRMetrics
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=m.toepperwien@ai.uni-hannover.de
#SBATCH --mail-type=END,FAIL
#SBATCH --get-user-env

# Load cluster-specific settings. CLUSTER env var is inherited from the submitting shell.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/configs/cluster/${CLUSTER:-luh}.sh"

#################################################################################
# AFR (Future-Random Advantage Separation) Analysis Job Script
#############################################################################
#
# Given a log directory (agent run_logs dir), auto-discovers the last phase with
# trained checkpoints (params_*.pkl) and runs AFR metrics across all seeds.
#
# Usage:
#   sbatch submit_afr.sh /path/to/run_logs
#   sbatch submit_afr.sh /path/to/run_logs --alpha 0.5 --n-batches 20
#   sbatch submit_afr.sh /path/to/run_logs --seed 3
#   sbatch submit_afr.sh /path/to/run_logs --output my_metrics.pkl
#   sbatch submit_afr.sh /path/to/run_logs --plots
#   sbatch submit_afr.sh /path/to/run_logs --no-plots
#
# Required input:
#   $1 : log dir (agent run_logs/..., e.g. .../CRL_antmaze-medium-navigate-v0/run_logs)
#
# Optional arguments:
#   --alpha <float>        : AFR temperature (default: 0.5)
#   --n-batches <int>      : Number of dataset batches per seed (default: 10)
#   --output <file>        : Output metrics filename (default: afr_metrics.pkl)
#   --plots                : Generate plots (default: true)
#   --no-plots             : Skip plots
#   --seed <int>           : Run only this seed (0..4)
#
################################################################################

set -euo pipefail

# ----------
# Parse arguments
# ----------
LOG_DIR=""
ALPHA=""
N_BATCHES=""
OUTPUT="afr_metrics.pkl"
GENERATE_PLOTS=true
SEED_SPEC=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --alpha) ALPHA="$2"; shift 2 ;;
        --alpha=*) ALPHA="${1#*=}"; shift ;;
        --n-batches) N_BATCHES="$2"; shift 2 ;;
        --n-batches=*) N_BATCHES="${1#*=}"; shift ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --output=*) OUTPUT="${1#*=}"; shift ;;
        --seed) SEED_SPEC="$2"; shift 2 ;;
        --seed=*) SEED_SPEC="${1#*=}"; shift ;;
        --plots) GENERATE_PLOTS=true; shift ;;
        --no-plots) GENERATE_PLOTS=false; shift ;;
        -*) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
        *)
            if [[ -z "$LOG_DIR" ]]; then
                LOG_DIR="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$LOG_DIR" ]]; then
    echo "Usage: sbatch $0 <log_dir> [--alpha F] [--n-batches N] [--output FILE] [--seed INT] [--plots/--no-plots]"
    exit 1
fi

if [[ ! -d "$LOG_DIR" ]]; then
    echo "[ERROR] Log directory does not exist: $LOG_DIR"
    exit 1
fi

# ----------
# Auto-discover: find the last phase with trained checkpoints
# ----------
PHASE_DIR=""
LATEST_PHASE=""

for phase_dir in "$LOG_DIR"/phase_*; do
    if [[ -d "$phase_dir" ]]; then
        pkl_count=$(find "$phase_dir" -name "params_*.pkl" 2>/dev/null | wc -l)
        if [[ "$pkl_count" -gt 0 ]]; then
            LATEST_PHASE="$phase_dir"
        fi
    fi
done

if [[ -z "$LATEST_PHASE" ]]; then
    echo "[ERROR] No phases with trained checkpoints found in $LOG_DIR"
    exit 1
fi

PHASE_DIR="$LATEST_PHASE"
PHASE_NAME=$(basename "$PHASE_DIR")

# ----------
# Cluster setup
# ----------
cluster_load_env

# ----------
# Setup output directory
# ----------
DATE=$(date -u +%Y%m%d_%H%M%S)
OUTPUT_DIR="$SCRIPT_DIR/afr_results_${DATE}_${PHASE_NAME}"
mkdir -p "$OUTPUT_DIR"

# ----------
# Print summary
# ----------
echo ""
echo "===================================="
echo "  AFR Metrics Computation"
echo "===================================="
echo "  Log dir:        $LOG_DIR"
echo "  Phase:          $PHASE_DIR"
echo "  Alpha:          ${ALPHA:-not used}"
echo "  N batches:      ${N_BATCHES:-10}"
echo "  Plots:          ${GENERATE_PLOTS}"
echo "  Output dir:     $OUTPUT_DIR"
echo "===================================="
echo ""

# ----------
# Run AFR metrics computation
# ----------
echo "--- Running AFR metrics ---"

CMD_ARGS=("--phase-dir" "$PHASE_DIR")
if [[ -n "$ALPHA" ]]; then
    CMD_ARGS+=("--alpha" "$ALPHA")
fi
if [[ -n "${N_BATCHES:-}" ]]; then
    CMD_ARGS+=("--n-batches" "$N_BATCHES")
fi
if [[ -n "${SEED_SPEC:-}" ]]; then
    CMD_ARGS+=("--seed" "$SEED_SPEC")
fi

METRICS_PATH="$OUTPUT_DIR/$OUTPUT"
CMD_ARGS+=("--output" "$METRICS_PATH")

if python "$SCRIPT_DIR/scripts/afr_metrics.py" "${CMD_ARGS[@]}"; then
    echo "Metrics saved to: $METRICS_PATH"
else
    echo "[ERROR] AFR metrics computation failed!"
    exit 1
fi

# ----------
# Generate plots if requested
# ----------
if [[ "$GENERATE_PLOTS" == "true" ]]; then
    echo ""
    echo "--- Generating AFR plots ---"
    PLOTS_DIR="$OUTPUT_DIR/plots"
    mkdir -p "$PLOTS_DIR"

    if python "$SCRIPT_DIR/analysis/advantages/f_metrics_afr.py" \
        --input "$METRICS_PATH" \
        --output-dir "$PLOTS_DIR"; then
        echo "Plots saved to: $PLOTS_DIR"
    else
        echo "[WARN] Plot generation failed"
    fi
fi

# ----------
# Summary
# ----------
echo ""
echo "===================================="
echo "  AFR Metrics Computation Complete"
echo "===================================="
echo "  Output dir:     $OUTPUT_DIR"
echo ""
ls -la "$OUTPUT_DIR"
echo "===================================="
echo ""
