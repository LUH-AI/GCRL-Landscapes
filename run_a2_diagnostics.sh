#!/usr/bin/env bash
# A2 Step 1 — advantage diagnostics (FR-AUC / gap / MRR) on the finished
# GCIVL n-step trees. NO TRAINING: every checkpoint already exists.
#
# Answers 7VBe Q5: "evaluate methods known to improve value learning quality,
# such as n-step returns ... and verify whether improvements in value quality
# are consistently reflected in higher FR-AUC and MRR scores."
#
# One array element per (env, n) arm. The n=1 baselines are NOT recomputed —
# the published campaign's advantages.parquet / afr_metrics.csv under $REFROOT
# are the paper's own numbers and are reused verbatim, so the horizon
# comparison is against exactly what the paper reports.
#
#   sbatch --array=0-3 run_a2_diagnostics.sh
#
#SBATCH --job-name=A2diag
#SBATCH --partition=ai,tnt
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/bigwork/nhwpmoha/GCRL-Landscapes/logs-a2-diagnostics/a2_%A_%a.log

set -uo pipefail
cd /bigwork/nhwpmoha/GCRL-Landscapes || exit 1
mkdir -p logs-a2-diagnostics outputs-a2

PY=.venv/bin/python

# arm_name|logdir
declare -a -r ARMS=(
  "gcivl-n3-cube|logs-rb-gcivl-n3-cube-single"
  "gcivl-n5-cube|logs-rb-gcivl-n5-cube-single"
  "gcivl-n3-antmaze-large|logs-rb-gcivl-n3-antmaze-large"
  "gcivl-n5-antmaze-large|logs-rb-gcivl-n5-antmaze-large"
)

idx=${SLURM_ARRAY_TASK_ID:-0}
if [[ $idx -ge ${#ARMS[@]} ]]; then echo "index ${idx} beyond arm list"; exit 0; fi
IFS='|' read -r arm logdir <<<"${ARMS[$idx]}"

cat="outputs-a2/catalog_${arm}.csv"
adv="outputs-a2/advantages_${arm}.parquet"
afr="outputs-a2/afr_${arm}.csv"

echo "=== A2 diagnostics: ${arm} ==="
echo "node   : $(hostname)"
echo "logdir : ${logdir}"
echo

echo "--- [1/3] catalog checkpoints"
$PY analysis/catalog_checkpoints.py --logdir "$logdir" --output "$cat" || exit 1
wc -l "$cat"

echo "--- [2/3] cross-goal advantage matrices"
$PY analysis/generate_advantages.py --catalog "$cat" --output "$adv" || exit 1

echo "--- [3/3] FR-AUC / gap / MRR"
$PY analysis/compute_afr_metrics.py --input "$adv" --output "$afr" || exit 1

echo
echo "=== done: ${afr} ==="
head -3 "$afr"
